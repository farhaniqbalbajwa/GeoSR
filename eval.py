import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from geosr.data import OLI2MSIDataset, PairedFolderDataset, ProbaVSingleOrMultiDataset, SEN2VENUSDataset
from geosr.metrics import ergas, psnr, sam, ssim
from geosr.model import GeoSRFM
from geosr.utils import rank0_print


def build_dataset(cfg):
    dsname = cfg["data"]["dataset"]
    root = cfg["data"]["root"]
    split = cfg["data"].get("split", "test")

    if dsname == "sen2venus":
        band_group = cfg["data"].get("band_group", "10m")
        return SEN2VENUSDataset(root=root, split=split, band_group=band_group, random_flip=False, random_rot90=False)
    if dsname == "oli2msi":
        return OLI2MSIDataset(root=root, split=split, random_flip=False, random_rot90=False)
    if dsname == "probav":
        band = cfg["data"].get("band", "NIR")
        num_frames = int(cfg["data"].get("num_frames", 1))
        return ProbaVSingleOrMultiDataset(root=root, split=split, band=band, num_frames=num_frames)
    if dsname == "paired_folder":
        scale = int(cfg["task"].get("scale", 4))
        return PairedFolderDataset(root=root, split=split, scale=scale, random_crop=False, patch_size_lr=128,
                                   random_flip=False, random_rot90=False)
    raise ValueError(dsname)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale = int(cfg["task"].get("scale", 4))

    model = GeoSRFM(
        in_channels=int(cfg["task"]["in_channels"]),
        out_channels=int(cfg["task"]["out_channels"]),
        scale=scale,
        dim=int(cfg["model"]["dim"]),
        depth=int(cfg["model"]["depth"]),
        num_heads=int(cfg["model"]["num_heads"]),
        window_size=int(cfg["model"]["window_size"]),
        mlp_ratio=float(cfg["model"]["mlp_ratio"]),
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    ds = build_dataset(cfg)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    psnr_m, ssim_m, sam_m, ergas_m = [], [], [], []
    for batch in tqdm(dl):
        lr = batch["lr"].to(device)
        if lr.ndim == 5:
            # PROBA-V multi-frame returns [B,T,1,H,W] — pick first for this single-image model
            lr = lr[:, 0]
        hr = batch["hr"].to(device)

        eff_scale = int(batch.get("scale", torch.tensor(scale)).max().item())
        sr = model(lr, mode="sr")

        psnr_m.append(psnr(sr, hr).cpu())
        ssim_m.append(ssim(sr, hr).cpu())
        sam_m.append(sam(sr, hr).cpu())
        ergas_m.append(ergas(sr, hr, scale=eff_scale).cpu())

    psnr_v = torch.cat(psnr_m).mean().item()
    ssim_v = torch.cat(ssim_m).mean().item()
    sam_v = torch.cat(sam_m).mean().item()
    ergas_v = torch.cat(ergas_m).mean().item()

    rank0_print(f"PSNR:  {psnr_v:.3f}")
    rank0_print(f"SSIM:  {ssim_v:.4f}")
    rank0_print(f"SAM:   {sam_v:.4f} rad")
    rank0_print(f"ERGAS: {ergas_v:.3f}")


if __name__ == "__main__":
    main()
