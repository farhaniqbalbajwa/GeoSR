import argparse
import os
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import yaml

from geosr.data import (
    OLI2MSIDataset,
    PairedFolderDataset,
    ProbaVSingleOrMultiDataset,
    SEN2VENUSDataset,
)
from geosr.metrics import ergas, sam
from geosr.model import GeoSRFM
from geosr.utils import (
    AverageMeter,
    get_rank,
    get_world_size,
    is_dist,
    rank0_print,
    save_checkpoint,
    set_seed,
)


def build_dataset(cfg: Dict):
    dsname = cfg["data"]["dataset"]
    root = cfg["data"]["root"]
    split = cfg["data"].get("split", "train")

    if dsname == "sen2venus":
        band_group = cfg["data"].get("band_group", "10m")
        return SEN2VENUSDataset(root=root, split=split, band_group=band_group)
    if dsname == "oli2msi":
        return OLI2MSIDataset(root=root, split=split)
    if dsname == "probav":
        band = cfg["data"].get("band", "NIR")
        num_frames = int(cfg["data"].get("num_frames", 1))
        return ProbaVSingleOrMultiDataset(root=root, split=split, band=band, num_frames=num_frames)
    if dsname == "paired_folder":
        scale = int(cfg["task"].get("scale", 4))
        return PairedFolderDataset(
            root=root,
            split=split,
            scale=scale,
            random_crop=bool(cfg["data"].get("random_crop", True)),
            patch_size_lr=int(cfg["data"].get("patch_size_lr", 128)),
            random_flip=bool(cfg["data"].get("random_flip", True)),
            random_rot90=bool(cfg["data"].get("random_rot90", True)),
        )
    raise ValueError(f"Unknown dataset: {dsname}")


def init_dist(cfg: Dict):
    if not cfg["dist"]["enabled"]:
        return
    torch.distributed.init_process_group(backend=cfg["dist"].get("backend", "nccl"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)


def lr_schedule(step: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    return base_lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--outdir", default="runs/geosr_fm")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    init_dist(cfg)
    set_seed(int(cfg.get("seed", 1337) + get_rank()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = cfg["task"]["mode"]
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
        dropout=float(cfg["model"].get("dropout", 0.0)),
        mae_mask_ratio=float(cfg.get("mae", {}).get("mask_ratio", 0.6)),
    ).to(device)

    if is_dist():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=False)

    ds = build_dataset(cfg)
    sampler = DistributedSampler(ds, shuffle=True) if is_dist() else None
    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(cfg["data"].get("num_workers", 8)),
        pin_memory=True,
        drop_last=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"].get("amp", True)))

    start_epoch = 0
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        (model.module if hasattr(model, "module") else model).load_state_dict(ckpt["model"], strict=True)
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0) + 1)
        global_step = int(ckpt.get("global_step", 0))
        rank0_print(f"Resumed from {args.resume} at epoch {start_epoch}")

    os.makedirs(args.outdir, exist_ok=True)

    for epoch in range(start_epoch, int(cfg["train"]["epochs"])):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(dl, disable=(get_rank() != 0))
        for batch in pbar:
            lr_now = lr_schedule(global_step, float(cfg["train"]["lr"]), int(cfg["train"]["warmup_steps"]))
            for g in opt.param_groups:
                g["lr"] = lr_now

            opt.zero_grad(set_to_none=True)

            if mode == "sr":
                lr = batch["lr"].to(device)    # [B,C,H,W]
                hr = batch["hr"].to(device)
                eff_scale = int(batch.get("scale", torch.tensor(scale)).max().item())

                with torch.cuda.amp.autocast(enabled=bool(cfg["train"].get("amp", True))):
                    sr = model(lr, mode="sr")

                    l1_w = float(cfg["loss"].get("l1", 1.0))
                    charbon_w = float(cfg["loss"].get("charbonnier", 0.0))
                    sam_w = float(cfg["loss"].get("sam", 0.0))
                    ergas_w = float(cfg["loss"].get("ergas", 0.0))

                    loss = 0.0
                    if l1_w > 0:
                        loss = loss + l1_w * F.l1_loss(sr, hr)
                    if charbon_w > 0:
                        eps = 1e-3
                        loss = loss + charbon_w * torch.mean(torch.sqrt((sr - hr) ** 2 + eps**2))
                    if sam_w > 0:
                        loss = loss + sam_w * sam(sr, hr).mean()
                    if ergas_w > 0:
                        loss = loss + ergas_w * ergas(sr, hr, scale=eff_scale).mean()

            elif mode == "mae":
                # expects batch dict with "hr" or "lr" treated as x
                x = (batch.get("hr", None) if "hr" in batch else batch["lr"]).to(device)
                with torch.cuda.amp.autocast(enabled=bool(cfg["train"].get("amp", True))):
                    pred, mask = (model.module if hasattr(model, "module") else model).forward_mae(x)
                    # reconstruct only masked regions
                    loss = torch.mean(((pred - x) ** 2) * mask)

            else:
                raise ValueError(f"Unknown mode: {mode}")

            scaler.scale(loss).backward()
            if float(cfg["train"].get("grad_clip_norm", 0.0)) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip_norm"]))
            scaler.step(opt)
            scaler.update()

            loss_meter.update(float(loss.item()))
            global_step += 1

            if get_rank() == 0:
                pbar.set_description(f"epoch {epoch} loss {loss_meter.avg:.4f} lr {lr_now:.2e}")

        if get_rank() == 0 and ((epoch + 1) % int(cfg["train"]["save_every"]) == 0):
            payload = {
                "model": (model.module if hasattr(model, "module") else model).state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": cfg,
            }
            save_checkpoint(os.path.join(args.outdir, f"ckpt_epoch_{epoch:04d}.pt"), payload)

    rank0_print("Training complete.")


if __name__ == "__main__":
    main()
