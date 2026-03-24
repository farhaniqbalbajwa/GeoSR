import csv
import glob
import os
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import cv2

try:
    import rasterio
    _HAS_RASTERIO = True
except Exception:
    _HAS_RASTERIO = False

try:
    import tifffile
    _HAS_TIFFFILE = True
except Exception:
    _HAS_TIFFFILE = False


def _read_png16(path: str) -> torch.Tensor:
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 2:
        im = im[..., None]
    im = im.astype(np.float32)
    # keep raw range; caller normalizes
    im = np.transpose(im, (2, 0, 1))  # C,H,W
    return torch.from_numpy(im)


def _read_geotiff(path: str) -> torch.Tensor:
    if _HAS_RASTERIO:
        with rasterio.open(path) as src:
            arr = src.read()  # [C,H,W]
        return torch.from_numpy(arr.astype(np.float32))
    if _HAS_TIFFFILE:
        arr = tifffile.imread(path)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3:
            # tifffile often gives H,W,C; heuristic:
            if arr.shape[0] < 16 and arr.shape[0] <= arr.shape[-1]:
                pass
            else:
                arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr.astype(np.float32))
    raise RuntimeError("Need rasterio or tifffile to read GeoTIFF.")


def _random_crop_pair(lr: torch.Tensor, hr: torch.Tensor, lr_size: int, scale: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # lr: [C, h, w], hr: [C, H, W]
    c, h, w = lr.shape
    if h < lr_size or w < lr_size:
        raise ValueError("LR patch smaller than crop size.")
    top = torch.randint(0, h - lr_size + 1, (1,)).item()
    left = torch.randint(0, w - lr_size + 1, (1,)).item()
    lr_crop = lr[:, top:top+lr_size, left:left+lr_size]
    hr_top, hr_left = top * scale, left * scale
    hr_crop = hr[:, hr_top:hr_top+lr_size*scale, hr_left:hr_left+lr_size*scale]
    return lr_crop, hr_crop


def _augment(lr: torch.Tensor, hr: torch.Tensor, flip: bool = True, rot90: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    if flip and torch.rand(()) < 0.5:
        lr = torch.flip(lr, dims=[2])
        hr = torch.flip(hr, dims=[2])
    if flip and torch.rand(()) < 0.5:
        lr = torch.flip(lr, dims=[1])
        hr = torch.flip(hr, dims=[1])
    if rot90:
        k = int(torch.randint(0, 4, (1,)).item())
        lr = torch.rot90(lr, k, dims=[1, 2])
        hr = torch.rot90(hr, k, dims=[1, 2])
    return lr, hr


class PairedFolderDataset(Dataset):
    """
    Generic paired dataset:
      root/
        train/
          lr/*.tif or *.png or *.npy
          hr/*.tif or *.png or *.npy
    """
    def __init__(self, root: str, split: str, scale: int, random_crop: bool = True, patch_size_lr: int = 128,
                 random_flip: bool = True, random_rot90: bool = True) -> None:
        super().__init__()
        self.scale = scale
        self.random_crop = random_crop
        self.patch_size_lr = patch_size_lr
        self.random_flip = random_flip
        self.random_rot90 = random_rot90

        lr_dir = os.path.join(root, split, "lr")
        hr_dir = os.path.join(root, split, "hr")
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*")))
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*")))
        assert len(self.lr_paths) == len(self.hr_paths), "LR/HR counts mismatch."

    def _read_any(self, path: str) -> torch.Tensor:
        ext = os.path.splitext(path)[1].lower()
        if ext in [".tif", ".tiff"]:
            return _read_geotiff(path)
        if ext in [".png"]:
            return _read_png16(path)
        if ext in [".npy"]:
            arr = np.load(path)
            if arr.ndim == 2:
                arr = arr[None, ...]
            return torch.from_numpy(arr.astype(np.float32))
        raise ValueError(f"Unsupported file: {path}")

    def __len__(self) -> int:
        return len(self.lr_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        lr = self._read_any(self.lr_paths[idx])
        hr = self._read_any(self.hr_paths[idx])
        # normalize to [0,1] if uint-like:
        if lr.max() > 1.5:
            lr = lr / 10000.0 if lr.max() > 1000 else lr / 65535.0
        if hr.max() > 1.5:
            hr = hr / 10000.0 if hr.max() > 1000 else hr / 65535.0

        if self.random_crop:
            lr, hr = _random_crop_pair(lr, hr, self.patch_size_lr, self.scale)
        lr, hr = _augment(lr, hr, self.random_flip, self.random_rot90)
        return {"lr": lr, "hr": hr}


class SEN2VENUSDataset(Dataset):
    """
    SEN2VENµS loader.
    Uses index.csv files per site; tensor files are .pt with shape [n,c,w,h]
    and require /10000 scaling back to reflectance floats.
    """
    def __init__(self, root: str, split: str, band_group: str = "10m", random_flip: bool = True, random_rot90: bool = True) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.random_flip = random_flip
        self.random_rot90 = random_rot90
        assert band_group in ["10m", "20m"]
        self.band_group = band_group

        # Build list of (tensor_lr_path, tensor_hr_path, patch_idx)
        self.samples: List[Tuple[str, str, int]] = []

        site_dirs = sorted([p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)])
        for site in site_dirs:
            index_path = os.path.join(site, "index.csv")
            if not os.path.exists(index_path):
                continue
            with open(index_path, "r", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    if band_group == "10m":
                        lr_name = row["tensor_10m_b2b3b4b8"]
                        hr_name = row["tensor_05m_b2b3b4b8"]
                    else:
                        lr_name = row["tensor_20m_b5b6b7b8a"]
                        hr_name = row["tensor_05m_b5b6b7b8a"]
                    lr_path = os.path.join(site, lr_name)
                    hr_path = os.path.join(site, hr_name)
                    nb = int(row["nb_patches"])
                    for i in range(nb):
                        self.samples.append((lr_path, hr_path, i))

        # Scale factor depends on group in this dataset
        self.scale = 2 if band_group == "10m" else 4

    @staticmethod
    @lru_cache(maxsize=16)
    def _load_pt(path: str) -> torch.Tensor:
        return torch.load(path, map_location="cpu")  # [n,c,w,h] int16-ish

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        lr_path, hr_path, i = self.samples[idx]
        lr_all = self._load_pt(lr_path).float() / 10000.0
        hr_all = self._load_pt(hr_path).float() / 10000.0
        lr = lr_all[i]  # [c,h,w]
        hr = hr_all[i]
        lr, hr = _augment(lr, hr, self.random_flip, self.random_rot90)
        return {"lr": lr, "hr": hr, "scale": torch.tensor(self.scale, dtype=torch.int32)}


class OLI2MSIDataset(Dataset):
    """
    OLI2MSI: train_lr/train_hr and test_lr/test_hr directories with GeoTIFF patches.
    """
    def __init__(self, root: str, split: str, random_flip: bool = True, random_rot90: bool = True) -> None:
        super().__init__()
        assert split in ["train", "test"]
        self.random_flip = random_flip
        self.random_rot90 = random_rot90
        self.scale = 3  # per dataset description

        lr_dir = os.path.join(root, f"{split}_lr")
        hr_dir = os.path.join(root, f"{split}_hr")
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.tif*")))
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.tif*")))
        assert len(self.lr_paths) == len(self.hr_paths)

    def __len__(self) -> int:
        return len(self.lr_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        lr = _read_geotiff(self.lr_paths[idx])
        hr = _read_geotiff(self.hr_paths[idx])
        # Normalize heuristics: many EO GeoTIFF are uint16 reflectance-scaled
        if lr.max() > 1.5:
            lr = lr / 10000.0 if lr.max() > 1000 else lr / 65535.0
        if hr.max() > 1.5:
            hr = hr / 10000.0 if hr.max() > 1000 else hr / 65535.0
        lr, hr = _augment(lr, hr, self.random_flip, self.random_rot90)
        return {"lr": lr, "hr": hr, "scale": torch.tensor(self.scale, dtype=torch.int32)}


class ProbaVSingleOrMultiDataset(Dataset):
    """
    PROBA-V dataset: scenes with HR.png, LRXXX.png, masks.
    This loader supports:
      - single-image SR: select one LR frame
      - multi-image SR: stack T LR frames as [T,1,H,W] (model needs adaptation; provided for completeness)
    """
    def __init__(self, root: str, split: str, band: str = "NIR", num_frames: int = 1) -> None:
        super().__init__()
        assert split in ["train", "test"]
        assert band in ["NIR", "RED"]
        self.num_frames = num_frames
        self.band = band
        self.scale = 3  # 128 -> 384 (per dataset description)

        base = os.path.join(root, split, band)
        self.scene_dirs = sorted([p for p in glob.glob(os.path.join(base, "imgset*")) if os.path.isdir(p)])

    def __len__(self) -> int:
        return len(self.scene_dirs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scene = self.scene_dirs[idx]
        lr_paths = sorted(glob.glob(os.path.join(scene, "LR*.png")))
        if self.num_frames > 1:
            chosen = lr_paths[: self.num_frames]
        else:
            chosen = [lr_paths[0]]

        lr_list = [_read_png16(p)[0:1] for p in chosen]  # force [1,H,W]
        lr = torch.stack(lr_list, dim=0)  # [T,1,128,128] (raw range)
        lr = lr / 65535.0  # normalize roughly

        out = {"lr": lr, "scale": torch.tensor(self.scale, dtype=torch.int32)}

        hr_path = os.path.join(scene, "HR.png")
        if os.path.exists(hr_path):
            hr = _read_png16(hr_path)[0:1] / 65535.0
            out["hr"] = hr
        return out
