from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def validate_waller_paths(waller_path: str, psf_path: str | None = None, train: bool | None = None) -> None:
    dataset_root = Path(waller_path)
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Waller dataset root not found: {dataset_root}. "
            "Please set data.waller_path to your WallerLabDataset directory."
        )

    required_dirs = ["diffuser_images", "ground_truth_lensed"]
    for dirname in required_dirs:
        dir_path = dataset_root / dirname
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(
                f"Missing required directory: {dir_path}. "
                "Expected WallerLabDataset layout with diffuser_images/ and ground_truth_lensed/."
            )

    csv_candidates = []
    if train is True:
        csv_candidates = [dataset_root / "dataset_train.csv"]
    elif train is False:
        csv_candidates = [dataset_root / "dataset_test.csv"]
    else:
        csv_candidates = [dataset_root / "dataset_train.csv", dataset_root / "dataset_test.csv"]

    if not any(path.exists() for path in csv_candidates):
        names = ", ".join(path.name for path in csv_candidates)
        raise FileNotFoundError(
            f"Missing required CSV file(s): {names} in {dataset_root}."
        )

    if psf_path is not None:
        psf_file = Path(psf_path)
        if not psf_file.exists():
            raise FileNotFoundError(
                f"PSF file not found: {psf_file}. "
                "Please set data.psf_path to your dataset psf.tiff."
            )


class WallerDataset(Dataset):
    def __init__(self, path: str, train: bool = False, transform_raw=None, transform_lab=None):
        self.path = Path(path)
        self.transform_raw = transform_raw
        self.transform_lab = transform_lab
        csv_name = "dataset_train.csv" if train else "dataset_test.csv"
        self.df = pd.read_csv(self.path / csv_name)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        filename = self.df.iloc[idx, 0]
        raw_path = str((self.path / "diffuser_images" / filename)).replace(".jpg.tiff", ".npy")
        lab_path = str((self.path / "ground_truth_lensed" / filename)).replace(".jpg.tiff", ".npy")

        raw = np.load(raw_path)
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        lab = np.load(lab_path)
        lab = cv2.cvtColor(lab, cv2.COLOR_BGR2RGB)

        if self.transform_raw is not None:
            raw = self.transform_raw(raw)
        if self.transform_lab is not None:
            lab = self.transform_lab(lab)

        return raw, lab
