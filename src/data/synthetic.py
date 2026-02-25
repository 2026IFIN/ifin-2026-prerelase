from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticLenslessSpec:
    num_samples: int
    channels: int
    height: int
    width: int
    seed: int


class SyntheticLenslessDataset(Dataset):
    def __init__(self, spec: SyntheticLenslessSpec) -> None:
        self.spec = spec
        generator = torch.Generator().manual_seed(spec.seed)
        self.measurements = torch.rand(
            spec.num_samples,
            spec.channels,
            spec.height,
            spec.width,
            generator=generator,
        )
        self.targets = torch.rand(
            spec.num_samples,
            spec.channels,
            spec.height,
            spec.width,
            generator=generator,
        )

    def __len__(self) -> int:
        return self.spec.num_samples

    def __getitem__(self, index: int):
        return self.measurements[index], self.targets[index]
