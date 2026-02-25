from __future__ import annotations

import torch


def psnr(img_pred: torch.Tensor, img_target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mse = torch.mean((img_pred - img_target) ** 2)
    return 10.0 * torch.log10(1.0 / (mse + eps))
