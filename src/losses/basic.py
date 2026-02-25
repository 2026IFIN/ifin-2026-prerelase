from __future__ import annotations

import torch
from torch import nn


class ReconstructionLoss(nn.Module):
    def __init__(self, use_lpips: bool = True, lpips_weight: float = 0.05) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.use_lpips = use_lpips
        self.lpips_weight = lpips_weight
        self.lpips = None
        if self.use_lpips:
            try:
                import lpips  # type: ignore

                self.lpips = lpips.LPIPS(net="vgg")
            except Exception:
                self.lpips = None

    def forward(
        self,
        img_recon: torch.Tensor,
        img_target: torch.Tensor,
        meas_recon: torch.Tensor | None = None,
        meas_input: torch.Tensor | None = None,
        train: bool = True,
        normalize: bool = True,
    ) -> torch.Tensor:
        loss = self.mse(img_recon, img_target)
        if self.lpips is not None:
            lpips_loss = torch.mean(self.lpips(img_recon, img_target, normalize=normalize))
            loss = loss + self.lpips_weight * lpips_loss
        return loss
