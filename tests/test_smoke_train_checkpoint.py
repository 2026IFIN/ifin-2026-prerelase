from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import torch
from torch import optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ifin import IFINNet, RB
from utils.checkpoint import load_checkpoint, load_model_state_compat, save_checkpoint
from utils.seed import seed_everything


def test_one_step_train_then_save_load() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(2026, deterministic=True)

    generator = torch.Generator(device="cpu").manual_seed(2026)
    psf = torch.rand(1, 1, 16, 16, generator=generator).to(device)

    model = IFINNet(
        in_channels=3,
        out_channels=3,
        psf=psf,
        height=32,
        width=32,
        dim=8,
        depth=2,
        block_cls=RB,
        k=1,
        repeat=True,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    meas_input = torch.rand(2, 3, 32, 32, generator=generator).to(device)
    img_target = torch.rand(2, 3, 32, 32, generator=generator).to(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    img_recon, meas_recon, iso_recon = model(meas_input)
    loss = criterion(img_recon, img_target) + 0.1 * criterion(meas_recon, meas_input)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss).item()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "smoke_ckpt.pth"
        save_checkpoint(
            ckpt_path,
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step_loss": float(loss.detach().cpu()),
            },
        )

        loaded = load_checkpoint(ckpt_path, map_location=device)

        reloaded_model = IFINNet(
            in_channels=3,
            out_channels=3,
            psf=psf,
            height=32,
            width=32,
            dim=8,
            depth=2,
            block_cls=RB,
            k=1,
            repeat=True,
        ).to(device)
        load_model_state_compat(reloaded_model, loaded, strict=True)

        reloaded_optimizer = optim.AdamW(reloaded_model.parameters(), lr=1e-4)
        reloaded_optimizer.load_state_dict(loaded["optimizer_state_dict"])

        reloaded_model.eval()
        with torch.no_grad():
            out_after_reload = reloaded_model(meas_input)

        for tensor in out_after_reload:
            assert tensor.shape == (2, 3, 32, 32)
            assert torch.isfinite(tensor).all().item()
