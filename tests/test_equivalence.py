from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ifin import IFINNet, RB
from utils.seed import seed_everything


def _build_inputs(device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(123)
    psf = torch.rand(1, 1, 16, 16, generator=generator, dtype=torch.float32).to(device)
    measurement = torch.rand(2, 3, 32, 32, generator=generator, dtype=torch.float32).to(device)
    return psf, measurement


def test_forward_equivalence_repeatability() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(4716, deterministic=True)

    psf, measurement = _build_inputs(device)

    model_a = IFINNet(
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

    model_b = IFINNet(
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

    model_b.load_state_dict(model_a.state_dict(), strict=True)
    model_a.eval()
    model_b.eval()

    with torch.no_grad():
        outputs_a = model_a(measurement)
        outputs_b = model_b(measurement)

    atol = 1e-6
    rtol = 1e-6
    max_diffs = []
    for tensor_a, tensor_b in zip(outputs_a, outputs_b):
        max_abs_diff = torch.max(torch.abs(tensor_a - tensor_b)).item()
        max_diffs.append(max_abs_diff)
        assert torch.allclose(tensor_a, tensor_b, atol=atol, rtol=rtol), (
            f"Forward mismatch detected. max_abs_diff={max_abs_diff}, atol={atol}, rtol={rtol}"
        )

    print({"max_abs_diffs": max_diffs, "atol": atol, "rtol": rtol, "device": str(device)})
