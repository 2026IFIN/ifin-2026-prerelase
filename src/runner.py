from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from data.synthetic import SyntheticLenslessDataset, SyntheticLenslessSpec
from data.waller import WallerDataset, validate_waller_paths
from losses.basic import ReconstructionLoss
from metrics.basic import psnr
from models import build_ifin_model
from utils.checkpoint import load_model_state_compat, save_checkpoint
from utils.seed import seed_everything


def _resolve_data_path(path_value: str) -> str:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    roots = [
        Path.cwd(),
        Path(__file__).resolve().parents[1],
        Path(__file__).resolve().parents[2],
    ]
    for root in roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return str(resolved)

    return str((Path.cwd() / candidate).resolve())


def _resolve_waller_and_psf_paths(data_cfg: Dict[str, Any]) -> tuple[str, str]:
    configured_waller = data_cfg.get("waller_path", "../../wallerlab/dataset")
    waller_path = _resolve_data_path(configured_waller)

    configured_psf = data_cfg.get("psf_path", "../../wallerlab/dataset/psf.tiff")
    psf_path = _resolve_data_path(configured_psf)
    if not Path(psf_path).exists():
        derived = str((Path(waller_path) / "psf.tiff").resolve())
        if Path(derived).exists():
            psf_path = derived

    return waller_path, psf_path


def _select_device(device_config: str) -> torch.device:
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def _build_psf(config: Dict[str, Any], device: torch.device) -> torch.Tensor:
    data_cfg = config["data"]
    model_cfg = config["model"]
    dataset_name = data_cfg.get("dataset", "waller")

    if dataset_name == "waller":
        waller_path, psf_path = _resolve_waller_and_psf_paths(data_cfg)
        validate_waller_paths(waller_path=waller_path, psf_path=psf_path)
        psf = cv2.imread(psf_path, 0)
        if psf is None:
            raise FileNotFoundError(f"Cannot read PSF image: {psf_path}")
        psf = cv2.resize(psf, (model_cfg["width"], model_cfg["height"]))
        psf = np.asarray(psf)
        psf = torch.from_numpy(psf).unsqueeze(0).unsqueeze(0).to(device)
        psf = psf / 255.0
        patch = int(data_cfg.get("psf_bg_patch", 15))
        psf_bg = torch.mean(psf[:, :, 0:patch, 0:patch])
        psf = psf - psf_bg
        psf[psf < 0] = 0
        return psf

    psf_size = data_cfg["psf_size"]
    generator = torch.Generator(device="cpu").manual_seed(config["seed"])
    psf = torch.rand(1, 1, psf_size, psf_size, generator=generator)
    return psf.to(device)


def _build_loader(config: Dict[str, Any], split: str) -> DataLoader:
    split_cfg = config[split]
    model_cfg = config["model"]
    data_cfg = config["data"]

    if data_cfg.get("dataset", "waller") == "waller":
        transformer_raw = transforms.Compose([transforms.ToTensor()])
        transformer_lab = transforms.Compose([transforms.ToTensor()])
        waller_path, psf_path = _resolve_waller_and_psf_paths(data_cfg)
        validate_waller_paths(
            waller_path=waller_path,
            psf_path=psf_path,
            train=(split == "train"),
        )
        dataset = WallerDataset(
            waller_path,
            train=(split == "train"),
            transform_raw=transformer_raw,
            transform_lab=transformer_lab,
        )
        return DataLoader(
            dataset,
            batch_size=split_cfg["batch_size"],
            shuffle=(split == "train"),
            num_workers=split_cfg.get("num_workers", 0),
            pin_memory=split_cfg.get("pin_memory", False),
        )

    spec = SyntheticLenslessSpec(
        num_samples=split_cfg["num_samples"],
        channels=model_cfg["in_channels"],
        height=model_cfg["height"],
        width=model_cfg["width"],
        seed=config["seed"] + (0 if split == "train" else 1),
    )
    dataset = SyntheticLenslessDataset(spec)
    return DataLoader(
        dataset,
        batch_size=split_cfg["batch_size"],
        shuffle=(split == "train"),
        num_workers=split_cfg.get("num_workers", 0),
        pin_memory=split_cfg.get("pin_memory", False),
    )


def train_one_epoch(config: Dict[str, Any]) -> Dict[str, float]:
    seed_everything(config["seed"], deterministic=config["deterministic"])
    device = _select_device(config["device"])
    train_loader = _build_loader(config, "train")

    model = build_ifin_model(config, _build_psf(config, device)).to(device)
    criterion = ReconstructionLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["optimizer"]["lr"])

    model.train()
    total_loss = 0.0
    total_psnr = 0.0

    for meas_input, img_target in train_loader:
        meas_input = meas_input.to(device)
        img_target = img_target.to(device)

        optimizer.zero_grad(set_to_none=True)
        img_recon, meas_recon, iso_recon = model(meas_input)
        loss = criterion(img_recon, img_target, meas_recon, meas_input)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total_loss += float(loss.detach().cpu())
            total_psnr += float(psnr(img_recon.detach(), img_target.detach()).cpu())

    avg_loss = total_loss / len(train_loader)
    avg_psnr = total_psnr / len(train_loader)

    if config["checkpoint"]["save"]:
        output_dir = Path(config["checkpoint"]["output_dir"])
        ckpt_path = output_dir / "ifin_smoke_last.pth"
        save_checkpoint(
            ckpt_path,
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            },
        )

    return {"loss": avg_loss, "psnr": avg_psnr, "device": str(device)}


def train(config: Dict[str, Any]) -> Dict[str, Any]:
    seed_everything(config["seed"], deterministic=config["deterministic"])
    device = _select_device(config["device"])
    train_loader = _build_loader(config, "train")

    model = build_ifin_model(config, _build_psf(config, device)).to(device)
    criterion = ReconstructionLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["optimizer"]["lr"])

    epochs = int(config.get("train", {}).get("epochs", 100))
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_psnr = 0.0

        for meas_input, img_target in train_loader:
            meas_input = meas_input.to(device)
            img_target = img_target.to(device)

            optimizer.zero_grad(set_to_none=True)
            img_recon, meas_recon, iso_recon = model(meas_input)
            loss = criterion(img_recon, img_target, meas_recon, meas_input)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_loss += float(loss.detach().cpu())
                total_psnr += float(psnr(img_recon.detach(), img_target.detach()).cpu())

        avg_loss = total_loss / len(train_loader)
        avg_psnr = total_psnr / len(train_loader)
        history.append({"epoch": epoch + 1, "loss": avg_loss, "psnr": avg_psnr})

        if config["checkpoint"]["save"]:
            output_dir = Path(config["checkpoint"]["output_dir"])
            ckpt_path = output_dir / "ifin_last.pth"
            save_checkpoint(
                ckpt_path,
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "metrics": {"loss": avg_loss, "psnr": avg_psnr},
                },
            )

    return {
        "epochs": epochs,
        "final_loss": history[-1]["loss"],
        "final_psnr": history[-1]["psnr"],
        "history": history,
        "device": str(device),
    }


def evaluate(config: Dict[str, Any], checkpoint_path: str | None = None) -> Dict[str, float]:
    seed_everything(config["seed"], deterministic=config["deterministic"])
    device = _select_device(config["device"])
    eval_loader = _build_loader(config, "eval")

    model = build_ifin_model(config, _build_psf(config, device)).to(device)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        load_model_state_compat(model, checkpoint, strict=True)

    model.eval()
    criterion = ReconstructionLoss()

    total_loss = 0.0
    total_psnr = 0.0
    with torch.no_grad():
        for meas_input, img_target in eval_loader:
            meas_input = meas_input.to(device)
            img_target = img_target.to(device)
            img_recon, meas_recon, iso_recon = model(meas_input)
            loss = criterion(img_recon, img_target, meas_recon, meas_input)
            total_loss += float(loss.detach().cpu())
            total_psnr += float(psnr(img_recon.detach(), img_target.detach()).cpu())

    return {
        "loss": total_loss / len(eval_loader),
        "psnr": total_psnr / len(eval_loader),
        "device": str(device),
    }


def infer(config: Dict[str, Any], checkpoint_path: str | None = None) -> Dict[str, Any]:
    seed_everything(config["seed"], deterministic=config["deterministic"])
    device = _select_device(config["device"])

    model = build_ifin_model(config, _build_psf(config, device)).to(device)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        load_model_state_compat(model, checkpoint, strict=True)

    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(config["seed"] + 99)
    meas_input = torch.rand(
        1,
        config["model"]["in_channels"],
        config["model"]["height"],
        config["model"]["width"],
        generator=generator,
    ).to(device)

    with torch.no_grad():
        img_recon, meas_recon, iso_recon = model(meas_input)

    return {
        "input_shape": tuple(meas_input.shape),
        "img_recon_shape": tuple(img_recon.shape),
        "meas_recon_shape": tuple(meas_recon.shape),
        "iso_recon_shape": tuple(iso_recon.shape),
        "device": str(device),
    }
