from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import re

import torch


def _normalize_state_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized[key[7:] if key.startswith("module.") else key] = value
    return normalized


def _remap_legacy_ifin_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remapped: Dict[str, torch.Tensor] = {}
    pattern_rules = [
        (r"^WieNerH\.", "initial_iso_operator."),
        (r"^start_w\.", "inverse_seed_block."),
        (r"^start_c\.", "forward_seed_block."),
        (r"^start_p\.", "psf_encoder."),
        (r"^refine_w\.", "inverse_refine_block."),
        (r"^out_w\.", "inverse_output_head."),
        (r"^refine_c\.", "forward_refine_block."),
        (r"^out_c\.", "forward_output_head."),
        (r"^down_layers\.(\d+)\.pool\.", r"down_blocks.\1.spatial_pool."),
        (r"^down_layers\.(\d+)\.caw_block\.W\.", r"down_blocks.\1.ifib_block.iso_operator."),
        (r"^down_layers\.(\d+)\.caw_block\.C\.", r"down_blocks.\1.ifib_block.fso_operator."),
        (r"^down_layers\.(\d+)\.caw_block\.conv1_w\.", r"down_blocks.\1.ifib_block.inverse_branch_block."),
        (r"^down_layers\.(\d+)\.caw_block\.conv1_c\.", r"down_blocks.\1.ifib_block.forward_branch_block."),
        (r"^down_layers\.(\d+)\.caw_block\.res_conv_w\.", r"down_blocks.\1.ifib_block.inverse_residual_projection."),
        (r"^down_layers\.(\d+)\.caw_block\.res_conv_c\.", r"down_blocks.\1.ifib_block.forward_residual_projection."),
        (r"^down_layers\.(\d+)\.caw_block\.alpha_c", r"down_blocks.\1.ifib_block.forward_self_mix"),
        (r"^down_layers\.(\d+)\.caw_block\.delta_c", r"down_blocks.\1.ifib_block.forward_cross_mix"),
        (r"^down_layers\.(\d+)\.caw_block\.alpha_w", r"down_blocks.\1.ifib_block.inverse_self_mix"),
        (r"^down_layers\.(\d+)\.caw_block\.delta_w", r"down_blocks.\1.ifib_block.inverse_cross_mix"),
        (r"^down_layers\.(\d+)\.conv_block\.", r"down_blocks.\1.psf_block."),
        (r"^up_layers\.(\d+)\.upw\.", r"up_blocks.\1.inverse_upsample."),
        (r"^up_layers\.(\d+)\.upc\.", r"up_blocks.\1.forward_upsample."),
        (r"^up_layers\.(\d+)\.caw_block\.W\.", r"up_blocks.\1.ifib_block.iso_operator."),
        (r"^up_layers\.(\d+)\.caw_block\.C\.", r"up_blocks.\1.ifib_block.fso_operator."),
        (r"^up_layers\.(\d+)\.caw_block\.conv1_w\.", r"up_blocks.\1.ifib_block.inverse_branch_block."),
        (r"^up_layers\.(\d+)\.caw_block\.conv1_c\.", r"up_blocks.\1.ifib_block.forward_branch_block."),
        (r"^up_layers\.(\d+)\.caw_block\.res_conv_w\.", r"up_blocks.\1.ifib_block.inverse_residual_projection."),
        (r"^up_layers\.(\d+)\.caw_block\.res_conv_c\.", r"up_blocks.\1.ifib_block.forward_residual_projection."),
        (r"^up_layers\.(\d+)\.caw_block\.alpha_c", r"up_blocks.\1.ifib_block.forward_self_mix"),
        (r"^up_layers\.(\d+)\.caw_block\.delta_c", r"up_blocks.\1.ifib_block.forward_cross_mix"),
        (r"^up_layers\.(\d+)\.caw_block\.alpha_w", r"up_blocks.\1.ifib_block.inverse_self_mix"),
        (r"^up_layers\.(\d+)\.caw_block\.delta_w", r"up_blocks.\1.ifib_block.inverse_cross_mix"),
        (r"^up_layers\.(\d+)\.convw1\.", r"up_blocks.\1.inverse_merge_projection."),
        (r"^up_layers\.(\d+)\.convc1\.", r"up_blocks.\1.forward_merge_projection."),
    ]
    for key, value in state_dict.items():
        new_key = key
        for pattern, replacement in pattern_rules:
            new_key = re.sub(pattern, replacement, new_key)
        remapped[new_key] = value
    return remapped


def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def load_model_state_compat(model: torch.nn.Module, checkpoint: Dict[str, Any], strict: bool = True) -> None:
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    normalized = _normalize_state_keys(state_dict)
    remapped = _remap_legacy_ifin_keys(normalized)
    try:
        model.load_state_dict(remapped, strict=strict)
    except RuntimeError:
        model.load_state_dict(normalized, strict=strict)
