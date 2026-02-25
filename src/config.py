from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | None = None, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    selected_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with selected_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if overrides:
        config = _deep_update(config, overrides)
    return config
