# src/config.py
import os
from pathlib import Path
from copy import deepcopy

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override values win."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(config_dir: str = "config", override: str = None) -> dict:
    """Load default config, optionally merge an override file, and apply env vars."""
    config_path = Path(config_dir)

    with open(config_path / "default.yaml") as f:
        cfg = yaml.safe_load(f)

    if override:
        override_path = config_path / f"{override}.yaml"
        with open(override_path) as f:
            override_cfg = yaml.safe_load(f)
        if override_cfg:
            cfg = _deep_merge(cfg, override_cfg)

    cfg["data_dir"] = os.environ.get("DATA_DIR", "./data")

    return cfg
