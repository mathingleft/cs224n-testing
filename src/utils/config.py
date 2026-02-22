from __future__ import annotations
import copy
from pathlib import Path
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config, resolving inheritance via the 'inherit' key."""
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "inherit" in config:
        parent_name = config.pop("inherit")
        parent_path = config_path.parent / f"{parent_name}.yaml"
        parent_config = load_config(parent_path)
        config = _deep_merge(parent_config, config)

    return config


def apply_overrides(config: dict, overrides: dict) -> dict:
    """Apply flat CLI overrides like {'model.name': 'foo'} into nested config."""
    result = copy.deepcopy(config)
    for key, value in overrides.items():
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return result
