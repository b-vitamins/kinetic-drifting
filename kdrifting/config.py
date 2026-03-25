"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a plain nested dictionary."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    yaml_module = cast(Any, yaml)
    raw = cast(object, yaml_module.safe_load(config_path.read_text(encoding="utf-8")))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected top-level mapping in {config_path}, got {type(raw)!r}")
    return cast(dict[str, Any], raw)
