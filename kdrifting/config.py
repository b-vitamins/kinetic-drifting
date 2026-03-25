"""Configuration loading helpers."""

from __future__ import annotations

from collections.abc import Mapping
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


def export_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the self-describing model config stored in artifacts and run metadata."""
    dataset_config = dict(cast(Mapping[str, Any], config.get("dataset", {})) or {})
    model_config = dict(cast(Mapping[str, Any], config.get("model", {})) or {})
    if "num_classes" in dataset_config and "num_classes" not in model_config:
        model_config["num_classes"] = int(dataset_config["num_classes"])
    return model_config
