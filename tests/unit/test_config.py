from __future__ import annotations

from pathlib import Path

from kdrifting.config import load_yaml_config


def test_load_yaml_config_reads_nested_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("outer:\n  inner: 3\n", encoding="utf-8")

    config = load_yaml_config(config_path)

    assert config == {"outer": {"inner": 3}}
