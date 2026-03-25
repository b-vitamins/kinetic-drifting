from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from kdrifting.hf import load_generator_model, load_mae_model, resolve_artifact_dir
from kdrifting.models.generator import DitGen
from kdrifting.models.mae import MAEResNet


def test_resolve_artifact_dir_accepts_run_root(tmp_path: Path) -> None:
    root = tmp_path / "run"
    artifact_dir = root / "params_ema"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "metadata.json").write_text("{}", encoding="utf-8")

    resolved = resolve_artifact_dir(str(root), kind="mae")

    assert resolved == artifact_dir.resolve()


def test_load_mae_model_restores_local_artifact(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "mae_artifact"
    artifact_dir.mkdir()
    model = MAEResNet(
        num_classes=7,
        in_channels=3,
        base_channels=16,
        patch_size=2,
        layers=(1, 1, 1, 1),
        input_patch_size=1,
    )
    torch.save(model.state_dict(), artifact_dir / "ema_model.pt")
    metadata = {
        "kind": "mae",
        "backend": "torch",
        "model_config": {
            "num_classes": 7,
            "in_channels": 3,
            "base_channels": 16,
            "patch_size": 2,
            "layers": [1, 1, 1, 1],
            "input_patch_size": 1,
        },
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    loaded_model, loaded_metadata = load_mae_model(str(artifact_dir))

    assert loaded_metadata["model_config"]["num_classes"] == 7
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded_model.state_dict()[key])


def test_load_generator_model_restores_local_artifact(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "generator_artifact"
    artifact_dir.mkdir()
    model_config: dict[str, Any] = {
        "cond_dim": 32,
        "num_classes": 11,
        "noise_classes": 8,
        "noise_coords": 4,
        "input_size": 8,
        "in_channels": 3,
        "patch_size": 2,
        "hidden_size": 32,
        "depth": 2,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "out_channels": 3,
        "n_cls_tokens": 2,
        "use_qknorm": True,
        "use_swiglu": True,
        "use_rope": True,
        "use_rmsnorm": True,
    }
    model = DitGen(**model_config)
    torch.save(model.state_dict(), artifact_dir / "ema_model.pt")
    metadata = {
        "kind": "gen",
        "backend": "torch",
        "model_config": model_config,
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    loaded_model, loaded_metadata = load_generator_model(str(artifact_dir))

    assert loaded_metadata["model_config"]["num_classes"] == 11
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded_model.state_dict()[key])
