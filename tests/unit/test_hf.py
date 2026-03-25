from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from kdrifting.hf import load_generator_model, load_mae_model, resolve_artifact_dir
from kdrifting.models.generator import DitGen
from kdrifting.models.mae import MAEResNet


def _offset_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    delta: float,
) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() + delta for key, value in state_dict.items()}


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


def test_load_mae_model_restores_local_checkpoint_directory_ema_state(tmp_path: Path) -> None:
    run_root = tmp_path / "mae_run"
    checkpoint_dir = run_root / "checkpoints"
    artifact_dir = run_root / "params_ema"
    checkpoint_dir.mkdir(parents=True)
    artifact_dir.mkdir()
    model = MAEResNet(
        num_classes=7,
        in_channels=3,
        base_channels=16,
        patch_size=2,
        layers=(1, 1, 1, 1),
        input_patch_size=1,
    )
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

    live_state = _offset_state_dict(model.state_dict(), delta=1.0)
    ema_state = _offset_state_dict(model.state_dict(), delta=2.0)
    torch.save(
        {
            "model": live_state,
            "optimizer": {},
            "ema_model": ema_state,
            "ema_decay": 0.95,
            "step": 5,
        },
        checkpoint_dir / "step_00000005.pt",
    )

    loaded_model, loaded_metadata = load_mae_model(str(checkpoint_dir))

    assert loaded_metadata["step"] == 5
    assert loaded_metadata["ema_decay"] == 0.95
    for key, value in ema_state.items():
        assert torch.equal(value, loaded_model.state_dict()[key])
        assert not torch.equal(value, live_state[key])


def test_load_generator_model_restores_explicit_checkpoint_file_ema_state(tmp_path: Path) -> None:
    run_root = tmp_path / "generator_run"
    checkpoint_dir = run_root / "checkpoints"
    artifact_dir = run_root / "params_ema"
    checkpoint_dir.mkdir(parents=True)
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
    metadata = {
        "kind": "gen",
        "backend": "torch",
        "model_config": deepcopy(model_config),
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    live_state = _offset_state_dict(model.state_dict(), delta=1.0)
    ema_state = _offset_state_dict(model.state_dict(), delta=3.0)
    checkpoint_path = checkpoint_dir / "step_00000012.pt"
    torch.save(
        {
            "model": live_state,
            "optimizer": {},
            "ema_model": ema_state,
            "ema_decay": 0.99,
            "step": 12,
        },
        checkpoint_path,
    )

    loaded_model, loaded_metadata = load_generator_model(str(checkpoint_path))

    assert loaded_metadata["step"] == 12
    assert loaded_metadata["ema_decay"] == 0.99
    for key, value in ema_state.items():
        assert torch.equal(value, loaded_model.state_dict()[key])
        assert not torch.equal(value, live_state[key])
