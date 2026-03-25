from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, cast

import optax
import torch

from kdrifting.checkpointing import restore_external_checkpoint
from kdrifting.export import export_model_artifact, export_training_checkpoint
from kdrifting.hf import load_generator_model, load_mae_model
from kdrifting.models.generator import DitGen
from kdrifting.models.mae import MAEResNet
from kdrifting.training.state import TrainState

UPSTREAM_ROOT = Path("/home/b/projects/drifting")
FLAX_CHECKPOINTS = cast(Any, importlib.import_module("flax.training.checkpoints"))
FLAX_SERIALIZATION = cast(Any, importlib.import_module("flax.serialization"))
JAX = cast(Any, importlib.import_module("jax"))
JNP = JAX.numpy


def _import_upstream(module_name: str) -> Any:
    root = str(UPSTREAM_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(module_name)


def _write_jax_artifact(
    root: Path,
    *,
    metadata: dict[str, Any],
    params: Any,
) -> Path:
    root.mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "ema_params.msgpack").write_bytes(FLAX_SERIALIZATION.msgpack_serialize(params))
    return root


def _write_jax_checkpoint_payload(
    root: Path,
    *,
    payload: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    checkpoints_dir = root / "checkpoints"
    params_ema_dir = root / "params_ema"
    checkpoints_dir.mkdir(parents=True)
    params_ema_dir.mkdir()
    (params_ema_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    FLAX_CHECKPOINTS.save_checkpoint(
        ckpt_dir=str(checkpoints_dir),
        target=payload,
        step=int(payload["step"]),
        overwrite=True,
    )
    return root


def _offset_tree(tree: Any, delta: float) -> Any:
    def add_delta(value: Any) -> Any:
        return value + delta

    return JAX.tree.map(add_delta, tree)


def _generator_ema_params(params: Any) -> Any:
    ema_params = _offset_tree(params, -0.05)
    ema_params["LightningDiT_0"]["pos_embed"] = params["LightningDiT_0"]["pos_embed"]
    return ema_params


def _build_optax_adam_state(
    params: Any,
    *,
    count: int,
    mu_value: float,
    nu_value: float,
) -> Any:
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=0.02, b1=0.85, b2=0.98)
    opt_state = cast(tuple[Any, ...], optimizer.init(params))
    adam_state = opt_state[0]

    def fill_mu(value: Any) -> Any:
        return JNP.full_like(value, mu_value)

    def fill_nu(value: Any) -> Any:
        return JNP.full_like(value, nu_value)

    mu = JAX.tree.map(fill_mu, params)
    nu = JAX.tree.map(fill_nu, params)
    new_adam_state = adam_state._replace(
        count=JNP.asarray(count, dtype=JNP.int32),
        mu=mu,
        nu=nu,
    )
    return (new_adam_state, *opt_state[1:])


def _assert_state_dict_equal(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(actual[key], value), key


def _assert_optimizer_state_equal(
    actual: torch.optim.Optimizer,
    expected: torch.optim.Optimizer,
) -> None:
    actual_state = actual.state_dict()
    expected_state = expected.state_dict()
    assert actual_state["param_groups"] == expected_state["param_groups"]

    actual_entries = cast(dict[int, dict[str, Any]], actual_state["state"])
    expected_entries = cast(dict[int, dict[str, Any]], expected_state["state"])
    assert actual_entries.keys() == expected_entries.keys()
    for key, expected_entry in expected_entries.items():
        actual_entry = actual_entries[key]
        assert actual_entry.keys() == expected_entry.keys()
        for name, expected_value in expected_entry.items():
            actual_value = actual_entry[name]
            if isinstance(expected_value, torch.Tensor):
                assert isinstance(actual_value, torch.Tensor)
                assert torch.equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value


def _make_mae_config() -> dict[str, Any]:
    return {
        "dataset": {
            "num_classes": 5,
        },
        "model": {
            "in_channels": 3,
            "base_channels": 64,
            "patch_size": 2,
            "layers": [1, 1, 1, 1],
            "input_patch_size": 1,
        },
        "optimizer": {
            "weight_decay": 0.02,
            "adam_b1": 0.85,
            "adam_b2": 0.98,
            "lr_schedule": {
                "learning_rate": 1e-3,
                "warmup_steps": 0,
                "total_steps": 8,
                "lr_schedule": "const",
            },
        },
        "train": {
            "ema_decay": 0.93,
        },
    }


def _make_generator_config() -> dict[str, Any]:
    return {
        "dataset": {
            "num_classes": 5,
        },
        "model": {
            "cond_dim": 32,
            "noise_classes": 4,
            "noise_coords": 2,
            "input_size": 8,
            "in_channels": 3,
            "patch_size": 2,
            "hidden_size": 32,
            "depth": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "out_channels": 3,
            "n_cls_tokens": 2,
            "use_qknorm": True,
            "use_swiglu": True,
            "use_rope": True,
            "use_rmsnorm": True,
        },
        "optimizer": {
            "weight_decay": 0.02,
            "adam_b1": 0.85,
            "adam_b2": 0.98,
            "lr_schedule": {
                "learning_rate": 1e-3,
                "warmup_steps": 0,
                "total_steps": 8,
                "lr_schedule": "const",
            },
        },
        "train": {
            "ema_decay": 0.97,
        },
    }


def _make_mae_state(config: dict[str, Any]) -> TrainState:
    model = MAEResNet(num_classes=int(config["dataset"]["num_classes"]), **dict(config["model"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=float(config["optimizer"]["weight_decay"]),
        betas=(
            float(config["optimizer"]["adam_b1"]),
            float(config["optimizer"]["adam_b2"]),
        ),
    )
    return TrainState.create(model, optimizer, ema_decay=float(config["train"]["ema_decay"]))


def _make_generator_state(config: dict[str, Any]) -> TrainState:
    model = DitGen(num_classes=int(config["dataset"]["num_classes"]), **dict(config["model"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=float(config["optimizer"]["weight_decay"]),
        betas=(
            float(config["optimizer"]["adam_b1"]),
            float(config["optimizer"]["adam_b2"]),
        ),
    )
    return TrainState.create(model, optimizer, ema_decay=float(config["train"]["ema_decay"]))


def test_export_model_artifact_converts_jax_mae_artifact_with_roundtrip_parity(
    tmp_path: Path,
) -> None:
    upstream_mae = _import_upstream("models.mae_model")
    metadata = {
        "kind": "mae",
        "backend": "jax",
        "format": "flax.msgpack",
        "step": 11,
        "ema_decay": 0.91,
        "model_config": {
            "num_classes": 5,
            "in_channels": 3,
            "base_channels": 64,
            "patch_size": 2,
            "layers": [1, 1, 1, 1],
            "input_patch_size": 1,
        },
    }
    upstream_model = upstream_mae._mae_from_metadata(metadata)
    variables = upstream_model.init(
        {
            "params": JAX.random.PRNGKey(0),
            "masking": JAX.random.PRNGKey(1),
            "dropout": JAX.random.PRNGKey(2),
        },
        **upstream_model.dummy_input(),
    )
    artifact_dir = _write_jax_artifact(
        tmp_path / "mae_artifact",
        metadata=metadata,
        params=variables["params"],
    )

    exported = export_model_artifact(
        init_from=str(artifact_dir),
        kind="mae",
        workdir=str(tmp_path / "mae_export"),
    )

    direct_model, _ = load_mae_model(str(artifact_dir))
    exported_model, exported_metadata = load_mae_model(exported["workdir"])

    assert exported_metadata["backend"] == "torch"
    assert exported_metadata["source_backend"] == "jax"
    assert exported_metadata["step"] == 11
    _assert_state_dict_equal(exported_model.state_dict(), direct_model.state_dict())


def test_export_model_artifact_converts_jax_generator_checkpoint_with_roundtrip_parity(
    tmp_path: Path,
) -> None:
    upstream_utils = _import_upstream("utils.hsdp_util")
    upstream_utils.set_global_mesh(1)
    upstream_generator = _import_upstream("models.generator")
    model_config = {
        "cond_dim": 32,
        "num_classes": 5,
        "noise_classes": 4,
        "noise_coords": 2,
        "input_size": 8,
        "in_channels": 3,
        "patch_size": 2,
        "hidden_size": 32,
        "depth": 1,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "out_channels": 3,
        "n_cls_tokens": 2,
        "use_qknorm": True,
        "use_swiglu": True,
        "use_rope": True,
        "use_rmsnorm": True,
    }
    metadata = {
        "kind": "gen",
        "backend": "jax",
        "format": "orbax.checkpoint",
        "step": 7,
        "ema_decay": 0.97,
        "model_config": dict(model_config),
    }
    upstream_model = upstream_generator.build_generator_from_config(model_config)
    variables = upstream_model.init(
        {"params": JAX.random.PRNGKey(0), "noise": JAX.random.PRNGKey(1)},
        **upstream_model.dummy_input(),
    )
    run_root = _write_jax_checkpoint_payload(
        tmp_path / "generator_run",
        payload={
            "params": variables["params"],
            "ema_params": _generator_ema_params(variables["params"]),
            "step": 7,
        },
        metadata=metadata,
    )

    exported = export_model_artifact(
        init_from=str(run_root),
        kind="gen",
        workdir=str(tmp_path / "generator_export"),
    )

    direct_model, _ = load_generator_model(str(run_root))
    exported_model, exported_metadata = load_generator_model(exported["workdir"])

    assert exported_metadata["backend"] == "torch"
    assert exported_metadata["source_backend"] == "jax"
    assert exported_metadata["step"] == 7
    _assert_state_dict_equal(exported_model.state_dict(), direct_model.state_dict())


def test_export_training_checkpoint_converts_jax_mae_run_with_roundtrip_resume_parity(
    tmp_path: Path,
) -> None:
    config = _make_mae_config()
    metadata = {
        "kind": "mae",
        "backend": "jax",
        "model_config": {
            "num_classes": int(config["dataset"]["num_classes"]),
            **dict(config["model"]),
        },
    }
    upstream_mae = _import_upstream("models.mae_model")
    upstream_model = upstream_mae._mae_from_metadata(metadata)
    variables = upstream_model.init(
        {
            "params": JAX.random.PRNGKey(0),
            "masking": JAX.random.PRNGKey(1),
            "dropout": JAX.random.PRNGKey(2),
        },
        **upstream_model.dummy_input(),
    )
    params = variables["params"]
    ema_params = _offset_tree(params, 0.125)
    run_root = _write_jax_checkpoint_payload(
        tmp_path / "mae_run",
        payload={
            "params": params,
            "ema_params": ema_params,
            "opt_state": _build_optax_adam_state(params, count=5, mu_value=0.25, nu_value=0.5),
            "ema_decay": 0.93,
            "step": 5,
        },
        metadata=metadata,
    )

    direct_state = restore_external_checkpoint(
        _make_mae_state(config),
        init_from=str(run_root),
        kind="mae",
    )
    exported = export_training_checkpoint(
        init_from=str(run_root),
        config=config,
        kind="mae",
        workdir=str(tmp_path / "mae_export"),
    )
    roundtrip_state = restore_external_checkpoint(
        _make_mae_state(config),
        init_from=exported["workdir"],
        kind="mae",
    )

    assert roundtrip_state.step == direct_state.step == 5
    assert roundtrip_state.ema_decay == direct_state.ema_decay == 0.93
    _assert_state_dict_equal(roundtrip_state.model.state_dict(), direct_state.model.state_dict())
    _assert_state_dict_equal(
        roundtrip_state.ema_model.state_dict(),
        direct_state.ema_model.state_dict(),
    )
    _assert_optimizer_state_equal(roundtrip_state.optimizer, direct_state.optimizer)

    metadata_path = Path(exported["workdir"]) / "metadata.json"
    root_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert root_metadata["optimizer_config"]["adam_b1"] == 0.85
    assert root_metadata["train_config"]["ema_decay"] == 0.93


def test_export_training_checkpoint_converts_jax_generator_run_with_roundtrip_resume_parity(
    tmp_path: Path,
) -> None:
    config = _make_generator_config()
    upstream_utils = _import_upstream("utils.hsdp_util")
    upstream_utils.set_global_mesh(1)
    upstream_generator = _import_upstream("models.generator")
    metadata = {
        "kind": "gen",
        "backend": "jax",
        "model_config": {
            "num_classes": int(config["dataset"]["num_classes"]),
            **dict(config["model"]),
        },
    }
    upstream_model = upstream_generator.build_generator_from_config(metadata["model_config"])
    variables = upstream_model.init(
        {"params": JAX.random.PRNGKey(0), "noise": JAX.random.PRNGKey(1)},
        **upstream_model.dummy_input(),
    )
    params = variables["params"]
    ema_params = _generator_ema_params(params)
    run_root = _write_jax_checkpoint_payload(
        tmp_path / "generator_run",
        payload={
            "params": params,
            "ema_params": ema_params,
            "opt_state": _build_optax_adam_state(params, count=7, mu_value=0.125, nu_value=0.25),
            "ema_decay": 0.97,
            "step": 7,
        },
        metadata=metadata,
    )

    direct_state = restore_external_checkpoint(
        _make_generator_state(config),
        init_from=str(run_root),
        kind="gen",
    )
    exported = export_training_checkpoint(
        init_from=str(run_root),
        config=config,
        kind="gen",
        workdir=str(tmp_path / "generator_export"),
    )
    roundtrip_state = restore_external_checkpoint(
        _make_generator_state(config),
        init_from=exported["workdir"],
        kind="gen",
    )

    assert roundtrip_state.step == direct_state.step == 7
    assert roundtrip_state.ema_decay == direct_state.ema_decay == 0.97
    _assert_state_dict_equal(roundtrip_state.model.state_dict(), direct_state.model.state_dict())
    _assert_state_dict_equal(
        roundtrip_state.ema_model.state_dict(),
        direct_state.ema_model.state_dict(),
    )
    _assert_optimizer_state_equal(roundtrip_state.optimizer, direct_state.optimizer)

    exported_model, exported_metadata = load_generator_model(exported["workdir"])
    assert exported_metadata["backend"] == "torch"
    _assert_state_dict_equal(exported_model.state_dict(), direct_state.ema_model.state_dict())
