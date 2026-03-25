from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import optax
import torch
from torch import nn

from kdrifting.checkpointing import (
    restore_checkpoint,
    restore_checkpoint_extra_state,
    restore_external_checkpoint,
    save_checkpoint,
)
from kdrifting.jax_artifacts import (
    convert_generator_jax_optimizer_tensors,
    convert_generator_jax_params,
    convert_mae_jax_optimizer_tensors,
    convert_mae_jax_params,
)
from kdrifting.memory_bank import ArrayMemoryBank
from kdrifting.models.generator import DitGen
from kdrifting.models.mae import MAEResNet
from kdrifting.training.state import TrainState

UPSTREAM_ROOT = Path("/home/b/projects/drifting")
FLAX_CHECKPOINTS: Any = importlib.import_module("flax.training.checkpoints")
JAX: Any = importlib.import_module("jax")
JNP: Any = JAX.numpy


def _make_state() -> TrainState:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = TrainState.create(model, optimizer, ema_decay=0.95)

    inputs = torch.tensor([[0.5, -1.0, 2.0]], dtype=torch.float32)
    loss = state.model(inputs).sum()
    loss.backward()
    state.optimizer.step()
    state.optimizer.zero_grad(set_to_none=True)
    state.update_ema()
    state.step = 7
    return state


def _assert_state_dict_equal(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(actual[key], value), key


def _import_upstream(module_name: str) -> Any:
    root = str(UPSTREAM_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(module_name)


def _assert_optimizer_state_matches(
    state: TrainState,
    *,
    step: int,
    exp_avg: dict[str, torch.Tensor],
    exp_avg_sq: dict[str, torch.Tensor],
) -> None:
    for name, parameter in state.model.named_parameters():
        opt_state = state.optimizer.state[parameter]
        assert float(cast(torch.Tensor, opt_state["step"]).item()) == float(step)
        assert torch.equal(cast(torch.Tensor, opt_state["exp_avg"]), exp_avg[name])
        assert torch.equal(cast(torch.Tensor, opt_state["exp_avg_sq"]), exp_avg_sq[name])


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


def _build_optax_adam_state(
    params: Any,
    *,
    count: int,
    mu_value: float,
    nu_value: float,
) -> Any:
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=0.01, b1=0.9, b2=0.999)
    opt_state = cast(tuple[Any, ...], optimizer.init(params))
    adam_state = opt_state[0]

    def fill_mu(value: Any) -> Any:
        return JNP.full_like(value, mu_value)

    def fill_nu(value: Any) -> Any:
        return JNP.full_like(value, nu_value)

    mu = JAX.tree.map(fill_mu, params)
    nu = JAX.tree.map(fill_nu, params)
    new_adam_state = adam_state._replace(count=JNP.asarray(count, dtype=JNP.int32), mu=mu, nu=nu)
    return (new_adam_state, *opt_state[1:])


def _write_jax_checkpoint_payload(root: Path, *, payload: dict[str, Any], step: int) -> Path:
    checkpoints_dir = root / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    FLAX_CHECKPOINTS.save_checkpoint(
        ckpt_dir=str(checkpoints_dir),
        target=payload,
        step=step,
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


def test_checkpoint_roundtrip_restores_train_state_and_extra_state(tmp_path: Path) -> None:
    state = _make_state()
    positive_bank = ArrayMemoryBank(num_classes=3, max_size=2, seed=17)
    positive_bank.add(
        np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ],
            dtype=np.float32,
        ),
        np.array([0, 1, 0], dtype=np.int32),
    )
    positive_bank.sample(np.array([0], dtype=np.int64), n_samples=2)
    extra_state = {"memory_bank_positive": positive_bank.state_dict()}

    save_checkpoint(state, workdir=str(tmp_path), keep=2, keep_every=1, extra_state=extra_state)

    restored = _make_state()
    restored = restore_checkpoint(restored, workdir=str(tmp_path), step=7)

    assert restored.step == 7
    assert restored.ema_decay == state.ema_decay
    _assert_state_dict_equal(restored.model.state_dict(), state.model.state_dict())
    _assert_state_dict_equal(restored.ema_model.state_dict(), state.ema_model.state_dict())

    restored_extra = restore_checkpoint_extra_state(workdir=str(tmp_path), step=7)
    assert restored_extra is not None

    restored_bank = ArrayMemoryBank(num_classes=1, max_size=1)
    restored_bank.load_state_dict(
        cast(dict[str, Any], restored_extra["memory_bank_positive"]),
    )

    assert restored_bank.seed == 17
    np.testing.assert_array_equal(restored_bank.count, positive_bank.count)
    np.testing.assert_array_equal(restored_bank.ptr, positive_bank.ptr)
    assert restored_bank.bank is not None
    assert positive_bank.bank is not None
    np.testing.assert_allclose(restored_bank.bank, positive_bank.bank)

    expected = positive_bank.sample(np.array([0, 1], dtype=np.int64), n_samples=2)
    actual = restored_bank.sample(np.array([0, 1], dtype=np.int64), n_samples=2)
    assert torch.equal(actual, expected)


def test_restore_external_torch_checkpoint_restores_full_train_state(tmp_path: Path) -> None:
    state = _make_state()
    save_checkpoint(state, workdir=str(tmp_path), keep=1)

    restored = _make_state()
    restored = restore_external_checkpoint(restored, init_from=str(tmp_path), kind="mae")

    assert restored.step == state.step
    assert restored.ema_decay == state.ema_decay
    _assert_state_dict_equal(restored.model.state_dict(), state.model.state_dict())
    _assert_state_dict_equal(restored.ema_model.state_dict(), state.ema_model.state_dict())
    _assert_optimizer_state_equal(restored.optimizer, state.optimizer)


def test_restore_external_jax_mae_checkpoint_restores_params_ema_and_adamw_state(
    tmp_path: Path,
) -> None:
    upstream_mae = _import_upstream("models.mae_model")
    metadata = {
            "kind": "mae",
            "backend": "jax",
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
    params = variables["params"]
    ema_params = _offset_tree(params, 0.125)
    opt_state = _build_optax_adam_state(params, count=5, mu_value=0.25, nu_value=0.5)
    run_root = _write_jax_checkpoint_payload(
        tmp_path / "mae_run",
        payload={
            "params": params,
            "ema_params": ema_params,
            "opt_state": opt_state,
            "ema_decay": 0.93,
            "step": 5,
        },
        step=5,
    )

    torch_model = MAEResNet(
        num_classes=5,
        in_channels=3,
        base_channels=64,
        patch_size=2,
        layers=(1, 1, 1, 1),
        input_patch_size=1,
    )
    torch_optimizer = torch.optim.AdamW(
        torch_model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    torch_state = TrainState.create(torch_model, torch_optimizer, ema_decay=0.1)

    restored = restore_external_checkpoint(torch_state, init_from=str(run_root), kind="mae")

    assert restored.step == 5
    assert restored.ema_decay == 0.93
    _assert_state_dict_equal(
        restored.model.state_dict(),
        convert_mae_jax_params(params, torch_model),
    )
    _assert_state_dict_equal(
        restored.ema_model.state_dict(),
        convert_mae_jax_params(ema_params, cast(MAEResNet, restored.ema_model)),
    )
    _assert_optimizer_state_matches(
        restored,
        step=5,
        exp_avg=convert_mae_jax_optimizer_tensors(params=opt_state[0].mu, model=torch_model),
        exp_avg_sq=convert_mae_jax_optimizer_tensors(params=opt_state[0].nu, model=torch_model),
    )


def test_restore_external_jax_generator_checkpoint_restores_params_ema_and_adamw_state(
    tmp_path: Path,
) -> None:
    upstream_utils = _import_upstream("utils.hsdp_util")
    upstream_utils.set_global_mesh(1)
    upstream_generator = _import_upstream("models.generator")
    model_config: dict[str, Any] = {
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
    upstream_model = upstream_generator.build_generator_from_config(model_config)
    variables = upstream_model.init(
        {"params": JAX.random.PRNGKey(0), "noise": JAX.random.PRNGKey(1)},
        **upstream_model.dummy_input(),
    )
    params = variables["params"]
    ema_params = _generator_ema_params(params)
    opt_state = _build_optax_adam_state(params, count=7, mu_value=0.125, nu_value=0.25)
    run_root = _write_jax_checkpoint_payload(
        tmp_path / "generator_run",
        payload={
            "params": params,
            "ema_params": ema_params,
            "opt_state": opt_state,
            "ema_decay": 0.97,
            "step": 7,
        },
        step=7,
    )

    torch_model = DitGen(**model_config)
    torch_optimizer = torch.optim.AdamW(
        torch_model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    torch_state = TrainState.create(torch_model, torch_optimizer, ema_decay=0.1)

    restored = restore_external_checkpoint(torch_state, init_from=str(run_root), kind="gen")

    assert restored.step == 7
    assert restored.ema_decay == 0.97
    _assert_state_dict_equal(
        restored.model.state_dict(),
        convert_generator_jax_params(params, torch_model),
    )
    _assert_state_dict_equal(
        restored.ema_model.state_dict(),
        convert_generator_jax_params(ema_params, cast(DitGen, restored.ema_model)),
    )
    _assert_optimizer_state_matches(
        restored,
        step=7,
        exp_avg=convert_generator_jax_optimizer_tensors(params=opt_state[0].mu, model=torch_model),
        exp_avg_sq=convert_generator_jax_optimizer_tensors(
            params=opt_state[0].nu,
            model=torch_model,
        ),
    )
