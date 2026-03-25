from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path
from typing import Any, cast

import flax.linen as flax_nn
import jax
import numpy as np
import optax
import torch
from torch import nn

from kdrifting.models.generator import DitGen
from kdrifting.models.mae import MAEResNet, build_activation_function
from kdrifting.schedules import create_learning_rate_fn
from kdrifting.training.generator import train_step as train_generator_step
from kdrifting.training.mae import train_step as train_mae_step
from kdrifting.training.state import TrainState

UPSTREAM_ROOT = Path("/home/b/projects/drifting")
JNP = cast(Any, jax.numpy)


def _identity_preprocess(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return batch


def _import_upstream(module_name: str) -> Any:
    root = str(UPSTREAM_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(module_name)


def _import_upstream_training_module(module_name: str) -> Any:
    utils_misc = _import_upstream("utils.misc")
    original_run_init = utils_misc.run_init
    utils_misc.run_init = lambda: None
    try:
        return importlib.import_module(module_name)
    finally:
        utils_misc.run_init = original_run_init


def _run_upstream_generator_train_step(
    train_step_fn: Any,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, Any]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Passing arguments 'a', 'a_min' or 'a_max' to jax.numpy.clip is deprecated.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="optax.global_norm is deprecated in favor of optax.tree.norm",
            category=DeprecationWarning,
        )
        return train_step_fn(*args, **kwargs)


def _identity_jax_batch(batch: Any) -> Any:
    return batch


def _init_scalar_05(key: Any) -> Any:
    del key
    return JNP.array(0.5, dtype=JNP.float32)


def _init_scalar_025(key: Any) -> Any:
    del key
    return JNP.array(0.25, dtype=JNP.float32)


class ToyMaeTorch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(
        self,
        *,
        x: torch.Tensor,
        labels: torch.Tensor,
        train: bool,
        generator: torch.Generator,
        **kwargs: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del train, generator, kwargs
        loss = self.weight * x.mean() + labels.to(dtype=torch.float32).mean() * 0.01
        return loss, {"toy": loss}


class ToyGeneratorTorch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

    def forward(
        self,
        labels: torch.Tensor,
        *,
        cfg_scale: torch.Tensor | float = 1.0,
        deterministic: bool,
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        del deterministic, generator
        cfg_tensor = torch.as_tensor(cfg_scale, device=labels.device, dtype=torch.float32)
        if cfg_tensor.ndim == 0:
            cfg_tensor = cfg_tensor.expand(labels.shape[0])
        label_term = labels.to(dtype=torch.float32)[:, None, None, None]
        cfg_term = cfg_tensor[:, None, None, None]
        samples = self.weight * (label_term + cfg_term)
        return {"samples": samples}


class ToyMaeJax(flax_nn.Module):
    @flax_nn.compact
    def __call__(
        self,
        *,
        x: Any,
        labels: Any,
        train: bool,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        del train, kwargs
        module = cast(Any, self)
        weight = module.param("weight", _init_scalar_05)
        loss = weight * JNP.mean(x) + JNP.mean(labels.astype(JNP.float32)) * 0.01
        return loss, {"toy": loss}


class ToyGeneratorJax(flax_nn.Module):
    @flax_nn.compact
    def __call__(
        self,
        *,
        c: Any,
        cfg_scale: Any = 1.0,
        train: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del train, kwargs
        module = cast(Any, self)
        weight = module.param("weight", _init_scalar_025)
        cfg = JNP.asarray(cfg_scale, dtype=JNP.float32)
        if cfg.ndim == 0:
            cfg = JNP.full_like(c, cfg)
        label_term = c.astype(JNP.float32)[:, None, None, None]
        cfg_term = cfg[:, None, None, None]
        samples = weight * (label_term + cfg_term)
        return {"samples": samples}


def _toy_feature_apply_torch(
    feature_params: Any,
    samples: torch.Tensor,
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    del feature_params, kwargs
    feature = samples.reshape(samples.shape[0], 1, -1)
    return {"toy": feature}


def _toy_feature_apply_jax(
    feature_params: Any,
    samples: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    del feature_params, kwargs
    feature = JNP.reshape(samples, (samples.shape[0], 1, -1))
    return {"toy": feature}


def test_mae_train_step_updates_state_and_returns_metrics() -> None:
    model = MAEResNet(
        num_classes=5,
        in_channels=3,
        base_channels=8,
        patch_size=2,
        layers=(1, 1, 1, 1),
        input_patch_size=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = TrainState.create(model, optimizer, ema_decay=0.9)
    schedule = create_learning_rate_fn(learning_rate=1e-3, warmup_steps=1, total_steps=10)
    batch = {
        "images": torch.randn(2, 32, 32, 3),
        "labels": torch.tensor([1, 2], dtype=torch.int64),
    }

    state, metrics = train_mae_step(
        state,
        batch,
        base_seed=123,
        forward_dict={"mask_ratio_min": 0.5, "mask_ratio_max": 0.5, "lambda_cls": 0.1},
        learning_rate_fn=schedule,
        preprocess_fn=_identity_preprocess,
    )

    assert state.step == 1
    assert "loss" in metrics
    assert "g_norm" in metrics
    assert "lr" in metrics


def test_mae_train_step_applies_schedule_and_matches_upstream_jax() -> None:
    upstream_utils = _import_upstream("utils.hsdp_util")
    upstream_utils.set_global_mesh(1)
    upstream_train_mae = _import_upstream_training_module("train_mae")
    upstream_builder = _import_upstream("utils.model_builder")

    torch_model = ToyMaeTorch()
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.0)
    torch_state = TrainState.create(torch_model, torch_optimizer, ema_decay=0.9)
    torch_schedule = create_learning_rate_fn(
        learning_rate=0.3,
        warmup_steps=1,
        total_steps=3,
        lr_schedule="const",
    )
    upstream_schedule = upstream_builder.create_learning_rate_fn(
        learning_rate=0.3,
        warmup_steps=1,
        total_steps=3,
        lr_schedule="const",
    )

    batch_torch = {
        "images": torch.tensor([[[[0.2]]], [[[0.4]]]], dtype=torch.float32),
        "labels": torch.tensor([1, 2], dtype=torch.int64),
    }

    jax_model = ToyMaeJax()
    jax_model_any = cast(Any, jax_model)
    jax_batch = {
        "images": np.asarray(batch_torch["images"]),
        "labels": np.asarray(batch_torch["labels"], dtype=np.int32),
    }
    variables = jax_model_any.init(
        {"params": jax.random.PRNGKey(0)},
        x=jax_batch["images"],
        labels=jax_batch["labels"],
        train=True,
    )
    jax_state = upstream_train_mae.TrainState.create(
        apply_fn=jax_model_any.apply,
        params=variables["params"],
        tx=optax.sgd(learning_rate=upstream_schedule),
        ema_params=variables["params"],
        ema_decay=0.9,
    )

    for step in range(2):
        torch_state, torch_metrics = train_mae_step(
            torch_state,
            batch_torch,
            base_seed=7,
            forward_dict={},
            learning_rate_fn=torch_schedule,
            preprocess_fn=_identity_preprocess,
            max_grad_norm=10.0,
        )
        jax_state, jax_metrics = _run_upstream_generator_train_step(
            upstream_train_mae.train_step,
            jax_state,
            jax_batch,
            rng_init=jax.random.PRNGKey(7),
            forward_dict={},
            learning_rate_fn=upstream_schedule,
            preprocess_fn=_identity_jax_batch,
            max_grad_norm=10.0,
        )
        expected_lr = torch_schedule(step)
        assert torch_state.optimizer.param_groups[0]["lr"] == expected_lr
        assert torch_metrics["lr"] == expected_lr
        np.testing.assert_allclose(
            np.array(torch_model.weight.detach().cpu().item(), dtype=np.float32),
            np.asarray(jax_state.params["weight"]),
            rtol=1e-6,
            atol=1e-6,
        )
        ema_weight = torch_state.ema_model.state_dict()["weight"]
        np.testing.assert_allclose(
            np.array(ema_weight.detach().cpu().item(), dtype=np.float32),
            np.asarray(jax_state.ema_params["weight"]),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.array(torch_metrics["loss"], dtype=np.float32),
            np.asarray(jax_metrics["loss"]),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.array(torch_metrics["lr"], dtype=np.float32),
            np.asarray(jax_metrics["lr"]),
            rtol=1e-7,
            atol=1e-7,
        )


def test_generator_train_step_runs_on_small_model() -> None:
    generator_model = DitGen(
        cond_dim=16,
        num_classes=5,
        noise_classes=4,
        noise_coords=2,
        input_size=8,
        in_channels=3,
        patch_size=2,
        hidden_size=16,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        out_channels=3,
        n_cls_tokens=2,
        use_qknorm=True,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
    )
    optimizer = torch.optim.AdamW(generator_model.parameters(), lr=1e-3)
    state = TrainState.create(generator_model, optimizer, ema_decay=0.9)
    feature_model = MAEResNet(
        num_classes=5,
        in_channels=3,
        base_channels=8,
        patch_size=2,
        layers=(1, 1, 1, 1),
        input_patch_size=1,
    )
    activation_fn = build_activation_function(feature_model)
    schedule = create_learning_rate_fn(learning_rate=1e-3, warmup_steps=1, total_steps=10)

    labels = torch.tensor([1, 2], dtype=torch.int64)
    samples = torch.randn(2, 2, 8, 8, 3)
    negative_samples = torch.randn(2, 1, 8, 8, 3)

    state, metrics = train_generator_step(
        state,
        labels=labels,
        samples=samples,
        negative_samples=negative_samples,
        feature_apply=activation_fn,
        learning_rate_fn=schedule,
        base_seed=321,
        gen_per_label=2,
        activation_kwargs={
            "patch_mean_size": [2],
            "patch_std_size": [2],
            "use_std": True,
            "use_mean": True,
            "every_k_block": 1,
        },
        loss_kwargs={"r_list": (0.05, 0.2)},
    )

    assert state.step == 1
    assert "loss" in metrics
    assert "g_norm" in metrics
    assert "lr" in metrics


def test_generator_train_step_applies_schedule_and_matches_upstream_jax() -> None:
    upstream_utils = _import_upstream("utils.hsdp_util")
    upstream_utils.set_global_mesh(1)
    upstream_train = _import_upstream_training_module("train")
    upstream_builder = _import_upstream("utils.model_builder")

    torch_model = ToyGeneratorTorch()
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.0)
    torch_state = TrainState.create(torch_model, torch_optimizer, ema_decay=0.9)
    torch_schedule = create_learning_rate_fn(
        learning_rate=0.2,
        warmup_steps=1,
        total_steps=3,
        lr_schedule="const",
    )
    upstream_schedule = upstream_builder.create_learning_rate_fn(
        learning_rate=0.2,
        warmup_steps=1,
        total_steps=3,
        lr_schedule="const",
    )

    labels_torch = torch.tensor([1, 2], dtype=torch.int64)
    samples_torch = torch.tensor(
        [
            [[[[0.2]]], [[[0.4]]]],
            [[[[0.6]]], [[[0.8]]]],
        ],
        dtype=torch.float32,
    )
    negative_samples_torch = torch.tensor(
        [
            [[[[0.1]]]],
            [[[[0.3]]]],
        ],
        dtype=torch.float32,
    )

    jax_model = ToyGeneratorJax()
    jax_model_any = cast(Any, jax_model)
    labels_jax = np.asarray(labels_torch, dtype=np.int32)
    samples_jax = np.asarray(samples_torch)
    negative_samples_jax = np.asarray(negative_samples_torch)
    variables = jax_model_any.init(
        {"params": jax.random.PRNGKey(0)},
        c=labels_jax,
        cfg_scale=np.ones_like(labels_jax, dtype=np.float32),
        train=True,
    )
    jax_state = upstream_train.TrainState.create(
        apply_fn=jax_model_any.apply,
        params=variables["params"],
        tx=optax.sgd(learning_rate=upstream_schedule),
        ema_params=variables["params"],
        ema_decay=0.9,
    )

    for step in range(2):
        torch_state, torch_metrics = train_generator_step(
            torch_state,
            labels=labels_torch,
            samples=samples_torch,
            negative_samples=negative_samples_torch,
            feature_apply=_toy_feature_apply_torch,
            learning_rate_fn=torch_schedule,
            base_seed=11,
            cfg_min=1.0,
            cfg_max=1.0,
            neg_cfg_pw=1.0,
            no_cfg_frac=0.0,
            gen_per_label=2,
            activation_kwargs={},
            loss_kwargs={"r_list": (0.05, 0.2)},
            max_grad_norm=10.0,
        )
        jax_state, jax_metrics = _run_upstream_generator_train_step(
            upstream_train.train_step,
            jax_state,
            labels_jax,
            samples_jax,
            negative_samples_jax,
            feature_params={},
            feature_apply=_toy_feature_apply_jax,
            rng_init=jax.random.PRNGKey(11),
            learning_rate_fn=upstream_schedule,
            cfg_min=1.0,
            cfg_max=1.0,
            neg_cfg_pw=1.0,
            no_cfg_frac=0.0,
            gen_per_label=2,
            activation_kwargs={},
            loss_kwargs={"R_list": (0.05, 0.2)},
            max_grad_norm=10.0,
        )
        expected_lr = torch_schedule(step)
        assert torch_state.optimizer.param_groups[0]["lr"] == expected_lr
        assert torch_metrics["lr"] == expected_lr
        np.testing.assert_allclose(
            np.array(torch_model.weight.detach().cpu().item(), dtype=np.float32),
            np.asarray(jax_state.params["weight"]),
            rtol=1e-6,
            atol=1e-6,
        )
        ema_weight = torch_state.ema_model.state_dict()["weight"]
        np.testing.assert_allclose(
            np.array(ema_weight.detach().cpu().item(), dtype=np.float32),
            np.asarray(jax_state.ema_params["weight"]),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.array(torch_metrics["loss"], dtype=np.float32),
            np.asarray(jax_metrics["loss"]),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.array(torch_metrics["lr"], dtype=np.float32),
            np.asarray(jax_metrics["lr"]),
            rtol=1e-7,
            atol=1e-7,
        )
