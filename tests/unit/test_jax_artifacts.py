from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from kdrifting.hf import load_generator_model, load_mae_model
from kdrifting.models.generator import DitGen
from kdrifting.models.mae import MAEResNet

UPSTREAM_ROOT = Path("/home/b/projects/drifting")
FLAX_SERIALIZATION = cast(Any, importlib.import_module("flax.serialization"))
JAX = cast(Any, importlib.import_module("jax"))


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
    root.mkdir()
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "ema_params.msgpack").write_bytes(FLAX_SERIALIZATION.msgpack_serialize(params))
    return root


def _torch_to_numpy(tensor: torch.Tensor) -> np.ndarray[Any, Any]:
    return tensor.detach().cpu().numpy()


def _torch_to_numpy_int32(tensor: torch.Tensor) -> np.ndarray[Any, Any]:
    return tensor.detach().cpu().to(dtype=torch.int32).numpy()


def _assert_close(
    actual: torch.Tensor,
    expected: Any,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    copied = np.array(np.asarray(expected), copy=True, order="C")
    expected_tensor = torch.tensor(copied, dtype=actual.dtype)
    assert actual.shape == expected_tensor.shape
    assert torch.allclose(actual.detach().cpu(), expected_tensor, atol=atol, rtol=rtol)


def test_load_mae_model_converts_jax_artifact_with_numerical_parity(tmp_path: Path) -> None:
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
    model = upstream_mae._mae_from_metadata(metadata)
    variables = model.init(
        {
            "params": JAX.random.PRNGKey(0),
            "masking": JAX.random.PRNGKey(1),
            "dropout": JAX.random.PRNGKey(2),
        },
        **model.dummy_input(),
    )
    artifact_dir = _write_jax_artifact(
        tmp_path / "mae_artifact",
        metadata=metadata,
        params=variables["params"],
    )

    loaded_model, loaded_metadata = load_mae_model(str(artifact_dir))
    loaded_model = cast(MAEResNet, loaded_model)

    assert loaded_metadata["backend"] == "jax"

    x = torch.linspace(-1.0, 1.0, steps=2 * 32 * 32 * 3, dtype=torch.float32).reshape(
        2,
        32,
        32,
        3,
    )
    labels = torch.tensor([1, 2], dtype=torch.int64)

    loss_torch, metrics_torch = loaded_model(
        x,
        labels,
        lambda_cls=0.25,
        mask_ratio_min=0.0,
        mask_ratio_max=0.0,
        train=False,
        generator=torch.Generator().manual_seed(0),
    )
    loss_jax, metrics_jax = model.apply(
        {"params": variables["params"]},
        _torch_to_numpy(x),
        _torch_to_numpy_int32(labels),
        lambda_cls=0.25,
        mask_ratio_min=0.0,
        mask_ratio_max=0.0,
        train=False,
        rngs={"masking": JAX.random.PRNGKey(0)},
    )

    _assert_close(loss_torch, loss_jax, atol=3e-4, rtol=1e-4)
    for key, value in metrics_torch.items():
        _assert_close(value, metrics_jax[key], atol=3e-4, rtol=1e-4)

    activations_torch = loaded_model.get_activations(
        x,
        patch_mean_size=[2],
        patch_std_size=[2],
        every_k_block=1,
    )
    activations_jax = cast(
        dict[str, Any],
        model.apply(
            {"params": variables["params"]},
            _torch_to_numpy(x),
            method=model.get_activations,
            patch_mean_size=[2],
            patch_std_size=[2],
            every_k_block=1,
        ),
    )
    assert set(activations_torch) == set(activations_jax)
    for key, value in activations_torch.items():
        _assert_close(value, activations_jax[key], atol=6e-3, rtol=1e-3)


def test_load_generator_model_converts_jax_artifact_with_numerical_parity(
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
        "model_config": model_config,
    }
    model = upstream_generator.build_generator_from_config(model_config)
    variables = model.init(
        {"params": JAX.random.PRNGKey(0), "noise": JAX.random.PRNGKey(1)},
        **model.dummy_input(),
    )
    artifact_dir = _write_jax_artifact(
        tmp_path / "generator_artifact",
        metadata=metadata,
        params=variables["params"],
    )

    loaded_model, loaded_metadata = load_generator_model(str(artifact_dir))
    loaded_model = cast(DitGen, loaded_model)

    assert loaded_metadata["backend"] == "jax"

    labels = torch.tensor([1, 2], dtype=torch.int64)
    noise_labels = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    cfg_scale = 1.5

    cond_torch = loaded_model.c_cfg_noise_to_cond(labels, cfg_scale, noise_labels)
    cond_jax = model.apply(
        {"params": variables["params"]},
        _torch_to_numpy_int32(labels),
        cfg_scale,
        _torch_to_numpy_int32(noise_labels),
        method=model.c_cfg_noise_to_cond,
    )
    _assert_close(cond_torch, cond_jax, atol=1e-5, rtol=1e-5)

    noise = torch.linspace(-0.5, 0.5, steps=2 * 8 * 8 * 3, dtype=torch.float32).reshape(
        2,
        8,
        8,
        3,
    )
    samples_torch = loaded_model.generate_image(noise, cond_torch, deterministic=True)
    samples_jax = model.apply(
        {"params": variables["params"]},
        _torch_to_numpy(noise),
        cond_jax,
        deterministic=True,
        method=model.generate_image,
    )
    _assert_close(samples_torch, samples_jax, atol=2e-5, rtol=2e-5)
