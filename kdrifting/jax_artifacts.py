"""JAX artifact loading and parameter conversion helpers."""

from __future__ import annotations

import importlib
import json
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from kdrifting.models.generator import DitGen, get_2d_sincos_pos_embed
from kdrifting.models.mae import MAEResNet

_LEGACY_METADATA_NAME = "ema_model.metadata.json"
_LEGACY_PARAMS_NAME = "ema_model.msgpack"
_PARAMS_NAME = "ema_params.msgpack"


def _flatten_tree(tree: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(tree, dict):
        for key, value in cast(dict[object, Any], tree).items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_tree(value, name))
        return flat
    flat[prefix] = tree
    return flat


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _metadata_candidates(artifact_dir: Path) -> tuple[Path, ...]:
    candidates = [artifact_dir / "metadata.json", artifact_dir / _LEGACY_METADATA_NAME]
    if artifact_dir.name == "checkpoints":
        candidates.extend(
            [
                artifact_dir.parent / "params_ema" / "metadata.json",
                artifact_dir.parent / "params_ema" / _LEGACY_METADATA_NAME,
                artifact_dir.parent / "metadata.json",
            ],
        )
    return tuple(candidates)


def read_jax_metadata(artifact_dir: Path) -> dict[str, Any]:
    """Read the best-effort metadata bundle for a JAX artifact or checkpoint."""
    for candidate in _metadata_candidates(artifact_dir):
        if candidate.is_file():
            return _load_json(candidate)
    return {}


def _import_flax_serialization() -> Any:
    return importlib.import_module("flax.serialization")


def _import_flax_checkpoints() -> Any:
    return importlib.import_module("flax.training.checkpoints")


def load_jax_init_entry(artifact_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Load raw JAX params plus metadata from a local artifact directory."""
    metadata = read_jax_metadata(artifact_dir)

    params_path = artifact_dir / _PARAMS_NAME
    if params_path.is_file():
        serialization = _import_flax_serialization()
        return serialization.msgpack_restore(params_path.read_bytes()), metadata

    legacy_params_path = artifact_dir / _LEGACY_PARAMS_NAME
    if legacy_params_path.is_file():
        serialization = _import_flax_serialization()
        legacy_metadata = metadata or read_jax_metadata(artifact_dir)
        return serialization.msgpack_restore(legacy_params_path.read_bytes()), legacy_metadata

    checkpoints = _import_flax_checkpoints()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sharding info not provided when restoring\\..*",
            category=UserWarning,
        )
        restored = checkpoints.restore_checkpoint(str(artifact_dir), target=None, step=None)
    if isinstance(restored, dict) and "params" in restored:
        return cast(Any, restored["params"]), metadata

    raise ValueError(f"Could not restore JAX params from {artifact_dir}")


def resolve_jax_checkpoint_dir(path: Path) -> Path:
    """Resolve a run root or checkpoint directory to the actual JAX checkpoint dir."""
    candidate = path.resolve()
    if candidate.name == "checkpoints" and candidate.is_dir():
        return candidate

    checkpoints_dir = candidate / "checkpoints"
    if checkpoints_dir.is_dir():
        return checkpoints_dir

    raise FileNotFoundError(f"Could not find a JAX checkpoint directory under {candidate}")


def load_jax_checkpoint_entry(checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a full JAX checkpoint payload plus sibling metadata."""
    checkpoint_dir = resolve_jax_checkpoint_dir(checkpoint_path)
    metadata = read_jax_metadata(checkpoint_dir)
    checkpoints = _import_flax_checkpoints()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sharding info not provided when restoring\\..*",
            category=UserWarning,
        )
        restored = checkpoints.restore_checkpoint(str(checkpoint_dir), target=None, step=None)
    if not isinstance(restored, dict):
        raise ValueError(f"Could not restore a JAX checkpoint payload from {checkpoint_dir}")
    return cast(dict[str, Any], restored), metadata


def _to_numpy(value: Any) -> np.ndarray[Any, Any]:
    array = np.asarray(value)
    if str(array.dtype) == "bfloat16":
        return array.astype(np.float32)
    return array


def _convert_leaf(source_key: str, source_value: Any, target_value: Tensor) -> Tensor:
    array = _to_numpy(source_value)
    target_shape = tuple(target_value.shape)

    if source_key.endswith(".kernel"):
        if array.ndim == 4:
            array = np.transpose(array, (3, 2, 0, 1))
        elif array.ndim == 2:
            array = np.transpose(array, (1, 0))

    if array.shape != target_shape:
        raise ValueError(
            f"Shape mismatch for {source_key}: source {array.shape} -> target {target_shape}",
        )

    copied = np.array(array, copy=True, order="C")
    return torch.tensor(copied, dtype=target_value.dtype)


def _convert_tree(
    flat_params: dict[str, Any],
    target_tensors: Mapping[str, Tensor],
    *,
    target_key_fn: Any,
    strict: bool,
    missing_label: str,
) -> dict[str, Tensor]:
    converted: dict[str, Tensor] = {}
    for source_key, source_value in flat_params.items():
        target_key = target_key_fn(source_key)
        if target_key is None:
            continue
        if target_key not in target_tensors:
            raise KeyError(f"Unmapped JAX parameter: {source_key} -> {target_key}")
        if target_key in converted:
            raise KeyError(f"Duplicate target parameter mapping: {target_key}")
        converted[target_key] = _convert_leaf(source_key, source_value, target_tensors[target_key])

    if strict:
        missing = sorted(set(target_tensors) - set(converted))
        if missing:
            raise KeyError(f"Missing converted {missing_label}: {missing}")
    return converted


def _mae_target_key(source_key: str) -> str:
    key = re.sub(r"encoder\.stages_(\d+)\.layers_(\d+)", r"encoder.stages.\1.\2", source_key)
    key = re.sub(
        r"encoder\.layer([1-4])_norm",
        lambda match: f"encoder.stage_norms.{int(match.group(1)) - 1}",
        key,
    )
    key = key.replace(".concat_norm_fn.", ".concat_norm.")
    if key.endswith(".kernel"):
        return f"{key.removesuffix('.kernel')}.weight"
    if key.endswith(".scale"):
        return f"{key.removesuffix('.scale')}.weight"
    return key


def convert_mae_jax_params(params: Any, model: MAEResNet) -> dict[str, Tensor]:
    """Convert upstream JAX MAE params to a PyTorch state dict."""
    flat_params = _flatten_tree(params)
    return _convert_tree(
        flat_params,
        model.state_dict(),
        target_key_fn=_mae_target_key,
        strict=True,
        missing_label="MAE parameters",
    )


def convert_mae_jax_optimizer_tensors(params: Any, model: MAEResNet) -> dict[str, Tensor]:
    """Convert MAE-shaped JAX optimizer tensors into torch named-parameter tensors."""
    flat_params = _flatten_tree(params)
    named_parameters = dict(model.named_parameters())
    return _convert_tree(
        flat_params,
        named_parameters,
        target_key_fn=_mae_target_key,
        strict=True,
        missing_label="MAE optimizer tensors",
    )


def _generator_target_key(source_key: str) -> str | None:
    if source_key == "Embed_0.embedding":
        return "class_embed.weight"
    if match := re.fullmatch(r"noise_embeds_(\d+)\.embedding", source_key):
        return f"noise_embeds.{match.group(1)}.weight"
    if match := re.fullmatch(
        r"TimestepEmbedder_0\.TorchLinear_(\d+)\.Dense_0\.(kernel|bias)",
        source_key,
    ):
        layer_index = int(match.group(1)) + 1
        param_name = "weight" if match.group(2) == "kernel" else "bias"
        return f"cfg_embedder.fc{layer_index}.linear.{param_name}"
    if source_key == "RMSNorm_0.weight":
        return "cfg_norm.weight"
    if match := re.fullmatch(
        r"LightningDiT_0\.TorchLinear_(\d+)\.Dense_0\.(kernel|bias)",
        source_key,
    ):
        layer_index = int(match.group(1))
        target_prefix = {0: "model.patch_embed", 1: "model.cls_proj"}.get(layer_index)
        if target_prefix is None:
            raise KeyError(f"Unexpected LightningDiT top-level linear index: {source_key}")
        param_name = "weight" if match.group(2) == "kernel" else "bias"
        return f"{target_prefix}.linear.{param_name}"
    if source_key == "LightningDiT_0.pos_embed":
        return None
    if source_key == "LightningDiT_0.cls_embed":
        return "model.cls_embed"
    if match := re.fullmatch(
        r"LightningDiT_0\.blocks_(\d+)\.(RMSNorm|LayerNorm)_(\d+)\.(weight|scale|bias)",
        source_key,
    ):
        block_index = int(match.group(1))
        norm_index = int(match.group(3)) + 1
        param_name = "weight" if match.group(4) in {"weight", "scale"} else "bias"
        return f"model.blocks.{block_index}.norm{norm_index}.{param_name}"
    if match := re.fullmatch(
        r"LightningDiT_0\.blocks_(\d+)\.TorchLinear_0\.Dense_0\.(kernel|bias)",
        source_key,
    ):
        block_index = int(match.group(1))
        param_name = "weight" if match.group(2) == "kernel" else "bias"
        return f"model.blocks.{block_index}.adaln.1.linear.{param_name}"
    if match := re.fullmatch(
        r"LightningDiT_0\.blocks_(\d+)\.Attention_0\.TorchLinear_(\d+)\.Dense_0\.(kernel|bias)",
        source_key,
    ):
        block_index = int(match.group(1))
        target_name = {0: "qkv", 1: "proj"}.get(int(match.group(2)))
        if target_name is None:
            raise KeyError(f"Unexpected attention linear index: {source_key}")
        param_name = "weight" if match.group(3) == "kernel" else "bias"
        return f"model.blocks.{block_index}.attn.{target_name}.linear.{param_name}"
    if match := re.fullmatch(
        r"LightningDiT_0\.blocks_(\d+)\.Attention_0\.(q_norm|k_norm)\.(weight|scale|bias)",
        source_key,
    ):
        block_index = int(match.group(1))
        param_name = "weight" if match.group(3) in {"weight", "scale"} else "bias"
        return f"model.blocks.{block_index}.attn.{match.group(2)}.{param_name}"
    if match := re.fullmatch(
        r"LightningDiT_0\.blocks_(\d+)\.SwiGLUFFN_0\.TorchLinear_(\d+)\.Dense_0\.(kernel|bias)",
        source_key,
    ):
        block_index = int(match.group(1))
        target_name = {0: "w1", 1: "w3", 2: "proj"}.get(int(match.group(2)))
        if target_name is None:
            raise KeyError(f"Unexpected SwiGLU linear index: {source_key}")
        param_name = "weight" if match.group(3) == "kernel" else "bias"
        return f"model.blocks.{block_index}.mlp.{target_name}.linear.{param_name}"
    if match := re.fullmatch(
        r"LightningDiT_0\.blocks_(\d+)\.StandardMLP_0\.TorchLinear_(\d+)\.Dense_0\.(kernel|bias)",
        source_key,
    ):
        block_index = int(match.group(1))
        target_name = {0: "fc1", 1: "fc2"}.get(int(match.group(2)))
        if target_name is None:
            raise KeyError(f"Unexpected MLP linear index: {source_key}")
        param_name = "weight" if match.group(3) == "kernel" else "bias"
        return f"model.blocks.{block_index}.mlp.{target_name}.linear.{param_name}"
    if match := re.fullmatch(
        r"LightningDiT_0\.FinalLayer_0\.(RMSNorm|LayerNorm)_0\.(weight|scale|bias)",
        source_key,
    ):
        param_name = "weight" if match.group(2) in {"weight", "scale"} else "bias"
        return f"model.final_layer.norm.{param_name}"
    if match := re.fullmatch(
        r"LightningDiT_0\.FinalLayer_0\.TorchLinear_(\d+)\.Dense_0\.(kernel|bias)",
        source_key,
    ):
        target_name = {0: "adaln.1.linear", 1: "proj.linear"}.get(int(match.group(1)))
        if target_name is None:
            raise KeyError(f"Unexpected final-layer linear index: {source_key}")
        param_name = "weight" if match.group(2) == "kernel" else "bias"
        return f"model.final_layer.{target_name}.{param_name}"
    raise KeyError(f"Unsupported generator parameter: {source_key}")


def _validate_generator_pos_embed(flat_params: dict[str, Any], model: DitGen) -> None:
    source_key = "LightningDiT_0.pos_embed"
    if source_key not in flat_params:
        return

    source_value = _to_numpy(flat_params[source_key]).astype(np.float32, copy=False)
    num_patches = source_value.shape[1]
    grid_size = int(round(num_patches**0.5))
    target_value = get_2d_sincos_pos_embed(model.model.hidden_size, grid_size)
    target_value = target_value.astype(np.float32, copy=False)[None, :, :]
    if not np.allclose(source_value, target_value, atol=1e-6, rtol=1e-6):
        raise ValueError(
            "Source JAX positional embedding does not match the PyTorch runtime buffer.",
        )


def convert_generator_jax_params(params: Any, model: DitGen) -> dict[str, Tensor]:
    """Convert upstream JAX generator params to a PyTorch state dict."""
    flat_params = _flatten_tree(params)
    _validate_generator_pos_embed(flat_params, model)
    return _convert_tree(
        flat_params,
        model.state_dict(),
        target_key_fn=_generator_target_key,
        strict=True,
        missing_label="generator parameters",
    )


def convert_generator_jax_optimizer_tensors(params: Any, model: DitGen) -> dict[str, Tensor]:
    """Convert generator-shaped JAX optimizer tensors into torch named-parameter tensors."""
    flat_params = _flatten_tree(params)
    named_parameters = dict(model.named_parameters())
    return _convert_tree(
        flat_params,
        named_parameters,
        target_key_fn=_generator_target_key,
        strict=True,
        missing_label="generator optimizer tensors",
    )
