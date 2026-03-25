"""Feature-model construction helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
from torch import Tensor, nn

from kdrifting.hf import load_mae_model
from kdrifting.models.convnext import ConvNextBase, ConvNextV2
from kdrifting.models.mae import MAEResNet, build_activation_function


def build_feature_model(
    *,
    path: str = "",
    use_convnext: bool = False,
    convnext_dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Build the feature model used by drift training."""
    if use_convnext:
        model = ConvNextBase(dtype=convnext_dtype)
        model.eval()
        return model
    if not path:
        raise ValueError("`path` is required when use_convnext=False.")
    model, _ = load_mae_model(path)
    model.eval()
    return model


def resolve_mae_path(feature_config: dict[str, Any]) -> str:
    """Resolve the MAE feature artifact path from source-compatible config fields."""
    mae_path = str(feature_config.get("mae_path", "")).strip()
    if mae_path:
        return mae_path
    if not bool(feature_config.get("use_mae", True)):
        return ""

    load_dict = dict(feature_config.get("load_dict", {}) or {})
    if str(load_dict.get("source", "hf")).strip().lower() == "local":
        return str(load_dict.get("path", "")).strip()

    hf_model_name = str(load_dict.get("hf_model_name", "")).strip()
    return f"hf://{hf_model_name}" if hf_model_name else ""


def build_feature_activation(
    *,
    feature_config: dict[str, Any],
    postprocess_fn: Callable[[Tensor], Tensor],
    device: torch.device,
) -> Callable[..., dict[str, Tensor]]:
    """Build the drift-training activation function from a feature config."""
    mae_model: MAEResNet | None = None
    convnext_model: ConvNextV2 | None = None
    use_mae = bool(feature_config.get("use_mae", True))
    use_convnext = bool(feature_config.get("use_convnext", False))
    if use_mae:
        mae_path = resolve_mae_path(feature_config)
        if not mae_path:
            raise ValueError(
                "feature.mae_path (or feature.load_dict.*) is required when use_mae=true.",
            )
        loaded_mae = cast(MAEResNet, build_feature_model(path=mae_path))
        loaded_mae.to(device)
        loaded_mae.eval()
        for parameter in loaded_mae.parameters():
            parameter.requires_grad_(False)
        mae_model = loaded_mae

    if use_convnext:
        convnext_dtype = (
            torch.bfloat16 if bool(feature_config.get("convnext_bf16", False)) else torch.float32
        )
        loaded_convnext = cast(
            ConvNextV2,
            build_feature_model(
                use_convnext=True,
                convnext_dtype=convnext_dtype,
            ),
        )
        loaded_convnext.to(device)
        loaded_convnext.eval()
        for parameter in loaded_convnext.parameters():
            parameter.requires_grad_(False)
        convnext_model = loaded_convnext

    return build_activation_function(
        mae_model,
        convnext_model=convnext_model,
        postprocess_fn=postprocess_fn,
    )
