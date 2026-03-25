"""Feature-model construction helpers."""

from __future__ import annotations

import torch
from torch import nn

from kdrifting.hf import load_mae_model
from kdrifting.models.convnext import ConvNextBase


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
