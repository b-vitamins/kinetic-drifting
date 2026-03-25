"""Shared tensor and module helpers for model implementations."""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn

DEFAULT_EPS: Final[float] = 1e-6


def bhwc_to_bchw(x: Tensor) -> Tensor:
    return x.permute(0, 3, 1, 2).contiguous()


def bchw_to_bhwc(x: Tensor) -> Tensor:
    return x.permute(0, 2, 3, 1).contiguous()


def safe_std(x: Tensor, dim: int | tuple[int, ...], eps: float = DEFAULT_EPS) -> Tensor:
    x32 = x.to(dtype=torch.float32)
    mean = x32.mean(dim=dim, keepdim=True)
    var = ((x32 - mean) ** 2).mean(dim=dim, keepdim=False)
    return torch.sqrt(torch.clamp(var, min=0.0) + eps)


class RMSNorm(nn.Module):
    """PyTorch RMSNorm matching the source implementation."""

    def __init__(
        self,
        dim: int,
        *,
        eps: float = DEFAULT_EPS,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        variance = x.to(dtype=torch.float32).pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            normed = normed * self.weight
        return normed.to(dtype=input_dtype)
