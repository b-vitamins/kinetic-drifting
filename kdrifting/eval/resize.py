"""Resize helper matching the source FID evaluation preprocessing."""

from __future__ import annotations

import torch
from torch import Tensor


def forward(image: Tensor) -> Tensor:
    """Resize a BCHW float tensor to 299x299 and normalize to the FID input range."""
    batch_size, channels, _, _ = image.shape
    theta = torch.eye(2, 3, device=image.device, dtype=image.dtype)
    theta0 = theta.unsqueeze(0).repeat(batch_size, 1, 1)
    grid = torch.nn.functional.affine_grid(
        theta0,
        [batch_size, channels, 299, 299],
        align_corners=False,
    )
    resized = torch.nn.functional.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return (resized - 128.0) / 128.0
