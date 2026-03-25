from __future__ import annotations

import torch

from kdrifting.data.imagenet import get_postprocess_fn


def test_pixel_postprocess_returns_bchw() -> None:
    postprocess = get_postprocess_fn(use_latent=False, use_cache=False, has_clip=True)
    images = torch.zeros((2, 8, 8, 3), dtype=torch.float32)

    out = postprocess(images)

    assert out.shape == (2, 3, 8, 8)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0
