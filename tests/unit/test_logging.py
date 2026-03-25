from __future__ import annotations

import numpy as np

from kdrifting.logging import WandbLogger


def test_normalize_images_accepts_bchw_float_input() -> None:
    images = np.ones((2, 3, 4, 5), dtype=np.float32) * 0.5

    normalized = WandbLogger.normalize_images(images)

    assert normalized.shape == (2, 4, 5, 3)
    assert normalized.dtype == np.uint8
    assert normalized[0, 0, 0, 0] == 127


def test_normalize_images_expands_single_channel_images() -> None:
    images = np.zeros((1, 4, 5, 1), dtype=np.float32)

    normalized = WandbLogger.normalize_images(images)

    assert normalized.shape == (1, 4, 5, 3)
