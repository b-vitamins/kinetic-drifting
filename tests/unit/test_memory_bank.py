from __future__ import annotations

import numpy as np
import torch

from kdrifting.memory_bank import ArrayMemoryBank


def test_memory_bank_overwrites_oldest_samples_per_class() -> None:
    bank = ArrayMemoryBank(num_classes=2, max_size=2)
    samples = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 0], dtype=np.int32)

    bank.add(samples, labels)

    assert bank.bank is not None
    np.testing.assert_allclose(bank.bank[0], np.array([[3.0, 30.0], [2.0, 20.0]], dtype=np.float32))
    np.testing.assert_array_equal(bank.count, np.array([2, 0], dtype=np.int32))


def test_memory_bank_sampling_returns_torch_tensor() -> None:
    np.random.seed(7)
    bank = ArrayMemoryBank(num_classes=2, max_size=4)
    bank.add(
        np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=np.float32,
        ),
        np.array([1, 1, 1, 1], dtype=np.int32),
    )

    sampled = bank.sample(np.array([1, 1], dtype=np.int64), n_samples=3)

    assert isinstance(sampled, torch.Tensor)
    assert sampled.shape == (2, 3, 2)
    assert sampled.dtype == torch.float32
