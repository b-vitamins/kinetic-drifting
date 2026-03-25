"""Memory-bank utilities used by generator training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch


def _to_numpy(array: Any) -> npt.NDArray[Any]:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


@dataclass(slots=True)
class ArrayMemoryBank:
    """Class-wise ring buffer for samples used during generator training."""

    num_classes: int = 1000
    max_size: int = 64
    dtype: np.dtype[Any] = field(default_factory=lambda: np.dtype(np.float32))
    bank: npt.NDArray[Any] | None = field(default=None, init=False)
    feature_shape: tuple[int, ...] | None = field(default=None, init=False)
    ptr: np.ndarray = field(init=False)
    count: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.ptr = np.zeros(self.num_classes, dtype=np.int32)
        self.count = np.zeros(self.num_classes, dtype=np.int32)

    def _init_bank(self, sample_shape: tuple[int, ...]) -> None:
        self.feature_shape = tuple(sample_shape)
        self.bank = np.zeros(
            (self.num_classes, self.max_size, *self.feature_shape),
            dtype=self.dtype,
        )

    def add(self, samples: Any, labels: Any) -> None:
        """Insert samples into their class-specific ring buffers."""
        samples_np = _to_numpy(samples)
        labels_np = _to_numpy(labels)
        if self.bank is None:
            self._init_bank(tuple(samples_np.shape[1:]))

        assert self.bank is not None
        for index, label_value in enumerate(labels_np):
            class_index = int(label_value)
            write_index = int(self.ptr[class_index])
            self.bank[class_index, write_index] = samples_np[index]
            self.ptr[class_index] = (write_index + 1) % self.max_size
            if self.count[class_index] < self.max_size:
                self.count[class_index] += 1

    def sample(
        self,
        labels: Any,
        n_samples: int,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Sample stored entries for each label."""
        if self.bank is None or self.feature_shape is None:
            raise RuntimeError("MemoryBank is empty. Call add() before sample().")

        labels_np = _to_numpy(labels).astype(np.int64, copy=False)
        batch_size = int(labels_np.shape[0])
        sample_indices = np.empty((batch_size, n_samples), dtype=np.int32)
        for index, label_value in enumerate(labels_np):
            class_index = int(label_value)
            valid = int(self.count[class_index])
            if valid <= 0:
                sample_indices[index] = np.zeros((n_samples,), dtype=np.int32)
            else:
                sample_indices[index] = np.random.choice(
                    valid,
                    n_samples,
                    replace=(valid < n_samples),
                )

        out = self.bank[labels_np[:, None], sample_indices]
        return torch.as_tensor(out, device=device)
