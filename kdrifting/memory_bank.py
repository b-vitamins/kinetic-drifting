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
    seed: int | None = None
    dtype: np.dtype[Any] = field(default_factory=lambda: np.dtype(np.float32))
    bank: npt.NDArray[Any] | None = field(default=None, init=False)
    feature_shape: tuple[int, ...] | None = field(default=None, init=False)
    ptr: np.ndarray = field(init=False)
    count: np.ndarray = field(init=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.ptr = np.zeros(self.num_classes, dtype=np.int32)
        self.count = np.zeros(self.num_classes, dtype=np.int32)
        self._rng = np.random.default_rng(self.seed)

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
                sample_indices[index] = self._rng.choice(
                    valid,
                    n_samples,
                    replace=(valid < n_samples),
                )

        out = self.bank[labels_np[:, None], sample_indices]
        return torch.as_tensor(out, device=device)

    def state_dict(self) -> dict[str, Any]:
        """Serialize the memory bank contents and RNG state."""
        return {
            "num_classes": self.num_classes,
            "max_size": self.max_size,
            "seed": self.seed,
            "dtype": self.dtype.str,
            "bank": self.bank,
            "feature_shape": self.feature_shape,
            "ptr": self.ptr.copy(),
            "count": self.count.copy(),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore the memory bank contents and RNG state."""
        self.num_classes = int(payload["num_classes"])
        self.max_size = int(payload["max_size"])
        seed = payload.get("seed")
        self.seed = None if seed is None else int(seed)
        self.dtype = np.dtype(str(payload["dtype"]))
        feature_shape = payload.get("feature_shape")
        self.feature_shape = None if feature_shape is None else tuple(int(v) for v in feature_shape)
        bank = payload.get("bank")
        self.bank = None if bank is None else np.asarray(bank, dtype=self.dtype)
        self.ptr = np.asarray(payload["ptr"], dtype=np.int32).copy()
        self.count = np.asarray(payload["count"], dtype=np.int32).copy()
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = payload["rng_state"]
