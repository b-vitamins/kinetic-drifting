"""Minimal distributed helpers for PyTorch-based execution."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar, cast

import torch.distributed as dist

T = TypeVar("T")


def is_distributed() -> bool:
    """Return whether ``torch.distributed`` is initialized."""
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    """Return the current process rank."""
    if not is_distributed():
        return 0
    return dist.get_rank()


def world_size() -> int:
    """Return the distributed world size."""
    if not is_distributed():
        return 1
    return dist.get_world_size()


def is_rank_zero() -> bool:
    """Return whether the current process is rank zero."""
    return rank() == 0


def barrier() -> None:
    """Synchronize all processes when distributed is active."""
    if is_distributed():
        dist_module: Any = dist
        barrier_fn = cast(Callable[[], object], dist_module.barrier)
        barrier_fn()


def all_gather_objects(obj: T) -> list[T]:
    """Gather an arbitrary Python object from all ranks."""
    if not is_distributed():
        return [obj]
    gathered: list[T | None] = [None for _ in range(world_size())]
    dist_module: Any = dist
    all_gather = cast(Callable[[list[Any], Any], None], dist_module.all_gather_object)
    all_gather(gathered, obj)
    return [item for item in gathered if item is not None]


def flatten_gathered(items: Iterable[list[T]]) -> list[T]:
    """Flatten a gathered ``list[list[T]]`` structure."""
    flattened: list[T] = []
    for chunk in items:
        flattened.extend(chunk)
    return flattened
