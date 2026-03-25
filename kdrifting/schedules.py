"""Learning-rate schedule helpers."""

from __future__ import annotations

import math
from typing import Protocol


class StepSchedule(Protocol):
    """Callable protocol for integer-step schedules."""

    def __call__(self, step: int) -> float: ...


def _linear_warmup(
    step: int,
    *,
    init_value: float,
    end_value: float,
    transition_steps: int,
) -> float:
    if transition_steps <= 0:
        return end_value
    frac = min(max(step, 0) / transition_steps, 1.0)
    return init_value + frac * (end_value - init_value)


def _cosine_decay(step: int, *, init_value: float, decay_steps: int, alpha: float) -> float:
    if decay_steps <= 0:
        return init_value
    clipped_step = min(max(step, 0), decay_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * clipped_step / decay_steps))
    decayed = (1.0 - alpha) * cosine_decay + alpha
    return init_value * decayed


def create_learning_rate_fn(
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    lr_schedule: str = "const",
) -> StepSchedule:
    """Create the warmup-plus-main learning-rate schedule used by the source repo."""
    init_value = 1e-6
    normalized_schedule = lr_schedule.lower()
    cosine_steps = max(total_steps - warmup_steps, 1)

    def schedule(step: int) -> float:
        step_value = int(step)
        if step_value < warmup_steps:
            return _linear_warmup(
                step_value,
                init_value=init_value,
                end_value=learning_rate,
                transition_steps=warmup_steps,
            )

        inner_step = step_value - warmup_steps
        if normalized_schedule in {"cos", "cosine"}:
            return _cosine_decay(
                inner_step,
                init_value=learning_rate,
                decay_steps=cosine_steps,
                alpha=1e-6,
            )
        if normalized_schedule == "const":
            return learning_rate
        raise NotImplementedError(normalized_schedule)

    return schedule
