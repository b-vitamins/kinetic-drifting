from __future__ import annotations

import math

from kdrifting.schedules import create_learning_rate_fn


def test_const_schedule_matches_expected_warmup_behavior() -> None:
    schedule = create_learning_rate_fn(
        learning_rate=0.2,
        warmup_steps=4,
        total_steps=20,
        lr_schedule="const",
    )

    assert math.isclose(schedule(0), 1e-6)
    assert math.isclose(schedule(2), 0.1000005)
    assert math.isclose(schedule(4), 0.2)
    assert math.isclose(schedule(17), 0.2)


def test_cosine_schedule_decays_after_warmup() -> None:
    schedule = create_learning_rate_fn(
        learning_rate=1.0,
        warmup_steps=2,
        total_steps=6,
        lr_schedule="cosine",
    )

    assert math.isclose(schedule(2), 1.0)
    assert schedule(4) < schedule(3)
    assert math.isclose(schedule(6), 1e-6, rel_tol=0.0, abs_tol=1e-10)
