from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from kdrifting.checkpointing import (
    restore_checkpoint,
    restore_checkpoint_extra_state,
    save_checkpoint,
)
from kdrifting.memory_bank import ArrayMemoryBank
from kdrifting.training.state import TrainState


def _make_state() -> TrainState:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = TrainState.create(model, optimizer, ema_decay=0.95)

    inputs = torch.tensor([[0.5, -1.0, 2.0]], dtype=torch.float32)
    loss = state.model(inputs).sum()
    loss.backward()
    state.optimizer.step()
    state.optimizer.zero_grad(set_to_none=True)
    state.update_ema()
    state.step = 7
    return state


def _assert_state_dict_equal(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(actual[key], value), key


def test_checkpoint_roundtrip_restores_train_state_and_extra_state(tmp_path: Path) -> None:
    state = _make_state()
    positive_bank = ArrayMemoryBank(num_classes=3, max_size=2, seed=17)
    positive_bank.add(
        np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ],
            dtype=np.float32,
        ),
        np.array([0, 1, 0], dtype=np.int32),
    )
    positive_bank.sample(np.array([0], dtype=np.int64), n_samples=2)
    extra_state = {"memory_bank_positive": positive_bank.state_dict()}

    save_checkpoint(state, workdir=str(tmp_path), keep=2, keep_every=1, extra_state=extra_state)

    restored = _make_state()
    restored = restore_checkpoint(restored, workdir=str(tmp_path), step=7)

    assert restored.step == 7
    assert restored.ema_decay == state.ema_decay
    _assert_state_dict_equal(restored.model.state_dict(), state.model.state_dict())
    _assert_state_dict_equal(restored.ema_model.state_dict(), state.ema_model.state_dict())

    restored_extra = restore_checkpoint_extra_state(workdir=str(tmp_path), step=7)
    assert restored_extra is not None

    restored_bank = ArrayMemoryBank(num_classes=1, max_size=1)
    restored_bank.load_state_dict(
        cast(dict[str, Any], restored_extra["memory_bank_positive"]),
    )

    assert restored_bank.seed == 17
    np.testing.assert_array_equal(restored_bank.count, positive_bank.count)
    np.testing.assert_array_equal(restored_bank.ptr, positive_bank.ptr)
    assert restored_bank.bank is not None
    assert positive_bank.bank is not None
    np.testing.assert_allclose(restored_bank.bank, positive_bank.bank)

    expected = positive_bank.sample(np.array([0, 1], dtype=np.int64), n_samples=2)
    actual = restored_bank.sample(np.array([0, 1], dtype=np.int64), n_samples=2)
    assert torch.equal(actual, expected)
