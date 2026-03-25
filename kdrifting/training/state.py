"""Mutable training state for PyTorch loops."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass(slots=True)
class TrainState:
    """Minimal PyTorch training state with EMA tracking."""

    model: nn.Module
    optimizer: Optimizer
    ema_model: nn.Module
    ema_decay: float = 0.999
    step: int = 0

    @classmethod
    def create(
        cls,
        model: nn.Module,
        optimizer: Optimizer,
        *,
        ema_decay: float = 0.999,
    ) -> TrainState:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
        return cls(
            model=model, optimizer=optimizer, ema_model=ema_model, ema_decay=float(ema_decay)
        )

    def update_ema(self) -> None:
        """Update EMA weights from the live model."""
        with torch.no_grad():
            ema_state = self.ema_model.state_dict()
            model_state = self.model.state_dict()
            for key, value in model_state.items():
                ema_value = ema_state[key]
                if not torch.is_floating_point(value):
                    ema_value.copy_(value)
                    continue
                ema_value.mul_(self.ema_decay).add_(value.detach(), alpha=1.0 - self.ema_decay)

    def increment_step(self) -> None:
        self.step += 1

    def state_dict(self) -> dict[str, Any]:
        """Serialize the training state."""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "ema_decay": self.ema_decay,
            "step": self.step,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Load a serialized training state."""
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.ema_model.load_state_dict(payload["ema_model"])
        self.ema_decay = float(payload["ema_decay"])
        self.step = int(payload["step"])
