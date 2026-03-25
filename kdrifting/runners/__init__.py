"""High-level training and inference runners."""

from kdrifting.runners.generator import train_generator, train_generator_from_config
from kdrifting.runners.mae import train_mae, train_mae_from_config

__all__ = [
    "train_generator",
    "train_generator_from_config",
    "train_mae",
    "train_mae_from_config",
]
