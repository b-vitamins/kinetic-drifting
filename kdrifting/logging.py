"""Logging helpers used by training and evaluation code."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from kdrifting.distributed import is_rank_zero


def log_for_0(message: str, *args: object) -> None:
    """Print a formatted message on rank zero only."""
    if is_rank_zero():
        if args:
            print(message % args, flush=True)
        else:
            print(message, flush=True)


class WandbLogger:
    """Minimal logger with wandb and offline JSONL backends."""

    def __init__(self) -> None:
        self.step = 0
        self.use_wandb = True
        self.log_every_k = 1
        self._buffer: dict[str, float] = {}
        self._count: dict[str, int] = {}
        self.offline_dir = Path("log")
        self._wandb: Any | None = None

    def set_logging(
        self,
        *,
        project: str | None = None,
        config: Any | None = None,
        entity: str | None = None,
        name: str | None = None,
        use_wandb: bool = True,
        offline_dir: str = "log",
        workdir: str | None = None,
        log_every_k: int = 1,
        allow_resume: bool = True,
        **kwargs: Any,
    ) -> None:
        """Configure the logger."""
        self.use_wandb = bool(use_wandb)
        self.log_every_k = int(log_every_k)
        workdir_path = Path(workdir).resolve() if workdir else None
        resolved_offline_dir = (
            workdir_path / "log"
            if workdir_path is not None and not self.use_wandb
            else Path(offline_dir)
        )
        self.offline_dir = resolved_offline_dir
        self.offline_dir.mkdir(parents=True, exist_ok=True)

        if not is_rank_zero():
            return

        if self.use_wandb:
            init_kwargs: dict[str, Any] = {
                "project": project,
                "entity": entity,
                "name": name,
                "config": config,
                "mode": "online",
                "reinit": True,
            }
            if allow_resume:
                init_kwargs["resume"] = "allow"
            init_kwargs.update(kwargs)
            import importlib

            self._wandb = importlib.import_module("wandb")
            self._wandb.init(**init_kwargs)

    def set_step(self, step: int) -> None:
        self.step = int(step)

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        reduced = {key: self._buffer[key] / max(1, self._count.get(key, 1)) for key in self._buffer}
        if self._wandb is not None:
            self._wandb.log(reduced, step=self.step)
        else:
            path = self.offline_dir / "metrics.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": self.step, **reduced}) + "\n")
        self._buffer.clear()
        self._count.clear()

    def log_dict(self, values: dict[str, Any]) -> None:
        if not is_rank_zero():
            return
        reduced: dict[str, float] = {}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    continue
                reduced[key] = float(value.detach().float().mean().cpu().item())
            elif isinstance(value, np.ndarray):
                if value.size == 0:
                    continue
                reduced[key] = float(np.asarray(value, dtype=np.float64).mean())
            elif isinstance(value, int | float):
                reduced[key] = float(value)
            elif isinstance(value, np.generic):
                reduced[key] = float(np.asarray(value, dtype=np.float64).item())
        for key, value in reduced.items():
            self._buffer[key] = self._buffer.get(key, 0.0) + value
            self._count[key] = self._count.get(key, 0) + 1
        if self.log_every_k <= 1 or (self.step % self.log_every_k == 0):
            self._flush_buffer()

    def log_dict_dir(self, prefix: str, values: dict[str, Any]) -> None:
        self.log_dict({f"{prefix}/{key}": value for key, value in values.items()})

    @staticmethod
    def normalize_images(images: np.ndarray | torch.Tensor) -> np.ndarray:
        array = (
            images.detach().cpu().numpy()
            if isinstance(images, torch.Tensor)
            else np.asarray(images)
        )
        if array.ndim == 3:
            array = array[None, ...]
        if array.ndim != 4:
            raise ValueError(f"Expected 3D or 4D image input, got shape {array.shape}")
        if array.shape[1] in {1, 3} and array.shape[-1] not in {1, 3}:
            array = np.transpose(array, (0, 2, 3, 1))
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        if array.shape[-1] != 3:
            raise ValueError(f"Expected RGB images after normalization, got shape {array.shape}")
        if array.dtype != np.uint8:
            array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
            array = np.clip(array, 0.0, 1.0)
            array = (array * 255.0).astype(np.uint8)
        return array

    @staticmethod
    def make_grid_image(images: np.ndarray, rows: int = 8) -> Image.Image:
        rows = max(1, int(rows))
        pil_images = [Image.fromarray(image) for image in images]
        cols = max(1, int(math.ceil(len(pil_images) / rows)))
        width, height = pil_images[0].size
        total = rows * cols
        if len(pil_images) < total:
            blank = Image.new("RGB", (width, height), color=(0, 0, 0))
            pil_images.extend([blank] * (total - len(pil_images)))
        grid = Image.new("RGB", (cols * width, rows * height))
        for index, image in enumerate(pil_images):
            row = index % rows
            col = index // rows
            grid.paste(image, (col * width, row * height))
        return grid

    def log_image(self, name: str, images: np.ndarray | torch.Tensor) -> None:
        if not is_rank_zero():
            return
        array = self.normalize_images(images)
        grid_image = self.make_grid_image(array)
        if self._wandb is not None:
            self._wandb.log({name: [self._wandb.Image(img) for img in array]}, step=self.step)
            self._wandb.log({f"{name}_grid": self._wandb.Image(grid_image)}, step=self.step)
            return
        output_dir = self.offline_dir / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        grid_image.save(output_dir / f"{name.replace('/', '_')}_step{self.step}.jpg", format="JPEG")

    def finish(self) -> None:
        self._flush_buffer()
        if self._wandb is not None and is_rank_zero():
            self._wandb.finish()


class NullLogger:
    """No-op logger implementation."""

    @staticmethod
    def log_dict(*args: object, **kwargs: object) -> None:
        del args, kwargs

    @staticmethod
    def log_image(*args: object, **kwargs: object) -> None:
        del args, kwargs

    @staticmethod
    def finish() -> None:
        return
