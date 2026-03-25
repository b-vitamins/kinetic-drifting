"""Construction helpers for models, data loaders, and optimizers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from kdrifting.data import create_imagenet_split
from kdrifting.distributed import world_size
from kdrifting.logging import WandbLogger
from kdrifting.schedules import create_learning_rate_fn


def _per_process_batch_size(total_batch_size: int) -> int:
    processes = world_size()
    if total_batch_size % processes != 0:
        raise ValueError(
            f"Expected batch size {total_batch_size} to be divisible by world size {processes}.",
        )
    return total_batch_size // processes


def build_model_dict(
    config: dict[str, Any],
    model_class: type[torch.nn.Module],
    *,
    workdir: str = "runs",
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Build a model, loaders, optimizer, and logger from a config dictionary."""
    dataset_cfg = dict(config["dataset"])
    model = model_class(num_classes=dataset_cfg["num_classes"], **dict(config["model"]))
    model.to(device)

    resolution = int(dataset_cfg["resolution"])
    use_aug = bool(dataset_cfg.get("use_aug", False))
    use_latent = bool(dataset_cfg.get("use_latent", False))
    use_cache = bool(dataset_cfg.get("use_cache", False))
    kwargs = dict(dataset_cfg.get("kwargs", {}))
    train_loader, preprocess_fn, postprocess_fn = create_imagenet_split(
        resolution=resolution,
        use_aug=use_aug,
        use_latent=use_latent,
        use_cache=use_cache,
        batch_size=_per_process_batch_size(int(dataset_cfg["batch_size"])),
        split="train",
        device=device,
        **kwargs,
    )
    eval_loader, _, _ = create_imagenet_split(
        resolution=resolution,
        use_aug=use_aug,
        use_latent=use_latent,
        use_cache=use_cache,
        batch_size=_per_process_batch_size(int(dataset_cfg["eval_batch_size"])),
        split="val",
        device=device,
        **kwargs,
    )

    optimizer_cfg = dict(config["optimizer"])
    lr_schedule_cfg = dict(optimizer_cfg["lr_schedule"])
    learning_rate_fn = create_learning_rate_fn(**lr_schedule_cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate_fn(0)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
        betas=(float(optimizer_cfg["adam_b1"]), float(optimizer_cfg["adam_b2"])),
    )

    logging_cfg = dict(config.get("logging", {}))
    use_wandb = bool(logging_cfg.pop("use_wandb", config.get("use_wandb", True)))
    logger = WandbLogger()
    output_root = Path(workdir).resolve()
    logger.set_logging(
        config=config,
        workdir=str(output_root),
        use_wandb=use_wandb,
        name=logging_cfg.get("name", output_root.name),
        **logging_cfg,
    )

    return {
        "model": model,
        "optimizer": optimizer,
        "logger": logger,
        "eval_loader": eval_loader,
        "train_loader": train_loader,
        "dataset_name": f"imagenet{resolution}",
        "preprocess_fn": preprocess_fn,
        "postprocess_fn": postprocess_fn,
        "train": dict(config.get("train", {})),
        "learning_rate_fn": learning_rate_fn,
        "feature": dict(config.get("feature", {})),
    }
