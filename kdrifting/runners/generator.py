"""Top-level generator training runner."""

from __future__ import annotations

import time
from typing import Any, cast

import torch
from tqdm.auto import tqdm

from kdrifting.checkpointing import save_checkpoint, save_params_ema_artifact
from kdrifting.config import load_yaml_config
from kdrifting.data import epoch0_sampler, get_postprocess_fn, infinite_sampler
from kdrifting.features import build_feature_activation
from kdrifting.logging import log_for_0
from kdrifting.memory_bank import ArrayMemoryBank
from kdrifting.model_builder import build_model_dict
from kdrifting.models.generator import DitGen
from kdrifting.runners.common import (
    PreprocessFn,
    create_or_restore_state,
    per_process_batch_size,
    prepare_preprocess_fn,
    select_device,
)
from kdrifting.training.generator import generate_step, train_step
from kdrifting.training.state import TrainState


def _sample_batch_indices(
    batch_size: int,
    target_size: int,
    *,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device.type if device.type != "mps" else "cpu")
    generator.manual_seed(seed)
    return torch.randperm(batch_size, generator=generator, device=device)[:target_size]


def _preview_metrics(images: torch.Tensor) -> dict[str, float]:
    return {
        "sample_mean": float(images.detach().float().mean().item()),
        "sample_std": float(images.detach().float().std(unbiased=False).item()),
    }


@torch.no_grad()
def evaluate_generator_model(
    model: torch.nn.Module,
    *,
    eval_loader: Any,
    preprocess_fn: PreprocessFn,
    postprocess_fn: Any,
    logger: Any,
    base_seed: int,
    step: int,
    cfg_scale: float,
) -> dict[str, float]:
    """Generate one preview batch and log generated and real images."""
    raw_batch = next(iter(epoch0_sampler(eval_loader)))
    prepared = preprocess_fn(raw_batch)
    labels = prepared["labels"]
    generated = generate_step(
        model,
        labels=labels,
        postprocess_fn=postprocess_fn,
        base_seed=base_seed,
        step=step,
        cfg_scale=cfg_scale,
    )
    real = postprocess_fn(prepared["images"])
    prefix = f"eval/cfg_{cfg_scale:g}"
    logger.log_image(f"{prefix}/generated", generated[:64])
    logger.log_image(f"{prefix}/real", real[:64])
    metrics = _preview_metrics(generated)
    metrics["cfg_scale"] = cfg_scale
    return metrics


def train_generator(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    logger: Any,
    eval_loader: Any,
    train_loader: Any,
    learning_rate_fn: Any,
    preprocess_fn: PreprocessFn,
    postprocess_fn: Any,
    feature_apply: Any,
    model_config: dict[str, Any],
    workdir: str,
    device: torch.device,
    train_batch_size: int,
    total_steps: int = 100000,
    save_per_step: int = 10000,
    eval_per_step: int = 5000,
    ema_decay: float = 0.999,
    seed: int = 42,
    pos_per_sample: int = 32,
    neg_per_sample: int = 16,
    forward_dict: dict[str, Any] | None = None,
    positive_bank_size: int = 64,
    negative_bank_size: int = 512,
    cfg_list: tuple[float, ...] | list[float] | None = None,
    activation_kwargs: dict[str, Any] | None = None,
    loss_kwargs: dict[str, Any] | None = None,
    max_grad_norm: float = 2.0,
    keep_every: int = 500000,
    keep_last: int = 2,
    init_from: str = "",
    push_per_step: int = 0,
    push_at_resume: int = 3000,
) -> TrainState:
    """Run the generator training loop."""
    prepared_preprocess = prepare_preprocess_fn(preprocess_fn, device)
    state = create_or_restore_state(
        model=model,
        optimizer=optimizer,
        ema_decay=ema_decay,
        workdir=workdir,
        init_from=init_from,
        kind="gen",
        device=device,
    )

    cfg_values = [float(value) for value in (cfg_list or (1.0,))]
    train_kwargs = dict(forward_dict or {})
    gen_per_label = int(train_kwargs.get("gen_per_label", 1))
    train_iter = infinite_sampler(train_loader, state.step)
    progress = tqdm(total=total_steps, initial=state.step, disable=False)
    initial_step = state.step
    memory_bank_positive = ArrayMemoryBank(num_classes=1000, max_size=positive_bank_size)
    memory_bank_negative = ArrayMemoryBank(num_classes=1, max_size=negative_bank_size)

    global_train_batch = int(train_batch_size)
    local_train_batch = per_process_batch_size(global_train_batch)

    log_for_0("Starting generator training loop...")
    while state.step < total_steps:
        step_start = time.perf_counter()
        step_index = state.step
        logger.set_step(step_index)

        goal = push_per_step
        if initial_step > 0 and step_index == initial_step:
            goal = push_per_step * push_at_resume

        pushed = 0
        last_batch: dict[str, torch.Tensor] | None = None
        while pushed < max(goal, 1):
            raw_batch = next(train_iter)
            prepared = prepared_preprocess(raw_batch)
            images = prepared["images"]
            labels = prepared["labels"]
            memory_bank_positive.add(images, labels)
            memory_bank_negative.add(images, labels * 0)
            pushed += int(images.shape[0])
            last_batch = prepared

        if last_batch is None:
            raise RuntimeError("Expected at least one batch while filling the memory bank.")

        labels = last_batch["labels"]
        images = last_batch["images"]
        if labels.shape[0] < local_train_batch:
            raise ValueError(
                f"Need at least {local_train_batch} labels in one batch, got {labels.shape[0]}.",
            )

        select_indices = _sample_batch_indices(
            labels.shape[0],
            local_train_batch,
            device=labels.device,
            seed=seed + step_index,
        )
        selected_labels = labels[select_indices]
        selected_images = images[select_indices]

        positive_samples = memory_bank_positive.sample(
            selected_labels,
            n_samples=pos_per_sample,
            device=device,
        )
        negative_samples = memory_bank_negative.sample(
            selected_labels * 0,
            n_samples=neg_per_sample,
            device=device,
        )

        state, metrics = train_step(
            state,
            labels=selected_labels,
            samples=positive_samples,
            negative_samples=negative_samples,
            feature_apply=feature_apply,
            learning_rate_fn=learning_rate_fn,
            base_seed=seed,
            activation_kwargs=activation_kwargs,
            loss_kwargs=loss_kwargs,
            max_grad_norm=max_grad_norm,
            **train_kwargs,
        )

        step_end = time.perf_counter()
        metrics["kimg"] = float(state.step * selected_images.shape[0] / 1000.0)
        metrics["forward_kimg"] = float(
            state.step * selected_images.shape[0] * gen_per_label / 1000.0,
        )
        metrics["time/total"] = step_end - step_start
        logger.log_dict(metrics)
        progress.update(1)

        if state.step % save_per_step == 0 or state.step == total_steps:
            save_checkpoint(state, workdir=workdir, keep=keep_last, keep_every=keep_every)
            save_params_ema_artifact(
                state,
                workdir=workdir,
                kind="gen",
                model_config=model_config,
            )

        if state.step % eval_per_step == 0 or state.step in {1, total_steps}:
            eval_cfgs = [cfg_values[0]] if state.step == 1 else cfg_values
            for cfg_scale in eval_cfgs:
                eval_metrics = evaluate_generator_model(
                    state.ema_model,
                    eval_loader=eval_loader,
                    preprocess_fn=prepared_preprocess,
                    postprocess_fn=postprocess_fn,
                    logger=logger,
                    base_seed=seed,
                    step=state.step,
                    cfg_scale=cfg_scale,
                )
                logger.log_dict_dir(f"eval_cfg_{cfg_scale:g}", eval_metrics)

    progress.close()
    logger.finish()
    return state


def train_generator_from_config(
    config: dict[str, Any],
    *,
    workdir: str = "runs",
    device: str | torch.device | None = None,
    init_from: str | None = None,
) -> TrainState:
    """Build the generator pipeline from config and run training."""
    runtime_device = select_device(device)
    model_dict = build_model_dict(config, DitGen, workdir=workdir, device=runtime_device)
    dataset_cfg = dict(config["dataset"])
    postprocess_fn_noclip = get_postprocess_fn(
        use_aug=bool(dataset_cfg.get("use_aug", False)),
        use_latent=bool(dataset_cfg.get("use_latent", False)),
        use_cache=bool(dataset_cfg.get("use_cache", False)),
        has_clip=False,
        device=runtime_device,
    )
    activation_fn = build_feature_activation(
        feature_config=dict(model_dict["feature"]),
        postprocess_fn=postprocess_fn_noclip,
        device=runtime_device,
    )
    train_kwargs = dict(model_dict["train"])
    if init_from is not None:
        train_kwargs["init_from"] = init_from
    return train_generator(
        model=cast(torch.nn.Module, model_dict["model"]),
        optimizer=cast(torch.optim.Optimizer, model_dict["optimizer"]),
        logger=model_dict["logger"],
        eval_loader=model_dict["eval_loader"],
        train_loader=model_dict["train_loader"],
        learning_rate_fn=model_dict["learning_rate_fn"],
        preprocess_fn=cast(PreprocessFn, model_dict["preprocess_fn"]),
        postprocess_fn=model_dict["postprocess_fn"],
        feature_apply=activation_fn,
        model_config=dict(config["model"]),
        workdir=workdir,
        device=runtime_device,
        **train_kwargs,
    )


def main(
    argv_config: str,
    *,
    workdir: str = "runs",
    device: str | None = None,
    init_from: str | None = None,
) -> TrainState:
    """CLI-oriented wrapper for generator training."""
    config = load_yaml_config(argv_config)
    return train_generator_from_config(
        config,
        workdir=workdir,
        device=device,
        init_from=init_from,
    )
