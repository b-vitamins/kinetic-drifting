"""Top-level MAE training runner."""

from __future__ import annotations

import copy
import time
from typing import Any, cast

import torch
from tqdm.auto import tqdm

from kdrifting.checkpointing import save_checkpoint, save_params_ema_artifact
from kdrifting.config import load_yaml_config
from kdrifting.data import epoch0_sampler, infinite_sampler
from kdrifting.logging import log_for_0
from kdrifting.model_builder import build_model_dict
from kdrifting.models.mae import MAEResNet
from kdrifting.runners.common import (
    PreprocessFn,
    RawBatch,
    average_metric_dicts,
    create_or_restore_state,
    prepare_preprocess_fn,
    select_device,
)
from kdrifting.training.mae import eval_step, train_step
from kdrifting.training.state import TrainState


def _batch_size(batch: RawBatch) -> int:
    return int(batch[1].shape[0])


def evaluate_mae_model(
    model: torch.nn.Module,
    eval_loader: Any,
    *,
    preprocess_fn: PreprocessFn,
    base_seed: int,
    step: int,
    forward_dict: dict[str, Any],
    eval_samples: int,
) -> dict[str, float]:
    """Evaluate an MAE model over at most ``eval_samples`` items."""
    metrics: list[tuple[dict[str, float], int]] = []
    seen = 0
    for eval_index, batch in enumerate(epoch0_sampler(eval_loader)):
        raw_batch = batch
        batch_size = _batch_size(raw_batch)
        remaining = eval_samples - seen
        if remaining <= 0:
            break
        metric = eval_step(
            model,
            raw_batch,
            base_seed=base_seed,
            step=step + eval_index,
            forward_dict=forward_dict,
            preprocess_fn=preprocess_fn,
        )
        weight = min(batch_size, remaining)
        metrics.append((metric, weight))
        seen += weight
    return average_metric_dicts(metrics)


def train_mae(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    logger: Any,
    eval_loader: Any,
    train_loader: Any,
    learning_rate_fn: Any,
    preprocess_fn: PreprocessFn,
    model_config: dict[str, Any],
    workdir: str,
    device: torch.device,
    total_steps: int = 100000,
    save_per_step: int = 10000,
    eval_per_step: int = 2000,
    eval_samples: int = 5000,
    ema_decay: float = 0.999,
    seed: int = 42,
    finetune_last_steps: int = 0,
    warmup_finetune: int = 1000,
    finetune_cls: float = 0.5,
    max_grad_norm: float = 2.0,
    keep_every: int = 500000,
    keep_last: int = 2,
    init_from: str = "",
    forward_dict: dict[str, Any] | None = None,
    eval_forward_dict: dict[str, Any] | None = None,
) -> TrainState:
    """Run the MAE training loop."""
    prepared_preprocess = prepare_preprocess_fn(preprocess_fn, device)
    train_forward = dict(forward_dict or {})
    eval_forward = dict(eval_forward_dict or train_forward)
    state = create_or_restore_state(
        model=model,
        optimizer=optimizer,
        ema_decay=ema_decay,
        workdir=workdir,
        init_from=init_from,
        kind="mae",
        device=device,
    )

    forward_zeros = copy.deepcopy(eval_forward)
    forward_zeros["mask_ratio_min"] = 0.0
    forward_zeros["mask_ratio_max"] = 0.0

    train_iter = infinite_sampler(train_loader, state.step)
    start_finetune_step = total_steps - finetune_last_steps
    wall_start = time.perf_counter()
    progress = tqdm(
        total=total_steps,
        initial=state.step,
        disable=False,
    )

    log_for_0("Starting MAE training loop...")
    while state.step < total_steps:
        step_start = time.perf_counter()
        step_index = state.step
        logger.set_step(step_index)

        raw_batch = next(train_iter)
        prepare_done = time.perf_counter()

        current_forward = copy.deepcopy(train_forward)
        if step_index >= start_finetune_step:
            warmup_progress = (step_index - start_finetune_step) / max(1, warmup_finetune)
            current_forward["lambda_cls"] = finetune_cls * min(1.0, warmup_progress)

        state, metrics = train_step(
            state,
            raw_batch,
            base_seed=seed,
            forward_dict=current_forward,
            learning_rate_fn=learning_rate_fn,
            preprocess_fn=prepared_preprocess,
            max_grad_norm=max_grad_norm,
        )

        step_end = time.perf_counter()
        processed_kimg = (state.step) * _batch_size(raw_batch) / 1000.0
        metrics["kimg"] = processed_kimg
        metrics["time/total"] = step_end - step_start
        metrics["time/prepare"] = prepare_done - step_start
        metrics["time/train"] = step_end - prepare_done
        metrics["time/per_step"] = (step_end - wall_start) / max(1, state.step)
        logger.log_dict(metrics)
        progress.update(1)

        if state.step % eval_per_step == 0:
            eval_metrics = evaluate_mae_model(
                state.model,
                eval_loader,
                preprocess_fn=prepared_preprocess,
                base_seed=seed,
                step=state.step,
                forward_dict=eval_forward,
                eval_samples=eval_samples,
            )
            logger.log_dict_dir("eval", eval_metrics)
            eval_ema_metrics = evaluate_mae_model(
                state.ema_model,
                eval_loader,
                preprocess_fn=prepared_preprocess,
                base_seed=seed,
                step=state.step,
                forward_dict=eval_forward,
                eval_samples=eval_samples,
            )
            logger.log_dict_dir(f"eval_ema_{state.ema_decay:g}", eval_ema_metrics)

            eval_nomask_metrics = evaluate_mae_model(
                state.model,
                eval_loader,
                preprocess_fn=prepared_preprocess,
                base_seed=seed,
                step=state.step,
                forward_dict=forward_zeros,
                eval_samples=eval_samples,
            )
            logger.log_dict_dir("eval_nomask", eval_nomask_metrics)
            eval_nomask_ema_metrics = evaluate_mae_model(
                state.ema_model,
                eval_loader,
                preprocess_fn=prepared_preprocess,
                base_seed=seed,
                step=state.step,
                forward_dict=forward_zeros,
                eval_samples=eval_samples,
            )
            logger.log_dict_dir(f"eval_ema_{state.ema_decay:g}_nomask", eval_nomask_ema_metrics)

        if state.step in {total_steps, start_finetune_step} or (
            state.step % save_per_step == 0 and state.step < start_finetune_step
        ):
            save_checkpoint(state, workdir=workdir, keep=keep_last, keep_every=keep_every)
            save_params_ema_artifact(
                state,
                workdir=workdir,
                kind="mae",
                model_config=model_config,
            )

    progress.close()
    logger.finish()
    return state


def train_mae_from_config(
    config: dict[str, Any],
    *,
    workdir: str = "runs",
    device: str | torch.device | None = None,
    init_from: str | None = None,
) -> TrainState:
    """Build the MAE pipeline from config and run training."""
    runtime_device = select_device(device)
    model_dict = build_model_dict(config, MAEResNet, workdir=workdir, device=runtime_device)
    train_kwargs = dict(model_dict["train"])
    if init_from is not None:
        train_kwargs["init_from"] = init_from
    return train_mae(
        model=cast(torch.nn.Module, model_dict["model"]),
        optimizer=cast(torch.optim.Optimizer, model_dict["optimizer"]),
        logger=model_dict["logger"],
        eval_loader=model_dict["eval_loader"],
        train_loader=model_dict["train_loader"],
        learning_rate_fn=model_dict["learning_rate_fn"],
        preprocess_fn=cast(PreprocessFn, model_dict["preprocess_fn"]),
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
    """CLI-oriented wrapper for MAE training."""
    config = load_yaml_config(argv_config)
    return train_mae_from_config(
        config,
        workdir=workdir,
        device=device,
        init_from=init_from,
    )
