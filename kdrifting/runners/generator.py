"""Top-level generator training runner."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, cast

import torch
from tqdm.auto import tqdm

from kdrifting.checkpointing import (
    restore_checkpoint_extra_state,
    save_checkpoint,
    save_params_ema_artifact,
)
from kdrifting.config import load_yaml_config
from kdrifting.data import get_postprocess_fn, infinite_sampler
from kdrifting.eval.generation import evaluate_fid
from kdrifting.features import build_feature_activation
from kdrifting.logging import log_for_0
from kdrifting.memory_bank import ArrayMemoryBank
from kdrifting.model_builder import build_model_dict
from kdrifting.models.generator import DitGen
from kdrifting.runners.common import (
    PreprocessFn,
    RawBatch,
    create_or_restore_state,
    per_process_batch_size,
    prepare_preprocess_fn,
    select_device,
)
from kdrifting.training.generator import generate_step, train_step
from kdrifting.training.state import TrainState

GeneratorEvalFn = Callable[..., dict[str, float]]


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


def _memory_bank_extra_state(
    positive_bank: ArrayMemoryBank,
    negative_bank: ArrayMemoryBank,
) -> dict[str, Any]:
    return {
        "memory_bank_positive": positive_bank.state_dict(),
        "memory_bank_negative": negative_bank.state_dict(),
    }


def _restore_memory_banks(
    *,
    workdir: str,
    positive_bank: ArrayMemoryBank,
    negative_bank: ArrayMemoryBank,
) -> bool:
    extra_state = restore_checkpoint_extra_state(workdir=workdir)
    if extra_state is None:
        return False

    positive_state = extra_state.get("memory_bank_positive")
    negative_state = extra_state.get("memory_bank_negative")
    if not isinstance(positive_state, dict) or not isinstance(negative_state, dict):
        return False

    positive_bank.load_state_dict(cast(dict[str, Any], positive_state))
    negative_bank.load_state_dict(cast(dict[str, Any], negative_state))
    return True


@torch.no_grad()
def evaluate_generator_model(
    model: torch.nn.Module,
    *,
    eval_loader: Any,
    preprocess_fn: PreprocessFn,
    postprocess_fn: Any,
    logger: Any,
    base_seed: int,
    dataset_name: str,
    num_samples: int,
    cfg_scale: float,
    log_folder: str,
    log_prefix: str,
    device: torch.device,
    eval_prc_recall: bool,
    eval_isc: bool,
    eval_fid_enabled: bool,
    evaluate_fn: GeneratorEvalFn | None = None,
) -> dict[str, float]:
    """Evaluate a generator model with source-compatible release metrics."""
    metric_fn = evaluate_fn or evaluate_fid

    def gen_func(batch: RawBatch, eval_index: int) -> torch.Tensor:
        labels = preprocess_fn(batch)["labels"]
        return generate_step(
            model,
            labels=labels,
            postprocess_fn=postprocess_fn,
            base_seed=base_seed,
            step=eval_index,
            cfg_scale=cfg_scale,
        )

    return metric_fn(
        dataset_name=dataset_name,
        gen_func=gen_func,
        eval_loader=eval_loader,
        logger=logger,
        num_samples=num_samples,
        log_folder=log_folder,
        log_prefix=log_prefix,
        eval_prc_recall=eval_prc_recall,
        eval_isc=eval_isc,
        eval_fid=eval_fid_enabled,
        device=device,
    )


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
    dataset_name: str,
    workdir: str,
    device: torch.device,
    train_batch_size: int,
    total_steps: int = 100000,
    save_per_step: int = 10000,
    eval_per_step: int = 5000,
    eval_samples: int = 50000,
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
    eval_prc_recall: bool = False,
    eval_isc: bool = True,
    eval_fid_enabled: bool = True,
    evaluate_fn: GeneratorEvalFn | None = None,
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
    memory_bank_positive = ArrayMemoryBank(
        num_classes=1000,
        max_size=positive_bank_size,
        seed=seed,
    )
    memory_bank_negative = ArrayMemoryBank(
        num_classes=1,
        max_size=negative_bank_size,
        seed=seed + 1,
    )
    restored_memory_banks = False
    if initial_step > 0:
        restored_memory_banks = _restore_memory_banks(
            workdir=workdir,
            positive_bank=memory_bank_positive,
            negative_bank=memory_bank_negative,
        )
        if restored_memory_banks:
            log_for_0("Restored generator memory banks from %s", workdir)

    global_train_batch = int(train_batch_size)
    local_train_batch = per_process_batch_size(global_train_batch)

    log_for_0("Starting generator training loop...")
    while state.step < total_steps:
        step_start = time.perf_counter()
        step_index = state.step
        logger.set_step(step_index)

        goal = push_per_step
        if initial_step > 0 and step_index == initial_step and not restored_memory_banks:
            goal = push_per_step * push_at_resume
            log_for_0("Generator resume warmup: refilling memory bank with goal=%d", goal)

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
            save_checkpoint(
                state,
                workdir=workdir,
                keep=keep_last,
                keep_every=keep_every,
                extra_state=_memory_bank_extra_state(
                    memory_bank_positive,
                    memory_bank_negative,
                ),
            )
            save_params_ema_artifact(
                state,
                workdir=workdir,
                kind="gen",
                model_config=model_config,
            )

        if state.step % eval_per_step == 0 or state.step in {1, total_steps}:
            is_sanity = state.step == 1
            sample_goal = 500 if is_sanity else eval_samples
            folder_prefix = "sanity" if is_sanity else "CFG"
            round_best_fid = float("inf")
            round_best_cfg = cfg_values[0]
            eval_cfgs = [cfg_values[0]] if is_sanity else cfg_values
            for cfg_scale in eval_cfgs:
                eval_metrics = evaluate_generator_model(
                    state.ema_model,
                    eval_loader=eval_loader,
                    preprocess_fn=prepared_preprocess,
                    postprocess_fn=postprocess_fn,
                    logger=logger,
                    base_seed=seed,
                    dataset_name=dataset_name,
                    num_samples=sample_goal,
                    cfg_scale=cfg_scale,
                    log_folder=f"{folder_prefix}{cfg_scale:g}",
                    log_prefix=f"EMA_{state.ema_decay:g}",
                    device=device,
                    eval_prc_recall=eval_prc_recall,
                    eval_isc=eval_isc,
                    eval_fid_enabled=eval_fid_enabled,
                    evaluate_fn=evaluate_fn,
                )
                fid_value = eval_metrics.get("fid", float("inf"))
                if fid_value < round_best_fid:
                    round_best_fid = fid_value
                    round_best_cfg = cfg_scale
            if not is_sanity and eval_fid_enabled:
                log_for_0(
                    "best_fid=%.4f best_cfg=%.1f (step=%d)",
                    round_best_fid,
                    round_best_cfg,
                    state.step,
                )
                logger.log_dict(
                    {
                        "best_fid": round_best_fid,
                        "best_cfg": round_best_cfg,
                    },
                )

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
        dataset_name=str(model_dict["dataset_name"]),
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
