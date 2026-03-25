"""Generator inference entrypoint and helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from kdrifting.data import create_imagenet_split, get_postprocess_fn
from kdrifting.eval.generation import evaluate_fid
from kdrifting.hf import load_generator_model
from kdrifting.logging import WandbLogger
from kdrifting.runners.common import save_image_grid, select_device
from kdrifting.training.generator import generate_step


def parse_labels(labels: str, *, num_samples: int) -> list[int]:
    """Parse a comma-separated label string and expand it to ``num_samples``."""
    values = [int(part.strip()) for part in labels.split(",") if part.strip()]
    if not values:
        values = [0]
    if num_samples <= 0:
        return values
    if len(values) >= num_samples:
        return values[:num_samples]
    repeats = (num_samples + len(values) - 1) // len(values)
    return (values * repeats)[:num_samples]


@torch.no_grad()
def run_inference(
    *,
    init_from: str,
    workdir: str = "runs/infer",
    cfg_scale: float = 1.0,
    num_samples: int = 64,
    labels: str = "",
    device: str | torch.device | None = None,
    seed: int = 0,
    json_out: str = "",
    noise_x: torch.Tensor | None = None,
    noise_labels: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Load a generator artifact, sample images, and save a preview grid."""
    if num_samples <= 0:
        raise ValueError(f"Expected num_samples > 0, got {num_samples}.")

    runtime_device = select_device(device)
    model, metadata = load_generator_model(init_from)
    model.to(runtime_device)
    model.eval()

    model_config = dict(metadata.get("model_config", {}) or {})
    use_latent = int(model_config.get("in_channels", 3)) == 4
    label_values = parse_labels(labels, num_samples=num_samples)
    label_tensor = torch.tensor(label_values, device=runtime_device, dtype=torch.int64)
    postprocess_fn = get_postprocess_fn(
        use_latent=use_latent,
        use_cache=False,
        has_clip=True,
        device=runtime_device,
    )
    samples = generate_step(
        model,
        labels=label_tensor,
        postprocess_fn=postprocess_fn,
        base_seed=seed,
        step=0,
        cfg_scale=cfg_scale,
        noise_x=noise_x,
        noise_labels=noise_labels,
    )

    output_dir = Path(workdir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "samples.pt"
    grid_path = save_image_grid(samples, output_dir / "samples_grid.png")
    torch.save(samples.detach().cpu(), sample_path)

    result = {
        "init_from": init_from,
        "cfg_scale": cfg_scale,
        "num_samples": len(label_values),
        "labels": label_values,
        "device": str(runtime_device),
        "sample_tensor": str(sample_path),
        "sample_grid": str(grid_path),
        "metadata": metadata,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if json_out:
        json_output_path = Path(json_out).expanduser().resolve()
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        json_output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


@torch.no_grad()
def run_fid_evaluation(
    *,
    init_from: str,
    workdir: str = "runs/infer",
    cfg_scale: float = 1.0,
    num_samples: int = 50000,
    eval_batch_size: int = 2048,
    device: str | torch.device | None = None,
    seed: int = 0,
    json_out: str = "",
    use_wandb: bool = False,
    wandb_entity: str | None = None,
    wandb_project: str = "release-fid",
    wandb_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate a generator artifact with release-style FID metrics."""
    if num_samples <= 0:
        raise ValueError(f"Expected num_samples > 0, got {num_samples}.")
    if eval_batch_size <= 0:
        raise ValueError(f"Expected eval_batch_size > 0, got {eval_batch_size}.")

    runtime_device = select_device(device)
    model, metadata = load_generator_model(init_from)
    model.to(runtime_device)
    model.eval()

    model_config = dict(metadata.get("model_config", {}) or {})
    use_latent = int(model_config.get("in_channels", 3)) == 4
    eval_loader, preprocess_fn, _ = create_imagenet_split(
        resolution=256,
        split="val",
        batch_size=eval_batch_size,
        use_aug=False,
        use_latent=False,
        use_cache=False,
        num_workers=0,
        device=runtime_device,
    )
    postprocess_fn = get_postprocess_fn(
        use_latent=False,
        use_cache=use_latent,
        has_clip=True,
        device=runtime_device,
    )

    def gen_func(batch: tuple[torch.Tensor, torch.Tensor], eval_index: int) -> torch.Tensor:
        labels = preprocess_fn(batch)["labels"].to(device=runtime_device)
        return generate_step(
            model,
            labels=labels,
            postprocess_fn=postprocess_fn,
            base_seed=seed,
            step=eval_index,
            cfg_scale=cfg_scale,
        )

    output_dir = Path(workdir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = WandbLogger()
    logger.set_logging(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_name or f"{Path(init_from).name}_fid",
        use_wandb=use_wandb,
        workdir=str(output_dir),
        log_every_k=1,
    )
    metrics = evaluate_fid(
        dataset_name="imagenet256",
        gen_func=gen_func,
        eval_loader=eval_loader,
        logger=logger,
        num_samples=num_samples,
        log_folder="fid_eval",
        log_prefix=f"cfg_{cfg_scale:g}",
        eval_prc_recall=(num_samples >= 50000),
        eval_isc=True,
        eval_fid=True,
        device=runtime_device,
    )
    logger.finish()

    result = {
        "init_from": init_from,
        "cfg_scale": cfg_scale,
        "num_samples": num_samples,
        "eval_batch_size": eval_batch_size,
        "device": str(runtime_device),
        "metadata": metadata,
        **metrics,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if json_out:
        json_output_path = Path(json_out).expanduser().resolve()
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        json_output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
