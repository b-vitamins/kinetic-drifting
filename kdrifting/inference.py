"""Generator inference entrypoint and helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from kdrifting.data import get_postprocess_fn
from kdrifting.hf import load_generator_model
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
