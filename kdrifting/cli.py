"""Top-level CLI entrypoint for repository tools."""

from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="kdrifting")
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the package version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command")

    mae_parser = subparsers.add_parser("train-mae", help="Run the MAE training loop.")
    mae_parser.add_argument("--config", required=True, help="Path to the MAE YAML config.")
    mae_parser.add_argument("--workdir", default="runs", help="Output directory.")
    mae_parser.add_argument("--device", default="auto", help="Torch device or 'auto'.")
    mae_parser.add_argument(
        "--init-from",
        default=None,
        help="Optional local or hf:// init artifact.",
    )

    gen_parser = subparsers.add_parser("train-gen", help="Run the generator training loop.")
    gen_parser.add_argument("--config", required=True, help="Path to the generator YAML config.")
    gen_parser.add_argument("--workdir", default="runs", help="Output directory.")
    gen_parser.add_argument("--device", default="auto", help="Torch device or 'auto'.")
    gen_parser.add_argument(
        "--init-from",
        default=None,
        help="Optional local or hf:// init artifact.",
    )

    infer_parser = subparsers.add_parser("infer", help="Run generator inference.")
    infer_parser.add_argument(
        "--init-from",
        required=True,
        help="Local or hf:// generator artifact.",
    )
    infer_parser.add_argument("--workdir", default="runs/infer", help="Output directory.")
    infer_parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale.",
    )
    infer_parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of samples to generate.",
    )
    infer_parser.add_argument(
        "--labels",
        default="",
        help="Comma-separated class labels. Defaults to zeros.",
    )
    infer_parser.add_argument("--device", default="auto", help="Torch device or 'auto'.")
    infer_parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    infer_parser.add_argument("--json-out", default="", help="Optional extra JSON output path.")

    fid_parser = subparsers.add_parser("eval-fid", help="Run release-style generator FID eval.")
    fid_parser.add_argument(
        "--init-from",
        required=True,
        help="Local or hf:// generator artifact.",
    )
    fid_parser.add_argument("--workdir", default="runs/infer", help="Output directory.")
    fid_parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale.",
    )
    fid_parser.add_argument(
        "--num-samples",
        type=int,
        default=50000,
        help="Number of evaluation samples to generate.",
    )
    fid_parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=2048,
        help="Evaluation batch size.",
    )
    fid_parser.add_argument("--device", default="auto", help="Torch device or 'auto'.")
    fid_parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    fid_parser.add_argument("--json-out", default="", help="Optional extra JSON output path.")
    fid_parser.add_argument("--use-wandb", action="store_true", help="Enable wandb logging.")
    fid_parser.add_argument("--wandb-entity", default=None, help="Optional wandb entity.")
    fid_parser.add_argument("--wandb-project", default="release-fid", help="wandb project name.")
    fid_parser.add_argument("--wandb-name", default=None, help="Optional wandb run name.")

    export_model_parser = subparsers.add_parser(
        "export-model",
        help="Export a source artifact into canonical torch EMA form.",
    )
    export_model_parser.add_argument(
        "--kind",
        choices=("mae", "gen"),
        required=True,
        help="Model family to export.",
    )
    export_model_parser.add_argument(
        "--init-from",
        required=True,
        help="Local or hf:// model source.",
    )
    export_model_parser.add_argument("--workdir", required=True, help="Output directory.")

    export_checkpoint_parser = subparsers.add_parser(
        "export-checkpoint",
        help="Convert an external training checkpoint into native torch format.",
    )
    export_checkpoint_parser.add_argument(
        "--kind",
        choices=("mae", "gen"),
        required=True,
        help="Model family to export.",
    )
    export_checkpoint_parser.add_argument(
        "--init-from",
        required=True,
        help="Local torch or JAX training checkpoint source.",
    )
    export_checkpoint_parser.add_argument(
        "--config",
        required=True,
        help="Path to the matching YAML config used to rebuild the optimizer and model.",
    )
    export_checkpoint_parser.add_argument("--workdir", required=True, help="Output directory.")
    export_checkpoint_parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for temporary reconstruction, or 'auto'.",
    )
    return parser


def main() -> None:
    """Run the top-level CLI."""
    args = build_parser().parse_args()
    if args.version:
        from kdrifting import __version__

        print(__version__)
        return

    if args.command == "train-mae":
        from kdrifting.runners.mae import main as train_mae_main

        train_mae_main(
            args.config,
            workdir=args.workdir,
            device=args.device,
            init_from=args.init_from,
        )
        return

    if args.command == "train-gen":
        from kdrifting.runners.generator import main as train_gen_main

        train_gen_main(
            args.config,
            workdir=args.workdir,
            device=args.device,
            init_from=args.init_from,
        )
        return

    if args.command == "infer":
        from kdrifting.inference import run_inference

        result = run_inference(
            init_from=args.init_from,
            workdir=args.workdir,
            cfg_scale=args.cfg_scale,
            num_samples=args.num_samples,
            labels=args.labels,
            device=args.device,
            seed=args.seed,
            json_out=args.json_out,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "eval-fid":
        from kdrifting.inference import run_fid_evaluation

        result = run_fid_evaluation(
            init_from=args.init_from,
            workdir=args.workdir,
            cfg_scale=args.cfg_scale,
            num_samples=args.num_samples,
            eval_batch_size=args.eval_batch_size,
            device=args.device,
            seed=args.seed,
            json_out=args.json_out,
            use_wandb=args.use_wandb,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "export-model":
        from kdrifting.export import export_model_artifact

        result = export_model_artifact(
            init_from=args.init_from,
            kind=args.kind,
            workdir=args.workdir,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "export-checkpoint":
        from kdrifting.config import load_yaml_config
        from kdrifting.export import export_training_checkpoint

        result = export_training_checkpoint(
            init_from=args.init_from,
            config=load_yaml_config(args.config),
            kind=args.kind,
            workdir=args.workdir,
            device=args.device,
        )
        print(json.dumps(result, indent=2))
        return

    build_parser().print_help()


if __name__ == "__main__":
    main()
