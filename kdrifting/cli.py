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

    build_parser().print_help()


if __name__ == "__main__":
    main()
