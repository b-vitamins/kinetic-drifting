"""Top-level CLI entrypoint for repository tools."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="kdrifting")
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the package version and exit.",
    )
    return parser


def main() -> None:
    """Run the top-level CLI."""
    args = build_parser().parse_args()
    if args.version:
        from kdrifting import __version__

        print(__version__)


if __name__ == "__main__":
    main()
