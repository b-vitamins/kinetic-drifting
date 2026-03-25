from __future__ import annotations

from kdrifting import __version__
from kdrifting.cli import build_parser


def test_package_version_is_defined() -> None:
    assert __version__ == "0.1.0"


def test_cli_parser_accepts_version_flag() -> None:
    args = build_parser().parse_args(["--version"])
    assert args.version is True
