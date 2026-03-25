from __future__ import annotations

import json
import sys
from typing import Any

from pytest import CaptureFixture, MonkeyPatch

from kdrifting.cli import build_parser, main


def test_cli_parser_accepts_train_subcommands() -> None:
    mae_args = build_parser().parse_args(["train-mae", "--config", "mae.yaml"])
    gen_args = build_parser().parse_args(["train-gen", "--config", "gen.yaml"])

    assert mae_args.command == "train-mae"
    assert mae_args.config == "mae.yaml"
    assert mae_args.device == "auto"
    assert gen_args.command == "train-gen"
    assert gen_args.config == "gen.yaml"
    assert gen_args.workdir == "runs"


def test_cli_parser_accepts_infer_subcommand() -> None:
    args = build_parser().parse_args(["infer", "--init-from", "artifact"])

    assert args.command == "infer"
    assert args.init_from == "artifact"
    assert args.num_samples == 64


def test_cli_parser_accepts_eval_fid_subcommand() -> None:
    args = build_parser().parse_args(["eval-fid", "--init-from", "artifact"])

    assert args.command == "eval-fid"
    assert args.init_from == "artifact"
    assert args.num_samples == 50000
    assert args.eval_batch_size == 2048


def test_cli_main_dispatches_inference(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_inference(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"sample_grid": "grid.png", "num_samples": 2}

    import kdrifting.inference as inference_module

    monkeypatch.setattr(inference_module, "run_inference", fake_run_inference)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kdrifting",
            "infer",
            "--init-from",
            "artifact",
            "--num-samples",
            "2",
            "--labels",
            "1,2",
        ],
    )

    main()

    assert captured["init_from"] == "artifact"
    assert captured["num_samples"] == 2
    assert captured["labels"] == "1,2"
    assert json.loads(capsys.readouterr().out)["sample_grid"] == "grid.png"


def test_cli_main_dispatches_fid_eval(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_fid_evaluation(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"fid": 1.25, "isc_mean": 2.0}

    import kdrifting.inference as inference_module

    monkeypatch.setattr(inference_module, "run_fid_evaluation", fake_run_fid_evaluation)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kdrifting",
            "eval-fid",
            "--init-from",
            "artifact",
            "--cfg-scale",
            "1.5",
            "--num-samples",
            "128",
        ],
    )

    main()

    assert captured["init_from"] == "artifact"
    assert captured["cfg_scale"] == 1.5
    assert captured["num_samples"] == 128
    assert json.loads(capsys.readouterr().out)["fid"] == 1.25
