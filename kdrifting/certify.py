"""Single-machine certification harness for parity-focused test suites."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "single-machine"
DEFAULT_PYTEST_CACHE_DIR = Path("/tmp/kdrifting-pytest-cache")


@dataclass(frozen=True, slots=True)
class CertificationCase:
    """One parity-focused certification case."""

    name: str
    description: str
    test_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Result of one executed certification case."""

    name: str
    description: str
    command: str
    duration_seconds: float
    returncode: int
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def certification_cases() -> tuple[CertificationCase, ...]:
    """Return the curated single-machine parity certification suite."""
    return (
        CertificationCase(
            name="artifact-parity",
            description="Model rebuild and checkpoint import parity against upstream JAX.",
            test_paths=(
                "tests/unit/test_jax_artifacts.py",
                "tests/unit/test_checkpointing.py",
                "tests/unit/test_export.py",
            ),
        ),
        CertificationCase(
            name="math-parity",
            description="Loss and metric math parity against upstream helpers.",
            test_paths=(
                "tests/unit/test_losses.py",
                "tests/unit/test_eval.py",
            ),
        ),
        CertificationCase(
            name="training-parity",
            description="Toy and real-model resumed training traces against upstream JAX.",
            test_paths=("tests/unit/test_training_steps.py",),
        ),
        CertificationCase(
            name="runtime-parity",
            description="Resume, runner, inference, and public evaluation-path parity.",
            test_paths=(
                "tests/unit/test_runners.py",
                "tests/unit/test_inference.py",
            ),
        ),
    )


def _run_command(*, argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )


def _git_output(*args: str) -> str:
    result = _run_command(argv=["git", *args], cwd=REPO_ROOT)
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def gather_system_info() -> dict[str, Any]:
    """Collect machine and repository facts for a certification report."""
    cuda_devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        cuda_module = cast(Any, torch.cuda)
        for index in range(torch.cuda.device_count()):
            props = cuda_module.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "memory_gb": round(int(props.total_memory) / 1024**3, 2),
                },
            )

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("rev-parse", "--abbrev-ref", "HEAD"),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": cuda_devices,
    }


def _case_command(
    case: CertificationCase,
    *,
    python_executable: str,
    pytest_cache_dir: Path,
) -> list[str]:
    return [
        python_executable,
        "-m",
        "pytest",
        "-q",
        *case.test_paths,
        "-o",
        f"cache_dir={pytest_cache_dir}",
    ]


def run_certification_case(
    case: CertificationCase,
    *,
    python_executable: str,
    pytest_cache_dir: Path,
) -> CaseResult:
    """Run one certification case and capture its output."""
    argv = _case_command(
        case,
        python_executable=python_executable,
        pytest_cache_dir=pytest_cache_dir,
    )
    start = time.perf_counter()
    completed = _run_command(argv=argv, cwd=REPO_ROOT)
    duration_seconds = time.perf_counter() - start
    return CaseResult(
        name=case.name,
        description=case.description,
        command=shlex.join(argv),
        duration_seconds=duration_seconds,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _result_payload(
    *,
    info: dict[str, Any],
    case_results: list[CaseResult],
) -> dict[str, Any]:
    return {
        "system": info,
        "overall_passed": all(result.passed for result in case_results),
        "cases": [
            {
                **asdict(result),
                "passed": result.passed,
            }
            for result in case_results
        ],
        "total_duration_seconds": sum(result.duration_seconds for result in case_results),
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    system = payload["system"]
    lines = [
        "# Single-Machine Certification Report",
        "",
        f"Generated at UTC: `{system['generated_at_utc']}`",
        f"Git commit: `{system['git_commit']}`",
        f"Git branch: `{system['git_branch']}`",
        f"Python: `{system['python_version']}` via `{system['python']}`",
        f"Torch: `{system['torch_version']}`",
        f"CUDA available: `{system['cuda_available']}`",
        f"CUDA device count: `{system['cuda_device_count']}`",
        "",
        "## Devices",
    ]
    if system["cuda_devices"]:
        lines.extend(
            [
                (
                    f"- GPU {device['index']}: `{device['name']}` with "
                    f"`{device['memory_gb']}` GiB VRAM"
                )
                for device in system["cuda_devices"]
            ],
        )
    else:
        lines.append("- No CUDA devices detected.")

    lines.extend(
        [
            "",
            "## Suite",
            "",
            f"Overall passed: `{payload['overall_passed']}`",
            f"Total duration (s): `{payload['total_duration_seconds']:.2f}`",
            "",
        ],
    )
    for case in payload["cases"]:
        lines.extend(
            [
                f"### {case['name']}",
                "",
                case["description"],
                "",
                f"- Passed: `{case['passed']}`",
                f"- Duration (s): `{case['duration_seconds']:.2f}`",
                f"- Command: `{case['command']}`",
                "",
                "```text",
                case["stdout"].rstrip() or "(no stdout)",
                "```",
            ],
        )
        stderr = case["stderr"].rstrip()
        if stderr:
            lines.extend(
                [
                    "",
                    "stderr:",
                    "```text",
                    stderr,
                    "```",
                ],
            )
        lines.append("")
    return "\n".join(lines)


def run_single_machine_certification(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    python_executable: str = sys.executable,
    pytest_cache_dir: str | Path = DEFAULT_PYTEST_CACHE_DIR,
) -> dict[str, Any]:
    """Run the curated single-machine certification suite and write a report."""
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(pytest_cache_dir).expanduser().resolve()
    info = gather_system_info()
    case_results = [
        run_certification_case(
            case,
            python_executable=python_executable,
            pytest_cache_dir=cache_dir,
        )
        for case in certification_cases()
    ]
    payload = _result_payload(info=info, case_results=case_results)

    json_path = resolved_output_dir / "single_machine_certification.json"
    markdown_path = resolved_output_dir / "single_machine_certification.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(payload) + "\n", encoding="utf-8")
    payload["json_report"] = str(json_path)
    payload["markdown_report"] = str(markdown_path)

    if not payload["overall_passed"]:
        raise RuntimeError(f"Certification failed; see {markdown_path}")
    return payload
