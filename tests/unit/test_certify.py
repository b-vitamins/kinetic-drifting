from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from kdrifting.certify import (
    CaseResult,
    CertificationCase,
    run_single_machine_certification,
)


def test_run_single_machine_certification_writes_reports(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    cases = (
        CertificationCase(
            name="case-a",
            description="first case",
            test_paths=("tests/unit/test_a.py",),
        ),
        CertificationCase(
            name="case-b",
            description="second case",
            test_paths=("tests/unit/test_b.py",),
        ),
    )

    def fake_system_info() -> dict[str, object]:
        return {
            "generated_at_utc": "2026-03-26T00:00:00+00:00",
            "git_commit": "abc123",
            "git_branch": "master",
            "python": "/tmp/python",
            "python_version": "3.11.9",
            "torch_version": "2.x",
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_devices": [{"index": 0, "name": "Fake GPU", "memory_gb": 12.0}],
        }

    def fake_run_case(
        case: CertificationCase,
        *,
        python_executable: str,
        pytest_cache_dir: Path,
    ) -> CaseResult:
        assert python_executable == "/tmp/python"
        assert pytest_cache_dir == Path("/tmp/fake-cache")
        return CaseResult(
            name=case.name,
            description=case.description,
            command=f"{python_executable} -m pytest {' '.join(case.test_paths)}",
            duration_seconds=1.25,
            returncode=0,
            stdout="1 passed",
            stderr="",
        )

    def fake_certification_cases() -> tuple[CertificationCase, ...]:
        return cases

    monkeypatch.setattr("kdrifting.certify.gather_system_info", fake_system_info)
    monkeypatch.setattr("kdrifting.certify.certification_cases", fake_certification_cases)
    monkeypatch.setattr("kdrifting.certify.run_certification_case", fake_run_case)

    result = run_single_machine_certification(
        output_dir=tmp_path,
        python_executable="/tmp/python",
        pytest_cache_dir="/tmp/fake-cache",
    )

    json_report = Path(result["json_report"])
    markdown_report = Path(result["markdown_report"])
    assert json_report.is_file()
    assert markdown_report.is_file()
    payload = json.loads(json_report.read_text(encoding="utf-8"))
    assert payload["overall_passed"] is True
    assert len(payload["cases"]) == 2
    assert "case-a" in markdown_report.read_text(encoding="utf-8")


def test_run_single_machine_certification_raises_on_failure(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    def fake_certification_cases() -> tuple[CertificationCase, ...]:
        return (
            CertificationCase(
                name="failing-case",
                description="failure case",
                test_paths=("tests/unit/test_fail.py",),
            ),
        )

    def fake_system_info() -> dict[str, object]:
        return {
            "generated_at_utc": "2026-03-26T00:00:00+00:00",
            "git_commit": "abc123",
            "git_branch": "master",
            "python": "/tmp/python",
            "python_version": "3.11.9",
            "torch_version": "2.x",
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_devices": [],
        }

    def fake_run_case(
        case: CertificationCase,
        *,
        python_executable: str,
        pytest_cache_dir: Path,
    ) -> CaseResult:
        del python_executable, pytest_cache_dir
        return CaseResult(
            name=case.name,
            description=case.description,
            command="pytest",
            duration_seconds=0.5,
            returncode=1,
            stdout="failed",
            stderr="traceback",
        )

    monkeypatch.setattr("kdrifting.certify.certification_cases", fake_certification_cases)
    monkeypatch.setattr("kdrifting.certify.gather_system_info", fake_system_info)
    monkeypatch.setattr("kdrifting.certify.run_certification_case", fake_run_case)

    with pytest.raises(RuntimeError, match="Certification failed"):
        run_single_machine_certification(output_dir=tmp_path)
