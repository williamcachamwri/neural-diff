#!/usr/bin/env python3
"""Execute all NEURAL DIFF static checks, backend test suites, coverage, and frontend tests.

Saves verbatim command logs, exit codes, and a structured JSON summary under `evidence/tests/`.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_EVIDENCE_DIR = REPO_ROOT / "evidence" / "tests"


def run_step(name: str, cmd: list[str], log_filename: str) -> dict[str, Any]:
    started = dt.datetime.now(dt.UTC).isoformat()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    ended = dt.datetime.now(dt.UTC).isoformat()
    log_path = TESTS_EVIDENCE_DIR / log_filename
    body = (
        f"=== Suite: {name} ===\n"
        f"Command: {' '.join(cmd)}\n"
        f"Started (UTC): {started}\n"
        f"Completed (UTC): {ended}\n"
        f"Exit Code: {proc.returncode}\n\n"
        f"--- STDOUT ---\n{proc.stdout}\n"
        f"--- STDERR ---\n{proc.stderr}\n"
    )
    log_path.write_text(body, encoding="utf-8")
    status_str = "PASS" if proc.returncode == 0 else "FAIL"
    print(f"[{status_str}] {name} (rc={proc.returncode}) -> {log_path.relative_to(REPO_ROOT)}")
    return {
        "name": name,
        "command": cmd,
        "started_at_utc": started,
        "completed_at_utc": ended,
        "exit_code": proc.returncode,
        "passed": proc.returncode == 0,
        "log_file": log_path.relative_to(REPO_ROOT).as_posix(),
    }


def main() -> int:
    TESTS_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    npm = shutil.which("npm") or "npm"

    suites: list[tuple[str, list[str], str]] = [
        ("ruff_lint", [py, "-m", "ruff", "check", "core", "adapters", "benchmarks", "api", "tests", "scripts"], "ruff-lint.log"),
        ("mypy_typecheck", [py, "-m", "mypy", "core", "adapters", "benchmarks", "api"], "mypy-typecheck.log"),
        ("unit_tests", [py, "-m", "pytest", "-m", "unit", "-v"], "unit-tests.log"),
        ("property_tests", [py, "-m", "pytest", "-m", "property", "-v"], "property-tests.log"),
        ("integration_tests", [py, "-m", "pytest", "-m", "integration", "-v"], "integration-tests.log"),
        ("regression_tests", [py, "-m", "pytest", "-m", "regression", "-v"], "regression-tests.log"),
        ("security_tests", [py, "-m", "pytest", "-m", "security", "-v"], "security-tests.log"),
        (
            "full_coverage",
            [
                py,
                "-m",
                "pytest",
                "--cov=core",
                "--cov=adapters",
                "--cov=benchmarks",
                "--cov=api",
                "--cov-report=term-missing",
                "-v",
            ],
            "coverage.log",
        ),
        ("frontend_typecheck", [npm, "--prefix", "frontend", "run", "typecheck"], "frontend-typecheck.log"),
        ("frontend_unit_tests", [npm, "--prefix", "frontend", "run", "test"], "frontend-vitest.log"),
        ("frontend_production_build", [npm, "--prefix", "frontend", "run", "build"], "frontend-build.log"),
    ]

    results = [run_step(name, cmd, log_file) for name, cmd, log_file in suites]
    summary = {
        "executed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "all_passed": all(r["passed"] for r in results),
        "suites": results,
    }
    summary_path = TESTS_EVIDENCE_DIR / "test_execution_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
