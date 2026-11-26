#!/usr/bin/env python3
"""Execute secret scanning and dependency vulnerability audits (Phase 13)."""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SECURITY_EVIDENCE_DIR = REPO_ROOT / "evidence" / "security"

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("github_pat", re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}")),
    ("github_fine_grained_pat", re.compile(r"github_pat_[A-Za-z0-9_]{40,}")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{32,}\b")),
    ("private_key_pem", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("slack_token", re.compile(r"xox[baprs]-[0-9A-Za-z-]{24,}")),
)


def run_secret_scan() -> dict[str, Any]:
    """Scan git-tracked and untracked workspace source files for credential patterns."""
    git_exe = shutil.which("git") or "git"
    proc = subprocess.run(
        [git_exe, "-C", str(REPO_ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    )
    files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    findings: list[dict[str, Any]] = []
    scanned_count = 0

    for rel in files:
        # Skip binary PNG/DB files
        if rel.lower().endswith((".png", ".jpg", ".ico", ".safetensors", ".pt", ".bin", ".db")):
            continue
        fp = REPO_ROOT / rel
        if not fp.is_file():
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        scanned_count += 1
        for line_no, line in enumerate(text.splitlines(), start=1):
            # Ignore regex pattern definitions inside this scanner itself
            if rel.replace("\\", "/") == "scripts/run_security_audit.py":
                continue
            for rule_name, pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    findings.append(
                        {
                            "rule": rule_name,
                            "file": rel,
                            "line": line_no,
                        }
                    )

    return {
        "scanned_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "files_scanned": scanned_count,
        "findings_count": len(findings),
        "findings": findings,
    }


def run_cmd_capture(cmd: list[str], log_path: Path, cwd: Path = REPO_ROOT) -> int:
    header = (
        f"=== Command: {' '.join(cmd)} ===\n"
        f"Executed (UTC): {dt.datetime.now(dt.UTC).isoformat()}\n\n"
    )
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    body = (
        header
        + "--- STDOUT ---\n"
        + proc.stdout
        + "\n--- STDERR ---\n"
        + proc.stderr
        + f"\nExit Code: {proc.returncode}\n"
    )
    log_path.write_text(body, encoding="utf-8")
    return proc.returncode


def main() -> int:
    if "--secret-scan-only" in sys.argv:
        secret_report = run_secret_scan()
        print(json.dumps(secret_report, indent=2))
        return 0 if secret_report["findings_count"] == 0 else 1

    SECURITY_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    secret_report = run_secret_scan()
    (SECURITY_EVIDENCE_DIR / "secret_scan.json").write_text(
        json.dumps(secret_report, indent=2), encoding="utf-8"
    )
    (SECURITY_EVIDENCE_DIR / "secret_scan.log").write_text(
        f"Secret scan completed at {secret_report['scanned_at_utc']}: "
        f"scanned={secret_report['files_scanned']} files, findings={secret_report['findings_count']}\n",
        encoding="utf-8",
    )

    pip_audit_rc = run_cmd_capture(
        [sys.executable, "-m", "pip_audit", "--skip-editable", "--desc"],
        SECURITY_EVIDENCE_DIR / "pip_audit.log",
    )

    npm_exe = shutil.which("npm") or "npm"
    npm_audit_rc = run_cmd_capture(
        [npm_exe, "--prefix", "frontend", "audit"],
        SECURITY_EVIDENCE_DIR / "npm_audit.log",
    )

    summary = {
        "secret_findings": secret_report["findings_count"],
        "pip_audit_exit_code": pip_audit_rc,
        "npm_audit_exit_code": npm_audit_rc,
    }
    print(json.dumps(summary, indent=2))
    return 0 if (secret_report["findings_count"] == 0 and pip_audit_rc == 0 and npm_audit_rc == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
