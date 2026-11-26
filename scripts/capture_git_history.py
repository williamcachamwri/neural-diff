#!/usr/bin/env python3
"""Capture verbatim Git commit count, history log, and working-tree status (Section 14 & 15)."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GIT_EVIDENCE_DIR = REPO_ROOT / "evidence" / "git"


def _git(args: list[str]) -> tuple[int, str, str]:
    git_exe = shutil.which("git") or "git"
    proc = subprocess.run(
        [git_exe, "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, proc.stdout, proc.stderr


def main() -> int:
    GIT_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    now_utc = dt.datetime.now(dt.UTC).isoformat()

    rc_cnt, out_cnt, err_cnt = _git(["rev-list", "--count", "HEAD"])
    rc_log, out_log, err_log = _git(["log", "--oneline"])
    rc_st, out_st, err_st = _git(["status", "--short"])

    commit_count = int(out_cnt.strip()) if rc_cnt == 0 and out_cnt.strip().isdigit() else 0

    history_body = (
        f"=== NEURAL DIFF GIT VERIFICATION ===\n"
        f"Captured (UTC): {now_utc}\n\n"
        f"$ git rev-list --count HEAD (rc={rc_cnt})\n{out_cnt}{err_cnt}\n"
        f"$ git status --short (rc={rc_st})\n{out_st or '(clean working tree)\n'}{err_st}\n"
        f"$ git log --oneline (rc={rc_log})\n{out_log}{err_log}\n"
    )
    (GIT_EVIDENCE_DIR / "git-history.log").write_text(history_body, encoding="utf-8")

    summary = {
        "captured_at_utc": now_utc,
        "commit_count": commit_count,
        "clean_working_tree": (rc_st == 0 and out_st.strip() == ""),
        "status_short": out_st.splitlines(),
        "recent_commits": out_log.splitlines()[:25],
    }
    (GIT_EVIDENCE_DIR / "git_verification.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Git commit count: {commit_count} | Clean tree: {summary['clean_working_tree']}")
    return 0 if (rc_cnt == 0 and rc_log == 0 and rc_st == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
