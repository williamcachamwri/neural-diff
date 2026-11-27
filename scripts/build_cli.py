"""Build script for the Charmbracelet `charm.land` v2 native Go CLI (`bin/ndiff`)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_DIR = REPO_ROOT / "cli"
BIN_DIR = REPO_ROOT / "bin"


def resolve_go_binary() -> str | None:
    found = shutil.which("go")
    if found:
        return found
    exe_name = "go.exe" if os.name == "nt" else "go"
    cached = REPO_ROOT / ".cache" / "go" / "bin" / exe_name
    if cached.is_file():
        return str(cached)
    return None


def main() -> int:
    go_bin = resolve_go_binary()
    if not go_bin:
        sys.stderr.write("Error: Go toolchain (>= 1.26) not found on PATH or in .cache/go/bin.\n")
        return 1

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    out_name = "ndiff.exe" if os.name == "nt" else "ndiff"
    out_path = BIN_DIR / out_name

    commit = "HEAD"
    git_bin = shutil.which("git")
    if git_bin:
        proc = subprocess.run(
            [git_bin, "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            commit = proc.stdout.strip()

    env = dict(os.environ)
    env.setdefault("GOPATH", str(REPO_ROOT / ".cache" / "gopath"))
    env.setdefault("GOCACHE", str(REPO_ROOT / ".cache" / "gocache"))

    cmd = [
        go_bin,
        "build",
        "-trimpath",
        f"-ldflags=-s -w -X main.commit={commit}",
        "-o",
        str(out_path),
        "./cmd/ndiff",
    ]
    res = subprocess.run(cmd, cwd=str(CLI_DIR), env=env, check=False)
    if res.returncode != 0:
        return res.returncode

    print(f"Built Charmbracelet CLI -> {out_path} ({out_path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
