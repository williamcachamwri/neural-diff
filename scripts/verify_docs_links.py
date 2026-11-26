#!/usr/bin/env python3
"""Verify all local relative file and screenshot links across Markdown documentation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parent.parent

FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_ATTR_RE = re.compile(r"""(?:src|href)=["']([^"']+)["']""")

SKIP_PREFIXES = (
    "http://",
    "https://",
    "mailto:",
    "tel:",
    "data:",
    "conversation://",
    "file://",
    "#",
)


def iter_markdown_files() -> list[Path]:
    candidates: list[Path] = []
    for root_doc in ("README.md", "CONTRIBUTING.md", "CHANGELOG.md"):
        p = REPO_ROOT / root_doc
        if p.is_file():
            candidates.append(p)
    for folder in ("docs", "evidence"):
        base = REPO_ROOT / folder
        if base.is_dir():
            candidates.extend(sorted(base.rglob("*.md")))
    return candidates


def strip_fenced_blocks(markdown_text: str) -> str:
    out_lines: list[str] = []
    in_fence = False
    fence_marker = ""
    for line in markdown_text.splitlines():
        m = FENCE_RE.match(line)
        if m:
            token = m.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = token[0] * len(token)
            elif line.lstrip().startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            continue
        if not in_fence:
            out_lines.append(line)
    return "\n".join(out_lines)


def clean_target(raw: str) -> str:
    target = raw.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    # Strip optional markdown link title: path/to/file.png "Title"
    if " " in target and (target.endswith('"') or target.endswith("'")):
        target = target.split(" ", 1)[0].strip()
    target = target.split("#", 1)[0].split("?", 1)[0].strip()
    return unquote(target)


def main() -> int:
    md_files = iter_markdown_files()
    broken: list[dict[str, str]] = []
    verified_count = 0

    for md_path in md_files:
        raw_text = md_path.read_text(encoding="utf-8")
        text = strip_fenced_blocks(raw_text)
        targets = MD_LINK_RE.findall(text) + HTML_ATTR_RE.findall(text)

        for raw_target in targets:
            stripped = raw_target.strip()
            if not stripped or stripped.startswith(SKIP_PREFIXES):
                continue
            rel_part = clean_target(stripped)
            if not rel_part:
                continue

            resolved = (
                (REPO_ROOT / rel_part.lstrip("/")).resolve()
                if rel_part.startswith("/")
                else (md_path.parent / rel_part).resolve()
            )
            if not resolved.exists():
                broken.append(
                    {
                        "source_doc": str(md_path.relative_to(REPO_ROOT)),
                        "target": raw_target,
                        "resolved": str(resolved),
                        "reason": "missing",
                    }
                )
            elif resolved.is_file() and resolved.stat().st_size == 0:
                broken.append(
                    {
                        "source_doc": str(md_path.relative_to(REPO_ROOT)),
                        "target": raw_target,
                        "resolved": str(resolved),
                        "reason": "empty_file",
                    }
                )
            else:
                verified_count += 1

    report = {
        "markdown_files_checked": len(md_files),
        "local_links_verified": verified_count,
        "broken_links_count": len(broken),
        "broken_links": broken,
    }
    print(json.dumps(report, indent=2))
    return 0 if not broken else 1


if __name__ == "__main__":
    raise SystemExit(main())
