#!/usr/bin/env python3
"""Verify `evidence/EVIDENCE_MANIFEST.json` against actual files on disk (Section 2.6).

Checks that every listed evidence file exists, matches its recorded byte size, and matches
its SHA-256 digest so no recorded artifact has silently changed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "evidence" / "EVIDENCE_MANIFEST.json"


def compute_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest_path: Path = MANIFEST_PATH) -> tuple[bool, list[str], int]:
    if not manifest_path.is_file():
        return False, [f"Manifest file does not exist: {manifest_path}"], 0

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = data.get("artifacts", [])
    errors: list[str] = []

    for item in artifacts:
        rel_path = item["relative_path"]
        target = REPO_ROOT / rel_path
        if not target.is_file():
            errors.append(f"MISSING: {rel_path}")
            continue
        actual_size = target.stat().st_size
        expected_size = int(item["size_bytes"])
        if actual_size != expected_size:
            errors.append(
                f"SIZE_MISMATCH: {rel_path} (expected {expected_size} bytes, got {actual_size} bytes)"
            )
        actual_hash = compute_sha256(target)
        expected_hash = str(item["sha256"])
        if actual_hash != expected_hash:
            errors.append(
                f"HASH_MISMATCH: {rel_path} (expected {expected_hash}, got {actual_hash})"
            )

    return len(errors) == 0, errors, len(artifacts)


def main() -> int:
    ok, errors, count = verify_manifest()
    if ok:
        print(
            f"VERIFIED: All {count} evidence artifacts in {MANIFEST_PATH.relative_to(REPO_ROOT)} "
            "exist and match recorded byte sizes and SHA-256 hashes."
        )
        return 0

    print(f"FAILED: {len(errors)} integrity discrepancy(ies) across {count} listed artifacts:")
    for err in errors:
        print(f"  - {err}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
