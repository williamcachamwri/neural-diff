#!/usr/bin/env python3
"""Generate `evidence/EVIDENCE_MANIFEST.json` from actual captured files on disk.

Note: SHA-256 digests establish post-capture file integrity and tamper detection;
they do not by themselves constitute independent cryptographic proof of event execution.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = REPO_ROOT / "evidence"
MANIFEST_PATH = EVIDENCE_DIR / "EVIDENCE_MANIFEST.json"

CATEGORY_METADATA: dict[str, tuple[str, str]] = {
    "environment": ("Environment Audit & Hardware Detection", "scripts/capture_environment.py"),
    "installation": ("Reproducible Dependency & Toolchain Provisioning", "uv sync / npm install"),
    "research": ("Ecosystem & Hugging Face Hub Live Verification", "Phase 1 & 2 Research Probes"),
    "backend": ("FastAPI Backend & Engine Verification", "API / Backend Integration"),
    "frontend": ("React TypeScript Application Live Screenshots", "scripts/capture_screenshots.py"),
    "model_comparisons": ("Real & Synthetic Model Comparison Artifacts", "ComparisonOrchestrator"),
    "benchmarks": ("Throughput, Memory, & Real Model Benchmarks", "scripts/run_benchmarks.py"),
    "tests": ("Automated Unit, Property, Integration, Regression, Security & E2E Logs", "scripts/run_all_tests.py"),
    "security": ("Secret Scanning & Dependency Vulnerability Audits", "scripts/run_security_audit.py"),
    "git": ("Repository Commit History & Working Tree Verification", "git log / git rev-list"),
    "releases": ("Release Readiness & Exported Sample Reports", "ReportEngine"),
    "evolution": ("Neural Evolution Tracking Multi-Gen DAG, Regressions & Exports", "scripts/capture_evolution_evidence.py"),
    "final_report": ("Final Engineering Acceptance Summary", "Phase 20 Final Verification"),
}


def compute_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest() -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for fp in sorted(EVIDENCE_DIR.rglob("*")):
        if not fp.is_file():
            continue
        if fp.resolve() == MANIFEST_PATH.resolve():
            continue
        if fp.name == ".gitkeep":
            continue

        rel_to_repo = fp.relative_to(REPO_ROOT).as_posix()
        rel_to_ev = fp.relative_to(EVIDENCE_DIR).parts
        category = rel_to_ev[0] if len(rel_to_ev) > 1 else "summary_document"
        related_feature, related_test = CATEGORY_METADATA.get(
            category,
            ("Project Evidence Index & Engineering Report", "Phase 17-20 Verification"),
        )
        stat = fp.stat()
        created_iso = dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.UTC).isoformat()

        artifacts.append(
            {
                "relative_path": rel_to_repo,
                "sha256": compute_sha256(fp),
                "size_bytes": stat.st_size,
                "captured_timestamp_utc": created_iso,
                "category": category,
                "related_feature": related_feature,
                "related_test_or_experiment": related_test,
                "verification_status": "VERIFIED",
            }
        )

    return {
        "schema_version": "1.0.0",
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "integrity_scope_notice": (
            "This manifest records post-capture SHA-256 hashes and byte sizes to detect silent "
            "file modification or truncation. It establishes post-capture file integrity, not "
            "independent cryptographic proof of the underlying execution event."
        ),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def main() -> int:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Generated {MANIFEST_PATH.relative_to(REPO_ROOT)} with {manifest['artifact_count']} artifacts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
