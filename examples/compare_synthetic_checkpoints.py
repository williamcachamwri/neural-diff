#!/usr/bin/env python3
"""Example 1: Deterministic synthetic checkpoint comparison and HTML/Markdown/JSON export."""

from __future__ import annotations

import tempfile
from pathlib import Path

from core.orchestrator import ComparisonOrchestrator, ComparisonRequestConfig
from core.report_engine.exporter import ReportEngine
from core.security import SecurityPolicy
from tests.fixtures.checkpoint_factory import make_synthetic_hf_checkpoint_pair


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dir_a, dir_b = make_synthetic_hf_checkpoint_pair(root)
        policy = SecurityPolicy(allowed_roots=[root])
        bundle = ComparisonOrchestrator(policy=policy).run(
            ComparisonRequestConfig(
                model_a=str(dir_a),
                model_b=str(dir_b),
                modules=["structural", "tensor", "neuron", "regression"],
                experiment_type="SYNTHETIC FIXTURE EXPERIMENT",
                preferred_device="cpu",
                local_files_only=True,
            )
        )
        paths = ReportEngine().export_all(bundle, root / "reports")
        print(f"Completed {bundle.comparison_id}: verdict={bundle.regression.verdict if bundle.regression else 'N/A'}")
        for fmt, p in paths.items():
            print(f"  {fmt}: {p.name} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
