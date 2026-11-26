#!/usr/bin/env python3
"""Example 2: Real open-access model comparison (Qwen2.5-0.5B Base vs Instruct)."""

from __future__ import annotations

from pathlib import Path

from core.orchestrator import ComparisonOrchestrator, ComparisonRequestConfig
from core.report_engine.exporter import ReportEngine


def main() -> None:
    cfg = ComparisonRequestConfig(
        model_a="Qwen/Qwen2.5-0.5B",
        model_b="Qwen/Qwen2.5-0.5B-Instruct",
        modules=["structural", "tensor", "neuron", "activation", "behavioral", "regression"],
        experiment_type="REAL MODEL EXPERIMENT",
        preferred_device="auto",
        behavioral_max_cases=8,
        max_new_tokens=96,
    )
    bundle = ComparisonOrchestrator().run(cfg)
    out = Path("data/example_reports")
    paths = ReportEngine().export_all(bundle, out)
    print(f"Completed REAL MODEL comparison {bundle.comparison_id}:")
    for fmt, p in paths.items():
        print(f"  {fmt}: {p}")


if __name__ == "__main__":
    main()
