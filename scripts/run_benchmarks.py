#!/usr/bin/env python3
"""Execute synthetic micro-benchmarks AND real Hugging Face model comparisons (Phases 15 & 16).

Strictly separates `SYNTHETIC FIXTURE EXPERIMENT` artifacts from `REAL MODEL EXPERIMENT`
artifacts, records actual hardware/runtime/VRAM metrics, and persists completed runs into
`data/neural_diff.db` so the web application displays genuine completed experiments.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Direct Hugging Face cache to workspace drive D:\ to protect C:\ free space
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))

from api.store import ExperimentStore  # noqa: E402
from benchmarks.runners.perf_runner import run_chunked_tensor_microbenchmarks  # noqa: E402
from core.exceptions import ModelAccessBlockedError, NeuralDiffError  # noqa: E402
from core.loaders.model_loader import ModelLoader  # noqa: E402
from core.orchestrator import ComparisonOrchestrator, ComparisonRequestConfig  # noqa: E402
from core.regression_engine.engine import RegressionBaseline, RegressionThresholds  # noqa: E402
from core.report_engine.exporter import ReportEngine  # noqa: E402
from core.resource_manager import ResourceManager  # noqa: E402
from core.tensor_engine.metrics import DiffTolerances  # noqa: E402
from tests.fixtures.checkpoint_factory import make_synthetic_hf_checkpoint_pair  # noqa: E402

BENCH_DIR = REPO_ROOT / "evidence" / "benchmarks"
MODEL_COMP_DIR = REPO_ROOT / "evidence" / "model_comparisons"
RELEASES_DIR = REPO_ROOT / "evidence" / "releases"


def persist_bundle_to_store(store: ExperimentStore, exporter: ReportEngine, bundle) -> None:
    store.create_job(
        comparison_id=bundle.comparison_id,
        created_at_utc=bundle.created_at_utc,
        experiment_type=bundle.experiment_type,
        model_a_id=bundle.model_a.model_id,
        model_b_id=bundle.model_b.model_id,
        selected_modules=bundle.selected_modules,
        total_steps=len(bundle.selected_modules) + 1,
    )
    summary_card = {
        "global_changed_pct": bundle.tensor_diff.global_changed_pct if bundle.tensor_diff else None,
        "global_mean_cosine": bundle.tensor_diff.global_mean_cosine if bundle.tensor_diff else None,
        "behavioral_pass_rate_a": (
            bundle.behavioral.overall_statistics.pass_rate_a if bundle.behavioral else None
        ),
        "behavioral_pass_rate_b": (
            bundle.behavioral.overall_statistics.pass_rate_b if bundle.behavioral else None
        ),
        "behavioral_score_delta": (
            bundle.behavioral.overall_statistics.mean_score_delta if bundle.behavioral else None
        ),
        "pooled_final_cka": (
            bundle.activation.cross_prompt_pooled_cka_final_layer if bundle.activation else None
        ),
        "duration_seconds": bundle.execution_telemetry.get("duration_seconds"),
    }
    store.save_completed(
        comparison_id=bundle.comparison_id,
        model_a_id=bundle.model_a.model_id,
        model_b_id=bundle.model_b.model_id,
        verdict=bundle.regression.verdict if bundle.regression else None,
        summary_dict=summary_card,
        bundle_json=exporter.to_json(bundle, include_full_tensor_records=True),
        markdown_report=exporter.to_markdown(bundle),
        html_report=exporter.to_html(bundle),
    )


def probe_gated_gemma3() -> dict[str, Any]:
    """Test `google/gemma-3-270m` access honestly, preserving gated blocker if unauthenticated."""
    loader = ModelLoader()
    try:
        with loader.load("google/gemma-3-270m") as handle:
            return {
                "model_id": "google/gemma-3-270m",
                "status": "VERIFIED",
                "metadata": handle.metadata.to_dict(),
            }
    except ModelAccessBlockedError as exc:
        return {
            "model_id": "google/gemma-3-270m",
            "status": "BLOCKED",
            "reason": exc.message,
            "details": exc.details,
            "reproduction_when_authorized": (
                'Set $env:HF_TOKEN="<token_with_accepted_gemma_license>" and run: '
                "uv run neural-diff inspect google/gemma-3-270m"
            ),
        }
    except NeuralDiffError as exc:
        return {
            "model_id": "google/gemma-3-270m",
            "status": "FAILED",
            "error": exc.to_dict(),
        }


def main() -> int:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_COMP_DIR.mkdir(parents=True, exist_ok=True)
    RELEASES_DIR.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = [
        f"=== NEURAL DIFF BENCHMARK & REAL MODEL EXECUTION ({dt.datetime.now(dt.UTC).isoformat()}) ==="
    ]

    def _log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    # Check disk headroom before downloading real models
    free_gb = shutil.disk_usage(REPO_ROOT).free / (1024**3)
    _log(f"Workspace free storage: {free_gb:.2f} GB")
    if free_gb < 4.0:
        _log("BLOCKED: Less than 4.0 GB free disk space; skipping Hub downloads.")
        return 1

    store = ExperimentStore()
    exporter = ReportEngine()
    orch = ComparisonOrchestrator()

    # 1. Synthetic Throughput & Memory Micro-Benchmarks (strictly separated!)
    _log("[1/5] Running synthetic chunked tensor & lazy Safetensors micro-benchmarks...")
    synth_perf = run_chunked_tensor_microbenchmarks()
    (BENCH_DIR / "synthetic_throughput_benchmarks.json").write_text(
        json.dumps(synth_perf, indent=2), encoding="utf-8"
    )
    _log("  -> Saved evidence/benchmarks/synthetic_throughput_benchmarks.json")

    # Also materialize a local synthetic fixture pair in `data/demo_fixtures/` for UI synthetic exploration
    demo_dir = REPO_ROOT / "data" / "demo_fixtures"
    synth_a, synth_b = make_synthetic_hf_checkpoint_pair(demo_dir)
    synth_bundle = orch.run(
        ComparisonRequestConfig(
            model_a=str(synth_a),
            model_b=str(synth_b),
            modules=["structural", "tensor", "neuron", "regression"],
            experiment_type="SYNTHETIC FIXTURE EXPERIMENT",
            preferred_device="cpu",
            local_files_only=True,
        ),
        comparison_id="ndiff-synth-demo",
    )
    persist_bundle_to_store(store, exporter, synth_bundle)
    (MODEL_COMP_DIR / "synthetic_fixture_comparison.json").write_text(
        exporter.to_json(synth_bundle), encoding="utf-8"
    )
    _log(f"  -> Completed synthetic fixture run {synth_bundle.comparison_id}")

    # 2. Probe Gemma 3 270M gated access status
    _log("[2/5] Probing google/gemma-3-270m upstream license/gate status...")
    gemma_probe = probe_gated_gemma3()
    (MODEL_COMP_DIR / "gemma3_270m_access_probe.json").write_text(
        json.dumps(gemma_probe, indent=2), encoding="utf-8"
    )
    _log(f"  -> google/gemma-3-270m status: {gemma_probe['status']}")

    # 3. REAL MODEL EXPERIMENT #1: SmolLM2-135M (Base) vs SmolLM2-135M-Instruct (Full 6-Module Suite)
    _log("[3/5] Executing REAL MODEL EXPERIMENT #1: HuggingFaceTB/SmolLM2-135M vs SmolLM2-135M-Instruct...")
    smol_cfg = ComparisonRequestConfig(
        model_a="HuggingFaceTB/SmolLM2-135M",
        model_b="HuggingFaceTB/SmolLM2-135M-Instruct",
        modules=["structural", "tensor", "neuron", "activation", "behavioral", "regression"],
        experiment_type="REAL MODEL EXPERIMENT",
        tolerances=DiffTolerances(atol=1e-6, rtol=1e-5),
        thresholds=RegressionThresholds(),
        preferred_device="auto",
        max_new_tokens=96,
        seed=42,
    )
    smol_bundle = orch.run(smol_cfg, comparison_id="ndiff-real-smollm2-135m")
    persist_bundle_to_store(store, exporter, smol_bundle)
    exporter.export_all(smol_bundle, MODEL_COMP_DIR / "real_smollm2_135m")
    _log(
        f"  -> SmolLM2-135M finished in {smol_bundle.execution_telemetry.get('duration_seconds')}s | "
        f"Changed={smol_bundle.tensor_diff.global_changed_pct if smol_bundle.tensor_diff else 'N/A'}% | "
        f"PassRate A={smol_bundle.behavioral.overall_statistics.pass_rate_a if smol_bundle.behavioral else 'N/A'} -> "
        f"B={smol_bundle.behavioral.overall_statistics.pass_rate_b if smol_bundle.behavioral else 'N/A'} | "
        f"Verdict={smol_bundle.regression.verdict if smol_bundle.regression else 'N/A'}"
    )

    # 4. REAL MODEL EXPERIMENT #2: Qwen/Qwen2.5-0.5B (Base) vs Qwen/Qwen2.5-0.5B-Instruct (Full 6-Module Suite)
    _log("[4/5] Executing REAL MODEL EXPERIMENT #2: Qwen/Qwen2.5-0.5B vs Qwen/Qwen2.5-0.5B-Instruct...")
    qwen_cfg = ComparisonRequestConfig(
        model_a="Qwen/Qwen2.5-0.5B",
        model_b="Qwen/Qwen2.5-0.5B-Instruct",
        modules=["structural", "tensor", "neuron", "activation", "behavioral", "regression"],
        experiment_type="REAL MODEL EXPERIMENT",
        tolerances=DiffTolerances(atol=1e-6, rtol=1e-5),
        thresholds=RegressionThresholds(),
        preferred_device="auto",
        max_new_tokens=96,
        seed=42,
    )
    qwen_bundle = orch.run(qwen_cfg, comparison_id="ndiff-real-qwen25-05b")
    persist_bundle_to_store(store, exporter, qwen_bundle)
    exporter.export_all(qwen_bundle, MODEL_COMP_DIR / "real_qwen25_05b")
    exporter.export_all(qwen_bundle, RELEASES_DIR)

    # Snapshot a real regression baseline from Qwen2.5-0.5B-Instruct in SQLite!
    if qwen_bundle.behavioral is not None:
        qwen_baseline = RegressionBaseline.from_behavioral_side(
            baseline_id="qwen2.5-0.5b-instruct-v1",
            behavioral=qwen_bundle.behavioral,
            side="b",
            tensor_count=qwen_bundle.model_b.tensor_count,
            total_parameters=qwen_bundle.model_b.total_parameters,
        )
        store.save_baseline(qwen_baseline.to_dict())

    _log(
        f"  -> Qwen2.5-0.5B finished in {qwen_bundle.execution_telemetry.get('duration_seconds')}s | "
        f"Changed={qwen_bundle.tensor_diff.global_changed_pct if qwen_bundle.tensor_diff else 'N/A'}% | "
        f"PassRate A={qwen_bundle.behavioral.overall_statistics.pass_rate_a if qwen_bundle.behavioral else 'N/A'} -> "
        f"B={qwen_bundle.behavioral.overall_statistics.pass_rate_b if qwen_bundle.behavioral else 'N/A'} | "
        f"Verdict={qwen_bundle.regression.verdict if qwen_bundle.regression else 'N/A'}"
    )

    # 5. REAL MODEL EXPERIMENT #3: Reverse Regression Test (Instruct -> Base) + Cross-Architecture Diff
    _log("[5/5] Executing REAL MODEL EXPERIMENT #3 (Cross-Architecture & Regression Detection)...")
    # 5a. Cross-architecture structural + activation diff: Qwen2.5-0.5B-Instruct vs SmolLM2-135M-Instruct
    cross_cfg = ComparisonRequestConfig(
        model_a="Qwen/Qwen2.5-0.5B-Instruct",
        model_b="HuggingFaceTB/SmolLM2-135M-Instruct",
        modules=["structural", "activation", "regression"],
        experiment_type="REAL MODEL EXPERIMENT",
        preferred_device="auto",
        local_files_only=True,
    )
    cross_bundle = orch.run(cross_cfg, comparison_id="ndiff-real-cross-arch")
    persist_bundle_to_store(store, exporter, cross_bundle)
    exporter.export_all(cross_bundle, MODEL_COMP_DIR / "real_cross_architecture")

    # 5b. Real regression detection: evaluating unaligned Base candidate against Instruct baseline!
    reg_demo_cfg = ComparisonRequestConfig(
        model_a="Qwen/Qwen2.5-0.5B-Instruct",
        model_b="Qwen/Qwen2.5-0.5B",
        modules=["structural", "tensor", "behavioral", "regression"],
        experiment_type="REAL MODEL EXPERIMENT",
        preferred_device="auto",
        behavioral_max_cases=16,
        max_new_tokens=96,
        local_files_only=True,
    )
    reg_bundle = orch.run(reg_demo_cfg, comparison_id="ndiff-real-regression-detected")
    persist_bundle_to_store(store, exporter, reg_bundle)
    exporter.export_all(reg_bundle, MODEL_COMP_DIR / "real_regression_detection")
    _log(
        f"  -> Real Regression Run {reg_bundle.comparison_id}: Verdict={reg_bundle.regression.verdict if reg_bundle.regression else 'N/A'} "
        f"({len(reg_bundle.regression.findings) if reg_bundle.regression else 0} findings, "
        f"{len(reg_bundle.regression.failure_reproductions) if reg_bundle.regression else 0} CLI reproductions)"
    )

    # Re-select Qwen2.5-0.5B Base -> Instruct as newest experiment by touching/updating summary if needed
    # (ExperimentHistory lists newest first; having both PASSED upgrade and FAILED_REGRESSION downgrade is rich!)
    real_perf_summary = {
        "captured_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "device": ResourceManager().resolve_device(),
        "experiments": [
            {
                "comparison_id": smol_bundle.comparison_id,
                "experiment_type": smol_bundle.experiment_type,
                "model_a": smol_bundle.model_a.model_id,
                "revision_a": smol_bundle.model_a.revision,
                "model_b": smol_bundle.model_b.model_id,
                "revision_b": smol_bundle.model_b.revision,
                "parameters_a": smol_bundle.model_a.total_parameters,
                "parameters_b": smol_bundle.model_b.total_parameters,
                "tensor_count": smol_bundle.model_a.tensor_count,
                "global_changed_pct": smol_bundle.tensor_diff.global_changed_pct if smol_bundle.tensor_diff else None,
                "global_mean_cosine": smol_bundle.tensor_diff.global_mean_cosine if smol_bundle.tensor_diff else None,
                "pooled_final_cka": smol_bundle.activation.cross_prompt_pooled_cka_final_layer if smol_bundle.activation else None,
                "pass_rate_a": smol_bundle.behavioral.overall_statistics.pass_rate_a if smol_bundle.behavioral else None,
                "pass_rate_b": smol_bundle.behavioral.overall_statistics.pass_rate_b if smol_bundle.behavioral else None,
                "mean_score_delta": smol_bundle.behavioral.overall_statistics.mean_score_delta if smol_bundle.behavioral else None,
                "verdict": smol_bundle.regression.verdict if smol_bundle.regression else None,
                "telemetry": smol_bundle.execution_telemetry,
            },
            {
                "comparison_id": qwen_bundle.comparison_id,
                "experiment_type": qwen_bundle.experiment_type,
                "model_a": qwen_bundle.model_a.model_id,
                "revision_a": qwen_bundle.model_a.revision,
                "model_b": qwen_bundle.model_b.model_id,
                "revision_b": qwen_bundle.model_b.revision,
                "parameters_a": qwen_bundle.model_a.total_parameters,
                "parameters_b": qwen_bundle.model_b.total_parameters,
                "tensor_count": qwen_bundle.model_a.tensor_count,
                "global_changed_pct": qwen_bundle.tensor_diff.global_changed_pct if qwen_bundle.tensor_diff else None,
                "global_mean_cosine": qwen_bundle.tensor_diff.global_mean_cosine if qwen_bundle.tensor_diff else None,
                "pooled_final_cka": qwen_bundle.activation.cross_prompt_pooled_cka_final_layer if qwen_bundle.activation else None,
                "pass_rate_a": qwen_bundle.behavioral.overall_statistics.pass_rate_a if qwen_bundle.behavioral else None,
                "pass_rate_b": qwen_bundle.behavioral.overall_statistics.pass_rate_b if qwen_bundle.behavioral else None,
                "mean_score_delta": qwen_bundle.behavioral.overall_statistics.mean_score_delta if qwen_bundle.behavioral else None,
                "verdict": qwen_bundle.regression.verdict if qwen_bundle.regression else None,
                "telemetry": qwen_bundle.execution_telemetry,
            },
            {
                "comparison_id": cross_bundle.comparison_id,
                "experiment_type": cross_bundle.experiment_type,
                "model_a": cross_bundle.model_a.model_id,
                "model_b": cross_bundle.model_b.model_id,
                "pooled_final_cka": cross_bundle.activation.cross_prompt_pooled_cka_final_layer if cross_bundle.activation else None,
                "verdict": cross_bundle.regression.verdict if cross_bundle.regression else None,
                "telemetry": cross_bundle.execution_telemetry,
            },
            {
                "comparison_id": reg_bundle.comparison_id,
                "experiment_type": reg_bundle.experiment_type,
                "model_a": reg_bundle.model_a.model_id,
                "model_b": reg_bundle.model_b.model_id,
                "pass_rate_a": reg_bundle.behavioral.overall_statistics.pass_rate_a if reg_bundle.behavioral else None,
                "pass_rate_b": reg_bundle.behavioral.overall_statistics.pass_rate_b if reg_bundle.behavioral else None,
                "verdict": reg_bundle.regression.verdict if reg_bundle.regression else None,
                "findings_count": len(reg_bundle.regression.findings) if reg_bundle.regression else 0,
                "telemetry": reg_bundle.execution_telemetry,
            },
        ],
    }
    (BENCH_DIR / "real_model_benchmarks.json").write_text(
        json.dumps(real_perf_summary, indent=2), encoding="utf-8"
    )
    (BENCH_DIR / "model-comparison.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    _log("Completed all synthetic and real model benchmarks successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
