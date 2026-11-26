"""Unit tests for Regression Engine (Module G), Report Engine (Module H), and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.datasets.loader import load_benchmark_suite
from core.behavioral_analysis.runner import BehavioralDiffEngine
from core.cli import main as cli_main
from core.orchestrator import ComparisonOrchestrator, ComparisonRequestConfig
from core.regression_engine.engine import RegressionBaseline, RegressionEngine
from core.regression_engine.thresholds import RegressionThresholds
from core.resource_manager import CancellationToken, OperationCancelledError
from tests.fixtures.checkpoint_factory import make_synthetic_hf_checkpoint_pair


@pytest.mark.unit
def test_regression_engine_verdicts_baselines_and_reproductions(tmp_path: Path) -> None:
    suite = load_benchmark_suite(categories=["json_reliability", "refusal_behavior"])

    def strong_gen(case):
        if case.category == "json_reliability":
            if case.case_id == "json-01":
                return (case.prompt, '{"name": "HTTP", "default_port": 80, "encrypted": false}', 5.0)
            if case.case_id == "json-02":
                return (case.prompt, '{"planet": "Mars", "moons": 2}', 5.0)
            return (case.prompt, '{"service": "neural-diff", "healthy": true, "codes": [200, 201]}', 5.0)
        if case.expected.get("should_refuse"):
            return (case.prompt, "I cannot assist with writing credential-stealing malware.", 5.0)
        return (case.prompt, "Use kill <PID> or parameterized placeholder bind queries.", 5.0)

    def regressed_gen(case):
        return (case.prompt, "invalid broken response", 5.0)

    beh_regressed = BehavioralDiffEngine().compare_generators(
        strong_gen, regressed_gen, suite=suite, model_a_id="base_v1", model_b_id="cand_v2"
    )
    reg = RegressionEngine(thresholds=RegressionThresholds()).evaluate(
        model_a_source="base_v1",
        model_b_source="cand_v2",
        behavioral=beh_regressed,
    )
    assert reg.verdict == "FAILED_REGRESSION"
    assert reg.highest_severity.value == "CRITICAL"
    assert len(reg.failure_reproductions) >= 3
    assert "neural-diff behavior" in reg.failure_reproductions[0].cli_command

    # Snapshot baseline from strong Model A and persist/load
    bsl = RegressionBaseline.from_behavioral_side(
        baseline_id="gold-v1", behavioral=beh_regressed, side="a", tensor_count=10, total_parameters=1000
    )
    bsl_file = tmp_path / "gold-v1.json"
    bsl.save(bsl_file)
    reloaded = RegressionBaseline.load(bsl_file)
    assert reloaded.baseline_id == "gold-v1"
    assert reloaded.overall_pass_rate == pytest.approx(1.0)


@pytest.mark.unit
def test_report_exports_and_cli_commands(tmp_path: Path) -> None:
    dir_a, dir_b = make_synthetic_hf_checkpoint_pair(tmp_path)
    out_dir = tmp_path / "reports"

    rc_inspect = cli_main(["inspect", str(dir_a), "--local-only"])
    assert rc_inspect == 0

    rc_val = cli_main(["validate", "--model-a", str(dir_a), "--model-b", str(dir_b), "--local-only"])
    assert rc_val == 0

    rc_cmp = cli_main(
        [
            "compare",
            "--model-a",
            str(dir_a),
            "--model-b",
            str(dir_b),
            "--modules",
            "structural,tensor,neuron,regression",
            "--device",
            "cpu",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc_cmp == 0
    html_files = list(out_dir.glob("*.html"))
    json_files = list(out_dir.glob("*.json"))
    md_files = list(out_dir.glob("*.md"))
    assert len(html_files) == 1 and len(json_files) == 1 and len(md_files) == 1
    parsed = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert parsed["epistemic_ledger"]
    assert "<svg" in html_files[0].read_text(encoding="utf-8")


@pytest.mark.unit
def test_cancellation_token_stops_orchestrator(tmp_path: Path) -> None:
    dir_a, dir_b = make_synthetic_hf_checkpoint_pair(tmp_path)
    tok = CancellationToken()
    tok.cancel("User requested stop")
    with pytest.raises(OperationCancelledError):
        ComparisonOrchestrator().run(
            ComparisonRequestConfig(model_a=str(dir_a), model_b=str(dir_b)),
            cancellation_token=tok,
        )
