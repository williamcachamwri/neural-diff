"""Regression Detection Engine (Module G) — baselines, thresholds, reproduction, and history."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.activation_analysis.comparator import ActivationDiffReport
from core.behavioral_analysis.runner import BehavioralDiffReport, PairedCaseComparison
from core.regression_engine.thresholds import (
    RegressionThresholds,
    classify_drop_severity,
)
from core.security import SecurityPolicy, read_validated_json, validate_safe_path
from core.structural_analysis.diff import StructuralDiffReport
from core.tensor_engine.comparator import TensorDiffReport
from core.types import EpistemicStatus, Severity


@dataclass(frozen=True)
class FailureReproduction:
    """Deterministic reproduction artifact for a regressed test case."""

    case_id: str
    category: str
    prompt: str
    baseline_output: str
    candidate_output: str
    baseline_score: float
    candidate_score: float
    rationale_candidate: str
    cli_command: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegressionFinding:
    """Single detected regression or warning across weight, activation, or behavioral checks."""

    finding_id: str
    domain: str  # "tensor" | "structural" | "activation" | "behavioral" | "baseline"
    severity: Severity
    epistemic_status: EpistemicStatus
    title: str
    description: str
    metric_name: str
    baseline_value: Any
    candidate_value: Any
    threshold_value: Any

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        data["epistemic_status"] = self.epistemic_status.value
        return data


@dataclass
class RegressionBaseline:
    """Versioned baseline snapshot for repeatable historical regression tracking."""

    baseline_id: str
    created_at_utc: str
    model_id: str
    revision: str | None
    suite_id: str
    suite_version: str
    overall_pass_rate: float
    overall_mean_score: float
    category_scores: dict[str, float]
    category_pass_rates: dict[str, float]
    tensor_count: int
    total_parameters: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, file_path: str | Path, policy: SecurityPolicy | None = None) -> RegressionBaseline:
        safe = validate_safe_path(file_path, policy=policy, must_exist=True)
        raw = read_validated_json(safe)
        return cls(
            baseline_id=str(raw["baseline_id"]),
            created_at_utc=str(raw["created_at_utc"]),
            model_id=str(raw["model_id"]),
            revision=raw.get("revision"),
            suite_id=str(raw["suite_id"]),
            suite_version=str(raw["suite_version"]),
            overall_pass_rate=float(raw["overall_pass_rate"]),
            overall_mean_score=float(raw["overall_mean_score"]),
            category_scores={str(k): float(v) for k, v in raw.get("category_scores", {}).items()},
            category_pass_rates={str(k): float(v) for k, v in raw.get("category_pass_rates", {}).items()},
            tensor_count=int(raw.get("tensor_count", 0)),
            total_parameters=int(raw.get("total_parameters", 0)),
        )

    @classmethod
    def from_behavioral_side(
        cls,
        *,
        baseline_id: str,
        behavioral: BehavioralDiffReport,
        side: str = "a",
        tensor_count: int = 0,
        total_parameters: int = 0,
    ) -> RegressionBaseline:
        use_b = side.lower() == "b"
        stats = behavioral.overall_statistics
        cat_scores = {
            c.category: (c.mean_score_b if use_b else c.mean_score_a)
            for c in behavioral.category_summaries
        }
        cat_pass = {
            c.category: (c.pass_rate_b if use_b else c.pass_rate_a)
            for c in behavioral.category_summaries
        }
        return cls(
            baseline_id=baseline_id,
            created_at_utc=dt.datetime.now(dt.UTC).isoformat(),
            model_id=behavioral.model_b_id if use_b else behavioral.model_a_id,
            revision=behavioral.revision_b if use_b else behavioral.revision_a,
            suite_id=behavioral.suite_id,
            suite_version=behavioral.suite_version,
            overall_pass_rate=stats.pass_rate_b if use_b else stats.pass_rate_a,
            overall_mean_score=stats.mean_score_b if use_b else stats.mean_score_a,
            category_scores=cat_scores,
            category_pass_rates=cat_pass,
            tensor_count=tensor_count,
            total_parameters=total_parameters,
        )


@dataclass
class RegressionReport:
    """Complete Module G regression evaluation verdict, findings, and failure reproductions."""

    verdict: str  # "PASSED" | "WARNING" | "FAILED_REGRESSION"
    highest_severity: Severity
    thresholds: RegressionThresholds
    findings: list[RegressionFinding]
    failure_reproductions: list[FailureReproduction]
    baseline_comparison: dict[str, Any] | None = None
    historical_trend: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epistemic_status": EpistemicStatus.OBSERVED.value,
            "verdict": self.verdict,
            "highest_severity": self.highest_severity.value,
            "thresholds": self.thresholds.to_dict(),
            "finding_counts": {
                sev.value: sum(1 for f in self.findings if f.severity == sev)
                for sev in Severity
            },
            "findings": [f.to_dict() for f in self.findings],
            "failure_reproductions": [r.to_dict() for r in self.failure_reproductions],
            "baseline_comparison": self.baseline_comparison,
            "historical_trend": self.historical_trend,
        }


_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class RegressionEngine:
    """Evaluates comparison outputs and optional saved baselines against threshold policies."""

    def __init__(self, thresholds: RegressionThresholds | None = None) -> None:
        self.thresholds = thresholds or RegressionThresholds()

    def evaluate(
        self,
        *,
        model_a_source: str,
        model_b_source: str,
        structural: StructuralDiffReport | None = None,
        tensor_diff: TensorDiffReport | None = None,
        activation: ActivationDiffReport | None = None,
        behavioral: BehavioralDiffReport | None = None,
        baseline: RegressionBaseline | None = None,
        historical_runs: list[dict[str, Any]] | None = None,
    ) -> RegressionReport:
        findings: list[RegressionFinding] = []
        reproductions: list[FailureReproduction] = []
        idx_counter = 0

        def _next_id(prefix: str) -> str:
            nonlocal idx_counter
            idx_counter += 1
            return f"{prefix}-{idx_counter:03d}"

        # 1. Numerical integrity checks (Tensor Diff)
        if tensor_diff is not None:
            if self.thresholds.forbid_nan_or_inf and tensor_diff.has_any_nan_or_inf:
                findings.append(
                    RegressionFinding(
                        finding_id=_next_id("REG-NUM"),
                        domain="tensor",
                        severity=Severity.CRITICAL,
                        epistemic_status=EpistemicStatus.OBSERVED,
                        title="Non-finite weights (NaN or Infinity) detected",
                        description="One or more checkpoint tensors contain NaN or +/-Infinity values.",
                        metric_name="has_any_nan_or_inf",
                        baseline_value=False,
                        candidate_value=True,
                        threshold_value=False,
                    )
                )
            if self.thresholds.forbid_missing_tensors and tensor_diff.missing_in_b_tensors > 0:
                findings.append(
                    RegressionFinding(
                        finding_id=_next_id("REG-STR"),
                        domain="structural",
                        severity=Severity.HIGH,
                        epistemic_status=EpistemicStatus.OBSERVED,
                        title="Missing checkpoint tensors in candidate model",
                        description=f"{tensor_diff.missing_in_b_tensors} tensor(s) present in Model A are absent in Model B.",
                        metric_name="missing_in_b_tensors",
                        baseline_value=0,
                        candidate_value=tensor_diff.missing_in_b_tensors,
                        threshold_value=0,
                    )
                )
            if self.thresholds.forbid_shape_mismatches and tensor_diff.shape_mismatch_tensors > 0:
                findings.append(
                    RegressionFinding(
                        finding_id=_next_id("REG-SHP"),
                        domain="structural",
                        severity=Severity.HIGH,
                        epistemic_status=EpistemicStatus.OBSERVED,
                        title="Shared tensor shape mismatches detected",
                        description=f"{tensor_diff.shape_mismatch_tensors} shared tensor(s) changed dimensions.",
                        metric_name="shape_mismatch_tensors",
                        baseline_value=0,
                        candidate_value=tensor_diff.shape_mismatch_tensors,
                        threshold_value=0,
                    )
                )
        elif (
            structural is not None
            and self.thresholds.forbid_missing_tensors
            and len(structural.removed_tensors) > 0
        ):
            findings.append(
                RegressionFinding(
                    finding_id=_next_id("REG-STR"),
                    domain="structural",
                    severity=Severity.HIGH,
                    epistemic_status=EpistemicStatus.OBSERVED,
                    title="Removed tensors detected in candidate model",
                    description=f"{len(structural.removed_tensors)} tensor(s) were removed.",
                    metric_name="removed_tensors",
                    baseline_value=0,
                    candidate_value=len(structural.removed_tensors),
                    threshold_value=0,
                )
            )

        # 2. Activation stability checks
        if activation is not None:
            cka = activation.cross_prompt_pooled_cka_final_layer
            if cka is not None and cka < self.thresholds.min_final_hidden_cka:
                findings.append(
                    RegressionFinding(
                        finding_id=_next_id("REG-ACT"),
                        domain="activation",
                        severity=Severity.MEDIUM,
                        epistemic_status=EpistemicStatus.CORRELATED,
                        title="Substantial final-layer representation divergence",
                        description=(
                            f"Cross-prompt final-layer Linear CKA ({cka:.4f}) is below policy threshold "
                            f"({self.thresholds.min_final_hidden_cka:.4f}); correlated with potential downstream drift."
                        ),
                        metric_name="cross_prompt_pooled_cka_final_layer",
                        baseline_value=1.0,
                        candidate_value=cka,
                        threshold_value=self.thresholds.min_final_hidden_cka,
                    )
                )

        # 3. Behavioral regression checks
        if behavioral is not None:
            stats = behavioral.overall_statistics
            pass_drop = round(stats.pass_rate_a - stats.pass_rate_b, 6)
            score_drop = round(stats.mean_score_a - stats.mean_score_b, 6)

            sev_pass = classify_drop_severity(
                pass_drop,
                warn_threshold=self.thresholds.max_overall_pass_rate_drop,
                critical_threshold=self.thresholds.critical_overall_pass_rate_drop,
            )
            if sev_pass is not None:
                findings.append(
                    RegressionFinding(
                        finding_id=_next_id("REG-BEH"),
                        domain="behavioral",
                        severity=sev_pass,
                        epistemic_status=EpistemicStatus.OBSERVED,
                        title="Overall behavioral pass-rate drop",
                        description=(
                            f"Pass rate decreased from {stats.pass_rate_a:.1%} to {stats.pass_rate_b:.1%} "
                            f"(drop={pass_drop:.1%})."
                        ),
                        metric_name="overall_pass_rate_drop",
                        baseline_value=stats.pass_rate_a,
                        candidate_value=stats.pass_rate_b,
                        threshold_value=self.thresholds.max_overall_pass_rate_drop,
                    )
                )

            if score_drop > self.thresholds.max_overall_score_drop and sev_pass is None:
                findings.append(
                    RegressionFinding(
                        finding_id=_next_id("REG-BEH"),
                        domain="behavioral",
                        severity=Severity.HIGH,
                        epistemic_status=EpistemicStatus.OBSERVED,
                        title="Overall behavioral mean score drop",
                        description=(
                            f"Mean deterministic score dropped from {stats.mean_score_a:.3f} to {stats.mean_score_b:.3f}."
                        ),
                        metric_name="overall_mean_score_drop",
                        baseline_value=stats.mean_score_a,
                        candidate_value=stats.mean_score_b,
                        threshold_value=self.thresholds.max_overall_score_drop,
                    )
                )

            for cat in behavioral.category_summaries:
                cat_drop = round(cat.mean_score_a - cat.mean_score_b, 6)
                cat_sev = classify_drop_severity(
                    cat_drop,
                    warn_threshold=self.thresholds.max_category_score_drop,
                    critical_threshold=self.thresholds.critical_category_score_drop,
                )
                if (
                    cat.category == "refusal_behavior"
                    and self.thresholds.zero_tolerance_refusal_regression
                    and cat.regressed_cases > 0
                ):
                    cat_sev = Severity.CRITICAL

                if cat_sev is not None:
                    findings.append(
                        RegressionFinding(
                            finding_id=_next_id("REG-CAT"),
                            domain="behavioral",
                            severity=cat_sev,
                            epistemic_status=EpistemicStatus.OBSERVED,
                            title=f"Behavioral regression in '{cat.category}'",
                            description=(
                                f"Category '{cat.category}' mean score changed {cat.mean_score_a:.3f} -> {cat.mean_score_b:.3f} "
                                f"({cat.regressed_cases} regressed case(s))."
                            ),
                            metric_name=f"category.{cat.category}.mean_score",
                            baseline_value=cat.mean_score_a,
                            candidate_value=cat.mean_score_b,
                            threshold_value=self.thresholds.max_category_score_drop,
                        )
                    )

            seed = behavioral.inference_parameters.get("seed", 42)
            for pcase in behavioral.cases:
                if pcase.status_change == "regressed":
                    reproductions.append(
                        self._build_reproduction(model_a_source, model_b_source, pcase, int(seed))
                    )

        # 4. Optional persisted baseline comparison
        baseline_comp: dict[str, Any] | None = None
        if baseline is not None and behavioral is not None:
            suite_match = (
                baseline.suite_id == behavioral.suite_id
                and baseline.suite_version == behavioral.suite_version
            )
            cand_pass = behavioral.overall_statistics.pass_rate_b
            cand_score = behavioral.overall_statistics.mean_score_b
            b_pass_drop = round(baseline.overall_pass_rate - cand_pass, 6)
            b_score_drop = round(baseline.overall_mean_score - cand_score, 6)

            baseline_comp = {
                "baseline_id": baseline.baseline_id,
                "baseline_model_id": baseline.model_id,
                "suite_compatible": suite_match,
                "baseline_pass_rate": baseline.overall_pass_rate,
                "candidate_pass_rate": cand_pass,
                "pass_rate_drop_vs_baseline": b_pass_drop,
                "baseline_mean_score": baseline.overall_mean_score,
                "candidate_mean_score": cand_score,
                "score_drop_vs_baseline": b_score_drop,
            }
            if not suite_match:
                findings.append(
                    RegressionFinding(
                        finding_id=_next_id("REG-BSL"),
                        domain="baseline",
                        severity=Severity.MEDIUM,
                        epistemic_status=EpistemicStatus.OBSERVED,
                        title="Baseline benchmark suite version mismatch",
                        description=(
                            f"Baseline uses {baseline.suite_id}@{baseline.suite_version} while run uses "
                            f"{behavioral.suite_id}@{behavioral.suite_version}."
                        ),
                        metric_name="suite_version",
                        baseline_value=f"{baseline.suite_id}@{baseline.suite_version}",
                        candidate_value=f"{behavioral.suite_id}@{behavioral.suite_version}",
                        threshold_value="exact_match",
                    )
                )
            elif b_pass_drop > self.thresholds.max_overall_pass_rate_drop:
                findings.append(
                    RegressionFinding(
                        finding_id=_next_id("REG-BSL"),
                        domain="baseline",
                        severity=Severity.HIGH,
                        epistemic_status=EpistemicStatus.OBSERVED,
                        title=f"Candidate regressed relative to baseline '{baseline.baseline_id}'",
                        description=(
                            f"Candidate pass rate ({cand_pass:.1%}) is {b_pass_drop:.1%} below baseline ({baseline.overall_pass_rate:.1%})."
                        ),
                        metric_name="pass_rate_vs_baseline",
                        baseline_value=baseline.overall_pass_rate,
                        candidate_value=cand_pass,
                        threshold_value=self.thresholds.max_overall_pass_rate_drop,
                    )
                )

        if not findings:
            highest = Severity.INFO
            verdict = "PASSED"
        else:
            highest = max(findings, key=lambda f: _SEVERITY_ORDER[f.severity]).severity
            if highest in (Severity.CRITICAL, Severity.HIGH):
                verdict = "FAILED_REGRESSION"
            else:
                verdict = "WARNING"

        return RegressionReport(
            verdict=verdict,
            highest_severity=highest,
            thresholds=self.thresholds,
            findings=findings,
            failure_reproductions=reproductions,
            baseline_comparison=baseline_comp,
            historical_trend=list(historical_runs or []),
        )

    @staticmethod
    def _build_reproduction(
        model_a: str,
        model_b: str,
        pcase: PairedCaseComparison,
        seed: int,
    ) -> FailureReproduction:
        cmd = (
            f'neural-diff behavior --model-a "{model_a}" --model-b "{model_b}" '
            f'--category "{pcase.case.category}" --seed {seed}'
        )
        return FailureReproduction(
            case_id=pcase.case.case_id,
            category=pcase.case.category,
            prompt=pcase.case.prompt,
            baseline_output=pcase.output_a,
            candidate_output=pcase.output_b,
            baseline_score=pcase.eval_a.score,
            candidate_score=pcase.eval_b.score,
            rationale_candidate=pcase.eval_b.rationale,
            cli_command=cmd,
        )
