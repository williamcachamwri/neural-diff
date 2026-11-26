"""Configurable threshold policies and severity rules for Regression Engine (Module G)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.types import Severity


@dataclass(frozen=True)
class RegressionThresholds:
    """Policy thresholds governing structural, numerical, activation, and behavioral gates."""

    policy_name: str = "production-default-v1"
    forbid_nan_or_inf: bool = True
    forbid_missing_tensors: bool = True
    forbid_shape_mismatches: bool = True
    max_overall_pass_rate_drop: float = 0.05
    critical_overall_pass_rate_drop: float = 0.15
    max_overall_score_drop: float = 0.05
    max_category_score_drop: float = 0.10
    critical_category_score_drop: float = 0.25
    zero_tolerance_refusal_regression: bool = True
    min_final_hidden_cka: float = 0.60

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_drop_severity(
    drop: float,
    *,
    warn_threshold: float,
    critical_threshold: float,
) -> Severity | None:
    """Classify a positive performance drop (`score_a - score_b`) into a `Severity` level."""
    if drop <= 0.0:
        return None
    if drop >= critical_threshold:
        return Severity.CRITICAL
    if drop >= warn_threshold:
        return Severity.HIGH
    if drop >= warn_threshold * 0.5:
        return Severity.MEDIUM
    return Severity.LOW
