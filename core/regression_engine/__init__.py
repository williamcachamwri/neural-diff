"""Regression engine exports (Module G)."""

from __future__ import annotations

from core.regression_engine.engine import (
    FailureReproduction,
    RegressionBaseline,
    RegressionEngine,
    RegressionFinding,
    RegressionReport,
)
from core.regression_engine.thresholds import (
    RegressionThresholds,
    classify_drop_severity,
)

__all__ = [
    "FailureReproduction",
    "RegressionBaseline",
    "RegressionEngine",
    "RegressionFinding",
    "RegressionReport",
    "RegressionThresholds",
    "classify_drop_severity",
]
