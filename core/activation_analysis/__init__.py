"""Activation diffing exports (Module E)."""

from __future__ import annotations

from core.activation_analysis.comparator import (
    DEFAULT_DIAGNOSTIC_PROMPTS,
    ActivationDiffEngine,
    ActivationDiffReport,
    CapturedPromptTrace,
    PromptActivationDiff,
    capture_module_traces,
)
from core.activation_analysis.metrics import (
    ActivationLayerComparison,
    compare_activation_tensors,
    compute_linear_cka,
)

__all__ = [
    "ActivationDiffEngine",
    "ActivationDiffReport",
    "ActivationLayerComparison",
    "CapturedPromptTrace",
    "DEFAULT_DIAGNOSTIC_PROMPTS",
    "PromptActivationDiff",
    "capture_module_traces",
    "compare_activation_tensors",
    "compute_linear_cka",
]
