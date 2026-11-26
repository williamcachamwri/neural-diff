"""Tensor comparison engine exports (Module B)."""

from __future__ import annotations

from core.tensor_engine.comparator import (
    LayerDiffSummary,
    TensorComparisonRecord,
    TensorDiffEngine,
    TensorDiffReport,
    TensorDiffStatus,
)
from core.tensor_engine.metrics import (
    DiffTolerances,
    TensorDiffMetrics,
    compare_tensors,
    compute_chunked_tensor_metrics,
)

__all__ = [
    "DiffTolerances",
    "LayerDiffSummary",
    "TensorComparisonRecord",
    "TensorDiffEngine",
    "TensorDiffMetrics",
    "TensorDiffReport",
    "TensorDiffStatus",
    "compare_tensors",
    "compute_chunked_tensor_metrics",
]
