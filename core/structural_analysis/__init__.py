"""Structural, compatibility, neuron, and attention analysis exports (Modules C & D)."""

from __future__ import annotations

from core.structural_analysis.compatibility import (
    CompatibilityIssue,
    CompatibilityReport,
    validate_compatibility,
)
from core.structural_analysis.diff import (
    DtypeChangeRecord,
    LayerStructuralDelta,
    RenamedTensorCandidate,
    ShapeChangeRecord,
    StructuralDiffEngine,
    StructuralDiffReport,
)
from core.structural_analysis.neuron_attention import (
    POLYSEMANTICITY_CAUTION,
    AttentionProjectionAnalysis,
    EmbeddingProjectionAnalysis,
    HeadSliceDiff,
    MLPBasisAnalysis,
    NeuronAttentionAnalyzer,
    NeuronAttentionReport,
    NormLayerAnalysis,
)

__all__ = [
    "AttentionProjectionAnalysis",
    "CompatibilityIssue",
    "CompatibilityReport",
    "DtypeChangeRecord",
    "EmbeddingProjectionAnalysis",
    "HeadSliceDiff",
    "LayerStructuralDelta",
    "MLPBasisAnalysis",
    "NeuronAttentionAnalyzer",
    "NeuronAttentionReport",
    "NormLayerAnalysis",
    "POLYSEMANTICITY_CAUTION",
    "RenamedTensorCandidate",
    "ShapeChangeRecord",
    "StructuralDiffEngine",
    "StructuralDiffReport",
    "validate_compatibility",
]
