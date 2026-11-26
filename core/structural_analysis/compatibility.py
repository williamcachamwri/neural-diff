"""Model compatibility validation across architectures, shapes, and vocabularies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.loaders.base import LoadedModel
from core.types import EpistemicStatus, Severity


@dataclass(frozen=True)
class CompatibilityIssue:
    """Single compatibility check finding."""

    code: str
    severity: Severity
    message: str
    field: str
    value_a: Any
    value_b: Any
    epistemic_status: EpistemicStatus = EpistemicStatus.OBSERVED

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        data["epistemic_status"] = self.epistemic_status.value
        return data


@dataclass
class CompatibilityReport:
    """Comprehensive compatibility assessment before running analysis modules."""

    compatible_for_direct_tensor_diff: bool
    compatible_for_structural_diff: bool
    compatible_for_head_neuron_analysis: bool
    compatible_for_activation_diff: bool
    compatible_for_behavioral_diff: bool
    shared_tensor_count: int
    shape_matched_tensor_count: int
    jaccard_key_overlap: float
    issues: list[CompatibilityIssue]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "epistemic_status": EpistemicStatus.OBSERVED.value,
            "compatible_for_direct_tensor_diff": self.compatible_for_direct_tensor_diff,
            "compatible_for_structural_diff": self.compatible_for_structural_diff,
            "compatible_for_head_neuron_analysis": self.compatible_for_head_neuron_analysis,
            "compatible_for_activation_diff": self.compatible_for_activation_diff,
            "compatible_for_behavioral_diff": self.compatible_for_behavioral_diff,
            "shared_tensor_count": self.shared_tensor_count,
            "shape_matched_tensor_count": self.shape_matched_tensor_count,
            "jaccard_key_overlap": self.jaccard_key_overlap,
            "issues": [i.to_dict() for i in self.issues],
            "summary": self.summary,
        }


def validate_compatibility(model_a: LoadedModel, model_b: LoadedModel) -> CompatibilityReport:
    """Validate structural, dimensional, and architectural compatibility between two models."""
    arch_a = model_a.metadata.architecture
    arch_b = model_b.metadata.architecture
    keys_a = set(model_a.tensors.keys())
    keys_b = set(model_b.tensors.keys())
    shared_keys = keys_a & keys_b
    union_keys = keys_a | keys_b
    jaccard = (len(shared_keys) / len(union_keys)) if union_keys else 1.0

    shape_matched = [
        k for k in shared_keys if model_a.tensors[k].shape == model_b.tensors[k].shape
    ]

    issues: list[CompatibilityIssue] = []

    # 1. Architecture family check
    if arch_a.model_type and arch_b.model_type and arch_a.model_type != arch_b.model_type:
        issues.append(
            CompatibilityIssue(
                code="ARCH_MODEL_TYPE_MISMATCH",
                severity=Severity.HIGH,
                message=(
                    f"Different `model_type` families ({arch_a.model_type} vs {arch_b.model_type}); "
                    "parameter naming conventions and basis spaces cannot be assumed equivalent."
                ),
                field="model_type",
                value_a=arch_a.model_type,
                value_b=arch_b.model_type,
            )
        )
    elif (
        arch_a.architecture_class
        and arch_b.architecture_class
        and arch_a.architecture_class != arch_b.architecture_class
    ):
        issues.append(
            CompatibilityIssue(
                code="ARCH_CLASS_MISMATCH",
                severity=Severity.MEDIUM,
                message=(
                    f"Architecture class changed ({arch_a.architecture_class} vs {arch_b.architecture_class})."
                ),
                field="architecture_class",
                value_a=arch_a.architecture_class,
                value_b=arch_b.architecture_class,
            )
        )

    # 2. Layer depth check
    layers_a = arch_a.num_hidden_layers
    layers_b = arch_b.num_hidden_layers
    if layers_a is None:
        l_idx_a = [d.layer_index for d in model_a.tensors.values() if d.layer_index is not None]
        layers_a = (max(l_idx_a) + 1) if l_idx_a else None
    if layers_b is None:
        l_idx_b = [d.layer_index for d in model_b.tensors.values() if d.layer_index is not None]
        layers_b = (max(l_idx_b) + 1) if l_idx_b else None

    if layers_a is not None and layers_b is not None and layers_a != layers_b:
        issues.append(
            CompatibilityIssue(
                code="LAYER_COUNT_MISMATCH",
                severity=Severity.HIGH,
                message=f"Hidden layer depth differs ({layers_a} vs {layers_b} layers).",
                field="num_hidden_layers",
                value_a=layers_a,
                value_b=layers_b,
            )
        )

    # 3. Hidden size & head configuration
    for field_name, label, sev in (
        ("hidden_size", "Hidden dimension", Severity.HIGH),
        ("intermediate_size", "MLP intermediate size", Severity.MEDIUM),
        ("num_attention_heads", "Query attention head count", Severity.MEDIUM),
        ("num_key_value_heads", "Key/Value head count (GQA)", Severity.MEDIUM),
    ):
        va = getattr(arch_a, field_name)
        vb = getattr(arch_b, field_name)
        if va is not None and vb is not None and va != vb:
            issues.append(
                CompatibilityIssue(
                    code=f"{field_name.upper()}_MISMATCH",
                    severity=sev,
                    message=f"{label} differs ({va} vs {vb}).",
                    field=field_name,
                    value_a=va,
                    value_b=vb,
                )
            )

    # 4. Vocabulary & dtype checks
    if arch_a.vocab_size is not None and arch_b.vocab_size is not None and arch_a.vocab_size != arch_b.vocab_size:
        issues.append(
            CompatibilityIssue(
                code="VOCAB_SIZE_MISMATCH",
                severity=Severity.MEDIUM,
                message=f"Vocabulary size changed ({arch_a.vocab_size} vs {arch_b.vocab_size}); embedding/lm_head shapes may differ.",
                field="vocab_size",
                value_a=arch_a.vocab_size,
                value_b=arch_b.vocab_size,
            )
        )

    if model_a.metadata.dtypes_present != model_b.metadata.dtypes_present:
        issues.append(
            CompatibilityIssue(
                code="DTYPE_DIFFERENCE",
                severity=Severity.LOW,
                message=(
                    f"Checkpoint numerical dtypes differ ({model_a.metadata.dtypes_present} vs "
                    f"{model_b.metadata.dtypes_present}); tensors will be promoted to float64 during diffing."
                ),
                field="dtypes_present",
                value_a=model_a.metadata.dtypes_present,
                value_b=model_b.metadata.dtypes_present,
            )
        )

    # 5. Tensor overlap checks
    shape_mismatches = len(shared_keys) - len(shape_matched)
    if shape_mismatches > 0:
        issues.append(
            CompatibilityIssue(
                code="TENSOR_SHAPE_MISMATCHES",
                severity=Severity.MEDIUM,
                message=f"{shape_mismatches} shared tensor(s) changed shape.",
                field="tensor_shapes",
                value_a=len(shared_keys),
                value_b=len(shape_matched),
            )
        )

    if len(shape_matched) == 0:
        issues.append(
            CompatibilityIssue(
                code="NO_ALIGNED_TENSORS",
                severity=Severity.CRITICAL,
                message="Zero shared tensors have matching shapes; elementwise weight diffing cannot compare pairs.",
                field="shared_tensors",
                value_a=len(keys_a),
                value_b=len(keys_b),
            )
        )

    direct_tensor_ok = len(shape_matched) > 0
    head_neuron_ok = direct_tensor_ok and (
        arch_a.model_type == arch_b.model_type or jaccard >= 0.5
    )

    has_hf_config_a = bool(arch_a.model_type or arch_a.architecture_class)
    has_hf_config_b = bool(arch_b.model_type or arch_b.architecture_class)
    activation_ok = has_hf_config_a and has_hf_config_b
    behavioral_ok = has_hf_config_a and has_hf_config_b

    if direct_tensor_ok and shape_mismatches == 0 and jaccard == 1.0:
        summary = (
            f"Fully aligned checkpoints ({len(shape_matched)}/{len(union_keys)} tensors match in name and shape)."
        )
    elif direct_tensor_ok:
        summary = (
            f"Partially aligned checkpoints ({len(shape_matched)}/{len(union_keys)} shape-compatible tensors; "
            f"Jaccard overlap={jaccard:.2%})."
        )
    else:
        summary = (
            "Incompatible parameter spaces (0 shape-matched shared tensors); structural and behavioral comparisons remain supported."
        )

    return CompatibilityReport(
        compatible_for_direct_tensor_diff=direct_tensor_ok,
        compatible_for_structural_diff=True,
        compatible_for_head_neuron_analysis=head_neuron_ok,
        compatible_for_activation_diff=activation_ok,
        compatible_for_behavioral_diff=behavioral_ok,
        shared_tensor_count=len(shared_keys),
        shape_matched_tensor_count=len(shape_matched),
        jaccard_key_overlap=round(jaccard, 6),
        issues=issues,
        summary=summary,
    )
