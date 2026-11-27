"""Domain types and enums for Neural Evolution Tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class TransformationType(StrEnum):
    """Supported post-training, compression, and merging transformation types (Section 1)."""

    BASE_MODEL = "Base Model"
    FINE_TUNING = "Fine-Tuning"
    INSTRUCTION_TUNING = "Instruction Tuning"
    SFT = "SFT"
    DPO = "DPO"
    RLHF = "RLHF"
    LORA = "LoRA"
    QLORA = "QLoRA"
    MODEL_MERGING = "Model Merging"
    QUANTIZATION = "Quantization"
    PRUNING = "Pruning"
    DISTILLATION = "Distillation"
    WEIGHT_MODIFICATION = "Weight Modification"
    UNKNOWN_TRANSFORMATION = "Unknown Transformation"


_TRANSFORMATION_ALIASES: dict[str, TransformationType] = {
    "base": TransformationType.BASE_MODEL,
    "base_model": TransformationType.BASE_MODEL,
    "base model": TransformationType.BASE_MODEL,
    "fine_tuning": TransformationType.FINE_TUNING,
    "fine-tuning": TransformationType.FINE_TUNING,
    "finetuning": TransformationType.FINE_TUNING,
    "instruction_tuning": TransformationType.INSTRUCTION_TUNING,
    "instruction tuning": TransformationType.INSTRUCTION_TUNING,
    "sft": TransformationType.SFT,
    "supervised_fine_tuning": TransformationType.SFT,
    "supervised fine-tuning": TransformationType.SFT,
    "dpo": TransformationType.DPO,
    "rlhf": TransformationType.RLHF,
    "ppo": TransformationType.RLHF,
    "grpo": TransformationType.RLHF,
    "lora": TransformationType.LORA,
    "qlora": TransformationType.QLORA,
    "model_merging": TransformationType.MODEL_MERGING,
    "model merging": TransformationType.MODEL_MERGING,
    "merge": TransformationType.MODEL_MERGING,
    "quantization": TransformationType.QUANTIZATION,
    "quantized": TransformationType.QUANTIZATION,
    "pruning": TransformationType.PRUNING,
    "pruned": TransformationType.PRUNING,
    "distillation": TransformationType.DISTILLATION,
    "distilled": TransformationType.DISTILLATION,
    "weight_modification": TransformationType.WEIGHT_MODIFICATION,
    "weight modification": TransformationType.WEIGHT_MODIFICATION,
    "unknown": TransformationType.UNKNOWN_TRANSFORMATION,
    "unknown_transformation": TransformationType.UNKNOWN_TRANSFORMATION,
    "unknown transformation": TransformationType.UNKNOWN_TRANSFORMATION,
}


def normalize_transformation_type(raw: str | TransformationType | None) -> TransformationType:
    """Resolve user-supplied strings or enum instances to canonical `TransformationType`."""
    if isinstance(raw, TransformationType):
        return raw
    if raw is None:
        return TransformationType.UNKNOWN_TRANSFORMATION
    cleaned = str(raw).strip()
    for member in TransformationType:
        if cleaned == member.value or cleaned.upper() == member.name:
            return member
    alias = _TRANSFORMATION_ALIASES.get(cleaned.lower())
    if alias is not None:
        return alias
    return TransformationType.UNKNOWN_TRANSFORMATION


class CheckpointValidationStatus(StrEnum):
    """Integrity and structural status of a registered checkpoint."""

    VERIFIED = "VERIFIED"
    ADAPTER_ONLY = "ADAPTER_ONLY"
    DUPLICATE_CONTENT = "DUPLICATE_CONTENT"
    CORRUPTED_OR_MISSING = "CORRUPTED_OR_MISSING"


@dataclass
class EvolutionProject:
    """Container organizing a DAG of evolving model checkpoints."""

    project_id: str
    name: str
    description: str
    base_checkpoint_id: str | None
    created_at_utc: str
    updated_at_utc: str
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RegisteredCheckpoint:
    """Complete verified provenance and structural record for a single checkpoint (Section 2)."""

    checkpoint_id: str
    project_id: str
    version_label: str
    model_name: str
    source_path: str
    resolved_path: str
    architecture: str | None
    model_family: str | None
    parameter_count: int
    layer_count: int | None
    hidden_size: int | None
    intermediate_size: int | None
    num_attention_heads: int | None
    num_key_value_heads: int | None
    vocab_size: int | None
    tensor_count: int
    checkpoint_format: str
    dtypes_present: list[str]
    file_size_bytes: int
    sha256_fingerprint: str
    tensor_metadata_hash: str
    parent_checkpoint_ids: list[str]
    transformation_type: TransformationType
    transformation_metadata: dict[str, Any]
    created_at_utc: str
    framework_info: dict[str, str]
    tokenizer_metadata: dict[str, Any]
    quantization_metadata: dict[str, Any]
    training_metadata: dict[str, Any]
    lora_metadata: dict[str, Any]
    experiment_notes: str
    validation_status: CheckpointValidationStatus
    duplicate_of_checkpoint_id: str | None = None
    security_notes: list[str] = field(default_factory=list)
    component_fingerprints: dict[str, str | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["transformation_type"] = self.transformation_type.value
        data["validation_status"] = self.validation_status.value
        return data


@dataclass
class CausalInterventionProvenance:
    """Strictly validated case-and-site counterfactual provenance record (Audit Item 1).

    Missing evidence fields default strictly to `False` / `None` and NEVER elevate an event
    to `CAUSALLY_VERIFIED`.
    """

    case_targeted: bool = False
    target_vs_foil_verified: bool = False
    negative_control_verified: bool = False
    verification_scope: str | None = None
    prompt_fingerprint: str | None = None
    failed_case_id: str | None = None
    layer_index: int | None = None
    intervention_type: str | None = None
    target_token: str | int | None = None
    foil_token: str | int | None = None
    negative_control_result: dict[str, Any] | float | None = None
    recovery_measurement: float | None = None
    clean_logit_diff: float | None = None
    corrupted_logit_diff: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvolutionRelationship:
    """Directed edge `parent -> child` in an evolution DAG (supports multi-parent merges)."""

    relationship_id: str
    project_id: str
    parent_checkpoint_id: str
    child_checkpoint_id: str
    transformation_type: TransformationType
    merge_weight: float | None = None
    transformation_metadata: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["transformation_type"] = self.transformation_type.value
        return data
