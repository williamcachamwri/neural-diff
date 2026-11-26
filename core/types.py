"""Canonical domain types, enums, and metadata structures for NEURAL DIFF."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class VerificationStatus(StrEnum):
    """Strict engineering verification status labels (Section 1.1)."""

    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY VERIFIED"
    NOT_VERIFIED = "NOT VERIFIED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    NOT_IMPLEMENTED = "NOT IMPLEMENTED"


class EpistemicStatus(StrEnum):
    """Epistemic classification distinguishing direct measurement from inference or causation."""

    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    CORRELATED = "CORRELATED"
    CAUSALLY_VERIFIED = "CAUSALLY VERIFIED"
    NOT_VERIFIED = "NOT VERIFIED"


class Severity(StrEnum):
    """Severity levels for structural incompatibilities and behavioral regressions."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class ModelFormat(StrEnum):
    """Supported checkpoint and directory ingestion formats."""

    SAFETENSORS_SINGLE = "safetensors_single"
    SAFETENSORS_SHARDED = "safetensors_sharded"
    PYTORCH_SAFE = "pytorch_weights_only"
    HF_LOCAL_DIR = "huggingface_local_dir"
    HF_HUB_CACHED = "huggingface_hub_cached"


class CanonicalRole(StrEnum):
    """Architecture-normalized parameter roles without assuming identical naming."""

    EMBEDDING = "embedding"
    ATTN_Q = "attn_q"
    ATTN_K = "attn_k"
    ATTN_V = "attn_v"
    ATTN_O = "attn_o"
    ATTN_QKV_FUSED = "attn_qkv_fused"
    MLP_GATE = "mlp_gate"
    MLP_UP = "mlp_up"
    MLP_DOWN = "mlp_down"
    MLP_FC = "mlp_fc"
    NORM = "norm"
    LM_HEAD = "lm_head"
    OTHER = "other"


@dataclass(frozen=True)
class TensorDescriptor:
    """Metadata extracted from a single checkpoint tensor without loading full values."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    num_elements: int
    byte_size: int
    shard_path: str | None = None
    layer_index: int | None = None
    canonical_role: CanonicalRole = CanonicalRole.OTHER

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        data["canonical_role"] = self.canonical_role.value
        return data


@dataclass
class ArchitectureMetadata:
    """Extracted transformer architecture and vocabulary metadata when available."""

    architecture_class: str | None = None
    model_type: str | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    vocab_size: int | None = None
    max_position_embeddings: int | None = None
    rms_norm_eps: float | None = None
    tie_word_embeddings: bool | None = None
    torch_dtype: str | None = None
    bos_token_id: int | list[int] | None = None
    eos_token_id: int | list[int] | None = None
    tokenizer_class: str | None = None
    tokenizer_vocab_size: int | None = None
    raw_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelMetadata:
    """Complete structural and provenance metadata for a loaded model or checkpoint."""

    model_id: str
    resolved_path: str
    format: ModelFormat
    revision: str | None
    tensor_count: int
    total_parameters: int
    total_bytes: int
    shard_count: int
    dtypes_present: list[str]
    architecture: ArchitectureMetadata
    custom_metadata: dict[str, str] = field(default_factory=dict)
    security_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "resolved_path": self.resolved_path,
            "format": self.format.value,
            "revision": self.revision,
            "tensor_count": self.tensor_count,
            "total_parameters": self.total_parameters,
            "total_bytes": self.total_bytes,
            "shard_count": self.shard_count,
            "dtypes_present": self.dtypes_present,
            "architecture": self.architecture.to_dict(),
            "custom_metadata": self.custom_metadata,
            "security_notes": self.security_notes,
        }


@dataclass(frozen=True)
class EpistemicClaim:
    """Explicitly labeled analytical finding."""

    statement: str
    epistemic_status: EpistemicStatus
    evidence_source: str
    metric_values: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "epistemic_status": self.epistemic_status.value,
            "evidence_source": self.evidence_source,
            "metric_values": self.metric_values,
        }
