"""Weight-tying verification between input token embeddings and output LM head."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.loaders.base import LoadedModel
from core.tensor_engine.metrics import DiffTolerances, compute_chunked_tensor_metrics
from core.types import CanonicalRole, EpistemicStatus


@dataclass(frozen=True)
class WeightTyingStatus:
    """Observed weight-tying relationship between embedding and LM-head tensors."""

    has_separate_lm_head_key: bool
    config_tie_word_embeddings: bool | None
    numerically_tied: bool
    embedding_key: str | None
    lm_head_key: str | None
    cosine_between_embed_and_head: float | None
    max_abs_diff_between_embed_and_head: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_weight_tying(model: LoadedModel, *, atol: float = 1e-6) -> WeightTyingStatus:
    """Check whether `embed_tokens` and `lm_head` share weights implicitly or numerically."""
    emb_keys = [
        k for k, d in model.tensors.items() if d.canonical_role == CanonicalRole.EMBEDDING and len(d.shape) == 2
    ]
    head_keys = [
        k for k, d in model.tensors.items() if d.canonical_role == CanonicalRole.LM_HEAD and len(d.shape) == 2
    ]
    cfg_tie = model.metadata.architecture.tie_word_embeddings

    emb_k = emb_keys[0] if emb_keys else None
    head_k = head_keys[0] if head_keys else None

    if emb_k is not None and head_k is None:
        return WeightTyingStatus(
            has_separate_lm_head_key=False,
            config_tie_word_embeddings=cfg_tie,
            numerically_tied=bool(cfg_tie is not False),
            embedding_key=emb_k,
            lm_head_key=None,
            cosine_between_embed_and_head=1.0 if cfg_tie is not False else None,
            max_abs_diff_between_embed_and_head=0.0 if cfg_tie is not False else None,
        )

    if emb_k is None or head_k is None:
        return WeightTyingStatus(
            has_separate_lm_head_key=head_k is not None,
            config_tie_word_embeddings=cfg_tie,
            numerically_tied=False,
            embedding_key=emb_k,
            lm_head_key=head_k,
            cosine_between_embed_and_head=None,
            max_abs_diff_between_embed_and_head=None,
        )

    if model.tensors[emb_k].shape != model.tensors[head_k].shape:
        return WeightTyingStatus(
            has_separate_lm_head_key=True,
            config_tie_word_embeddings=cfg_tie,
            numerically_tied=False,
            embedding_key=emb_k,
            lm_head_key=head_k,
            cosine_between_embed_and_head=None,
            max_abs_diff_between_embed_and_head=None,
        )

    metrics = compute_chunked_tensor_metrics(
        model.iter_tensor_chunks(emb_k),
        model.iter_tensor_chunks(head_k),
        tolerances=DiffTolerances(atol=atol, rtol=0.0),
    )
    return WeightTyingStatus(
        has_separate_lm_head_key=True,
        config_tie_word_embeddings=cfg_tie,
        numerically_tied=metrics.within_tolerance,
        embedding_key=emb_k,
        lm_head_key=head_k,
        cosine_between_embed_and_head=metrics.cosine_similarity,
        max_abs_diff_between_embed_and_head=metrics.max_abs_diff,
    )


def compare_weight_tying(model_a: LoadedModel, model_b: LoadedModel) -> dict[str, Any]:
    """Compare weight-tying state across Model A and Model B (detects accidental untying during SFT)."""
    ta = inspect_weight_tying(model_a)
    tb = inspect_weight_tying(model_b)
    return {
        "epistemic_status": EpistemicStatus.OBSERVED.value,
        "model_a": ta.to_dict(),
        "model_b": tb.to_dict(),
        "tying_changed": ta.numerically_tied != tb.numerically_tied,
    }
