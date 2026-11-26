"""Neuron, Attention Head, Embedding, and Normalization Analysis (Module D)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from core.loaders.base import LoadedModel
from core.resource_manager import CancellationToken
from core.types import CanonicalRole, EpistemicStatus

POLYSEMANTICITY_CAUTION = (
    "Basis coordinate indices in MLP, attention, and embedding matrices are commonly polysemantic "
    "under superposition. Observed norm or cosine shifts at index k do NOT establish a monosemantic "
    "feature or causal behavioral mechanism without separate sparse decomposition and causal intervention."
)


@dataclass(frozen=True)
class HeadSliceDiff:
    """Observed weight shift for a single attention head sub-projection."""

    head_index: int
    cosine_similarity: float
    relative_l2: float
    frobenius_norm_a: float
    frobenius_norm_b: float
    max_abs_diff: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AttentionProjectionAnalysis:
    """Head-decomposed analysis of an attention projection matrix (`q`, `k`, `v`, `o`)."""

    tensor_name: str
    layer_index: int | None
    role: str
    num_heads: int
    head_dim: int
    mean_head_cosine: float
    min_head_cosine: float
    max_head_relative_l2: float
    most_diverged_heads: list[HeadSliceDiff]
    all_heads: list[HeadSliceDiff]

    def to_dict(self, *, include_all_heads: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tensor_name": self.tensor_name,
            "layer_index": self.layer_index,
            "role": self.role,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "mean_head_cosine": self.mean_head_cosine,
            "min_head_cosine": self.min_head_cosine,
            "max_head_relative_l2": self.max_head_relative_l2,
            "most_diverged_heads": [h.to_dict() for h in self.most_diverged_heads],
        }
        if include_all_heads:
            payload["all_heads"] = [h.to_dict() for h in self.all_heads]
        return payload


@dataclass
class MLPBasisAnalysis:
    """Basis-row/column shift analysis for an MLP projection matrix."""

    tensor_name: str
    layer_index: int | None
    role: str
    basis_axis: int
    basis_dimension: int
    mean_basis_cosine: float
    p95_relative_l2: float
    max_relative_l2: float
    top_shifted_indices: list[dict[str, float | int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EmbeddingProjectionAnalysis:
    """Per-token vector shift summary for embedding or LM-head matrices."""

    tensor_name: str
    role: str
    vocab_size: int
    hidden_size: int
    mean_token_cosine: float
    min_token_cosine: float
    changed_token_rows: int
    changed_token_pct: float
    top_shifted_token_ids: list[dict[str, float | int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NormLayerAnalysis:
    """Scale/bias parameter shift for RMSNorm or LayerNorm vectors."""

    tensor_name: str
    layer_index: int | None
    dimension: int
    mean_a: float
    mean_b: float
    mean_scale_shift: float
    max_abs_diff: float
    relative_l2: float
    cosine_similarity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NeuronAttentionReport:
    """Complete Module D report covering attention heads, MLP coordinates, embeddings, and norms."""

    epistemic_status: EpistemicStatus
    epistemic_caution: str
    attention_projections: list[AttentionProjectionAnalysis]
    mlp_projections: list[MLPBasisAnalysis]
    embeddings_and_heads: list[EmbeddingProjectionAnalysis]
    normalization_layers: list[NormLayerAnalysis]
    skipped_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "epistemic_status": self.epistemic_status.value,
            "epistemic_caution": self.epistemic_caution,
            "counts": {
                "attention_projections": len(self.attention_projections),
                "mlp_projections": len(self.mlp_projections),
                "embeddings_and_heads": len(self.embeddings_and_heads),
                "normalization_layers": len(self.normalization_layers),
            },
            "attention_projections": [a.to_dict() for a in self.attention_projections],
            "mlp_projections": [m.to_dict() for m in self.mlp_projections],
            "embeddings_and_heads": [e.to_dict() for e in self.embeddings_and_heads],
            "normalization_layers": [n.to_dict() for n in self.normalization_layers],
            "skipped_notes": self.skipped_notes,
        }


def _vector_batch_metrics(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute vectorized row-wise (cosine, rel_l2, norm_a, norm_b, max_abs) for 2D matrices `[N, D]`."""
    a = mat_a.to(torch.float32)
    b = mat_b.to(torch.float32)
    diff = b - a
    norm_a = torch.linalg.vector_norm(a, ord=2, dim=1)
    norm_b = torch.linalg.vector_norm(b, ord=2, dim=1)
    norm_diff = torch.linalg.vector_norm(diff, ord=2, dim=1)
    dot = (a * b).sum(dim=1)

    denom = (norm_a * norm_b).clamp_min(eps)
    cos = (dot / denom).clamp(-1.0, 1.0)
    both_zero = (norm_a == 0.0) & (norm_b == 0.0)
    one_zero = (norm_a == 0.0) ^ (norm_b == 0.0)
    cos = torch.where(both_zero, torch.ones_like(cos), cos)
    cos = torch.where(one_zero, torch.zeros_like(cos), cos)

    rel_l2 = torch.where(
        norm_diff == 0.0,
        torch.zeros_like(norm_diff),
        norm_diff / norm_a.clamp_min(eps),
    )
    max_abs = diff.abs().max(dim=1).values
    return cos, rel_l2, norm_a, norm_b, max_abs


class NeuronAttentionAnalyzer:
    """Analyzes head-sliced attention weights, MLP basis coordinates, embeddings, and norms."""

    def __init__(self, *, vocab_batch_rows: int = 8192, top_k: int = 10) -> None:
        self.vocab_batch_rows = max(256, int(vocab_batch_rows))
        self.top_k = max(1, int(top_k))

    def analyze(
        self,
        model_a: LoadedModel,
        model_b: LoadedModel,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> NeuronAttentionReport:
        arch_a = model_a.metadata.architecture
        arch_b = model_b.metadata.architecture
        q_heads = (
            arch_a.num_attention_heads
            if arch_a.num_attention_heads and arch_a.num_attention_heads == arch_b.num_attention_heads
            else (arch_a.num_attention_heads or arch_b.num_attention_heads)
        )
        kv_heads = (
            arch_a.num_key_value_heads
            if arch_a.num_key_value_heads and arch_a.num_key_value_heads == arch_b.num_key_value_heads
            else (arch_a.num_key_value_heads or arch_b.num_key_value_heads or q_heads)
        )

        attn_results: list[AttentionProjectionAnalysis] = []
        mlp_results: list[MLPBasisAnalysis] = []
        emb_results: list[EmbeddingProjectionAnalysis] = []
        norm_results: list[NormLayerAnalysis] = []
        skipped: list[str] = []

        descs_a = model_a.tensors
        descs_b = model_b.tensors

        for name, da in descs_a.items():
            if cancellation_token is not None:
                cancellation_token.check()
            db = descs_b.get(name)
            if db is None or da.shape != db.shape:
                continue

            role = da.canonical_role
            # 1. Normalization 1D vectors
            if role == CanonicalRole.NORM and len(da.shape) == 1 and da.num_elements > 0:
                ta = model_a.get_tensor(name).to(torch.float64)
                tb = model_b.get_tensor(name).to(torch.float64)
                if not (torch.isfinite(ta).all() and torch.isfinite(tb).all()):
                    skipped.append(f"Skipped norm '{name}' due to non-finite values.")
                    continue
                diff = tb - ta
                na = float(torch.linalg.vector_norm(ta).item())
                nb = float(torch.linalg.vector_norm(tb).item())
                nd = float(torch.linalg.vector_norm(diff).item())
                cos = 1.0 if (na == 0.0 and nb == 0.0) else (
                    0.0 if (na == 0.0 or nb == 0.0) else max(-1.0, min(1.0, float(torch.dot(ta, tb).item()) / (na * nb)))
                )
                rel_l2 = 0.0 if nd == 0.0 else nd / max(na, 1e-12)
                norm_results.append(
                    NormLayerAnalysis(
                        tensor_name=name,
                        layer_index=da.layer_index,
                        dimension=da.shape[0],
                        mean_a=round(float(ta.mean().item()), 8),
                        mean_b=round(float(tb.mean().item()), 8),
                        mean_scale_shift=round(float(diff.mean().item()), 8),
                        max_abs_diff=round(float(diff.abs().max().item()), 8),
                        relative_l2=round(rel_l2, 8),
                        cosine_similarity=round(cos, 8),
                    )
                )
                continue

            # Remaining analyses operate on 2D matrices
            if len(da.shape) != 2 or da.num_elements == 0:
                continue

            # 2. Attention Q/K/V/O projections
            if role in (
                CanonicalRole.ATTN_Q,
                CanonicalRole.ATTN_K,
                CanonicalRole.ATTN_V,
                CanonicalRole.ATTN_O,
            ):
                heads = q_heads if role in (CanonicalRole.ATTN_Q, CanonicalRole.ATTN_O) else kv_heads
                if not heads or heads <= 0:
                    skipped.append(f"Skipped head slicing for '{name}': head count unavailable in config.")
                    continue
                split_dim = 1 if role == CanonicalRole.ATTN_O else 0
                if da.shape[split_dim] % heads != 0:
                    skipped.append(
                        f"Skipped head slicing for '{name}': shape {list(da.shape)} axis {split_dim} not divisible by {heads} heads."
                    )
                    continue
                head_dim = da.shape[split_dim] // heads
                ta = model_a.get_tensor(name)
                tb = model_b.get_tensor(name)
                if split_dim == 0:
                    # [num_heads, head_dim * in_dim]
                    ha = ta.reshape(heads, -1)
                    hb = tb.reshape(heads, -1)
                else:
                    # [out_dim, num_heads, head_dim] -> [num_heads, out_dim * head_dim]
                    ha = ta.reshape(da.shape[0], heads, head_dim).permute(1, 0, 2).reshape(heads, -1)
                    hb = tb.reshape(da.shape[0], heads, head_dim).permute(1, 0, 2).reshape(heads, -1)

                cos_v, rel_l2_v, norm_a_v, norm_b_v, max_abs_v = _vector_batch_metrics(ha, hb)
                head_slices = [
                    HeadSliceDiff(
                        head_index=h_idx,
                        cosine_similarity=round(float(cos_v[h_idx].item()), 8),
                        relative_l2=round(float(rel_l2_v[h_idx].item()), 8),
                        frobenius_norm_a=round(float(norm_a_v[h_idx].item()), 6),
                        frobenius_norm_b=round(float(norm_b_v[h_idx].item()), 6),
                        max_abs_diff=round(float(max_abs_v[h_idx].item()), 8),
                    )
                    for h_idx in range(heads)
                ]
                ranked_heads = sorted(head_slices, key=lambda h: h.relative_l2, reverse=True)
                attn_results.append(
                    AttentionProjectionAnalysis(
                        tensor_name=name,
                        layer_index=da.layer_index,
                        role=role.value,
                        num_heads=heads,
                        head_dim=head_dim,
                        mean_head_cosine=round(float(cos_v.mean().item()), 8),
                        min_head_cosine=round(float(cos_v.min().item()), 8),
                        max_head_relative_l2=round(float(rel_l2_v.max().item()), 8),
                        most_diverged_heads=ranked_heads[: min(5, heads)],
                        all_heads=head_slices,
                    )
                )
                continue

            # 3. MLP Projections
            if role in (
                CanonicalRole.MLP_GATE,
                CanonicalRole.MLP_UP,
                CanonicalRole.MLP_DOWN,
                CanonicalRole.MLP_FC,
            ):
                ta = model_a.get_tensor(name)
                tb = model_b.get_tensor(name)
                # For down_proj [hidden, intermediate], intermediate basis is axis 1; otherwise axis 0
                basis_axis = 1 if role == CanonicalRole.MLP_DOWN else 0
                mat_a = ta.transpose(0, 1).contiguous() if basis_axis == 1 else ta
                mat_b = tb.transpose(0, 1).contiguous() if basis_axis == 1 else tb

                m_cos, m_rel_l2, m_norm_a, m_norm_b, _max_abs = _vector_batch_metrics(mat_a, mat_b)
                basis_dim = int(mat_a.shape[0])
                p95_rel = float(torch.quantile(m_rel_l2, 0.95).item()) if basis_dim > 0 else 0.0
                k = min(self.top_k, basis_dim)
                top_vals, top_idxs = torch.topk(m_rel_l2, k=k)
                top_list = [
                    {
                        "basis_index": int(top_idxs[i].item()),
                        "relative_l2": round(float(top_vals[i].item()), 8),
                        "cosine_similarity": round(float(m_cos[top_idxs[i]].item()), 8),
                        "norm_a": round(float(m_norm_a[top_idxs[i]].item()), 6),
                        "norm_b": round(float(m_norm_b[top_idxs[i]].item()), 6),
                    }
                    for i in range(k)
                ]
                mlp_results.append(
                    MLPBasisAnalysis(
                        tensor_name=name,
                        layer_index=da.layer_index,
                        role=role.value,
                        basis_axis=basis_axis,
                        basis_dimension=basis_dim,
                        mean_basis_cosine=round(float(m_cos.mean().item()), 8),
                        p95_relative_l2=round(p95_rel, 8),
                        max_relative_l2=round(float(m_rel_l2.max().item()), 8),
                        top_shifted_indices=top_list,
                    )
                )
                continue

            # 4. Embeddings & LM Head (processed in row batches to cap RAM)
            if role in (CanonicalRole.EMBEDDING, CanonicalRole.LM_HEAD):
                emb_results.append(self._analyze_embedding_matrix(model_a, model_b, name, da.shape, role.value))

        return NeuronAttentionReport(
            epistemic_status=EpistemicStatus.OBSERVED,
            epistemic_caution=POLYSEMANTICITY_CAUTION,
            attention_projections=attn_results,
            mlp_projections=mlp_results,
            embeddings_and_heads=emb_results,
            normalization_layers=norm_results,
            skipped_notes=skipped,
        )

    def _analyze_embedding_matrix(
        self,
        model_a: LoadedModel,
        model_b: LoadedModel,
        name: str,
        shape: tuple[int, ...],
        role_str: str,
    ) -> EmbeddingProjectionAnalysis:
        vocab_size, hidden_size = shape[0], shape[1]
        ta = model_a.get_tensor(name)
        tb = model_b.get_tensor(name)

        sum_cos = 0.0
        min_cos = 1.0
        changed_rows = 0
        top_heap: list[tuple[float, int, float]] = []

        for start in range(0, vocab_size, self.vocab_batch_rows):
            end = min(vocab_size, start + self.vocab_batch_rows)
            ba = ta[start:end]
            bb = tb[start:end]
            cos, rel_l2, _na, _nb, max_abs = _vector_batch_metrics(ba, bb)
            sum_cos += float(cos.sum().item())
            batch_min = float(cos.min().item())
            if batch_min < min_cos:
                min_cos = batch_min
            changed_rows += int((max_abs > 1e-6).sum().item())

            k = min(self.top_k, end - start)
            vals, idxs = torch.topk(rel_l2, k=k)
            for i in range(k):
                local_idx = int(idxs[i].item())
                top_heap.append(
                    (
                        float(vals[i].item()),
                        start + local_idx,
                        float(cos[local_idx].item()),
                    )
                )

        top_heap.sort(key=lambda item: item[0], reverse=True)
        top_tokens = [
            {
                "token_id": tok_id,
                "relative_l2": round(rl2, 8),
                "cosine_similarity": round(cval, 8),
            }
            for rl2, tok_id, cval in top_heap[: self.top_k]
            if rl2 > 0.0
        ]

        return EmbeddingProjectionAnalysis(
            tensor_name=name,
            role=role_str,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            mean_token_cosine=round(sum_cos / vocab_size, 8) if vocab_size > 0 else 1.0,
            min_token_cosine=round(min_cos, 8),
            changed_token_rows=changed_rows,
            changed_token_pct=round((changed_rows / vocab_size) * 100.0, 6) if vocab_size > 0 else 0.0,
            top_shifted_token_ids=top_tokens,
        )
