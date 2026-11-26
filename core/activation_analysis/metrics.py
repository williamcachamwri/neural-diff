"""Activation space comparison metrics including Linear CKA and token-wise drift."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from core.statistics.distributions import (
    DistributionShiftSummary,
    TensorDistributionSummary,
    compute_distribution_shift,
    summarize_tensor,
)


def compute_linear_cka(act_a: torch.Tensor, act_b: torch.Tensor, eps: float = 1e-12) -> float | None:
    """Compute Linear Centered Kernel Alignment (Kornblith et al., 2019) over `[N, D_a]` vs `[N, D_b]`.

    Requires `N >= 2` aligned sample/token rows. Invariant to orthogonal transforms and isotropic scaling,
    and valid even when feature dimensions `D_a != D_b`.
    """
    x = act_a.detach().cpu().to(torch.float64)
    y = act_b.detach().cpu().to(torch.float64)
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    if y.ndim > 2:
        y = y.reshape(-1, y.shape[-1])
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    if y.ndim == 1:
        y = y.unsqueeze(-1)

    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        return None
    if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
        return None

    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    yt_x = y.transpose(0, 1) @ x
    xt_x = x.transpose(0, 1) @ x
    yt_y = y.transpose(0, 1) @ y

    hsic_xy = float((yt_x * yt_x).sum().item())
    norm_x = math.sqrt(max(0.0, float((xt_x * xt_x).sum().item())))
    norm_y = math.sqrt(max(0.0, float((yt_y * yt_y).sum().item())))

    if norm_x <= eps and norm_y <= eps:
        return 1.0
    if norm_x <= eps or norm_y <= eps:
        return 0.0
    cka = hsic_xy / (norm_x * norm_y)
    return round(max(0.0, min(1.0, cka)), 8)


@dataclass(frozen=True)
class ActivationLayerComparison:
    """Comparison metrics for a single captured activation hook point (hidden, attn, or mlp)."""

    layer_name: str
    component_type: str  # "hidden_state" | "attention_output" | "mlp_intermediate" | "logits"
    layer_index: int | None
    shape_a: list[int]
    shape_b: list[int]
    shapes_aligned: bool
    token_length_aligned: bool
    cosine_similarity: float | None
    relative_l2: float | None
    max_abs_diff: float | None
    linear_cka: float | None
    per_token_cosine: list[float] | None
    per_token_l2: list[float] | None
    most_diverged_token_index: int | None
    distribution_a: TensorDistributionSummary
    distribution_b: TensorDistributionSummary
    distribution_shift: DistributionShiftSummary

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["distribution_a"] = self.distribution_a.to_dict()
        data["distribution_b"] = self.distribution_b.to_dict()
        data["distribution_shift"] = self.distribution_shift.to_dict()
        return data


def compare_activation_tensors(
    *,
    layer_name: str,
    component_type: str,
    layer_index: int | None,
    act_a: torch.Tensor,
    act_b: torch.Tensor,
    eps: float = 1e-12,
) -> ActivationLayerComparison:
    """Compare two activation tensors `[B, T, D]` or `[T, D]` across direct and representation metrics."""
    a = act_a.detach().cpu().to(torch.float32)
    b = act_b.detach().cpu().to(torch.float32)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a.squeeze(0)
    if b.ndim == 3 and b.shape[0] == 1:
        b = b.squeeze(0)

    shape_a = list(a.shape)
    shape_b = list(b.shape)
    shapes_aligned = shape_a == shape_b
    token_aligned = len(shape_a) >= 2 and len(shape_b) >= 2 and shape_a[0] == shape_b[0]

    dist_a = summarize_tensor(a)
    dist_b = summarize_tensor(b)
    shift = compute_distribution_shift(dist_a, dist_b)

    cka = compute_linear_cka(a, b, eps=eps) if token_aligned else None

    if not shapes_aligned or a.numel() == 0:
        return ActivationLayerComparison(
            layer_name=layer_name,
            component_type=component_type,
            layer_index=layer_index,
            shape_a=shape_a,
            shape_b=shape_b,
            shapes_aligned=False,
            token_length_aligned=token_aligned,
            cosine_similarity=None,
            relative_l2=None,
            max_abs_diff=None,
            linear_cka=cka,
            per_token_cosine=None,
            per_token_l2=None,
            most_diverged_token_index=None,
            distribution_a=dist_a,
            distribution_b=dist_b,
            distribution_shift=shift,
        )

    a64 = a.reshape(-1).to(torch.float64)
    b64 = b.reshape(-1).to(torch.float64)
    if not (torch.isfinite(a64).all() and torch.isfinite(b64).all()):
        return ActivationLayerComparison(
            layer_name=layer_name,
            component_type=component_type,
            layer_index=layer_index,
            shape_a=shape_a,
            shape_b=shape_b,
            shapes_aligned=True,
            token_length_aligned=token_aligned,
            cosine_similarity=None,
            relative_l2=None,
            max_abs_diff=None,
            linear_cka=None,
            per_token_cosine=None,
            per_token_l2=None,
            most_diverged_token_index=None,
            distribution_a=dist_a,
            distribution_b=dist_b,
            distribution_shift=shift,
        )

    diff = b64 - a64
    norm_a = float(torch.linalg.vector_norm(a64).item())
    norm_b = float(torch.linalg.vector_norm(b64).item())
    norm_diff = float(torch.linalg.vector_norm(diff).item())
    max_abs = float(diff.abs().max().item())

    if norm_a <= eps and norm_b <= eps:
        cosine = 1.0
    elif norm_a <= eps or norm_b <= eps:
        cosine = 0.0
    else:
        cosine = max(-1.0, min(1.0, float(torch.dot(a64, b64).item()) / (norm_a * norm_b)))

    rel_l2 = 0.0 if norm_diff == 0.0 else norm_diff / max(norm_a, eps)

    per_tok_cos: list[float] | None = None
    per_tok_l2: list[float] | None = None
    max_tok_idx: int | None = None

    if a.ndim == 2:
        t_a = a.to(torch.float64)
        t_b = b.to(torch.float64)
        t_diff = t_b - t_a
        t_na = torch.linalg.vector_norm(t_a, dim=1)
        t_nb = torch.linalg.vector_norm(t_b, dim=1)
        t_nd = torch.linalg.vector_norm(t_diff, dim=1)
        t_dot = (t_a * t_b).sum(dim=1)
        denom = (t_na * t_nb).clamp_min(eps)
        t_cos = (t_dot / denom).clamp(-1.0, 1.0)
        both_z = (t_na <= eps) & (t_nb <= eps)
        one_z = (t_na <= eps) ^ (t_nb <= eps)
        t_cos = torch.where(both_z, torch.ones_like(t_cos), t_cos)
        t_cos = torch.where(one_z, torch.zeros_like(t_cos), t_cos)
        per_tok_cos = [round(float(v), 6) for v in t_cos.tolist()]
        per_tok_l2 = [round(float(v), 6) for v in t_nd.tolist()]
        if len(per_tok_l2) > 0:
            max_tok_idx = int(torch.argmax(t_nd).item())

    return ActivationLayerComparison(
        layer_name=layer_name,
        component_type=component_type,
        layer_index=layer_index,
        shape_a=shape_a,
        shape_b=shape_b,
        shapes_aligned=True,
        token_length_aligned=token_aligned,
        cosine_similarity=round(cosine, 8),
        relative_l2=round(rel_l2, 8),
        max_abs_diff=round(max_abs, 8),
        linear_cka=cka,
        per_token_cosine=per_tok_cos,
        per_token_l2=per_tok_l2,
        most_diverged_token_index=max_tok_idx,
        distribution_a=dist_a,
        distribution_b=dist_b,
        distribution_shift=shift,
    )
