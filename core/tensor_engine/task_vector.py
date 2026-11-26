"""Task-vector SVD singular spectrum and Shannon effective-rank analyzer (LoRA vs Full SFT)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TaskVectorSpectrum:
    """Spectral summary of a 2D parameter update `delta = W_B - W_A`."""

    shape: list[int]
    max_possible_rank: int
    shannon_effective_rank: float
    rank_at_90pct_energy: int
    rank_at_99pct_energy: int
    spectral_norm: float
    frobenius_norm: float
    top_singular_values: list[float]
    inferred_update_regime: str  # "zero_update" | "low_rank_adapter" | "full_rank_update"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_task_vector_spectrum(
    weight_a: torch.Tensor,
    weight_b: torch.Tensor,
    *,
    top_k_sv: int = 8,
    low_rank_ratio_threshold: float = 0.25,
) -> TaskVectorSpectrum:
    """Compute singular value spectrum and Shannon effective rank (Roy & Vetterli, 2007) of `W_B - W_A`."""
    if weight_a.ndim != 2 or tuple(weight_a.shape) != tuple(weight_b.shape):
        raise ValueError("`analyze_task_vector_spectrum` requires two matching 2D matrices.")

    delta = (weight_b.detach().cpu().to(torch.float64)) - (weight_a.detach().cpu().to(torch.float64))
    max_rank = min(int(delta.shape[0]), int(delta.shape[1]))
    if max_rank == 0 or not bool(torch.isfinite(delta).all().item()):
        return TaskVectorSpectrum(
            shape=list(delta.shape),
            max_possible_rank=max_rank,
            shannon_effective_rank=0.0,
            rank_at_90pct_energy=0,
            rank_at_99pct_energy=0,
            spectral_norm=0.0,
            frobenius_norm=0.0,
            top_singular_values=[],
            inferred_update_regime="zero_update",
        )

    sv = torch.linalg.svdvals(delta)
    frob = float(torch.linalg.vector_norm(sv).item())
    if frob <= 1e-12:
        return TaskVectorSpectrum(
            shape=list(delta.shape),
            max_possible_rank=max_rank,
            shannon_effective_rank=0.0,
            rank_at_90pct_energy=0,
            rank_at_99pct_energy=0,
            spectral_norm=0.0,
            frobenius_norm=0.0,
            top_singular_values=[0.0] * min(top_k_sv, max_rank),
            inferred_update_regime="zero_update",
        )

    l1_sum = float(sv.sum().item())
    probs = (sv / l1_sum).clamp_min(1e-18)
    entropy = -float((probs * probs.log()).sum().item())
    eff_rank = math.exp(entropy)

    sq = sv * sv
    cum_energy = torch.cumsum(sq, dim=0) / sq.sum()
    r90 = int(torch.searchsorted(cum_energy, 0.90).item()) + 1
    r99 = int(torch.searchsorted(cum_energy, 0.99).item()) + 1
    r90 = min(r90, max_rank)
    r99 = min(r99, max_rank)

    regime = (
        "low_rank_adapter"
        if (max_rank >= 8 and r99 <= max(2, int(max_rank * low_rank_ratio_threshold)))
        else "full_rank_update"
    )

    return TaskVectorSpectrum(
        shape=list(delta.shape),
        max_possible_rank=max_rank,
        shannon_effective_rank=round(eff_rank, 6),
        rank_at_90pct_energy=r90,
        rank_at_99pct_energy=r99,
        spectral_norm=round(float(sv[0].item()), 8),
        frobenius_norm=round(frob, 8),
        top_singular_values=[round(float(x), 8) for x in sv[:top_k_sv].tolist()],
        inferred_update_regime=regime,
    )
