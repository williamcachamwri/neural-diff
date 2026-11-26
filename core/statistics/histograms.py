"""Aligned-bin empirical histogram and Total Variation Distance (TVD) calculator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class PairedHistogram:
    """Shared-bin empirical histogram comparing value densities of two tensors."""

    bins: int
    bin_edges: list[float]
    counts_a: list[int]
    counts_b: list[int]
    density_a: list[float]
    density_b: list[float]
    total_variation_distance: float
    jensen_shannon_divergence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_paired_histogram(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    *,
    bins: int = 32,
) -> PairedHistogram:
    """Compute shared-edge histograms, Total Variation Distance, and JS divergence."""
    num_bins = max(4, min(256, int(bins)))
    fa = tensor_a.detach().cpu().reshape(-1).to(torch.float64)
    fb = tensor_b.detach().cpu().reshape(-1).to(torch.float64)
    fa = fa[torch.isfinite(fa)]
    fb = fb[torch.isfinite(fb)]

    if fa.numel() == 0 or fb.numel() == 0:
        edges = np.linspace(-1.0, 1.0, num_bins + 1).tolist()
        zeros_i = [0] * num_bins
        zeros_f = [0.0] * num_bins
        return PairedHistogram(
            bins=num_bins,
            bin_edges=[round(float(x), 8) for x in edges],
            counts_a=zeros_i,
            counts_b=zeros_i,
            density_a=zeros_f,
            density_b=zeros_f,
            total_variation_distance=0.0,
            jensen_shannon_divergence=0.0,
        )

    na = fa.numpy()
    nb = fb.numpy()
    lo = min(float(na.min()), float(nb.min()))
    hi = max(float(na.max()), float(nb.max()))
    if hi <= lo:
        lo -= 0.5
        hi += 0.5

    bin_edges = np.linspace(lo, hi, num_bins + 1)
    ca, _ = np.histogram(na, bins=bin_edges)
    cb, _ = np.histogram(nb, bins=bin_edges)

    pa = ca.astype(np.float64) / max(1, int(ca.sum()))
    pb = cb.astype(np.float64) / max(1, int(cb.sum()))

    tvd = float(0.5 * np.abs(pa - pb).sum())
    eps = 1e-15
    pa_s = np.clip(pa, eps, 1.0)
    pb_s = np.clip(pb, eps, 1.0)
    pa_s /= pa_s.sum()
    pb_s /= pb_s.sum()
    m = 0.5 * (pa_s + pb_s)
    js = float(0.5 * np.sum(pa_s * np.log(pa_s / m)) + 0.5 * np.sum(pb_s * np.log(pb_s / m)))

    return PairedHistogram(
        bins=num_bins,
        bin_edges=[round(float(x), 8) for x in bin_edges.tolist()],
        counts_a=[int(x) for x in ca.tolist()],
        counts_b=[int(x) for x in cb.tolist()],
        density_a=[round(float(x), 8) for x in pa.tolist()],
        density_b=[round(float(x), 8) for x in pb.tolist()],
        total_variation_distance=round(max(0.0, min(1.0, tvd)), 8),
        jensen_shannon_divergence=round(max(0.0, js), 8),
    )
