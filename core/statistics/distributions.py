"""Numerical distribution statistics and distribution-shift metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


def _safe_float(val: float) -> float | None:
    if math.isnan(val) or math.isinf(val):
        return None
    return float(val)


@dataclass(frozen=True)
class TensorDistributionSummary:
    """Summary of a tensor's numerical value distribution."""

    total_elements: int
    finite_elements: int
    nan_count: int
    pos_inf_count: int
    neg_inf_count: int
    zero_count: int
    sparsity_ratio: float
    mean: float | None
    std: float | None
    min_val: float | None
    max_val: float | None
    p05: float | None
    p50: float | None
    p95: float | None
    p99: float | None
    l2_norm: float | None
    sampled_quantiles: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DistributionShiftSummary:
    """Comparison of two numerical tensor distributions (valid even across differing shapes)."""

    mean_delta: float | None
    std_delta: float | None
    std_ratio: float | None
    median_delta: float | None
    iqr_or_p90_span_delta: float | None
    max_quantile_abs_shift: float | None
    sparsity_delta: float
    wasserstein_1d_approx: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_tensor_chunks(
    chunks: Iterable[torch.Tensor],
    *,
    zero_threshold: float = 0.0,
    max_exact_quantile_elements: int = 2_097_152,
    sample_seed: int = 42,
) -> TensorDistributionSummary:
    """Compute exact moments/norms and exact-or-sampled quantiles over flattened tensor chunks."""
    total_elements = 0
    finite_elements = 0
    nan_count = 0
    pos_inf_count = 0
    neg_inf_count = 0
    zero_count = 0

    sum_x = 0.0
    sum_x2 = 0.0
    min_val = math.inf
    max_val = -math.inf

    rng = np.random.default_rng(sample_seed)
    sample_buffers: list[np.ndarray] = []
    collected_finite = 0
    sampled = False

    for raw_chunk in chunks:
        flat = raw_chunk.detach().cpu().reshape(-1)
        n = int(flat.numel())
        if n == 0:
            continue
        total_elements += n

        if flat.is_floating_point() or flat.is_complex():
            f64 = flat.to(torch.float64)
            isnan = torch.isnan(f64)
            isposinf = torch.isposinf(f64)
            isneginf = torch.isneginf(f64)
            nan_c = int(isnan.sum().item())
            pinf_c = int(isposinf.sum().item())
            ninf_c = int(isneginf.sum().item())
            nan_count += nan_c
            pos_inf_count += pinf_c
            neg_inf_count += ninf_c
            finite_vals = f64[torch.isfinite(f64)] if (nan_c or pinf_c or ninf_c) else f64
        else:
            finite_vals = flat.to(torch.float64)

        fn = int(finite_vals.numel())
        if fn == 0:
            continue
        finite_elements += fn

        if zero_threshold <= 0.0:
            zero_count += int((finite_vals == 0.0).sum().item())
        else:
            zero_count += int((finite_vals.abs() <= zero_threshold).sum().item())

        sum_x += float(finite_vals.sum().item())
        sum_x2 += float(torch.dot(finite_vals, finite_vals).item())
        c_min = float(finite_vals.min().item())
        c_max = float(finite_vals.max().item())
        if c_min < min_val:
            min_val = c_min
        if c_max > max_val:
            max_val = c_max

        arr = finite_vals.numpy()
        if not sampled and (collected_finite + fn) <= max_exact_quantile_elements:
            sample_buffers.append(arr.copy())
            collected_finite += fn
        else:
            sampled = True
            # Deterministic uniform stride/random subsample to bound memory
            target_keep = max(1024, min(fn, max_exact_quantile_elements // 8))
            if fn > target_keep:
                idx = rng.choice(fn, size=target_keep, replace=False)
                arr = arr[idx]
            sample_buffers.append(arr.copy())
            collected_finite += len(arr)
            if collected_finite > max_exact_quantile_elements:
                merged = np.concatenate(sample_buffers)
                idx = rng.choice(len(merged), size=max_exact_quantile_elements // 2, replace=False)
                sample_buffers = [merged[idx]]
                collected_finite = len(sample_buffers[0])

    sparsity = (zero_count / total_elements) if total_elements > 0 else 0.0

    if finite_elements == 0:
        return TensorDistributionSummary(
            total_elements=total_elements,
            finite_elements=0,
            nan_count=nan_count,
            pos_inf_count=pos_inf_count,
            neg_inf_count=neg_inf_count,
            zero_count=zero_count,
            sparsity_ratio=round(sparsity, 8),
            mean=None,
            std=None,
            min_val=None,
            max_val=None,
            p05=None,
            p50=None,
            p95=None,
            p99=None,
            l2_norm=None,
            sampled_quantiles=sampled,
        )

    mean = sum_x / finite_elements
    var = max(0.0, (sum_x2 / finite_elements) - (mean * mean))
    std = math.sqrt(var)
    l2_norm = math.sqrt(max(0.0, sum_x2))

    merged_sample = np.concatenate(sample_buffers) if sample_buffers else np.empty(0, dtype=np.float64)
    q05, q50, q95, q99 = np.quantile(merged_sample, [0.05, 0.50, 0.95, 0.99]).tolist()

    return TensorDistributionSummary(
        total_elements=total_elements,
        finite_elements=finite_elements,
        nan_count=nan_count,
        pos_inf_count=pos_inf_count,
        neg_inf_count=neg_inf_count,
        zero_count=zero_count,
        sparsity_ratio=round(sparsity, 8),
        mean=_safe_float(mean),
        std=_safe_float(std),
        min_val=_safe_float(min_val),
        max_val=_safe_float(max_val),
        p05=_safe_float(q05),
        p50=_safe_float(q50),
        p95=_safe_float(q95),
        p99=_safe_float(q99),
        l2_norm=_safe_float(l2_norm),
        sampled_quantiles=sampled,
    )


def summarize_tensor(
    tensor: torch.Tensor,
    *,
    zero_threshold: float = 0.0,
) -> TensorDistributionSummary:
    """Convenience wrapper summarizing an in-memory PyTorch tensor."""
    return summarize_tensor_chunks([tensor], zero_threshold=zero_threshold)


def compute_distribution_shift(
    dist_a: TensorDistributionSummary,
    dist_b: TensorDistributionSummary,
) -> DistributionShiftSummary:
    """Compute interpretable distribution changes between Tensor A and Tensor B."""
    sparsity_delta = dist_b.sparsity_ratio - dist_a.sparsity_ratio
    if dist_a.finite_elements == 0 or dist_b.finite_elements == 0:
        return DistributionShiftSummary(
            mean_delta=None,
            std_delta=None,
            std_ratio=None,
            median_delta=None,
            iqr_or_p90_span_delta=None,
            max_quantile_abs_shift=None,
            sparsity_delta=round(sparsity_delta, 8),
            wasserstein_1d_approx=None,
        )

    assert dist_a.mean is not None and dist_b.mean is not None
    assert dist_a.std is not None and dist_b.std is not None
    assert dist_a.p05 is not None and dist_b.p05 is not None
    assert dist_a.p50 is not None and dist_b.p50 is not None
    assert dist_a.p95 is not None and dist_b.p95 is not None
    assert dist_a.p99 is not None and dist_b.p99 is not None

    mean_delta = dist_b.mean - dist_a.mean
    std_delta = dist_b.std - dist_a.std
    if dist_a.std > 1e-15:
        std_ratio: float | None = dist_b.std / dist_a.std
    elif dist_b.std <= 1e-15:
        std_ratio = 1.0
    else:
        std_ratio = None

    median_delta = dist_b.p50 - dist_a.p50
    span_a = dist_a.p95 - dist_a.p05
    span_b = dist_b.p95 - dist_b.p05
    span_delta = span_b - span_a

    q_diffs = [
        abs(dist_b.p05 - dist_a.p05),
        abs(dist_b.p50 - dist_a.p50),
        abs(dist_b.p95 - dist_a.p95),
        abs(dist_b.p99 - dist_a.p99),
    ]
    max_q_shift = max(q_diffs)
    # Quantile-grid approximation to 1D Wasserstein-1 distance: E_u[|F_B^{-1}(u) - F_A^{-1}(u)|]
    w1_approx = float(sum(q_diffs[:3]) / 3.0)

    return DistributionShiftSummary(
        mean_delta=_safe_float(mean_delta),
        std_delta=_safe_float(std_delta),
        std_ratio=_safe_float(std_ratio) if std_ratio is not None else None,
        median_delta=_safe_float(median_delta),
        iqr_or_p90_span_delta=_safe_float(span_delta),
        max_quantile_abs_shift=_safe_float(max_q_shift),
        sparsity_delta=round(sparsity_delta, 8),
        wasserstein_1d_approx=_safe_float(w1_approx),
    )
