"""Exact & chunked mathematical tensor comparison metrics (Module B)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import Any

import torch

from core.resource_manager import CancellationToken


def _json_safe_float(val: float | None) -> float | None:
    if val is None or math.isnan(val) or math.isinf(val):
        return None
    return float(val)


@dataclass(frozen=True)
class DiffTolerances:
    """Configurable numerical tolerances for tensor comparison."""

    atol: float = 1e-6
    rtol: float = 1e-5
    zero_threshold: float = 0.0
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.atol < 0 or self.rtol < 0 or self.zero_threshold < 0 or self.eps <= 0:
            raise ValueError("Tolerances must be non-negative and `eps` must be strictly positive.")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class TensorDiffMetrics:
    """Complete numerical comparison measurements between two aligned tensors."""

    num_elements: int
    finite_pair_elements: int
    non_finite_pair_elements: int
    nan_count_a: int
    nan_count_b: int
    inf_count_a: int
    inf_count_b: int
    exact_equal: bool
    within_tolerance: bool
    max_abs_diff: float | None
    mean_abs_diff: float | None
    rmse: float | None
    l2_diff: float | None
    l2_norm_a: float | None
    l2_norm_b: float | None
    relative_l2: float | None
    relative_l2_infinite: bool
    cosine_similarity: float | None
    changed_elements: int
    changed_pct: float
    zero_to_nonzero: int
    nonzero_to_zero: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def iter_aligned_chunk_pairs(
    iter_a: Iterable[torch.Tensor],
    iter_b: Iterable[torch.Tensor],
    chunk_elements: int = 1_048_576,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Align two arbitrary 1D tensor chunk iterators into equal-length chunk pairs."""
    step = max(1, int(chunk_elements))
    it_a = iter(iter_a)
    it_b = iter(iter_b)
    buf_a: torch.Tensor | None = None
    buf_b: torch.Tensor | None = None

    while True:
        while (buf_a is None or buf_a.numel() == 0):
            try:
                nxt = next(it_a).detach().cpu().reshape(-1)
                if nxt.numel() > 0:
                    buf_a = nxt
                else:
                    buf_a = None
                    continue
            except StopIteration:
                buf_a = None
                break

        while (buf_b is None or buf_b.numel() == 0):
            try:
                nxt = next(it_b).detach().cpu().reshape(-1)
                if nxt.numel() > 0:
                    buf_b = nxt
                else:
                    buf_b = None
                    continue
            except StopIteration:
                buf_b = None
                break

        if buf_a is None and buf_b is None:
            break
        if buf_a is None or buf_b is None:
            raise ValueError("Aligned chunk streams terminated with mismatched total element counts.")

        take = min(step, int(buf_a.numel()), int(buf_b.numel()))
        yield buf_a[:take], buf_b[:take]
        buf_a = buf_a[take:] if buf_a.numel() > take else None
        buf_b = buf_b[take:] if buf_b.numel() > take else None


def compute_chunked_tensor_metrics(
    chunks_a: Iterable[torch.Tensor],
    chunks_b: Iterable[torch.Tensor],
    *,
    tolerances: DiffTolerances | None = None,
    chunk_elements: int = 1_048_576,
    cancellation_token: CancellationToken | None = None,
) -> TensorDiffMetrics:
    """Compute exact float64-accumulated comparison metrics across streaming chunk pairs."""
    tol = tolerances or DiffTolerances()

    total_elements = 0
    finite_pairs = 0
    non_finite_pairs = 0
    nan_a = 0
    nan_b = 0
    inf_a = 0
    inf_b = 0

    exact_equal = True
    within_tol = True
    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    sum_sq_diff = 0.0
    sum_sq_a = 0.0
    sum_sq_b = 0.0
    dot_ab = 0.0

    changed_elements = 0
    zero_to_nonzero = 0
    nonzero_to_zero = 0

    for ca, cb in iter_aligned_chunk_pairs(chunks_a, chunks_b, chunk_elements=chunk_elements):
        if cancellation_token is not None:
            cancellation_token.check()

        n = int(ca.numel())
        total_elements += n

        a64 = ca.to(torch.float64)
        b64 = cb.to(torch.float64)

        a_nan = torch.isnan(a64)
        b_nan = torch.isnan(b64)
        a_inf = torch.isinf(a64)
        b_inf = torch.isinf(b64)

        nan_a += int(a_nan.sum().item())
        nan_b += int(b_nan.sum().item())
        inf_a += int(a_inf.sum().item())
        inf_b += int(b_inf.sum().item())

        finite_mask = ~(a_nan | b_nan | a_inf | b_inf)
        fn = int(finite_mask.sum().item())
        nf = n - fn
        finite_pairs += fn
        non_finite_pairs += nf

        # Non-finite indices: equal only when both +inf or both -inf; NaN != NaN mathematically
        if nf > 0:
            nf_mask = ~finite_mask
            a_nf = a64[nf_mask]
            b_nf = b64[nf_mask]
            same_pos_inf = torch.isposinf(a_nf) & torch.isposinf(b_nf)
            same_neg_inf = torch.isneginf(a_nf) & torch.isneginf(b_nf)
            nf_same = same_pos_inf | same_neg_inf
            nf_diff_count = int((~nf_same).sum().item())
            if nf_diff_count > 0:
                exact_equal = False
                within_tol = False
                changed_elements += nf_diff_count
            # Zero transitions involving +/-Inf against finite zero
            a_zero_nf = a_nf.abs() <= tol.zero_threshold
            b_zero_nf = b_nf.abs() <= tol.zero_threshold
            b_non_zero_nf = torch.isinf(b_nf) | (b_nf.abs() > tol.zero_threshold)
            a_non_zero_nf = torch.isinf(a_nf) | (a_nf.abs() > tol.zero_threshold)
            zero_to_nonzero += int((a_zero_nf & b_non_zero_nf).sum().item())
            nonzero_to_zero += int((a_non_zero_nf & b_zero_nf).sum().item())

        if fn == 0:
            continue

        af = a64[finite_mask] if nf > 0 else a64
        bf = b64[finite_mask] if nf > 0 else b64

        if exact_equal and not bool(torch.equal(af, bf)):
            exact_equal = False

        diff = bf - af
        abs_diff = diff.abs()

        chunk_max = float(abs_diff.max().item())
        if chunk_max > max_abs_diff:
            max_abs_diff = chunk_max

        sum_abs_diff += float(abs_diff.sum().item())
        sum_sq_diff += float(torch.dot(diff, diff).item())
        sum_sq_a += float(torch.dot(af, af).item())
        sum_sq_b += float(torch.dot(bf, bf).item())
        dot_ab += float(torch.dot(af, bf).item())

        exceeded = abs_diff > (tol.atol + tol.rtol * af.abs())
        exc_count = int(exceeded.sum().item())
        if exc_count > 0:
            within_tol = False
            changed_elements += exc_count

        if tol.zero_threshold <= 0.0:
            a_zero = af == 0.0
            b_zero = bf == 0.0
        else:
            a_zero = af.abs() <= tol.zero_threshold
            b_zero = bf.abs() <= tol.zero_threshold

        zero_to_nonzero += int((a_zero & (~b_zero)).sum().item())
        nonzero_to_zero += int(((~a_zero) & b_zero).sum().item())

    if total_elements == 0:
        return TensorDiffMetrics(
            num_elements=0,
            finite_pair_elements=0,
            non_finite_pair_elements=0,
            nan_count_a=0,
            nan_count_b=0,
            inf_count_a=0,
            inf_count_b=0,
            exact_equal=True,
            within_tolerance=True,
            max_abs_diff=0.0,
            mean_abs_diff=0.0,
            rmse=0.0,
            l2_diff=0.0,
            l2_norm_a=0.0,
            l2_norm_b=0.0,
            relative_l2=0.0,
            relative_l2_infinite=False,
            cosine_similarity=1.0,
            changed_elements=0,
            changed_pct=0.0,
            zero_to_nonzero=0,
            nonzero_to_zero=0,
        )

    changed_pct = round((changed_elements / total_elements) * 100.0, 6)

    if finite_pairs == 0:
        return TensorDiffMetrics(
            num_elements=total_elements,
            finite_pair_elements=0,
            non_finite_pair_elements=non_finite_pairs,
            nan_count_a=nan_a,
            nan_count_b=nan_b,
            inf_count_a=inf_a,
            inf_count_b=inf_b,
            exact_equal=exact_equal,
            within_tolerance=within_tol,
            max_abs_diff=None,
            mean_abs_diff=None,
            rmse=None,
            l2_diff=None,
            l2_norm_a=None,
            l2_norm_b=None,
            relative_l2=None,
            relative_l2_infinite=False,
            cosine_similarity=1.0 if exact_equal else None,
            changed_elements=changed_elements,
            changed_pct=changed_pct,
            zero_to_nonzero=zero_to_nonzero,
            nonzero_to_zero=nonzero_to_zero,
        )

    mean_abs = sum_abs_diff / finite_pairs
    rmse = math.sqrt(max(0.0, sum_sq_diff / finite_pairs))
    l2_diff = math.sqrt(max(0.0, sum_sq_diff))
    norm_a = math.sqrt(max(0.0, sum_sq_a))
    norm_b = math.sqrt(max(0.0, sum_sq_b))

    rel_l2_inf = False
    if l2_diff == 0.0:
        rel_l2: float | None = 0.0
    elif norm_a == 0.0:
        rel_l2 = None
        rel_l2_inf = True
    else:
        rel_l2 = l2_diff / max(norm_a, tol.eps)

    if norm_a == 0.0 and norm_b == 0.0:
        cosine: float | None = 1.0
    elif norm_a == 0.0 or norm_b == 0.0:
        cosine = 0.0
    else:
        raw_cos = dot_ab / (norm_a * norm_b)
        cosine = max(-1.0, min(1.0, raw_cos))

    return TensorDiffMetrics(
        num_elements=total_elements,
        finite_pair_elements=finite_pairs,
        non_finite_pair_elements=non_finite_pairs,
        nan_count_a=nan_a,
        nan_count_b=nan_b,
        inf_count_a=inf_a,
        inf_count_b=inf_b,
        exact_equal=exact_equal,
        within_tolerance=within_tol,
        max_abs_diff=_json_safe_float(max_abs_diff),
        mean_abs_diff=_json_safe_float(mean_abs),
        rmse=_json_safe_float(rmse),
        l2_diff=_json_safe_float(l2_diff),
        l2_norm_a=_json_safe_float(norm_a),
        l2_norm_b=_json_safe_float(norm_b),
        relative_l2=_json_safe_float(rel_l2),
        relative_l2_infinite=rel_l2_inf,
        cosine_similarity=_json_safe_float(cosine),
        changed_elements=changed_elements,
        changed_pct=changed_pct,
        zero_to_nonzero=zero_to_nonzero,
        nonzero_to_zero=nonzero_to_zero,
    )


def compare_tensors(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    *,
    tolerances: DiffTolerances | None = None,
    chunk_elements: int = 1_048_576,
) -> TensorDiffMetrics:
    """Compare two in-memory PyTorch tensors of identical shape (any numeric/bool dtype)."""
    if tuple(tensor_a.shape) != tuple(tensor_b.shape):
        raise ValueError(
            f"Shape mismatch in `compare_tensors`: {tuple(tensor_a.shape)} vs {tuple(tensor_b.shape)}. "
            "Use `TensorDiffEngine` for heterogeneous/missing tensor comparison."
        )
    return compute_chunked_tensor_metrics(
        [tensor_a],
        [tensor_b],
        tolerances=tolerances,
        chunk_elements=chunk_elements,
    )
