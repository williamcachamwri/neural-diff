"""Unit tests for Module B Tensor Diff metrics and numerical distributions."""

from __future__ import annotations

import math

import pytest
import torch

from core.statistics.distributions import compute_distribution_shift, summarize_tensor
from core.tensor_engine.metrics import (
    DiffTolerances,
    compare_tensors,
    compute_chunked_tensor_metrics,
    iter_aligned_chunk_pairs,
)


@pytest.mark.unit
def test_identical_tensors_exact_equality() -> None:
    a = torch.tensor([[1.0, -2.5, 0.0], [3.25, 4.0, -0.5]], dtype=torch.float32)
    m = compare_tensors(a, a.clone())
    assert m.exact_equal is True
    assert m.within_tolerance is True
    assert m.max_abs_diff == 0.0
    assert m.mean_abs_diff == 0.0
    assert m.rmse == 0.0
    assert m.relative_l2 == 0.0
    assert m.cosine_similarity == pytest.approx(1.0)
    assert m.changed_elements == 0
    assert m.changed_pct == 0.0
    assert m.zero_to_nonzero == 0
    assert m.nonzero_to_zero == 0


@pytest.mark.unit
def test_known_hand_calculated_differences() -> None:
    # a = [3, 4, 0, 2], ||a||_2 = sqrt(9+16+0+4) = sqrt(29)
    # b = [3, 0, 5, 4], diff = [0, -4, 5, 2]
    # abs_diff = [0, 4, 5, 2] -> max=5, mean=11/4=2.75
    # sq_diff = 0 + 16 + 25 + 4 = 45 -> rmse = sqrt(45/4), l2_diff = sqrt(45)
    a = torch.tensor([3.0, 4.0, 0.0, 2.0], dtype=torch.float64)
    b = torch.tensor([3.0, 0.0, 5.0, 4.0], dtype=torch.float64)
    m = compare_tensors(a, b, tolerances=DiffTolerances(atol=1e-6, rtol=1e-5))
    assert m.exact_equal is False
    assert m.max_abs_diff == pytest.approx(5.0)
    assert m.mean_abs_diff == pytest.approx(2.75)
    assert m.rmse == pytest.approx(math.sqrt(45.0 / 4.0))
    assert m.relative_l2 == pytest.approx(math.sqrt(45.0) / math.sqrt(29.0))
    expected_cos = (9.0 + 0.0 + 0.0 + 8.0) / (math.sqrt(29.0) * math.sqrt(50.0))
    assert m.cosine_similarity == pytest.approx(expected_cos)
    assert m.changed_elements == 3
    assert m.changed_pct == pytest.approx(75.0)
    assert m.zero_to_nonzero == 1
    assert m.nonzero_to_zero == 1


@pytest.mark.unit
def test_zero_vector_and_nan_inf_edge_cases() -> None:
    z = torch.zeros(6, dtype=torch.float32)
    nz = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    m_zero_both = compare_tensors(z, z.clone())
    assert m_zero_both.cosine_similarity == 1.0
    assert m_zero_both.relative_l2 == 0.0

    m_zero_one = compare_tensors(z, nz)
    assert m_zero_one.cosine_similarity == 0.0
    assert m_zero_one.relative_l2 is None
    assert m_zero_one.relative_l2_infinite is True

    # NaN and Infinity handling
    t_a = torch.tensor([1.0, float("nan"), float("inf"), float("-inf"), 2.0])
    t_b = torch.tensor([1.0, float("nan"), float("inf"), 5.0, 3.0])
    m_nf = compare_tensors(t_a, t_b)
    assert m_nf.exact_equal is False  # NaN != NaN
    assert m_nf.nan_count_a == 1
    assert m_nf.nan_count_b == 1
    assert m_nf.inf_count_a == 2
    assert m_nf.inf_count_b == 1
    assert m_nf.finite_pair_elements == 2  # indices 0 and 4
    assert m_nf.max_abs_diff == pytest.approx(1.0)


@pytest.mark.unit
def test_bfloat16_and_heterogeneous_chunk_alignment() -> None:
    g = torch.Generator().manual_seed(99)
    a_bf16 = torch.randn(500, generator=g).to(torch.bfloat16)
    b_fp16 = (a_bf16.to(torch.float32) + 0.02).to(torch.float16)

    # Irregular chunk splits on A vs B must align element-for-element
    chunks_a = [a_bf16[:113], a_bf16[113:311], a_bf16[311:]]
    chunks_b = [b_fp16[:77], b_fp16[77:250], b_fp16[250:499], b_fp16[499:]]

    aligned_total = sum(ca.numel() for ca, _cb in iter_aligned_chunk_pairs(chunks_a, chunks_b, chunk_elements=64))
    assert aligned_total == 500

    m_chunked = compute_chunked_tensor_metrics(chunks_a, chunks_b, chunk_elements=64)
    m_direct = compare_tensors(a_bf16, b_fp16)
    assert m_chunked.rmse == pytest.approx(m_direct.rmse, rel=1e-12)
    assert m_chunked.cosine_similarity == pytest.approx(m_direct.cosine_similarity, rel=1e-12)


@pytest.mark.unit
def test_distribution_summary_and_shift() -> None:
    a = torch.linspace(-1.0, 1.0, steps=1001)
    b = a * 2.0 + 0.5
    da = summarize_tensor(a)
    db = summarize_tensor(b)
    shift = compute_distribution_shift(da, db)
    assert da.mean == pytest.approx(0.0, abs=1e-6)
    assert db.mean == pytest.approx(0.5, abs=1e-6)
    assert shift.mean_delta == pytest.approx(0.5, abs=1e-6)
    assert shift.std_ratio == pytest.approx(2.0, rel=1e-5)
    assert shift.wasserstein_1d_approx is not None and shift.wasserstein_1d_approx > 0.4
