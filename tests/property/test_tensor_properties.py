"""Property-based mathematical invariant tests using Hypothesis (Section 8)."""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from core.activation_analysis.metrics import compute_linear_cka
from core.tensor_engine.metrics import (
    DiffTolerances,
    compare_tensors,
    compute_chunked_tensor_metrics,
)

finite_float = st.floats(
    min_value=-1e4,
    max_value=1e4,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


@pytest.mark.property
@settings(max_examples=60, deadline=None)
@given(values=st.lists(finite_float, min_size=1, max_size=128))
def test_tensor_reflexivity_identity(values: list[float]) -> None:
    t = torch.tensor(values, dtype=torch.float32)
    m = compare_tensors(t, t.clone())
    assert m.exact_equal is True
    assert m.within_tolerance is True
    assert m.max_abs_diff == 0.0
    assert m.mean_abs_diff == 0.0
    assert m.rmse == 0.0
    assert m.relative_l2 == 0.0
    assert m.cosine_similarity == pytest.approx(1.0, abs=1e-7)
    assert m.changed_elements == 0
    assert m.zero_to_nonzero == 0
    assert m.nonzero_to_zero == 0


@pytest.mark.property
@settings(max_examples=60, deadline=None)
@given(
    pairs=st.lists(st.tuples(finite_float, finite_float), min_size=1, max_size=128),
)
def test_tensor_symmetry_invariants(pairs: list[tuple[float, float]]) -> None:
    a = torch.tensor([p[0] for p in pairs], dtype=torch.float32)
    b = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
    tol = DiffTolerances(atol=1e-5, rtol=0.0)  # Pure atol makes changed_elements symmetric too!

    mab = compare_tensors(a, b, tolerances=tol)
    mba = compare_tensors(b, a, tolerances=tol)

    assert mab.exact_equal == mba.exact_equal
    assert mab.max_abs_diff == pytest.approx(mba.max_abs_diff, rel=1e-12, abs=1e-12)
    assert mab.mean_abs_diff == pytest.approx(mba.mean_abs_diff, rel=1e-12, abs=1e-12)
    assert mab.rmse == pytest.approx(mba.rmse, rel=1e-12, abs=1e-12)
    assert mab.cosine_similarity == pytest.approx(mba.cosine_similarity, rel=1e-12, abs=1e-12)
    assert mab.changed_elements == mba.changed_elements
    assert mab.zero_to_nonzero == mba.nonzero_to_zero
    assert mab.nonzero_to_zero == mba.zero_to_nonzero


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(
    values=st.lists(finite_float, min_size=2, max_size=96),
    scale=st.floats(min_value=0.05, max_value=50.0, allow_nan=False, allow_infinity=False),
)
def test_cosine_positive_scale_and_negation_invariance(values: list[float], scale: float) -> None:
    a = torch.tensor(values, dtype=torch.float64)
    if float(torch.linalg.vector_norm(a).item()) < 1e-6:
        return
    m_pos = compare_tensors(a, a * scale)
    m_neg = compare_tensors(a, -a * scale)
    assert m_pos.cosine_similarity == pytest.approx(1.0, abs=1e-7)
    assert m_neg.cosine_similarity == pytest.approx(-1.0, abs=1e-7)


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(
    triples=st.lists(
        st.tuples(finite_float, finite_float, finite_float),
        min_size=1,
        max_size=80,
    )
)
def test_l2_triangle_inequality(triples: list[tuple[float, float, float]]) -> None:
    a = torch.tensor([t[0] for t in triples], dtype=torch.float64)
    b = torch.tensor([t[1] for t in triples], dtype=torch.float64)
    c = torch.tensor([t[2] for t in triples], dtype=torch.float64)

    dab = compare_tensors(a, b).l2_diff or 0.0
    dbc = compare_tensors(b, c).l2_diff or 0.0
    dac = compare_tensors(a, c).l2_diff or 0.0
    assert dac <= (dab + dbc) + 1e-7 * max(1.0, dab + dbc)


@pytest.mark.property
@settings(max_examples=40, deadline=None)
@given(
    pairs=st.lists(st.tuples(finite_float, finite_float), min_size=5, max_size=200),
    chunk_size=st.integers(min_value=1, max_value=37),
)
def test_chunk_partition_invariance(
    pairs: list[tuple[float, float]],
    chunk_size: int,
) -> None:
    a = torch.tensor([p[0] for p in pairs], dtype=torch.float32)
    b = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
    full = compare_tensors(a, b)
    chunked = compute_chunked_tensor_metrics([a], [b], chunk_elements=chunk_size)
    assert chunked.exact_equal == full.exact_equal
    assert chunked.changed_elements == full.changed_elements
    assert chunked.max_abs_diff == pytest.approx(full.max_abs_diff, rel=1e-12, abs=1e-12)
    assert chunked.rmse == pytest.approx(full.rmse, rel=1e-12, abs=1e-12)
    assert chunked.cosine_similarity == pytest.approx(full.cosine_similarity, rel=1e-12, abs=1e-12)


@pytest.mark.property
@settings(max_examples=30, deadline=None)
@given(
    angle=st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False, allow_infinity=False),
    scale=st.floats(min_value=0.2, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_linear_cka_planar_rotation_invariance(angle: float, scale: float) -> None:
    x = torch.tensor(
        [[-2.0, 0.5], [-1.0, -1.5], [0.5, 2.0], [1.5, -0.5], [2.5, 1.0]],
        dtype=torch.float64,
    )
    rot = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=torch.float64,
    )
    y = (x @ rot) * scale
    cka = compute_linear_cka(x, y)
    assert cka == pytest.approx(1.0, abs=1e-7)
