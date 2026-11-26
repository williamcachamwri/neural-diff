"""Hypothesis property tests verifying total boundedness across all 8 behavioral evaluators."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from benchmarks.datasets.loader import load_benchmark_suite
from benchmarks.evaluators.registry import evaluate_case


@pytest.mark.property
@settings(max_examples=40, deadline=None)
@given(candidate_text=st.text(min_size=0, max_size=400))
def test_all_eight_evaluators_never_raise_and_bound_scores_in_unit_interval(
    candidate_text: str,
) -> None:
    suite = load_benchmark_suite()
    # Test one representative case per category across arbitrary fuzzed Unicode strings
    seen_cats: set[str] = set()
    for case in suite.cases:
        if case.category in seen_cats:
            continue
        seen_cats.add(case.category)
        res = evaluate_case(case, candidate_text)
        assert math.isfinite(res.score)
        assert 0.0 <= res.score <= 1.0
        assert isinstance(res.passed, bool)
        assert res.evaluator_type == "deterministic"
