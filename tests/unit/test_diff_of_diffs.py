from __future__ import annotations

import pytest

from core.regression_engine.diff_of_diffs import RunSnapshot, compare_run_history


def test_history_flags_net_regression_and_locates_worst_step() -> None:
    runs = [
        RunSnapshot("r1", 0.5, {"math": 0.8, "json": 0.9}),
        RunSnapshot("r2", 0.6, {"math": 0.4, "json": 0.95}),
        RunSnapshot("r3", 0.7, {"math": 0.5, "json": 1.0}),
    ]
    cmp = compare_run_history(runs, drop_tolerance=0.05)
    assert cmp.regressing_categories == ["math"]
    math = next(t for t in cmp.trends if t.category == "math")
    assert math.net_delta == pytest.approx(-0.3)
    assert math.largest_single_step_drop == pytest.approx(0.4)
    assert math.drop_between == ("r1", "r2")
    assert not math.monotonic_non_decreasing
    js = next(t for t in cmp.trends if t.category == "json")
    assert js.monotonic_non_decreasing and js.drop_between is None
    assert cmp.to_dict()["epistemic_status"] == "OBSERVED"


def test_history_requires_two_runs_and_valid_tolerance() -> None:
    with pytest.raises(ValueError):
        compare_run_history([RunSnapshot("only", 1.0, {"a": 1.0})])
    two = [RunSnapshot("a", 1.0, {"x": 1.0}), RunSnapshot("b", 1.0, {"x": 1.0})]
    with pytest.raises(ValueError):
        compare_run_history(two, drop_tolerance=-1)


def test_categories_present_in_one_run_only_are_skipped() -> None:
    runs = [RunSnapshot("a", 1.0, {"x": 1.0}), RunSnapshot("b", 1.0, {"x": 1.0, "new": 0.2})]
    cmp = compare_run_history(runs)
    assert [t.category for t in cmp.trends] == ["x"]
