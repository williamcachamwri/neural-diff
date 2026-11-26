"""Diff-of-diffs: compare successive behavioral runs to expose per-category trends over time."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.types import EpistemicStatus


@dataclass(frozen=True)
class RunSnapshot:
    """Minimal behavioral summary of one historical run (taken from a stored comparison bundle)."""

    run_id: str
    pass_rate: float
    category_scores: dict[str, float]


@dataclass(frozen=True)
class CategoryTrend:
    category: str
    first_score: float
    last_score: float
    net_delta: float
    largest_single_step_drop: float
    drop_between: tuple[str, str] | None
    monotonic_non_decreasing: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["drop_between"] = list(self.drop_between) if self.drop_between else None
        return data


@dataclass(frozen=True)
class HistoryComparison:
    run_ids: list[str]
    pass_rate_series: list[float]
    trends: list[CategoryTrend]
    regressing_categories: list[str]
    epistemic_status: str = EpistemicStatus.OBSERVED.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_ids": self.run_ids,
            "pass_rate_series": self.pass_rate_series,
            "trends": [t.to_dict() for t in self.trends],
            "regressing_categories": self.regressing_categories,
            "epistemic_status": self.epistemic_status,
        }


def compare_run_history(
    snapshots: list[RunSnapshot],
    *,
    drop_tolerance: float = 0.05,
) -> HistoryComparison:
    """Order-preserving comparison of >= 2 runs; flags categories whose last score fell by > `drop_tolerance`."""
    if len(snapshots) < 2:
        raise ValueError("`compare_run_history` needs at least two runs.")
    if drop_tolerance < 0:
        raise ValueError("`drop_tolerance` must be non-negative.")

    categories = sorted(set().union(*(s.category_scores.keys() for s in snapshots)))
    trends: list[CategoryTrend] = []
    for cat in categories:
        series = [(s.run_id, s.category_scores[cat]) for s in snapshots if cat in s.category_scores]
        if len(series) < 2:
            continue
        worst_drop = 0.0
        drop_between: tuple[str, str] | None = None
        monotonic = True
        for (id_prev, v_prev), (id_cur, v_cur) in zip(series, series[1:], strict=False):
            step = v_cur - v_prev
            if step < 0:
                monotonic = False
            if -step > worst_drop:
                worst_drop = -step
                drop_between = (id_prev, id_cur)
        trends.append(
            CategoryTrend(
                category=cat,
                first_score=series[0][1],
                last_score=series[-1][1],
                net_delta=series[-1][1] - series[0][1],
                largest_single_step_drop=worst_drop,
                drop_between=drop_between,
                monotonic_non_decreasing=monotonic,
            )
        )

    regressing = [t.category for t in trends if t.net_delta < -drop_tolerance]
    return HistoryComparison(
        run_ids=[s.run_id for s in snapshots],
        pass_rate_series=[s.pass_rate for s in snapshots],
        trends=trends,
        regressing_categories=regressing,
    )
