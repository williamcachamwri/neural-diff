"""Paired statistical significance & confidence intervals for behavioral comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.stats import binomtest


@dataclass(frozen=True)
class PairedStatisticalSummary:
    """Paired statistical evaluation of Model B minus Model A across identical test cases."""

    n_cases: int
    mean_score_a: float
    mean_score_b: float
    mean_score_delta: float
    bootstrap_ci95_low: float
    bootstrap_ci95_high: float
    pass_rate_a: float
    pass_rate_b: float
    pass_rate_delta: float
    b_wins: int
    a_wins: int
    ties: int
    mcnemar_b_only_pass: int
    mcnemar_a_only_pass: int
    mcnemar_p_value: float
    statistically_significant_95: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_paired_significance(
    scores_a: list[float],
    scores_b: list[float],
    passed_a: list[bool],
    passed_b: list[bool],
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> PairedStatisticalSummary:
    """Compute paired score delta 95% bootstrap CI and exact McNemar binomial p-value."""
    if not (len(scores_a) == len(scores_b) == len(passed_a) == len(passed_b)):
        raise ValueError("Paired score and pass lists must have identical length.")

    n = len(scores_a)
    if n == 0:
        return PairedStatisticalSummary(
            n_cases=0,
            mean_score_a=0.0,
            mean_score_b=0.0,
            mean_score_delta=0.0,
            bootstrap_ci95_low=0.0,
            bootstrap_ci95_high=0.0,
            pass_rate_a=0.0,
            pass_rate_b=0.0,
            pass_rate_delta=0.0,
            b_wins=0,
            a_wins=0,
            ties=0,
            mcnemar_b_only_pass=0,
            mcnemar_a_only_pass=0,
            mcnemar_p_value=1.0,
            statistically_significant_95=False,
        )

    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)
    deltas = arr_b - arr_a

    mean_a = float(arr_a.mean())
    mean_b = float(arr_b.mean())
    mean_delta = float(deltas.mean())

    if n == 1 or float(np.std(deltas)) == 0.0:
        ci_low = mean_delta
        ci_high = mean_delta
    else:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(max(200, n_bootstrap), n))
        boot_means = deltas[idx].mean(axis=1)
        ci_low, ci_high = (float(v) for v in np.quantile(boot_means, [0.025, 0.975]))

    pr_a = sum(1 for p in passed_a if p) / n
    pr_b = sum(1 for p in passed_b if p) / n

    b_wins = sum(1 for da in deltas if da > 1e-9)
    a_wins = sum(1 for da in deltas if da < -1e-9)
    ties = n - b_wins - a_wins

    b_only = sum(1 for pa, pb in zip(passed_a, passed_b, strict=True) if (not pa) and pb)
    a_only = sum(1 for pa, pb in zip(passed_a, passed_b, strict=True) if pa and (not pb))
    discordant = b_only + a_only
    if discordant == 0:
        p_val = 1.0
    else:
        p_val = float(binomtest(b_only, n=discordant, p=0.5, alternative="two-sided").pvalue)

    ci_excludes_zero = (ci_low > 0.0) or (ci_high < 0.0)
    sig = bool((n >= 5 and ci_excludes_zero) or (p_val < 0.05))

    return PairedStatisticalSummary(
        n_cases=n,
        mean_score_a=round(mean_a, 6),
        mean_score_b=round(mean_b, 6),
        mean_score_delta=round(mean_delta, 6),
        bootstrap_ci95_low=round(ci_low, 6),
        bootstrap_ci95_high=round(ci_high, 6),
        pass_rate_a=round(pr_a, 6),
        pass_rate_b=round(pr_b, 6),
        pass_rate_delta=round(pr_b - pr_a, 6),
        b_wins=b_wins,
        a_wins=a_wins,
        ties=ties,
        mcnemar_b_only_pass=b_only,
        mcnemar_a_only_pass=a_only,
        mcnemar_p_value=round(p_val, 6),
        statistically_significant_95=sig,
    )
