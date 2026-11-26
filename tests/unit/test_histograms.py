"""Unit tests for shared-bin histograms and Total Variation Distance."""

from __future__ import annotations

import pytest
import torch

from core.statistics.histograms import compute_paired_histogram


@pytest.mark.unit
def test_paired_histogram_identical_and_disjoint_distributions() -> None:
    x = torch.linspace(-2.0, 2.0, steps=500)
    h_same = compute_paired_histogram(x, x.clone(), bins=16)
    assert h_same.total_variation_distance == pytest.approx(0.0)
    assert h_same.jensen_shannon_divergence == pytest.approx(0.0, abs=1e-7)
    assert h_same.counts_a == h_same.counts_b

    # Disjoint support -> TVD = 1.0
    left = torch.full((200,), -10.0)
    right = torch.full((200,), 10.0)
    h_disjoint = compute_paired_histogram(left, right, bins=10)
    assert h_disjoint.total_variation_distance == pytest.approx(1.0)
    assert h_disjoint.jensen_shannon_divergence > 0.6
