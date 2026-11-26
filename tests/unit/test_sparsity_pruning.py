"""Unit tests for structured row/column pruning and 2:4 semi-structured sparsity."""

from __future__ import annotations

import pytest
import torch

from core.structural_analysis.sparsity_pruning import analyze_matrix_sparsity


@pytest.mark.unit
def test_structured_pruning_and_2_to_4_sparsity_detection() -> None:
    dense = torch.ones((8, 12))
    p_dense = analyze_matrix_sparsity(dense)
    assert p_dense.element_sparsity_ratio == 0.0
    assert p_dense.is_2_to_4_semi_structured is False

    # Construct exact 2:4 semi-structured sparse matrix (2 zeros in every 4-element block)
    w24 = torch.tensor([[1.0, 0.0, -2.0, 0.0, 0.0, 3.0, 0.0, 4.0]] * 6)
    p24 = analyze_matrix_sparsity(w24)
    assert p24.element_sparsity_ratio == pytest.approx(0.5)
    assert p24.compliant_2_to_4_groups_ratio == pytest.approx(1.0)
    assert p24.is_2_to_4_semi_structured is True

    # Zero out an entire row and column
    pruned = torch.ones((4, 4))
    pruned[2, :] = 0.0
    pruned[:, 1] = 0.0
    p_struct = analyze_matrix_sparsity(pruned)
    assert p_struct.zero_rows == 1
    assert p_struct.zero_cols == 1
