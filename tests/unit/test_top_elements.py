"""Unit tests for top-k coordinate delta extraction."""

from __future__ import annotations

import pytest
import torch

from core.tensor_engine.top_elements import extract_top_k_element_deltas


@pytest.mark.unit
def test_extract_top_k_element_deltas_2d_and_3d() -> None:
    a = torch.zeros((3, 4), dtype=torch.float32)
    b = a.clone()
    b[2, 1] = -7.5
    b[0, 3] = 4.25
    b[1, 2] = 1.5

    top2 = extract_top_k_element_deltas(a, b, k=2)
    assert len(top2) == 2
    assert top2[0].coordinates == [2, 1]
    assert top2[0].abs_diff == pytest.approx(7.5)
    assert top2[0].signed_diff == pytest.approx(-7.5)
    assert top2[1].coordinates == [0, 3]
    assert top2[1].abs_diff == pytest.approx(4.25)

    # 3D shape check and mismatch guard
    a3 = torch.ones((2, 2, 2))
    b3 = a3.clone()
    b3[1, 0, 1] = 10.0
    res3 = extract_top_k_element_deltas(a3, b3, k=1)
    assert res3[0].coordinates == [1, 0, 1]

    with pytest.raises(ValueError):
        extract_top_k_element_deltas(torch.zeros((2, 3)), torch.zeros((3, 2)))
