"""Structured row/column pruning and NVIDIA 2:4 semi-structured sparsity analyzer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SparsityPruningProfile:
    """Observed unstructured, row/column structured, and 2:4 semi-structured sparsity metrics."""

    shape: list[int]
    element_sparsity_ratio: float
    zero_rows: int
    zero_cols: int
    compliant_2_to_4_groups_ratio: float | None
    is_2_to_4_semi_structured: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_matrix_sparsity(
    weight: torch.Tensor,
    *,
    zero_threshold: float = 0.0,
) -> SparsityPruningProfile:
    """Detect zeroed channels/rows/cols and 2:4 semi-structured sparsity compliance."""
    if weight.ndim != 2:
        raise ValueError("`analyze_matrix_sparsity` expects a 2D weight matrix.")

    mat = weight.detach().cpu().to(torch.float64)
    rows, cols = int(mat.shape[0]), int(mat.shape[1])
    if rows == 0 or cols == 0:
        return SparsityPruningProfile(
            shape=[rows, cols],
            element_sparsity_ratio=0.0,
            zero_rows=0,
            zero_cols=0,
            compliant_2_to_4_groups_ratio=None,
            is_2_to_4_semi_structured=False,
        )

    is_zero = (mat == 0.0) if zero_threshold <= 0.0 else (mat.abs() <= zero_threshold)
    elem_sparsity = float(is_zero.sum().item()) / (rows * cols)
    z_rows = int(is_zero.all(dim=1).sum().item())
    z_cols = int(is_zero.all(dim=0).sum().item())

    ratio_24: float | None = None
    semi_24 = False
    if cols % 4 == 0:
        groups = is_zero.reshape(rows, cols // 4, 4)
        zeros_per_group = groups.sum(dim=-1)
        compliant = zeros_per_group >= 2
        ratio_24 = round(float(compliant.to(torch.float64).mean().item()), 6)
        semi_24 = bool(compliant.all().item()) and elem_sparsity >= 0.45

    return SparsityPruningProfile(
        shape=[rows, cols],
        element_sparsity_ratio=round(elem_sparsity, 8),
        zero_rows=z_rows,
        zero_cols=z_cols,
        compliant_2_to_4_groups_ratio=ratio_24,
        is_2_to_4_semi_structured=semi_24,
    )
