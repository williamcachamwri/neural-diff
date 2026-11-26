"""Top-K individual coordinate difference extractor for Tensor Diff Engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ElementDeltaRecord:
    """Exact multi-dimensional index and values for one high-magnitude parameter change."""

    flat_index: int
    coordinates: list[int]
    value_a: float
    value_b: float
    abs_diff: float
    signed_diff: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_top_k_element_deltas(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    *,
    k: int = 10,
) -> list[ElementDeltaRecord]:
    """Return the `k` coordinates with the largest finite absolute difference `|b - a|`."""
    if tuple(tensor_a.shape) != tuple(tensor_b.shape):
        raise ValueError(
            f"Shape mismatch in `extract_top_k_element_deltas`: {tuple(tensor_a.shape)} vs {tuple(tensor_b.shape)}."
        )
    if tensor_a.numel() == 0 or k <= 0:
        return []

    orig_shape = tuple(int(d) for d in tensor_a.shape)
    fa = tensor_a.detach().cpu().reshape(-1).to(torch.float64)
    fb = tensor_b.detach().cpu().reshape(-1).to(torch.float64)

    finite_mask = torch.isfinite(fa) & torch.isfinite(fb)
    if not bool(finite_mask.any().item()):
        return []

    signed = fb - fa
    abs_d = torch.where(finite_mask, signed.abs(), torch.full_like(signed, -1.0))

    actual_k = min(int(k), int(finite_mask.sum().item()))
    vals, flat_idxs = torch.topk(abs_d, k=actual_k)

    records: list[ElementDeltaRecord] = []
    for abs_v, f_idx in zip(vals.tolist(), flat_idxs.tolist(), strict=True):
        if abs_v < 0.0:
            continue
        coords = _unravel_index(int(f_idx), orig_shape)
        va = float(fa[f_idx].item())
        vb = float(fb[f_idx].item())
        records.append(
            ElementDeltaRecord(
                flat_index=int(f_idx),
                coordinates=coords,
                value_a=va,
                value_b=vb,
                abs_diff=float(abs_v),
                signed_diff=float(vb - va),
            )
        )
    return records


def _unravel_index(flat_idx: int, shape: tuple[int, ...]) -> list[int]:
    if not shape:
        return []
    coords = [0] * len(shape)
    rem = flat_idx
    for axis in range(len(shape) - 1, -1, -1):
        dim = shape[axis]
        coords[axis] = rem % dim
        rem //= dim
    return coords
