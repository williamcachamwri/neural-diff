"""Residual-stream activation patching (interventional) between a clean and a corrupted prompt.

For each decoder block, the *last-token* output of the clean run is written into the corrupted run and the
effect on a chosen logit difference is measured. Because this is an intervention on one specific prompt pair,
the finding is labelled CAUSALLY VERIFIED *for that pair and that site only*; it is not a general claim about
what the layer "represents".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from core.types import EpistemicStatus


@dataclass(frozen=True)
class PatchSiteEffect:
    layer: int
    patched_logit_diff: float
    recovery_fraction: float  # 0 = no effect vs corrupted run, 1 = fully restores the clean logit diff

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PatchingResult:
    clean_logit_diff: float
    corrupted_logit_diff: float
    sites: list[PatchSiteEffect]
    epistemic_status: str = EpistemicStatus.CAUSALLY_VERIFIED.value
    scope: str = "single prompt pair, last-token residual stream, one block at a time"

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean_logit_diff": self.clean_logit_diff,
            "corrupted_logit_diff": self.corrupted_logit_diff,
            "epistemic_status": self.epistemic_status,
            "scope": self.scope,
            "sites": [s.to_dict() for s in self.sites],
        }


def _logit_diff(logits: torch.Tensor, correct_id: int, wrong_id: int) -> float:
    last = logits[0, -1].double()
    return float((last[correct_id] - last[wrong_id]).item())


@torch.no_grad()
def run_residual_patching(
    model: nn.Module,
    layers: nn.ModuleList,
    clean_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
    *,
    correct_token_id: int,
    wrong_token_id: int,
) -> PatchingResult:
    """Patch the clean last-token block output into the corrupted run, one block at a time."""
    if clean_ids.shape != corrupted_ids.shape or clean_ids.ndim != 2 or clean_ids.shape[0] != 1:
        raise ValueError("Clean and corrupted inputs must both be shaped (1, T) with identical T.")

    captured: dict[int, torch.Tensor] = {}
    handles = []

    def _make_capture(idx: int):  # type: ignore[no-untyped-def]
        def hook(_m: nn.Module, _i: Any, out: Any) -> None:
            hs = out[0] if isinstance(out, tuple) else out
            captured[idx] = hs[:, -1, :].detach().clone()

        return hook

    for i, blk in enumerate(layers):
        handles.append(blk.register_forward_hook(_make_capture(i)))
    try:
        clean_logits = model(input_ids=clean_ids).logits
    finally:
        for h in handles:
            h.remove()
    corrupted_logits = model(input_ids=corrupted_ids).logits

    clean_d = _logit_diff(clean_logits, correct_token_id, wrong_token_id)
    corr_d = _logit_diff(corrupted_logits, correct_token_id, wrong_token_id)
    denom = clean_d - corr_d

    sites: list[PatchSiteEffect] = []
    for i, blk in enumerate(layers):

        def _patch(_m: nn.Module, _inp: Any, out: Any, idx: int = i) -> Any:
            hs = out[0] if isinstance(out, tuple) else out
            hs = hs.clone()
            hs[:, -1, :] = captured[idx].to(hs.dtype)
            return (hs, *out[1:]) if isinstance(out, tuple) else hs

        handle = blk.register_forward_hook(_patch)
        try:
            patched = _logit_diff(model(input_ids=corrupted_ids).logits, correct_token_id, wrong_token_id)
        finally:
            handle.remove()
        recovery = 0.0 if abs(denom) < 1e-12 else (patched - corr_d) / denom
        sites.append(PatchSiteEffect(layer=i, patched_logit_diff=patched, recovery_fraction=recovery))

    return PatchingResult(clean_logit_diff=clean_d, corrupted_logit_diff=corr_d, sites=sites)
