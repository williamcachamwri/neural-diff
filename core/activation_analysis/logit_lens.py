"""Logit lens: project each layer's last-token residual stream through the final norm and LM head.

Observational only (EpistemicStatus.OBSERVED): it shows what the unembedding would read at each depth,
not what the layer computes causally.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from core.types import EpistemicStatus


@dataclass(frozen=True)
class LensLayerReading:
    layer: int
    top_token_ids: list[int]
    top_probs: list[float]
    kl_to_final: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LogitLensResult:
    readings: list[LensLayerReading]
    epistemic_status: str = EpistemicStatus.OBSERVED.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "epistemic_status": self.epistemic_status,
            "readings": [r.to_dict() for r in self.readings],
        }


@torch.no_grad()
def run_logit_lens(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    final_norm: nn.Module,
    lm_head: nn.Module,
    top_k: int = 5,
) -> LogitLensResult:
    """Decode the last position's hidden state after every decoder block (HF-style `output_hidden_states`)."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("`run_logit_lens` expects a single non-empty sequence shaped (1, T).")

    out = model(input_ids=input_ids, output_hidden_states=True)
    hidden_states = out.hidden_states  # (embeddings, block 1, ..., block N [already final-normed in HF])
    if hidden_states is None or len(hidden_states) < 2:
        raise ValueError("Model did not return per-layer hidden states.")

    final_logp = torch.log_softmax(out.logits[0, -1].double(), dim=-1)
    readings: list[LensLayerReading] = []
    n_blocks = len(hidden_states) - 1
    for layer in range(1, n_blocks + 1):
        h = hidden_states[layer][:, -1:, :]
        # HF applies the final norm inside the last hidden state already; avoid normalising twice.
        projected = h if layer == n_blocks else final_norm(h)
        logits = lm_head(projected)[0, -1].double()
        logp = torch.log_softmax(logits, dim=-1)
        probs = logp.exp()
        k = min(top_k, probs.numel())
        top = torch.topk(probs, k)
        kl = float(torch.sum(probs * (logp - final_logp)).item())
        readings.append(
            LensLayerReading(
                layer=layer - 1,
                top_token_ids=[int(i) for i in top.indices.tolist()],
                top_probs=[float(p) for p in top.values.tolist()],
                kl_to_final=max(0.0, kl),
            )
        )
    return LogitLensResult(readings=readings)
