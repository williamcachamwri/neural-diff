"""Unit tests distinguishing low-rank LoRA updates from full-rank fine-tuning via SVD."""

from __future__ import annotations

import pytest
import torch

from core.tensor_engine.task_vector import analyze_task_vector_spectrum


@pytest.mark.unit
def test_svd_effective_rank_distinguishes_lora_from_full_rank() -> None:
    g = torch.Generator().manual_seed(77)
    base = torch.randn((64, 64), generator=g, dtype=torch.float32)

    # Rank-4 LoRA update: delta = B @ A
    lora_b = torch.randn((64, 4), generator=g) * 0.2
    lora_a = torch.randn((4, 64), generator=g) * 0.2
    lora_merged = base + (lora_b @ lora_a)

    spec_lora = analyze_task_vector_spectrum(base, lora_merged)
    assert spec_lora.rank_at_99pct_energy <= 4
    assert spec_lora.shannon_effective_rank <= 4.05
    assert spec_lora.inferred_update_regime == "low_rank_adapter"

    # Full-rank isotropic Gaussian perturbation
    full_ft = base + torch.randn((64, 64), generator=g) * 0.1
    spec_full = analyze_task_vector_spectrum(base, full_ft)
    assert spec_full.rank_at_99pct_energy > 40
    assert spec_full.shannon_effective_rank > 35.0
    assert spec_full.inferred_update_regime == "full_rank_update"
