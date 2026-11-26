from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from core.activation_analysis.patching import run_residual_patching


def _tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(1)
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    return LlamaForCausalLM(cfg).eval()


@pytest.mark.unit
def test_patching_last_block_restores_clean_logit_diff() -> None:
    """Patching the final block's last-token output must reproduce the clean last-position logits exactly,
    because everything downstream of that site at that position is a pure function of the patched vector."""
    model = _tiny_llama()
    clean = torch.tensor([[3, 7, 11, 5]])
    corrupted = torch.tensor([[3, 9, 12, 5]])
    res = run_residual_patching(
        model, model.model.layers, clean, corrupted, correct_token_id=10, wrong_token_id=20
    )
    assert len(res.sites) == 3
    assert res.sites[-1].recovery_fraction == pytest.approx(1.0, abs=1e-4)
    assert res.clean_logit_diff != pytest.approx(res.corrupted_logit_diff)
    assert res.to_dict()["epistemic_status"] == "CAUSALLY VERIFIED"
    assert "single prompt pair" in res.scope


@pytest.mark.unit
def test_patching_identical_prompts_gives_zero_recovery_and_validates_shapes() -> None:
    model = _tiny_llama()
    same = torch.tensor([[1, 2, 3]])
    res = run_residual_patching(model, model.model.layers, same, same, correct_token_id=1, wrong_token_id=2)
    assert all(s.recovery_fraction == 0.0 for s in res.sites)
    with pytest.raises(ValueError):
        run_residual_patching(
            model, model.model.layers, same, torch.tensor([[1, 2]]), correct_token_id=1, wrong_token_id=2
        )
