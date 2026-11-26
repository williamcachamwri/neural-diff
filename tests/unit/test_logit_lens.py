from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from core.activation_analysis.logit_lens import run_logit_lens


def _tiny_llama(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
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
def test_logit_lens_final_layer_matches_model_output() -> None:
    model = _tiny_llama()
    ids = torch.tensor([[3, 7, 11, 5]])
    res = run_logit_lens(model, ids, final_norm=model.model.norm, lm_head=model.lm_head, top_k=3)
    assert len(res.readings) == 3
    last = res.readings[-1]
    expected = torch.softmax(model(input_ids=ids).logits[0, -1].double(), dim=-1)
    top = torch.topk(expected, 3)
    assert last.top_token_ids == top.indices.tolist()
    assert last.kl_to_final == pytest.approx(0.0, abs=1e-9)
    assert all(r.kl_to_final >= 0 for r in res.readings)
    assert res.to_dict()["epistemic_status"] == "OBSERVED"


@pytest.mark.unit
def test_logit_lens_rejects_batches_and_empty_input() -> None:
    model = _tiny_llama()
    with pytest.raises(ValueError):
        run_logit_lens(model, torch.zeros((2, 3), dtype=torch.long), final_norm=model.model.norm, lm_head=model.lm_head)
    with pytest.raises(ValueError):
        run_logit_lens(model, torch.zeros((1, 0), dtype=torch.long), final_norm=model.model.norm, lm_head=model.lm_head)
