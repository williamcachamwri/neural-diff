"""Unit tests for Activation Diff Engine and Linear CKA (Module E)."""

from __future__ import annotations

import pytest
import torch

from core.activation_analysis.comparator import ActivationDiffEngine
from core.activation_analysis.metrics import compute_linear_cka
from tests.fixtures.checkpoint_factory import TinyCausalModel


@pytest.mark.unit
def test_linear_cka_orthogonal_and_scale_invariance() -> None:
    g = torch.Generator().manual_seed(7)
    x = torch.randn((24, 12), generator=g, dtype=torch.float64)

    # Random orthogonal matrix via QR decomposition
    q, _ = torch.linalg.qr(torch.randn((12, 12), generator=g, dtype=torch.float64))
    y = (x @ q) * 3.75

    cka_rot = compute_linear_cka(x, y)
    assert cka_rot == pytest.approx(1.0, abs=1e-7)

    # Cross-dimensional CKA (D_a=12 vs D_b=20 embedding of same subspace)
    pad_proj = torch.zeros((12, 20), dtype=torch.float64)
    pad_proj[:12, :12] = q
    y_wide = x @ pad_proj
    assert compute_linear_cka(x, y_wide) == pytest.approx(1.0, abs=1e-7)


@pytest.mark.unit
def test_activation_diff_engine_on_tiny_causal_modules() -> None:
    torch.manual_seed(101)
    mod_a = TinyCausalModel(vocab_size=64, hidden_size=16, num_layers=2)
    mod_b = TinyCausalModel(vocab_size=64, hidden_size=16, num_layers=2)
    mod_b.load_state_dict(mod_a.state_dict())

    prompts = [
        ("prompt one", [1, 5, 9, 12, 3], ["a", "b", "c", "d", "e"]),
        ("prompt two", [2, 8, 14, 7], ["f", "g", "h", "i"]),
    ]

    engine = ActivationDiffEngine(seed=42)
    identical_rep = engine.compare_modules(mod_a, mod_b, prompts)
    assert identical_rep.prompt_count == 2
    assert identical_rep.mean_hidden_cosine_by_layer[0] == pytest.approx(1.0)
    assert identical_rep.mean_hidden_cosine_by_layer[1] == pytest.approx(1.0)
    assert identical_rep.mean_final_token_js_divergence == pytest.approx(0.0, abs=1e-7)

    # Perturb Layer 1 weights and verify Layer 0 stays identical while Layer 1 diverges
    with torch.no_grad():
        mod_b.layers[1].self_attn.weight.add_(0.35)

    pert_rep = engine.compare_modules(mod_a, mod_b, prompts)
    assert pert_rep.mean_hidden_cosine_by_layer[0] == pytest.approx(1.0)
    assert pert_rep.mean_hidden_cosine_by_layer[1] is not None
    assert pert_rep.mean_hidden_cosine_by_layer[1] < 0.9999
    assert pert_rep.mean_final_token_js_divergence is not None
    assert pert_rep.mean_final_token_js_divergence > 0.0
    # Verify hooks were removed cleanly after capture
    for sub in mod_a.modules():
        assert len(sub._forward_hooks) == 0
