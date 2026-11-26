"""Unit tests for weight-tying inspection and untying transition detection."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from core.loaders.model_loader import load_model
from core.security import SecurityPolicy
from core.structural_analysis.tie_weights import compare_weight_tying


@pytest.mark.unit
def test_detect_accidental_weight_untying_between_checkpoints(tmp_path: Path) -> None:
    policy = SecurityPolicy(allowed_roots=[tmp_path])
    emb = torch.randn((32, 16), generator=torch.Generator().manual_seed(1))

    fa = tmp_path / "tied.safetensors"
    fb = tmp_path / "untied.safetensors"
    save_file({"model.embed_tokens.weight": emb, "lm_head.weight": emb.clone()}, str(fa))
    save_file({"model.embed_tokens.weight": emb, "lm_head.weight": emb + 0.25}, str(fb))

    with load_model(fa, policy=policy) as ha, load_model(fb, policy=policy) as hb:
        res = compare_weight_tying(ha, hb)
        assert res["model_a"]["numerically_tied"] is True
        assert res["model_b"]["numerically_tied"] is False
        assert res["tying_changed"] is True
