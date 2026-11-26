"""Unit tests for Structural Diff (Module C) and Neuron/Attention Analyzer (Module D)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from adapters.architectures.registry import classify_tensor_role
from core.loaders.model_loader import load_model
from core.structural_analysis.diff import StructuralDiffEngine
from core.structural_analysis.neuron_attention import (
    POLYSEMANTICITY_CAUTION,
    NeuronAttentionAnalyzer,
)
from core.types import CanonicalRole, EpistemicStatus
from tests.fixtures.checkpoint_factory import make_synthetic_hf_checkpoint_pair


@pytest.mark.unit
def test_canonical_tensor_role_classification() -> None:
    assert classify_tensor_role("model.layers.7.self_attn.q_proj.weight") == (7, CanonicalRole.ATTN_Q)
    assert classify_tensor_role("transformer.h.2.attn.c_attn.weight") == (2, CanonicalRole.ATTN_QKV_FUSED)
    assert classify_tensor_role("model.layers.0.mlp.gate_proj.weight") == (0, CanonicalRole.MLP_GATE)
    assert classify_tensor_role("model.layers.3.mlp.down_proj.weight") == (3, CanonicalRole.MLP_DOWN)
    assert classify_tensor_role("model.embed_tokens.weight") == (None, CanonicalRole.EMBEDDING)
    assert classify_tensor_role("lm_head.weight") == (None, CanonicalRole.LM_HEAD)
    assert classify_tensor_role("model.layers.1.input_layernorm.weight") == (1, CanonicalRole.NORM)


@pytest.mark.unit
def test_structural_diff_renames_shapes_dtypes_and_cross_arch_guard(tmp_path: Path) -> None:
    dir_a = tmp_path / "m_a"
    dir_b = tmp_path / "m_b"
    dir_a.mkdir()
    dir_b.mkdir()

    shared_q = torch.randn(16, 16)
    renamed_w = torch.randn(16, 16)
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": shared_q,
            "model.layers.0.self_attn.o_proj.weight": renamed_w,
            "model.embed_tokens.weight": torch.randn(50, 16),
            "legacy_extra.weight": torch.ones(4, 4),
        },
        str(dir_a / "model.safetensors"),
    )
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": shared_q.to(torch.float16),  # dtype change
            "model.layers.0.attn.out_proj.weight": renamed_w.clone(),  # renamed ATTN_O
            "model.embed_tokens.weight": torch.randn(60, 16),  # vocab shape change
            "added_adapter.weight": torch.zeros(8, 8),
        },
        str(dir_b / "model.safetensors"),
    )

    with load_model(dir_a) as ha, load_model(dir_b) as hb:
        rep = StructuralDiffEngine().compare(ha, hb)
        assert len(rep.shape_changes) == 1
        assert rep.shape_changes[0].name == "model.embed_tokens.weight"
        assert len(rep.dtype_changes) == 1
        assert len(rep.inferred_renames) == 1
        assert rep.inferred_renames[0].old_name == "model.layers.0.self_attn.o_proj.weight"
        assert rep.inferred_renames[0].new_name == "model.layers.0.attn.out_proj.weight"
        assert rep.inferred_renames[0].epistemic_status == EpistemicStatus.INFERRED
        assert len(rep.removed_tensors) == 1
        assert len(rep.added_tensors) == 1

    # Distinct model_type architectures must NOT trigger false rename pairings
    (dir_a / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    (dir_b / "config.json").write_text(json.dumps({"model_type": "gpt2"}), encoding="utf-8")
    with load_model(dir_a) as ha2, load_model(dir_b) as hb2:
        rep_cross = StructuralDiffEngine().compare(ha2, hb2)
        assert len(rep_cross.inferred_renames) == 0
        assert any(i.code == "ARCH_MODEL_TYPE_MISMATCH" for i in rep_cross.compatibility.issues)


@pytest.mark.unit
def test_neuron_attention_head_and_mlp_analysis(tmp_path: Path) -> None:
    dir_a, dir_b = make_synthetic_hf_checkpoint_pair(
        tmp_path, vocab_size=96, hidden_size=32, intermediate_size=64, num_layers=2, num_heads=4, num_kv_heads=2
    )
    with load_model(dir_a) as ha, load_model(dir_b) as hb:
        report = NeuronAttentionAnalyzer().analyze(ha, hb)
        assert report.epistemic_status == EpistemicStatus.OBSERVED
        assert POLYSEMANTICITY_CAUTION in report.epistemic_caution
        # 2 layers * 4 projections (q, k, v, o) = 8
        assert len(report.attention_projections) == 8
        q0 = next(a for a in report.attention_projections if a.tensor_name.endswith("layers.0.self_attn.q_proj.weight"))
        k0 = next(a for a in report.attention_projections if a.tensor_name.endswith("layers.0.self_attn.k_proj.weight"))
        assert q0.num_heads == 4 and q0.head_dim == 8
        assert k0.num_heads == 2 and k0.head_dim == 8  # GQA head count respected!
        # 2 layers * 3 MLP projections = 6
        assert len(report.mlp_projections) == 6
        assert len(report.embeddings_and_heads) == 2
        assert len(report.normalization_layers) == 5
