"""Deterministic golden-fixture regression tests (Section 8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.loaders.model_loader import load_model
from core.security import SecurityPolicy
from core.tensor_engine.comparator import TensorDiffEngine
from core.tensor_engine.metrics import DiffTolerances
from tests.fixtures.checkpoint_factory import make_synthetic_hf_checkpoint_pair


@pytest.mark.regression
def test_deterministic_synthetic_pair_golden_metrics(tmp_path: Path) -> None:
    """Verify exact deterministic numerical outputs on fixed-seed checkpoint fixtures."""
    policy = SecurityPolicy(allowed_roots=[tmp_path])
    dir_a, dir_b = make_synthetic_hf_checkpoint_pair(
        tmp_path,
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        perturbation_scale=0.05,
    )

    with load_model(dir_a, policy=policy) as ha, load_model(dir_b, policy=policy) as hb:
        engine = TensorDiffEngine(tolerances=DiffTolerances(atol=1e-6, rtol=1e-5), chunk_elements=512)
        rep1 = engine.compare(ha, hb)
        rep2 = engine.compare(ha, hb)

    assert rep1.total_union_tensors == 21
    assert rep1.identical_tensors == 1  # model.layers.0.input_layernorm.weight
    assert rep1.modified_tensors == 20
    # Strict repeatability across runs
    assert rep1.global_changed_pct == rep2.global_changed_pct
    assert rep1.global_mean_cosine == rep2.global_mean_cosine
    assert rep1.global_mean_relative_l2 == rep2.global_mean_relative_l2
    assert rep1.global_max_abs_diff == rep2.global_max_abs_diff
    assert rep1.top_diverged_names == rep2.top_diverged_names
    # Layer 1 received 3x perturbation relative to Layer 0
    l0 = rep1.layer_summaries[0]
    l1 = rep1.layer_summaries[1]
    assert l1.mean_relative_l2 is not None and l0.mean_relative_l2 is not None
    assert l1.mean_relative_l2 > l0.mean_relative_l2 * 2.0
