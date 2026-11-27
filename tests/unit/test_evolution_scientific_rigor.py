"""Scientific rigor unit tests for Neural Evolution Tracking (Audit P1-1..P2-5).

Covers:
1. Strict 4-tier epistemic separation (`OBSERVED`, `CORRELATED`, `CAUSAL SUPPORT`, `CAUSALLY VERIFIED`).
2. Small-sample statistical safeguards (`is_exploratory`, `statistically_significant=False` when `n < 8`),
   Holm-Bonferroni step-down multiple-comparison adjustment, and `regression_kind` separation.
3. Exact O(V+E) DP branch counting (`count_total_branches`), lazy `iter_branches`, explicit `truncated=True`
   metadata when exceeding `max_paths`, and complete-DAG edge fallback in `RegressionOriginEngine`.
4. 5-component SHA-256 checkpoint fingerprinting (`weights`, `architecture_config`, `tokenizer`, `adapter`, `metadata`)
   ensuring tokenizer mutations invalidate fingerprints and behavioral evaluation caches.
5. Complete 7-way topological pair scope taxonomy (`identical`, `consecutive`, `base_to_current`,
   `ancestor_to_descendant`, `descendant_to_ancestor`, `branch_to_branch`, `unrelated`).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from core.evolution.behavioral_evolution import (
    build_behavioral_checkpoint_cache_key,
    holm_bonferroni_adjust,
)
from core.evolution.dag import EvolutionDAG
from core.evolution.registry import inspect_and_build_checkpoint_record
from core.evolution.regression_origin import RegressionOriginEngine
from core.evolution.store import EvolutionStore
from core.evolution.types import EvolutionRelationship, RegisteredCheckpoint, TransformationType
from core.security import SecurityPolicy
from core.types import EpistemicStatus
from tests.fixtures.evolution_factory import create_known_evolution_checkpoints


def _dummy_ckpt(cid: str, label: str, idx: int = 0) -> RegisteredCheckpoint:
    return RegisteredCheckpoint(
        checkpoint_id=cid,
        project_id="proj-rigor",
        version_label=label,
        model_name=label,
        source_path=f"/tmp/{cid}",
        resolved_path=f"/tmp/{cid}",
        architecture="LlamaForCausalLM",
        model_family="llama",
        parameter_count=1024,
        layer_count=2,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        tensor_count=10,
        checkpoint_format="huggingface_local_dir",
        dtypes_present=["F32"],
        file_size_bytes=4096,
        sha256_fingerprint=f"{idx:064x}",
        tensor_metadata_hash="meta001",
        parent_checkpoint_ids=[],
        transformation_type=TransformationType.SFT,
        transformation_metadata={},
        created_at_utc=f"2026-11-27T00:00:{idx:02d}+00:00",
        framework_info={},
        tokenizer_metadata={},
        quantization_metadata={},
        training_metadata={},
        lora_metadata={},
        experiment_notes="",
        validation_status=inspect_and_build_checkpoint_record.__globals__[
            "CheckpointValidationStatus"
        ].VERIFIED,
    )


@pytest.mark.unit
def test_epistemic_hierarchy_never_conflates_diagnostic_patching_with_case_verification() -> None:
    """P1-1: Diagnostic transition patching >=20% yields CAUSAL SUPPORT, NEVER CAUSALLY VERIFIED."""
    c0 = _dummy_ckpt("c0", "Base", 0)
    c1 = _dummy_ckpt("c1", "Quantized", 1)
    rel = EvolutionRelationship(
        relationship_id="r01",
        project_id="proj-rigor",
        parent_checkpoint_id="c0",
        child_checkpoint_id="c1",
        transformation_type=TransformationType.QUANTIZATION,
    )
    dag = EvolutionDAG([c0, c1], [rel])

    beh_payload = {
        "checkpoint_evaluations": [
            {
                "checkpoint_id": "c0",
                "version_label": "Base",
                "pass_rate": 1.0,
                "mean_score": 1.0,
                "categories": {
                    "code_preservation": {"pass_rate": 1.0, "mean_score": 1.0, "sample_size": 3},
                    "vietnamese": {"pass_rate": 1.0, "mean_score": 1.0, "sample_size": 3},
                },
                "case_results": [
                    {"case_id": "code-01", "category": "code_preservation", "passed": True, "score": 1.0},
                    {"case_id": "vi-01", "category": "vietnamese", "passed": True, "score": 1.0},
                ],
            },
            {
                "checkpoint_id": "c1",
                "version_label": "Quantized",
                "pass_rate": 0.0,
                "mean_score": 0.2,
                "categories": {
                    "code_preservation": {"pass_rate": 0.0, "mean_score": 0.1, "sample_size": 3},
                    "vietnamese": {"pass_rate": 0.0, "mean_score": 0.3, "sample_size": 3},
                },
                "case_results": [
                    {"case_id": "code-01", "category": "code_preservation", "passed": False, "score": 0.1},
                    {"case_id": "vi-01", "category": "vietnamese", "passed": False, "score": 0.3},
                ],
            },
        ],
        "consecutive_transitions": [
            {
                "source_checkpoint_id": "c0",
                "target_checkpoint_id": "c1",
                "transformation_type": "Quantization",
                "sample_size": 6,
                "newly_introduced_failures": ["code-01", "vi-01"],
                "paired_statistics": {
                    "mcnemar_p_value": 0.031,
                    "bootstrap_ci95_low": -0.9,
                    "bootstrap_ci95_high": -0.5,
                },
                "category_deltas": [
                    {
                        "category": "code_preservation",
                        "sample_size": 3,
                        "mean_score_before": 1.0,
                        "mean_score_after": 0.1,
                        "pass_rate_before": 1.0,
                        "pass_rate_after": 0.0,
                        "mcnemar_p_value": 0.25,
                        "wilson_ci95_after": [0.0, 0.56],
                    },
                    {
                        "category": "vietnamese",
                        "sample_size": 3,
                        "mean_score_before": 1.0,
                        "mean_score_after": 0.3,
                        "pass_rate_before": 1.0,
                        "pass_rate_after": 0.0,
                        "mcnemar_p_value": 0.25,
                        "wilson_ci95_after": [0.0, 0.56],
                    },
                ],
            }
        ],
    }

    # Case A: Pure behavioral regression with ZERO tensor/representation covariates -> OBSERVED
    engine = RegressionOriginEngine()
    ev_obs = engine.detect(
        dag=dag,
        behavioral_evolution=beh_payload,
        tensor_transitions=[],
        representation_evolution=None,
    )
    assert all(e["epistemic_status"] == EpistemicStatus.OBSERVED.value for e in ev_obs)

    # Case B: Diagnostic transition patching recovered 65% logit margin -> CAUSAL SUPPORT (NOT CAUSALLY VERIFIED!)
    rep_with_diag_patch = {
        "consecutive_transitions": [
            {
                "source_checkpoint_id": "c0",
                "target_checkpoint_id": "c1",
                "mean_cka": 0.84,
                "mean_drift": 0.16,
                "cka_by_layer": {"0": 0.9, "1": 0.78},
                "causal_patching": {
                    "most_causal_layer": 1,
                    "peak_recovery_fraction": 0.65,
                    "scope": "diagnostic prompt only",
                },
            }
        ]
    }
    ev_support = engine.detect(
        dag=dag,
        behavioral_evolution=beh_payload,
        tensor_transitions=[],
        representation_evolution=rep_with_diag_patch,
    )
    for e in ev_support:
        assert e["epistemic_status"] == EpistemicStatus.CAUSAL_SUPPORT.value
        assert e["causal_verification"]["causal_verification_exists"] is False
        assert e["causal_verification"]["causal_support_exists"] is True

    # Case C: Case-targeted counterfactual patching verified ONLY on `code-01` (with control_verified=True)
    # -> `code_preservation` (and `overall`) get CAUSALLY VERIFIED, while `vietnamese` stays CAUSAL SUPPORT!
    ev_targeted = engine.detect(
        dag=dag,
        behavioral_evolution=beh_payload,
        tensor_transitions=[],
        representation_evolution=rep_with_diag_patch,
        case_interventions={
            ("c0", "c1", "code-01"): {
                "most_causal_layer": 1,
                "peak_recovery_fraction": 0.72,
                "clean_logit_diff": 4.2,
                "corrupted_logit_diff": -1.1,
                "control_verified": True,
            }
        },
    )
    by_cat = {e["affected_category"]: e for e in ev_targeted}
    assert by_cat["code_preservation"]["epistemic_status"] == EpistemicStatus.CAUSALLY_VERIFIED.value
    assert by_cat["code_preservation"]["causal_verification"]["causal_verification_exists"] is True
    assert by_cat["code_preservation"]["causal_verification"]["verified_case_ids"] == ["code-01"]
    # `vietnamese` was NOT case-patched -> MUST NOT inherit CAUSALLY VERIFIED!
    assert by_cat["vietnamese"]["epistemic_status"] == EpistemicStatus.CAUSAL_SUPPORT.value
    assert by_cat["vietnamese"]["causal_verification"]["causal_verification_exists"] is False


@pytest.mark.unit
def test_small_sample_safeguards_holm_bonferroni_and_regression_kind_separation() -> None:
    """P1-2: Small samples (n=3 < 8) are exploratory & never statistically significant; Holm-Bonferroni holds."""
    raw = [0.01, 0.04, 0.03, 0.20]
    adj = holm_bonferroni_adjust(raw)
    # Sorted raw: 0.01 (*4=0.04), 0.03 (*3=0.09), 0.04 (*2=0.08 -> max(0.09,0.08)=0.09), 0.20 (*1=0.20)
    assert adj == [0.04, 0.09, 0.09, 0.2]

    c0 = _dummy_ckpt("c0", "SFT", 0)
    c1 = _dummy_ckpt("c1", "Merged", 1)
    rel = EvolutionRelationship("r01", "proj-rigor", "c0", "c1", TransformationType.MODEL_MERGING)
    dag = EvolutionDAG([c0, c1], [rel])

    # Score dropped 0.90 -> 0.65 while pass rate stayed 0.667 -> 0.667 on n=3 cases
    beh_score_only = {
        "checkpoint_evaluations": [
            {
                "checkpoint_id": "c0",
                "version_label": "SFT",
                "pass_rate": 0.666667,
                "mean_score": 0.90,
                "categories": {"json_reliability": {"pass_rate": 0.666667, "mean_score": 0.90, "sample_size": 3}},
                "case_results": [],
            },
            {
                "checkpoint_id": "c1",
                "version_label": "Merged",
                "pass_rate": 0.666667,
                "mean_score": 0.65,
                "categories": {"json_reliability": {"pass_rate": 0.666667, "mean_score": 0.65, "sample_size": 3}},
                "case_results": [],
            },
        ],
        "consecutive_transitions": [
            {
                "source_checkpoint_id": "c0",
                "target_checkpoint_id": "c1",
                "transformation_type": "Model Merging",
                "sample_size": 3,
                "newly_introduced_failures": [],
                "paired_statistics": {
                    "mcnemar_p_value": 0.001,  # Even if raw p were tiny, n=3 < 8 MUST block significance!
                    "bootstrap_ci95_low": -0.3,
                    "bootstrap_ci95_high": -0.2,
                },
                "category_deltas": [
                    {
                        "category": "json_reliability",
                        "sample_size": 3,
                        "mean_score_before": 0.90,
                        "mean_score_after": 0.65,
                        "pass_rate_before": 0.666667,
                        "pass_rate_after": 0.666667,
                        "mcnemar_p_value": 0.001,
                        "wilson_ci95_after": [0.20, 0.93],
                    }
                ],
            }
        ],
    }

    engine = RegressionOriginEngine(min_category_sample_size=8, min_overall_sample_size=20)
    evs = engine.detect(
        dag=dag,
        behavioral_evolution=beh_score_only,
        tensor_transitions=[],
        representation_evolution=None,
    )
    cat_ev = next(e for e in evs if e["affected_category"] == "json_reliability")
    assert cat_ev["regression_kind"] == "score_only"
    assert cat_ev["statistical_confidence"]["is_exploratory"] is True
    assert cat_ev["statistical_confidence"]["statistically_significant"] is False
    assert cat_ev["statistical_confidence"]["confidence_level"] == "exploratory_small_sample"


@pytest.mark.unit
def test_diamond_dag_over_128_branches_reports_truncation_and_misses_zero_edges() -> None:
    """P1-3: 8-stage diamond DAG has exact 2^8 = 256 branches; truncating at 32 warns & still scans all edges."""
    # Structure: s0 -> (a1, b1) -> m1 -> (a2, b2) -> m2 ... x 8 stages => 2^8 = 256 root-to-leaf paths!
    nodes: list[RegisteredCheckpoint] = [_dummy_ckpt("m0", "Root", 0)]
    rels: list[EvolutionRelationship] = []
    idx = 1
    for stage in range(1, 9):
        prev_m = f"m{stage - 1}"
        a_id = f"a{stage}"
        b_id = f"b{stage}"
        m_id = f"m{stage}"
        nodes.extend(
            [
                _dummy_ckpt(a_id, a_id, idx),
                _dummy_ckpt(b_id, b_id, idx + 1),
                _dummy_ckpt(m_id, m_id, idx + 2),
            ]
        )
        idx += 3
        rels.extend(
            [
                EvolutionRelationship(f"r-{prev_m}-{a_id}", "p", prev_m, a_id, TransformationType.SFT),
                EvolutionRelationship(f"r-{prev_m}-{b_id}", "p", prev_m, b_id, TransformationType.SFT),
                EvolutionRelationship(f"r-{a_id}-{m_id}", "p", a_id, m_id, TransformationType.MODEL_MERGING),
                EvolutionRelationship(f"r-{b_id}-{m_id}", "p", b_id, m_id, TransformationType.MODEL_MERGING),
            ]
        )

    dag = EvolutionDAG(nodes, rels)
    assert dag.count_total_branches() == 256

    meta_32 = dag.enumerate_branches_with_meta(max_paths=32)
    assert meta_32.total_branch_count == 256
    assert meta_32.analyzed_branch_count == 32
    assert meta_32.truncated is True

    g_payload = dag.to_graph_payload(max_branches=32)
    assert g_payload["branch_count"] == 256
    assert g_payload["analyzed_branch_count"] == 32
    assert g_payload["branches_truncated"] is True

    # Place a regression ONLY on `b8 -> m8` (which is only reached on the 256th DFS branch!)
    # With `max_branches=16`, the first 16 DFS branches all take `a8 -> m8` and never traverse `b8 -> m8`,
    # YET `RegressionOriginEngine`'s complete-DAG edge fallback MUST still detect `b8 -> m8`!
    evals = [
        {
            "checkpoint_id": n.checkpoint_id,
            "version_label": n.version_label,
            "pass_rate": 0.20 if n.checkpoint_id == "m8" else 0.90,
            "mean_score": 0.20 if n.checkpoint_id == "m8" else 0.90,
            "categories": {},
            "case_results": [],
        }
        for n in nodes
    ]
    beh_Payload = {
        "checkpoint_evaluations": evals,
        "consecutive_transitions": [
            {
                "source_checkpoint_id": "b8",
                "target_checkpoint_id": "m8",
                "transformation_type": "Model Merging",
                "sample_size": 24,
                "newly_introduced_failures": ["case-x"],
                "paired_statistics": {
                    "mcnemar_p_value": 0.002,
                    "bootstrap_ci95_low": -0.8,
                    "bootstrap_ci95_high": -0.6,
                },
                "category_deltas": [],
            }
        ],
    }
    reg_eng = RegressionOriginEngine(max_branches=16)
    found = reg_eng.detect(
        dag=dag,
        behavioral_evolution=beh_Payload,
        tensor_transitions=[],
        representation_evolution=None,
    )
    assert reg_eng.last_branch_enumeration is not None
    assert reg_eng.last_branch_enumeration.truncated is True
    assert any(
        e["previous_checkpoint_id"] == "b8" and e["first_affected_checkpoint_id"] == "m8"
        for e in found
    )


@pytest.mark.unit
def test_tokenizer_artifact_sha256_coverage_and_7_way_pair_scope_classification(
    tmp_path: Path,
) -> None:
    """P2-4 & P2-5: Tokenizer edits change SHA-256 + eval cache key; pair scopes cover all 7 relations."""
    ckpts = create_known_evolution_checkpoints(tmp_path)
    dir_tok_v1 = tmp_path / "tok_v1"
    dir_tok_v2 = tmp_path / "tok_v2"
    shutil.copytree(ckpts["a"], dir_tok_v1)
    shutil.copytree(ckpts["a"], dir_tok_v2)

    (dir_tok_v1 / "tokenizer.json").write_text(json.dumps({"vocab": {"hello": 1}}), encoding="utf-8")
    (dir_tok_v2 / "tokenizer.json").write_text(
        json.dumps({"vocab": {"hello": 1, "<|new_tok|>": 2}}), encoding="utf-8"
    )

    policy = SecurityPolicy(allowed_roots=[tmp_path])
    r1 = inspect_and_build_checkpoint_record(
        project_id="p-tok", source=dir_tok_v1, version_label="v1", policy=policy, local_files_only=True
    )
    r2 = inspect_and_build_checkpoint_record(
        project_id="p-tok",
        source=dir_tok_v2,
        version_label="v2",
        existing_project_checkpoints=[r1],
        policy=policy,
        local_files_only=True,
    )

    # Weights SHA-256 is identical, but tokenizer_sha256 and composite sha256_fingerprint MUST differ!
    assert r1.component_fingerprints["weights_sha256"] == r2.component_fingerprints["weights_sha256"]
    assert (
        r1.component_fingerprints["tokenizer_sha256"]
        != r2.component_fingerprints["tokenizer_sha256"]
    )
    assert r1.sha256_fingerprint != r2.sha256_fingerprint
    assert build_behavioral_checkpoint_cache_key(r1) != build_behavioral_checkpoint_cache_key(r2)
    assert r2.transformation_metadata.get("identical_weights_to") == r1.checkpoint_id

    # P2-5: Verify all 7 topological scopes on Base -> SFT -> DPO -> Quant, SFT_B, plus Disjoint Root
    store = EvolutionStore(db_path=tmp_path / "scopes.db")
    proj = store.create_project(name="Scopes")
    n_base = _dummy_ckpt("base", "Base", 0)
    n_sft = _dummy_ckpt("sft", "SFT", 1)
    n_dpo = _dummy_ckpt("dpo", "DPO", 2)
    n_quant = _dummy_ckpt("quant", "Quant", 3)
    n_branch = _dummy_ckpt("sft_b", "SFT-B", 4)
    n_disjoint = _dummy_ckpt("other_root", "OtherRoot", 5)

    dag = EvolutionDAG(
        [n_base, n_sft, n_dpo, n_quant, n_branch, n_disjoint],
        [
            EvolutionRelationship("e1", proj.project_id, "base", "sft", TransformationType.SFT),
            EvolutionRelationship("e2", proj.project_id, "sft", "dpo", TransformationType.DPO),
            EvolutionRelationship("e3", proj.project_id, "dpo", "quant", TransformationType.QUANTIZATION),
            EvolutionRelationship("e4", proj.project_id, "base", "sft_b", TransformationType.SFT),
        ],
    )
    assert dag.classify_pair_scope("sft", "sft", "base") == "identical"
    assert dag.classify_pair_scope("sft", "dpo", "base") == "consecutive"
    assert dag.classify_pair_scope("base", "quant", "base") == "base_to_current"
    assert dag.classify_pair_scope("sft", "quant", "base") == "ancestor_to_descendant"
    assert dag.classify_pair_scope("quant", "sft", "base") == "descendant_to_ancestor"
    assert dag.classify_pair_scope("dpo", "sft_b", "base") == "branch_to_branch"
    assert dag.classify_pair_scope("sft", "other_root", "base") == "unrelated"
