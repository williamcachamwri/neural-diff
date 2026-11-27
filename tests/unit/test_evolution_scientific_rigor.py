"""Focused scientific rigor & adversarial validation suite for Neural Evolution Tracking.

Covers all final audit requirements:
1. Causal Verification Provenance (`case_targeted`, `target_vs_foil_verified`,
   `negative_control_verified`, `verification_scope`, full provenance storage, missing-never-True,
   and strict case + intervention-site scoping).
2. Adversarial DAG Regression Fallback (multiple roots, multiple branch & merge points, >= 512
   root-to-leaf paths, single-branch regression, shared-branch regression, introduced-and-recovered
   regression + re-regression, cross-merge non-contamination, and divergent leaf outcomes).
3. Actual Filesystem 5-Component SHA-256 Validation & Behavioral Cache Invalidation (independent
   mutations of model weights, architecture config, tokenizer JSON, chat template, adapter weights,
   and tensor metadata, plus missing files, duplicate files, and creation-order-independent traversal).
4. Statistical Validation against independent `math.comb` reference formulas (raw McNemar p-values,
   Holm-Bonferroni adjusted p-values, stable ordering, tied p-values, empty hypothesis families,
   small-sample gates & binomial floor power limits, `score_only`, and `pass_rate_only` regressions).
5. Complete 7-way topological pair scope taxonomy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

from core.evolution.behavioral_evolution import (
    BehavioralEvolutionEngine,
    build_behavioral_checkpoint_cache_key,
    holm_bonferroni_adjust,
)
from core.evolution.dag import EvolutionDAG
from core.evolution.registry import (
    compute_checkpoint_component_fingerprints,
    inspect_and_build_checkpoint_record,
    revalidate_checkpoint_health,
)
from core.evolution.regression_origin import (
    RegressionOriginEngine,
    validate_causal_verification_provenance,
)
from core.evolution.store import EvolutionStore
from core.evolution.types import (
    CausalInterventionProvenance,
    CheckpointValidationStatus,
    EvolutionRelationship,
    RegisteredCheckpoint,
    TransformationType,
)
from core.security import SecurityPolicy
from core.statistics.significance import compute_paired_significance
from core.types import EpistemicStatus


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
        layer_count=4,
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
        created_at_utc=f"2026-11-27T00:{(idx // 60) % 60:02d}:{idx % 60:02d}+00:00",
        framework_info={},
        tokenizer_metadata={},
        quantization_metadata={},
        training_metadata={},
        lora_metadata={},
        experiment_notes="",
        validation_status=CheckpointValidationStatus.VERIFIED,
    )


def _reference_mcnemar_two_sided(passed_a: list[bool], passed_b: list[bool]) -> float:
    """Independent exact two-sided binomial McNemar reference using `math.comb` only."""
    b_only = sum(1 for pa, pb in zip(passed_a, passed_b, strict=True) if (not pa) and pb)
    a_only = sum(1 for pa, pb in zip(passed_a, passed_b, strict=True) if pa and (not pb))
    n = b_only + a_only
    if n == 0:
        return 1.0
    k = min(b_only, a_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return min(1.0, 2.0 * tail)


def _reference_holm_bonferroni(raw_ps: list[float]) -> list[float]:
    """Independent textbook Holm (1979) step-down reference implementation."""
    m = len(raw_ps)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda idx: (raw_ps[idx], idx))
    out = [0.0] * m
    for r, idx in enumerate(order):
        # Adjusted p_(r) = max_{j <= r} min(1, (m - j) * p_(j))
        val = max(min(1.0, (m - j) * raw_ps[order[j]]) for j in range(r + 1))
        out[idx] = round(val, 6)
    return out


# ==============================================================================
# 1. CAUSAL VERIFICATION PROVENANCE TESTS
# ==============================================================================


@pytest.mark.unit
def test_causal_verification_provenance_strict_conditions_and_case_site_scoping() -> None:
    """Audit Item 1: All 4 provenance conditions must be explicitly True; missing never defaults True."""
    code_prompt = "Write a Python function fib(n) returning the nth Fibonacci number."
    vi_prompt = "Giải thích ngắn gọn định lý Pythagoras bằng tiếng Việt."
    code_fp = hashlib.sha256(code_prompt.encode("utf-8")).hexdigest()
    vi_fp = hashlib.sha256(vi_prompt.encode("utf-8")).hexdigest()

    valid_code_record: dict[str, Any] = {
        "case_targeted": True,
        "target_vs_foil_verified": True,
        "negative_control_verified": True,
        "verification_scope": "case:code-01|layer:2|intervention:residual_stream_patch",
        "prompt_fingerprint": code_fp,
        "failed_case_id": "code-01",
        "layer_index": 2,
        "intervention_type": "residual_stream_patch",
        "target_token": "def",
        "foil_token": "class",
        "negative_control_result": {
            "control_layer_index": 0,
            "control_recovery_fraction": 0.03,
        },
        "recovery_measurement": 0.74,
        "clean_logit_diff": 4.5,
        "corrupted_logit_diff": -1.2,
    }

    # Baseline valid check
    ok, norm, errs = validate_causal_verification_provenance(
        valid_code_record,
        expected_case_id="code-01",
        expected_prompt_fingerprints={code_fp},
    )
    assert ok is True
    assert errs == []
    assert norm["case_targeted"] is True
    assert norm["target_vs_foil_verified"] is True
    assert norm["negative_control_verified"] is True
    assert norm["verification_scope_valid"] is True
    assert norm["prompt_fingerprint"] == code_fp
    assert norm["failed_case_id"] == "code-01"
    assert norm["layer_index"] == 2
    assert norm["intervention_type"] == "residual_stream_patch"
    assert norm["target_token"] == "def"
    assert norm["foil_token"] == "class"
    assert norm["negative_control_result"] == {
        "control_layer_index": 0,
        "control_recovery_fraction": 0.03,
    }
    assert norm["recovery_measurement"] == 0.74

    # Every single missing, truthy-string, or invalid provenance field MUST fail (never default True!)
    invalid_mutations: list[dict[str, Any]] = [
        {k: v for k, v in valid_code_record.items() if k != "case_targeted"},
        {**valid_code_record, "case_targeted": 1},  # Non-bool truthy int rejected
        {k: v for k, v in valid_code_record.items() if k != "target_vs_foil_verified"},
        {**valid_code_record, "target_vs_foil_verified": "true"},  # Non-bool string rejected
        {k: v for k, v in valid_code_record.items() if k != "negative_control_verified"},
        {k: v for k, v in valid_code_record.items() if k != "verification_scope"},
        {**valid_code_record, "verification_scope": ""},
        {**valid_code_record, "verification_scope": "diagnostic_prompt_transition_only"},
        {**valid_code_record, "verification_scope": "category"},
        {**valid_code_record, "verification_scope": "case:vi-01|layer:2"},  # Wrong case scope
        {**valid_code_record, "verification_scope": "case:code-01|layer:3"},  # Wrong layer scope
        {**valid_code_record, "failed_case_id": "vi-01"},
        {k: v for k, v in valid_code_record.items() if k != "prompt_fingerprint"},
        {**valid_code_record, "prompt_fingerprint": vi_fp},  # Wrong prompt hash
        {k: v for k, v in valid_code_record.items() if k != "layer_index"},
        {**valid_code_record, "layer_index": True},  # Bool disguised as int
        {k: v for k, v in valid_code_record.items() if k != "intervention_type"},
        {k: v for k, v in valid_code_record.items() if k != "target_token"},
        {**valid_code_record, "foil_token": "def"},  # Identical target & foil
        {k: v for k, v in valid_code_record.items() if k != "negative_control_result"},
        {
            **valid_code_record,
            "negative_control_result": {"control_layer_index": 0, "control_recovery_fraction": 0.45},
        },
        {k: v for k, v in valid_code_record.items() if k != "recovery_measurement"},
        {**valid_code_record, "recovery_measurement": 0.12},  # Below 0.20 recovery floor
        # Legacy `control_verified=True` payload without explicit 4-condition provenance MUST fail:
        {
            "most_causal_layer": 2,
            "peak_recovery_fraction": 0.80,
            "clean_logit_diff": 4.5,
            "corrupted_logit_diff": -1.2,
            "control_verified": True,
        },
    ]
    for mutated in invalid_mutations:
        is_ok, norm_mut, mut_errs = validate_causal_verification_provenance(
            mutated,
            expected_case_id="code-01",
            expected_prompt_fingerprints={code_fp},
        )
        assert is_ok is False, f"Expected failure for {mutated}, got valid!"
        assert norm_mut["is_causally_verified"] is False
        assert len(mut_errs) >= 1

    # End-to-end event scoping in RegressionOriginEngine
    c0 = _dummy_ckpt("c0", "Base", 0)
    c1 = _dummy_ckpt("c1", "Quantized", 1)
    dag = EvolutionDAG(
        [c0, c1],
        [EvolutionRelationship("r01", "proj-rigor", "c0", "c1", TransformationType.QUANTIZATION)],
    )
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
                    {
                        "case_id": "code-01",
                        "category": "code_preservation",
                        "prompt": code_prompt,
                        "passed": True,
                        "score": 1.0,
                    },
                    {
                        "case_id": "vi-01",
                        "category": "vietnamese",
                        "prompt": vi_prompt,
                        "passed": True,
                        "score": 1.0,
                    },
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
                    {
                        "case_id": "code-01",
                        "category": "code_preservation",
                        "prompt": code_prompt,
                        "passed": False,
                        "score": 0.1,
                    },
                    {
                        "case_id": "vi-01",
                        "category": "vietnamese",
                        "prompt": vi_prompt,
                        "passed": False,
                        "score": 0.3,
                    },
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
                    "mcnemar_p_value": 0.03125,
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

    rep_with_diag = {
        "consecutive_transitions": [
            {
                "source_checkpoint_id": "c0",
                "target_checkpoint_id": "c1",
                "mean_cka": 0.84,
                "mean_drift": 0.16,
                "cka_by_layer": {"0": 0.9, "2": 0.75},
                "causal_patching": {
                    "case_targeted": False,
                    "verification_scope": "diagnostic_prompt_transition_only",
                    "most_causal_layer": 2,
                    "peak_recovery_fraction": 0.65,
                },
            }
        ]
    }

    engine = RegressionOriginEngine()

    # Only `code-01` verified -> `code_preservation` is CAUSALLY VERIFIED (strictly scoped to `code-01` @ L2),
    # while `vietnamese` AND `overall` (which includes unverified `vi-01`) MUST stay CAUSAL SUPPORT!
    ev_partial = engine.detect(
        dag=dag,
        behavioral_evolution=beh_payload,
        tensor_transitions=[],
        representation_evolution=rep_with_diag,
        case_interventions={("c0", "c1", "code-01"): valid_code_record},
    )
    by_cat = {e["affected_category"]: e for e in ev_partial}

    code_ev = by_cat["code_preservation"]
    assert code_ev["epistemic_status"] == EpistemicStatus.CAUSALLY_VERIFIED.value
    cv = code_ev["causal_verification"]
    assert cv["causal_verification_exists"] is True
    assert cv["case_targeted"] is True
    assert cv["target_vs_foil_verified"] is True
    assert cv["negative_control_verified"] is True
    assert cv["verification_scope"] == (
        f"case:code-01|prompt_sha256:{code_fp[:16]}|layer:2|intervention:residual_stream_patch"
    )
    assert cv["prompt_fingerprint"] == code_fp
    assert cv["failed_case_id"] == "code-01"
    assert cv["layer_index"] == 2
    assert cv["intervention_type"] == "residual_stream_patch"
    assert cv["target_token"] == "def"
    assert cv["foil_token"] == "class"
    assert cv["recovery_measurement"] == 0.74

    assert by_cat["vietnamese"]["epistemic_status"] == EpistemicStatus.CAUSAL_SUPPORT.value
    assert by_cat["vietnamese"]["causal_verification"]["causal_verification_exists"] is False

    # `overall` has `["code-01", "vi-01"]` -> partial verification MUST NOT elevate `overall` to CAUSALLY VERIFIED
    assert by_cat["overall"]["epistemic_status"] == EpistemicStatus.CAUSAL_SUPPORT.value
    assert by_cat["overall"]["causal_verification"]["causal_verification_exists"] is False
    assert by_cat["overall"]["causal_verification"]["partially_verified"] is True
    assert by_cat["overall"]["causal_verification"]["verified_case_ids"] == ["code-01"]
    assert by_cat["overall"]["causal_verification"]["unverified_case_ids"] == ["vi-01"]

    # Once `vi-01` ALSO receives independent valid provenance, `overall` becomes CAUSALLY VERIFIED
    valid_vi_prov = CausalInterventionProvenance(
        case_targeted=True,
        target_vs_foil_verified=True,
        negative_control_verified=True,
        verification_scope="case_and_intervention_site",
        prompt_fingerprint=vi_fp,
        failed_case_id="vi-01",
        layer_index=3,
        intervention_type="mlp_out_patch",
        target_token="Pythagoras",
        foil_token="Newton",
        negative_control_result={"control_layer_index": 1, "control_recovery_fraction": 0.02},
        recovery_measurement=0.68,
        clean_logit_diff=3.8,
        corrupted_logit_diff=-0.9,
    )
    ev_full = engine.detect(
        dag=dag,
        behavioral_evolution=beh_payload,
        tensor_transitions=[],
        representation_evolution=rep_with_diag,
        case_interventions={
            ("c0", "c1", "code-01"): valid_code_record,
            ("c0", "c1", "vi-01"): valid_vi_prov,
        },
    )
    full_by_cat = {e["affected_category"]: e for e in ev_full}
    assert full_by_cat["overall"]["epistemic_status"] == EpistemicStatus.CAUSALLY_VERIFIED.value
    assert full_by_cat["vietnamese"]["epistemic_status"] == EpistemicStatus.CAUSALLY_VERIFIED.value


# ==============================================================================
# 2. ADVERSARIAL DAG REGRESSION FALLBACK TESTS
# ==============================================================================


@pytest.mark.unit
def test_adversarial_multi_root_multi_merge_512_path_dag_fallback_and_recovery() -> None:
    """Audit Item 2: Multi-root, multi-branch, multi-merge >=256-path DAG fallback & recovery."""
    nodes: list[RegisteredCheckpoint] = []
    rels: list[EvolutionRelationship] = []
    idx = 0

    def _add_node(nid: str) -> None:
        nonlocal idx
        nodes.append(_dummy_ckpt(nid, nid, idx))
        idx += 1

    def _add_edge(u: str, v: str, ttype: TransformationType = TransformationType.SFT) -> None:
        rels.append(EvolutionRelationship(f"rel-{u}-{v}", "proj-rigor", u, v, ttype))

    # 3 distinct roots (`r1`, `r2` merging into diamond cascade `m0`, plus disjoint clean root `r3`)
    for nid in ("r1", "u1", "r2", "v1", "v1_isolated_leaf", "m0", "r3", "r3_leaf"):
        _add_node(nid)
    _add_edge("r1", "u1", TransformationType.SFT)
    _add_edge("r2", "v1", TransformationType.SFT)
    _add_edge("v1", "v1_isolated_leaf", TransformationType.PRUNING)
    _add_edge("u1", "m0", TransformationType.MODEL_MERGING)
    _add_edge("v1", "m0", TransformationType.MODEL_MERGING)
    _add_edge("r3", "r3_leaf", TransformationType.QUANTIZATION)

    # 7-stage diamond cascade from `m0` -> `m7` (2 roots * 2^7 = 256 paths at `m7`)
    for s in range(1, 8):
        prev_m = f"m{s - 1}"
        a_id, b_id, m_id = f"a{s}", f"b{s}", f"m{s}"
        for nid in (a_id, b_id, m_id):
            _add_node(nid)
        _add_edge(prev_m, a_id, TransformationType.SFT)
        _add_edge(prev_m, b_id, TransformationType.DPO)
        _add_edge(a_id, m_id, TransformationType.MODEL_MERGING)
        _add_edge(b_id, m_id, TransformationType.MODEL_MERGING)

    # Diverging leaf subtrees after `m7` -> 256 * 3 = 768 paths + 1 (`v1_isolated_leaf`) + 1 (`r3_leaf`) = 770 paths!
    for nid in ("sub_heal", "leaf_healed", "leaf_broken", "b_tail", "leaf_tail_only"):
        _add_node(nid)
    _add_edge("m7", "sub_heal", TransformationType.SFT)
    _add_edge("sub_heal", "leaf_healed", TransformationType.DPO)
    _add_edge("m7", "leaf_broken", TransformationType.QUANTIZATION)
    _add_edge("m7", "b_tail", TransformationType.PRUNING)
    _add_edge("b_tail", "leaf_tail_only", TransformationType.PRUNING)

    dag = EvolutionDAG(nodes, rels)
    total_paths = dag.count_total_branches()
    assert total_paths == 770  # >= 256 paths across 3 roots, 10 branch points, 8 merge points
    assert len(dag.roots()) == 3
    assert len(dag.leaves()) == 5

    # Construct category scores testing all adversarial scenarios simultaneously:
    # 1. `single_branch_cat`: regresses ONLY on `b_tail -> leaf_tail_only` (the very last DFS leaf!).
    # 2. `shared_multi_leaf_cat`: regresses on shared trunk `m2 -> a3` AND `m2 -> b3`? Wait: let's
    #    regress `shared_multi_leaf_cat` on `a3 -> m3` & `b3 -> m3` OR on `m2 -> a3` (`0.95 -> 0.50`)
    #    where `m3..m7` stay `0.50`, `leaf_broken` & `leaf_tail_only` stay `0.50` ("regressed"),
    #    while `sub_heal -> leaf_healed` RECOVERS back to `0.95` ("recovered") -> different leaf outcomes!
    # 3. `recover_then_regress_cat`: regresses on `r1 -> u1` (`0.95 -> 0.45`), RECOVERS at merge `m0`
    #    (`0.95`), stays `0.95` until `m5 -> b6` where it REGRESSES AGAIN (`0.95 -> 0.40`), then
    #    recovers at `m6` (`0.95`)!
    # 4. `cross_merge_isolation_cat`: regresses on `r1 -> u1` (`0.95 -> 0.50`) and stays `0.78` at `m0`
    #    (still regressed relative to `r1=0.95` since `0.95 - 0.78 = 0.17 >= 0.10`), whereas root `r2`
    #    starts at `0.82 -> v1 (0.82) -> m0 (0.78)` (drop `0.04 < 0.10`, NEVER regressed on `r2`!).
    #    Then `m0 (0.78) -> a1 (0.40)` drops `0.38`: MUST be detected via `r2`'s clean lineage even
    #    when `max_branches=8` truncates all `r2` paths (since `r1` alone has 384 paths before `r2` is reached)!
    cats = [
        "single_branch_cat",
        "shared_multi_leaf_cat",
        "recover_then_regress_cat",
        "cross_merge_isolation_cat",
    ]

    node_cat_scores: dict[str, dict[str, float]] = {
        n.checkpoint_id: dict.fromkeys(cats, 0.95) for n in nodes
    }

    # Scenario 1: regression ONLY on `v1 -> v1_isolated_leaf` (under root `r2` -> path #385,
    # completely beyond `max_branches=8`!)
    node_cat_scores["v1_isolated_leaf"]["single_branch_cat"] = 0.35

    # Scenario 2: shared regression introduced on `m2 -> a3`, persisting through `m3..m7`,
    # surviving on `leaf_broken` & `leaf_tail_only`, but RECOVERING on `sub_heal` & `leaf_healed`!
    node_cat_scores["a3"]["shared_multi_leaf_cat"] = 0.50
    for downstream_id in (
        "m3",
        "a4",
        "b4",
        "m4",
        "a5",
        "b5",
        "m5",
        "a6",
        "b6",
        "m6",
        "a7",
        "b7",
        "m7",
        "leaf_broken",
        "b_tail",
        "leaf_tail_only",
    ):
        node_cat_scores[downstream_id]["shared_multi_leaf_cat"] = 0.50
    node_cat_scores["sub_heal"]["shared_multi_leaf_cat"] = 0.95
    node_cat_scores["leaf_healed"]["shared_multi_leaf_cat"] = 0.95

    # Scenario 3: introduced on `r1 -> u1`, recovered at `m0`, re-introduced on `m5 -> b6`, recovered at `m6`
    node_cat_scores["u1"]["recover_then_regress_cat"] = 0.45
    node_cat_scores["b6"]["recover_then_regress_cat"] = 0.40

    # Scenario 4: cross-merge lineage isolation (`r1` regresses at `r1->u1` and stays `0.78` at `m0`;
    # `r2=0.82 -> v1=0.82 -> m0=0.78` is clean; `m0=0.78 -> a1=0.40` regresses on `r2`'s lineage!)
    node_cat_scores["r1"]["cross_merge_isolation_cat"] = 0.95
    node_cat_scores["u1"]["cross_merge_isolation_cat"] = 0.50
    node_cat_scores["r2"]["cross_merge_isolation_cat"] = 0.82
    node_cat_scores["v1"]["cross_merge_isolation_cat"] = 0.82
    node_cat_scores["m0"]["cross_merge_isolation_cat"] = 0.78
    node_cat_scores["a1"]["cross_merge_isolation_cat"] = 0.40

    checkpoint_evals: list[dict[str, Any]] = []
    for n in nodes:
        cid = n.checkpoint_id
        c_map = node_cat_scores[cid]
        mean_sc = round(sum(c_map.values()) / len(cats), 6)
        checkpoint_evals.append(
            {
                "checkpoint_id": cid,
                "version_label": cid,
                "pass_rate": mean_sc,
                "mean_score": mean_sc,
                "categories": {
                    cat: {"pass_rate": sc, "mean_score": sc, "sample_size": 12}
                    for cat, sc in c_map.items()
                },
                "case_results": [],
            }
        )

    eval_by_id = {ev["checkpoint_id"]: ev for ev in checkpoint_evals}
    consecutive_transitions: list[dict[str, Any]] = []
    for rel in rels:
        u, v = rel.parent_checkpoint_id, rel.child_checkpoint_id
        ua, vb = eval_by_id[u], eval_by_id[v]
        consecutive_transitions.append(
            {
                "source_checkpoint_id": u,
                "target_checkpoint_id": v,
                "transformation_type": rel.transformation_type.value,
                "sample_size": 48,
                "newly_introduced_failures": [],
                "paired_statistics": {
                    "mcnemar_p_value": 0.01,
                    "bootstrap_ci95_low": -0.4,
                    "bootstrap_ci95_high": -0.1,
                },
                "category_deltas": [
                    {
                        "category": cat,
                        "sample_size": 12,
                        "mean_score_before": ua["categories"][cat]["mean_score"],
                        "mean_score_after": vb["categories"][cat]["mean_score"],
                        "pass_rate_before": ua["categories"][cat]["pass_rate"],
                        "pass_rate_after": vb["categories"][cat]["pass_rate"],
                        "mcnemar_p_value": 0.01,
                        "wilson_ci95_after": [0.2, 0.8],
                    }
                    for cat in cats
                ],
            }
        )

    beh_payload = {
        "checkpoint_evaluations": checkpoint_evals,
        "consecutive_transitions": consecutive_transitions,
    }

    # Run with aggressive truncation (`max_branches=8`, < 1.1% of the 769 paths!) vs full (`max_branches=1000`)
    eng_trunc = RegressionOriginEngine(max_branches=8)
    ev_trunc = eng_trunc.detect(
        dag=dag,
        behavioral_evolution=beh_payload,
        tensor_transitions=[],
        representation_evolution=None,
    )
    assert eng_trunc.last_branch_enumeration is not None
    assert eng_trunc.last_branch_enumeration.truncated is True
    assert eng_trunc.last_branch_enumeration.total_branch_count == 770
    assert eng_trunc.last_branch_enumeration.analyzed_branch_count == 8

    eng_full = RegressionOriginEngine(max_branches=1000)
    ev_full = eng_full.detect(
        dag=dag,
        behavioral_evolution=beh_payload,
        tensor_transitions=[],
        representation_evolution=None,
    )
    assert eng_full.last_branch_enumeration is not None
    assert eng_full.last_branch_enumeration.truncated is False

    # 1. Zero missed transitions & zero duplicate events: origin triples match 100% between truncated & full!
    trunc_keys = [
        (e["previous_checkpoint_id"], e["first_affected_checkpoint_id"], e["affected_category"])
        for e in ev_trunc
    ]
    full_keys = [
        (e["previous_checkpoint_id"], e["first_affected_checkpoint_id"], e["affected_category"])
        for e in ev_full
    ]
    assert len(trunc_keys) == len(set(trunc_keys)), "Duplicate regression origin events emitted!"
    assert set(trunc_keys) == set(full_keys)

    by_origin = {
        (e["previous_checkpoint_id"], e["first_affected_checkpoint_id"], e["affected_category"]): e
        for e in ev_trunc
    }

    # 2. Single-branch regression at path #385 (`v1 -> v1_isolated_leaf`) detected under `max_branches=8`!
    ev_single = by_origin[("v1", "v1_isolated_leaf", "single_branch_cat")]
    assert ev_single["discovered_via_fallback"] is True
    assert ev_single["survives_to_leaf"] is True
    assert ev_single["surviving_leaf_ids"] == ["v1_isolated_leaf"]
    assert ev_single["recovered_leaf_ids"] == []
    assert "r3_leaf" not in ev_single["leaf_outcomes"]  # Disjoint root `r3` never mixed in!

    # 3. Shared regression (`m2 -> a3`) emitted ONCE and records divergent leaf outcomes accurately!
    ev_shared = by_origin[("m2", "a3", "shared_multi_leaf_cat")]
    assert ev_shared["survives_to_leaf"] is True
    assert sorted(ev_shared["surviving_leaf_ids"]) == ["leaf_broken", "leaf_tail_only"]
    assert ev_shared["recovered_leaf_ids"] == ["leaf_healed"]
    assert ev_shared["leaf_outcomes"] == {
        "leaf_healed": "recovered",
        "leaf_broken": "regressed",
        "leaf_tail_only": "regressed",
    }
    # Continuation edges (`a3 -> m3 -> a4 ...`) MUST NOT be emitted as duplicate origins for `m2->a3`,
    # whereas `b3 -> m3` (0.95 -> 0.50 on the clean `m2 -> b3` sibling path) IS properly detected!
    assert ("a3", "m3", "shared_multi_leaf_cat") not in by_origin
    assert ("b3", "m3", "shared_multi_leaf_cat") in by_origin

    # 4. Regression introduced (`r1 -> u1`), recovered (`m0`), and subsequently re-introduced (`m5 -> b6`)!
    ev_first = by_origin[("r1", "u1", "recover_then_regress_cat")]
    assert "m0" in ev_first["recovered_at_checkpoint_ids"]
    assert ev_first["survives_to_leaf"] is False
    ev_reintro = by_origin[("m5", "b6", "recover_then_regress_cat")]
    assert "m6" in ev_reintro["recovered_at_checkpoint_ids"]
    assert ev_reintro["survives_to_leaf"] is False

    # 5. Cross-merge history non-contamination: `m0 -> a1` detected via `r2`'s clean lineage even when
    #    `max_branches=8` only enumerated `r1` paths (where `r1 -> u1 -> m0` was still regressed)!
    assert ("r1", "u1", "cross_merge_isolation_cat") in by_origin
    assert ("m0", "a1", "cross_merge_isolation_cat") in by_origin
    assert by_origin[("m0", "a1", "cross_merge_isolation_cat")]["discovered_via_fallback"] is True


# ==============================================================================
# 3. FILESYSTEM 5-COMPONENT FINGERPRINT & CACHE INVALIDATION TESTS
# ==============================================================================


def _seed_full_hf_checkpoint_dir(target_dir: Path) -> None:
    """Create a complete HuggingFace-style checkpoint directory with all 5 artifact domains."""
    target_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        "model.embed_tokens.weight": torch.arange(64, dtype=torch.float32).reshape(16, 4),
        "model.layers.0.mlp.down_proj.weight": torch.linspace(-0.5, 0.5, steps=16, dtype=torch.float32).reshape(4, 4),
        "lm_head.weight": torch.ones((16, 4), dtype=torch.float32) * 0.25,
    }
    save_file(
        tensors,
        str(target_dir / "model.safetensors"),
        metadata={"format": "pt", "training_stage": "sft_v1"},
    )
    (target_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "vocab_size": 16,
                "rms_norm_eps": 1e-5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (target_dir / "generation_config.json").write_text(
        json.dumps({"bos_token_id": 1, "eos_token_id": 2}), encoding="utf-8"
    )
    (target_dir / "tokenizer.json").write_text(
        json.dumps({"version": "1.0", "model": {"vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3}}}),
        encoding="utf-8",
    )
    (target_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": 512,
                "chat_template": "{% for m in messages %}{{ m['content'] }}{% endfor %}",
            }
        ),
        encoding="utf-8",
    )
    (target_dir / "chat_template.jinja").write_text(
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}",
        encoding="utf-8",
    )
    (target_dir / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 4, "lora_alpha": 8, "target_modules": ["down_proj"]}),
        encoding="utf-8",
    )
    adapter_tensors = {
        "base_model.model.layers.0.mlp.down_proj.lora_A.weight": torch.full((2, 4), 0.1, dtype=torch.float32),
        "base_model.model.layers.0.mlp.down_proj.lora_B.weight": torch.full((4, 2), 0.2, dtype=torch.float32),
    }
    save_file(adapter_tensors, str(target_dir / "adapter_model.safetensors"))


@pytest.mark.unit
def test_filesystem_5_component_fingerprints_cache_invalidation_missing_duplicates_and_traversal(
    tmp_path: Path,
) -> None:
    """Audit Item 3: Independent mutations across all 6 artifact types + missing, duplicates, traversal."""
    policy = SecurityPolicy(allowed_roots=[tmp_path])
    store = EvolutionStore(tmp_path / "fp_cache.db")

    base_dir = tmp_path / "ckpt_base"
    _seed_full_hf_checkpoint_dir(base_dir)

    base_rec = inspect_and_build_checkpoint_record(
        project_id="proj-fp",
        source=base_dir,
        version_label="base",
        policy=policy,
        local_files_only=True,
    )
    base_comp = base_rec.component_fingerprints
    for key in (
        "weights_sha256",
        "architecture_config_sha256",
        "tokenizer_sha256",
        "adapter_sha256",
        "metadata_sha256",
    ):
        assert isinstance(base_comp[key], str) and len(str(base_comp[key])) == 64

    # Seed actual SQLite behavioral evaluation cache entry for `base_rec`
    suite_fp = "suite_sha256_v1"
    inf_fp = "inf_sha256_v1"
    base_cache_key = build_behavioral_checkpoint_cache_key(base_rec)
    store.save_cached_checkpoint_eval(
        base_cache_key, suite_fp, inf_fp, {"pass_rate": 0.75, "checkpoint_id": base_rec.checkpoint_id}
    )
    assert store.get_cached_checkpoint_eval(base_cache_key, suite_fp, inf_fp) is not None

    def _clone_and_verify_single_component_change(
        name: str,
        mutate_fn: Any,
        expected_changed_key: str,
    ) -> None:
        cand_dir = tmp_path / f"mut_{name}"
        shutil.copytree(base_dir, cand_dir)
        mutate_fn(cand_dir)

        mut_rec = inspect_and_build_checkpoint_record(
            project_id="proj-fp",
            source=cand_dir,
            version_label=name,
            policy=policy,
            local_files_only=True,
        )
        # 1. Exact expected component fingerprint changed; ALL other 4 component hashes stay identical!
        for k in (
            "weights_sha256",
            "architecture_config_sha256",
            "tokenizer_sha256",
            "adapter_sha256",
            "metadata_sha256",
        ):
            if k == expected_changed_key:
                assert mut_rec.component_fingerprints[k] != base_comp[k], (
                    f"{name}: expected {k} to change"
                )
            else:
                assert mut_rec.component_fingerprints[k] == base_comp[k], (
                    f"{name}: unexpected mutation in {k}"
                )

        # 2. Composite SHA-256 & behavioral cache key invalidated (SQLite cache miss!)
        assert mut_rec.sha256_fingerprint != base_rec.sha256_fingerprint
        mut_cache_key = build_behavioral_checkpoint_cache_key(mut_rec)
        assert mut_cache_key != base_cache_key
        assert store.get_cached_checkpoint_eval(mut_cache_key, suite_fp, inf_fp) is None

        # 3. Tampering `base_dir` copy in-place is caught by `revalidate_checkpoint_health`
        tamper_dir = tmp_path / f"tamper_{name}"
        shutil.copytree(base_dir, tamper_dir)
        tamper_rec = inspect_and_build_checkpoint_record(
            project_id="proj-fp",
            source=tamper_dir,
            version_label=f"t_{name}",
            policy=policy,
            local_files_only=True,
        )
        mutate_fn(tamper_dir)
        st, err = revalidate_checkpoint_health(tamper_rec, policy=policy)
        assert st == CheckpointValidationStatus.CORRUPTED_OR_MISSING
        assert err is not None and expected_changed_key in err

    # (a) Modify Model weights ONLY
    def _mut_weights(d: Path) -> None:
        ts = load_file(str(d / "model.safetensors"))
        ts["model.layers.0.mlp.down_proj.weight"] = ts["model.layers.0.mlp.down_proj.weight"] + 0.125
        save_file(
            ts,
            str(d / "model.safetensors"),
            metadata={"format": "pt", "training_stage": "sft_v1"},
        )

    _clone_and_verify_single_component_change("model_weights", _mut_weights, "weights_sha256")

    # (b) Modify Architecture configuration ONLY (`config.json`)
    def _mut_arch_cfg(d: Path) -> None:
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        cfg["rms_norm_eps"] = 1e-6
        (d / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    _clone_and_verify_single_component_change(
        "arch_config", _mut_arch_cfg, "architecture_config_sha256"
    )

    # (c) Modify Tokenizer JSON ONLY (`tokenizer.json`)
    def _mut_tok_json(d: Path) -> None:
        tok = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
        tok["model"]["vocab"]["<|im_end|>"] = 4
        (d / "tokenizer.json").write_text(json.dumps(tok), encoding="utf-8")

    _clone_and_verify_single_component_change("tokenizer_json", _mut_tok_json, "tokenizer_sha256")

    # (d) Modify Chat template ONLY (`chat_template.jinja`)
    def _mut_chat_tmpl(d: Path) -> None:
        (d / "chat_template.jinja").write_text(
            "<|im_start|>{{ messages[0]['content'] }}<|im_end|>\n", encoding="utf-8"
        )

    _clone_and_verify_single_component_change("chat_template", _mut_chat_tmpl, "tokenizer_sha256")

    # (e) Modify Adapter weights ONLY (`adapter_model.safetensors`)
    def _mut_adapter_weights(d: Path) -> None:
        ats = load_file(str(d / "adapter_model.safetensors"))
        ats["base_model.model.layers.0.mlp.down_proj.lora_B.weight"] += 0.5
        save_file(ats, str(d / "adapter_model.safetensors"))

    _clone_and_verify_single_component_change(
        "adapter_weights", _mut_adapter_weights, "adapter_sha256"
    )

    # (f) Modify Tensor metadata ONLY (`__metadata__` header & reshaped `(4,4) -> (2,8)` with bit-identical bytes!)
    def _mut_tensor_metadata_header(d: Path) -> None:
        ts = load_file(str(d / "model.safetensors"))
        save_file(
            ts,
            str(d / "model.safetensors"),
            metadata={"format": "pt", "training_stage": "dpo_v2"},
        )

    _clone_and_verify_single_component_change(
        "tensor_metadata_header", _mut_tensor_metadata_header, "metadata_sha256"
    )

    def _mut_tensor_shape_metadata(d: Path) -> None:
        ts = load_file(str(d / "model.safetensors"))
        # Reshape (4, 4) -> (2, 8) without changing a single underlying float32 weight byte!
        ts["model.layers.0.mlp.down_proj.weight"] = ts[
            "model.layers.0.mlp.down_proj.weight"
        ].reshape(2, 8)
        save_file(
            ts,
            str(d / "model.safetensors"),
            metadata={"format": "pt", "training_stage": "sft_v1"},
        )

    _clone_and_verify_single_component_change(
        "tensor_shape_metadata", _mut_tensor_shape_metadata, "metadata_sha256"
    )

    # (g) Coexisting model weights + adapter Safetensors: mutating adapter tensor metadata or adapter
    #     tensor shapes MUST update `metadata_sha256` (while keeping `weights_sha256` & `adapter_sha256` intact!)
    def _mut_adapter_metadata_header(d: Path) -> None:
        ats = load_file(str(d / "adapter_model.safetensors"))
        save_file(
            ats,
            str(d / "adapter_model.safetensors"),
            metadata={"peft_stage": "qlora_recalibrated_v2"},
        )

    _clone_and_verify_single_component_change(
        "adapter_metadata_header", _mut_adapter_metadata_header, "metadata_sha256"
    )

    def _mut_adapter_tensor_shape(d: Path) -> None:
        ats = load_file(str(d / "adapter_model.safetensors"))
        ats["base_model.model.layers.0.mlp.down_proj.lora_A.weight"] = ats[
            "base_model.model.layers.0.mlp.down_proj.lora_A.weight"
        ].reshape(1, 8)
        save_file(ats, str(d / "adapter_model.safetensors"))

    _clone_and_verify_single_component_change(
        "adapter_tensor_shape", _mut_adapter_tensor_shape, "metadata_sha256"
    )

    # --- Missing optional files tests ---
    bare_dir = tmp_path / "bare_dir"
    bare_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base_dir / "model.safetensors", bare_dir / "model.safetensors")
    bare_rec = inspect_and_build_checkpoint_record(
        project_id="proj-fp",
        source=bare_dir,
        version_label="bare",
        policy=policy,
        local_files_only=True,
    )
    assert bare_rec.component_fingerprints["weights_sha256"] == base_comp["weights_sha256"]
    assert bare_rec.component_fingerprints["architecture_config_sha256"] is None
    assert bare_rec.component_fingerprints["tokenizer_sha256"] is None
    assert bare_rec.component_fingerprints["adapter_sha256"] is None
    # `bare_dir` lacks `adapter_model.safetensors`, so its `metadata_sha256` differs from `base_comp`
    assert bare_rec.component_fingerprints["metadata_sha256"] is not None
    assert bare_rec.component_fingerprints["metadata_sha256"] != base_comp["metadata_sha256"]

    with pytest.raises(FileNotFoundError):
        compute_checkpoint_component_fingerprints(tmp_path / "does_not_exist_dir")

    missing_dir = tmp_path / "will_be_deleted"
    shutil.copytree(base_dir, missing_dir)
    del_rec = inspect_and_build_checkpoint_record(
        project_id="proj-fp",
        source=missing_dir,
        version_label="del",
        policy=policy,
        local_files_only=True,
    )
    (missing_dir / "model.safetensors").unlink()
    st_del, err_del = revalidate_checkpoint_health(del_rec, policy=policy)
    assert st_del == CheckpointValidationStatus.CORRUPTED_OR_MISSING
    assert err_del is not None

    # --- Duplicate files & duplicate checkpoint detection ---
    exact_clone_dir = tmp_path / "exact_clone"
    shutil.copytree(base_dir, exact_clone_dir)
    dup_rec = inspect_and_build_checkpoint_record(
        project_id="proj-fp",
        source=exact_clone_dir,
        version_label="clone",
        existing_project_checkpoints=[base_rec],
        policy=policy,
        local_files_only=True,
    )
    assert dup_rec.validation_status == CheckpointValidationStatus.DUPLICATE_CONTENT
    assert dup_rec.duplicate_of_checkpoint_id == base_rec.checkpoint_id

    dup_shard_dir = tmp_path / "dup_shard_inside"
    shutil.copytree(base_dir, dup_shard_dir)
    shutil.copyfile(dup_shard_dir / "model.safetensors", dup_shard_dir / "extra_copy.safetensors")
    sha_dup_shard, comp_dup_shard, _ = compute_checkpoint_component_fingerprints(dup_shard_dir)
    assert comp_dup_shard["weights_sha256"] != base_comp["weights_sha256"]
    assert comp_dup_shard["metadata_sha256"] != base_comp["metadata_sha256"]
    assert sha_dup_shard != base_rec.sha256_fingerprint

    # --- Deterministic directory traversal across creation order, nested subdirs & mtimes ---
    order_a = tmp_path / "walk_order_a"
    order_b = tmp_path / "walk_order_b"
    rel_files = [
        ("shards/model-00001-of-00002.safetensors", (base_dir / "model.safetensors").read_bytes()),
        ("shards/model-00002-of-00002.safetensors", (base_dir / "model.safetensors").read_bytes()),
        ("config.json", (base_dir / "config.json").read_bytes()),
        ("tokenizer/tokenizer.json", (base_dir / "tokenizer.json").read_bytes()),
        ("tokenizer/chat_template.jinja", (base_dir / "chat_template.jinja").read_bytes()),
        ("lora/adapter_config.json", (base_dir / "adapter_config.json").read_bytes()),
        ("lora/adapter_model.safetensors", (base_dir / "adapter_model.safetensors").read_bytes()),
    ]
    for idx_f, (rel_p, raw_bytes) in enumerate(rel_files):
        fa = order_a / rel_p
        fa.parent.mkdir(parents=True, exist_ok=True)
        fa.write_bytes(raw_bytes)
        os.utime(fa, (1_700_000_000 + idx_f * 10, 1_700_000_000 + idx_f * 10))

    for idx_f, (rel_p, raw_bytes) in enumerate(reversed(rel_files)):
        fb = order_b / rel_p
        fb.parent.mkdir(parents=True, exist_ok=True)
        fb.write_bytes(raw_bytes)
        os.utime(fb, (1_800_000_000 + idx_f * 97, 1_800_000_000 + idx_f * 97))

    sha_a, comp_a, bytes_a = compute_checkpoint_component_fingerprints(order_a)
    sha_b, comp_b, bytes_b = compute_checkpoint_component_fingerprints(order_b)
    assert sha_a == sha_b
    assert comp_a == comp_b
    assert bytes_a == bytes_b


# ==============================================================================
# 4. STATISTICAL VALIDATION TESTS
# ==============================================================================


@pytest.mark.unit
def test_holm_bonferroni_mcnemar_reference_ties_empty_small_samples_and_regression_kinds() -> None:
    """Audit Item 4: Reference p-values, Holm-Bonferroni ties/ordering/empty, small samples & kinds."""
    # 1. Empty hypothesis family
    assert holm_bonferroni_adjust([]) == []
    empty_sig = compute_paired_significance([], [], [], [])
    assert empty_sig.n_cases == 0
    assert empty_sig.mcnemar_p_value == 1.0
    assert empty_sig.statistically_significant_95 is False

    # 2. Exact comparison against independent textbook Holm-Bonferroni across unsorted & tied families
    families = [
        [0.01, 0.04, 0.03, 0.20],
        [0.20, 0.03, 0.04, 0.01],  # Reversed order -> stable permutation invariance!
        [0.02, 0.02, 0.50],  # Tied leading p-values
        [0.01, 0.02, 0.02, 0.02, 0.15],  # 3-way internal tie
        [0.05, 0.05, 0.05, 0.05],  # All-tied family
        [0.0001, 0.004, 0.019, 0.021, 0.021, 0.45, 0.90, 1.0],
    ]
    for fam in families:
        got = holm_bonferroni_adjust(fam)
        expected = _reference_holm_bonferroni(fam)
        assert got == expected, f"Mismatch for {fam}: got {got} vs ref {expected}"
        # Equal raw p-values MUST always receive identical adjusted p-values
        for i in range(len(fam)):
            for j in range(i + 1, len(fam)):
                if fam[i] == fam[j]:
                    assert got[i] == got[j]

    # 3. Verify raw two-sided McNemar p-values against independent `math.comb` formula
    paired_vectors = [
        ([True] * 8, [False] * 8),  # d=8 -> 2 * 0.5^8 = 0.0078125 -> 0.007812
        ([True] * 6 + [False] * 2, [False] * 8),  # d=6 -> 2 * 0.5^6 = 0.03125
        ([True] * 10, [False] * 9 + [True]),  # d=9 -> 2 * 0.5^9 = 0.00390625
        ([True, True, False, True], [False, True, True, False]),  # a_only=2, b_only=1, n=3 -> p=1.0
        ([True, True, False, False], [True, True, False, False]),  # discordant=0 -> p=1.0
    ]
    for pa, pb in paired_vectors:
        sa = [1.0 if x else 0.0 for x in pa]
        sb = [1.0 if x else 0.0 for x in pb]
        sig = compute_paired_significance(sa, sb, pa, pb)
        ref_p = round(_reference_mcnemar_two_sided(pa, pb), 6)
        assert abs(sig.mcnemar_p_value - ref_p) <= 1e-6

    # 4. End-to-end BehavioralEvolutionEngine -> RegressionOriginEngine statistical validation:
    #    - `small_cat` (n=5 < 8): exploratory, NEVER statistically_significant even if raw p=0.001
    #    - `score_only_cat` (n=10 >= 8): pass_rate stays 1.0 -> 1.0 (discordant=0 -> raw_p=1.0),
    #      mean_score drops 0.95 -> 0.65 -> `score_only`, NOT statistically_significant under McNemar
    #    - `pass_rate_only_cat` (n=10 >= 8): scores slip 0.81 -> 0.78 across all 10 cases so mean_score
    #      drop is 0.03 (< 0.10), while binary pass_rate drops 1.0 -> 0.0 (d=10 -> raw_p = 2*0.5^10 = 0.001953)!
    #      Adjusted across m=3 categories: 3 * 0.001953 = 0.005859 < 0.05 -> `pass_rate_only` & `statistically_significant=True`,
    #      AND `survives_to_leaf=True` when leaf stays at pass_rate=0.0!
    c0 = _dummy_ckpt("c0", "Base", 0)
    c1 = _dummy_ckpt("c1", "Quantized", 1)
    c2 = _dummy_ckpt("c2", "Leaf", 2)
    dag = EvolutionDAG(
        [c0, c1, c2],
        [
            EvolutionRelationship("r01", "proj-rigor", "c0", "c1", TransformationType.QUANTIZATION),
            EvolutionRelationship("r12", "proj-rigor", "c1", "c2", TransformationType.PRUNING),
        ],
    )

    def _build_cases(stage: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        # small_cat: n=5, all pass on c0, all fail on c1 & c2
        for i in range(5):
            ok = stage == "c0"
            rows.append(
                {
                    "case_id": f"sm-{i}",
                    "category": "small_cat",
                    "output": "ok" if ok else "bad",
                    "passed": ok,
                    "score": 1.0 if ok else 0.0,
                }
            )
        # score_only_cat: n=10, all pass=True at every stage, score 0.95 -> 0.65 -> 0.65
        for i in range(10):
            sc = 0.95 if stage == "c0" else 0.65
            rows.append(
                {
                    "case_id": f"so-{i}",
                    "category": "score_only_cat",
                    "output": f"ans-{stage}",
                    "passed": True,
                    "score": sc,
                }
            )
        # pass_rate_only_cat: n=10, score 0.81 (passed=True) -> 0.78 (passed=False) -> 0.78 (passed=False)
        for i in range(10):
            ok = stage == "c0"
            sc = 0.81 if ok else 0.78
            rows.append(
                {
                    "case_id": f"pro-{i}",
                    "category": "pass_rate_only_cat",
                    "output": "pass" if ok else "fail",
                    "passed": ok,
                    "score": sc,
                }
            )
        return rows

    evals_list: list[dict[str, Any]] = []
    for cid in ("c0", "c1", "c2"):
        r_list = _build_cases(cid)
        by_c: dict[str, list[dict[str, Any]]] = {}
        for r in r_list:
            by_c.setdefault(r["category"], []).append(r)
        cats_summary = {
            cat: {
                "category": cat,
                "sample_size": len(items),
                "passed_count": sum(1 for x in items if x["passed"]),
                "pass_rate": round(sum(1 for x in items if x["passed"]) / len(items), 6),
                "pass_rate_wilson_ci95": [0.1, 0.9],
                "mean_score": round(sum(x["score"] for x in items) / len(items), 6),
            }
            for cat, items in by_c.items()
        }
        evals_list.append(
            {
                "checkpoint_id": cid,
                "version_label": cid,
                "test_case_ids": [r["case_id"] for r in r_list],
                "sample_size": len(r_list),
                "pass_rate": round(sum(1 for r in r_list if r["passed"]) / len(r_list), 6),
                "mean_score": round(sum(r["score"] for r in r_list) / len(r_list), 6),
                "categories": cats_summary,
                "case_results": r_list,
            }
        )

    beh_eng = BehavioralEvolutionEngine()
    tr_01 = beh_eng.compare_checkpoint_evals(
        evals_list[0], evals_list[1], scope="consecutive", transformation_type="Quantization"
    )
    tr_12 = beh_eng.compare_checkpoint_evals(
        evals_list[1], evals_list[2], scope="consecutive", transformation_type="Pruning"
    )

    reg_eng = RegressionOriginEngine(min_category_sample_size=8, min_overall_sample_size=20)
    evs = reg_eng.detect(
        dag=dag,
        behavioral_evolution={
            "checkpoint_evaluations": evals_list,
            "consecutive_transitions": [tr_01, tr_12],
        },
        tensor_transitions=[],
        representation_evolution=None,
    )
    cat_events = {e["affected_category"]: e for e in evs}

    # Check `small_cat` (n=5 < 8)
    sm_ev = cat_events["small_cat"]
    assert sm_ev["statistical_confidence"]["is_exploratory"] is True
    assert sm_ev["statistical_confidence"]["statistically_significant"] is False
    assert sm_ev["statistical_confidence"]["raw_p_value"] == round(2 * (0.5**5), 6)  # 0.0625

    # Check `score_only_cat` (n=10 >= 8, discordant=0 -> raw_p=1.0, adj_p=1.0)
    so_ev = cat_events["score_only_cat"]
    assert so_ev["regression_kind"] == "score_only"
    assert so_ev["statistical_confidence"]["is_exploratory"] is False
    assert so_ev["statistical_confidence"]["raw_p_value"] == 1.0
    assert so_ev["statistical_confidence"]["adjusted_p_value"] == 1.0
    assert so_ev["statistical_confidence"]["statistically_significant"] is False
    assert "score_only" in so_ev["statistical_confidence"]["statistical_limitations"]

    # Check `pass_rate_only_cat` (n=10 >= 8, d=10 -> raw_p=2*0.5^10=0.001953, m=3 -> adj_p=0.005859 < 0.05)
    pro_ev = cat_events["pass_rate_only_cat"]
    assert pro_ev["regression_kind"] == "pass_rate_only"
    assert pro_ev["statistical_confidence"]["is_exploratory"] is False
    assert pro_ev["statistical_confidence"]["raw_p_value"] == round(2 * (0.5**10), 6)
    assert pro_ev["statistical_confidence"]["adjusted_p_value"] == round(
        3 * round(2 * (0.5**10), 6), 6
    )
    assert pro_ev["statistical_confidence"]["statistically_significant"] is True
    assert pro_ev["survives_to_leaf"] is True
    assert pro_ev["surviving_leaf_ids"] == ["c2"]

    # Check empty category hypothesis family (`category_deltas=[]`)
    tr_empty_cats = {**tr_01, "category_deltas": []}
    evs_empty_fam = reg_eng.detect(
        dag=EvolutionDAG([c0, c1], [EvolutionRelationship("r01", "p", "c0", "c1", TransformationType.QUANTIZATION)]),
        behavioral_evolution={
            "checkpoint_evaluations": evals_list[:2],
            "consecutive_transitions": [tr_empty_cats],
        },
        tensor_transitions=[],
        representation_evolution=None,
    )
    assert len(evs_empty_fam) == 1
    assert evs_empty_fam[0]["affected_category"] == "overall"


# ==============================================================================
# 5. 7-WAY PAIR SCOPE CLASSIFICATION TEST
# ==============================================================================


@pytest.mark.unit
def test_7_way_pair_scope_classification() -> None:
    c_base = _dummy_ckpt("base", "Base", 0)
    c_sft_a = _dummy_ckpt("sft_a", "SFT-A", 1)
    c_sft_b = _dummy_ckpt("sft_b", "SFT-B", 2)
    c_quant = _dummy_ckpt("quant", "Quant", 3)
    c_other_root = _dummy_ckpt("ext_root", "ExtRoot", 4)
    rels = [
        EvolutionRelationship("r1", "p", "base", "sft_a", TransformationType.SFT),
        EvolutionRelationship("r2", "p", "base", "sft_b", TransformationType.SFT),
        EvolutionRelationship("r3", "p", "sft_a", "quant", TransformationType.QUANTIZATION),
    ]
    dag = EvolutionDAG([c_base, c_sft_a, c_sft_b, c_quant, c_other_root], rels)

    assert dag.classify_pair_scope("sft_a", "sft_a", "base") == "identical"
    assert dag.classify_pair_scope("base", "sft_a", "base") == "consecutive"
    assert dag.classify_pair_scope("sft_a", "quant", "base") == "consecutive"
    assert dag.classify_pair_scope("base", "quant", "base") == "base_to_current"
    assert (
        dag.classify_pair_scope("base", "quant", "ext_root") == "ancestor_to_descendant"
    )
    # Multi-hop non-base ancestor (`Base -> SFT-A -> Quant -> Pruned`)
    c_pruned = _dummy_ckpt("pruned", "Pruned", 5)
    dag2 = EvolutionDAG(
        [c_base, c_sft_a, c_sft_b, c_quant, c_pruned, c_other_root],
        [*rels, EvolutionRelationship("r4", "p", "quant", "pruned", TransformationType.PRUNING)],
    )
    assert dag2.classify_pair_scope("sft_a", "pruned", "base") == "ancestor_to_descendant"
    assert dag2.classify_pair_scope("pruned", "sft_a", "base") == "descendant_to_ancestor"
    assert dag2.classify_pair_scope("sft_a", "sft_b", "base") == "branch_to_branch"
    assert dag2.classify_pair_scope("sft_a", "ext_root", "base") == "unrelated"


@pytest.mark.unit
def test_score_only_regression_never_claims_causal_status_and_separates_three_flags() -> None:
    """Audit HIGH-A: Separate `behavioral_regression_observed`, `score_regression_observed`, and `causal_intervention_verified`."""
    c0 = _dummy_ckpt("c0", "v1a-SFT", 0)
    c1 = _dummy_ckpt("c1", "v2-Merged", 1)
    dag = EvolutionDAG(
        [c0, c1],
        [EvolutionRelationship("r01", "proj-rigor", "c0", "c1", TransformationType.MODEL_MERGING)],
    )

    json_p = "Return valid JSON with keys name and count."
    inst_p = "Answer in exactly two bullet points."
    json_fp = hashlib.sha256(json_p.encode("utf-8")).hexdigest()
    inst_fp = hashlib.sha256(inst_p.encode("utf-8")).hexdigest()

    # `json_reliability`: pass rate unchanged (`66.7% -> 66.7%`, `json-01` failed while `json-02` passed),
    # while rubric mean_score dropped `0.925 -> 0.641667` (`score_only`).
    # `instruction_following`: pass rate dropped `66.7% -> 0.0%` (`inst-01` failed), mean_score `0.81 -> 0.78` (`pass_rate_only`).
    beh_payload = {
        "checkpoint_evaluations": [
            {
                "checkpoint_id": "c0",
                "version_label": "v1a-SFT",
                "pass_rate": 0.666667,
                "mean_score": 0.8675,
                "categories": {
                    "json_reliability": {"pass_rate": 0.666667, "mean_score": 0.925, "sample_size": 3},
                    "instruction_following": {"pass_rate": 0.666667, "mean_score": 0.81, "sample_size": 3},
                },
                "case_results": [
                    {"case_id": "json-01", "category": "json_reliability", "prompt": json_p, "passed": True, "score": 1.0},
                    {"case_id": "inst-01", "category": "instruction_following", "prompt": inst_p, "passed": True, "score": 0.81},
                ],
            },
            {
                "checkpoint_id": "c1",
                "version_label": "v2-Merged",
                "pass_rate": 0.333333,
                "mean_score": 0.710833,
                "categories": {
                    "json_reliability": {"pass_rate": 0.666667, "mean_score": 0.641667, "sample_size": 3},
                    "instruction_following": {"pass_rate": 0.0, "mean_score": 0.78, "sample_size": 3},
                },
                "case_results": [
                    {"case_id": "json-01", "category": "json_reliability", "prompt": json_p, "passed": False, "score": 0.25},
                    {"case_id": "inst-01", "category": "instruction_following", "prompt": inst_p, "passed": False, "score": 0.78},
                ],
            },
        ],
        "consecutive_transitions": [
            {
                "source_checkpoint_id": "c0",
                "target_checkpoint_id": "c1",
                "transformation_type": "Model Merging",
                "sample_size": 6,
                "newly_introduced_failures": ["json-01", "inst-01"],
                "paired_statistics": {
                    "mcnemar_p_value": 0.5,
                    "bootstrap_ci95_low": -0.35,
                    "bootstrap_ci95_high": -0.05,
                },
                "category_deltas": [
                    {
                        "category": "json_reliability",
                        "sample_size": 3,
                        "mean_score_before": 0.925,
                        "mean_score_after": 0.641667,
                        "pass_rate_before": 0.666667,
                        "pass_rate_after": 0.666667,
                        "mcnemar_p_value": 1.0,
                        "wilson_ci95_after": [0.20, 0.93],
                    },
                    {
                        "category": "instruction_following",
                        "sample_size": 3,
                        "mean_score_before": 0.81,
                        "mean_score_after": 0.78,
                        "pass_rate_before": 0.666667,
                        "pass_rate_after": 0.0,
                        "mcnemar_p_value": 0.5,
                        "wilson_ci95_after": [0.0, 0.56],
                    },
                ],
            }
        ],
    }

    rep_payload = {
        "consecutive_transitions": [
            {
                "source_checkpoint_id": "c0",
                "target_checkpoint_id": "c1",
                "mean_cka": 0.85,
                "mean_drift": 0.15,
                "cka_by_layer": {"0": 0.9, "1": 0.8},
                "causal_patching": {
                    "case_targeted": False,
                    "verification_scope": "diagnostic_prompt_transition_only",
                    "most_causal_layer": 1,
                    "peak_recovery_fraction": 0.82,
                },
            }
        ]
    }

    # Supply valid 4-condition counterfactual provenance on BOTH `json-01` and `inst-01`
    interventions = {
        ("c0", "c1", "json-01"): {
            "case_targeted": True,
            "target_vs_foil_verified": True,
            "negative_control_verified": True,
            "verification_scope": "case_and_intervention_site",
            "prompt_fingerprint": json_fp,
            "failed_case_id": "json-01",
            "layer_index": 1,
            "intervention_type": "residual_stream_patch",
            "target_token": "{",
            "foil_token": "Sure",
            "negative_control_result": {"control_layer_index": 0, "control_recovery_fraction": 0.02},
            "recovery_measurement": 0.88,
            "clean_logit_diff": 5.0,
            "corrupted_logit_diff": -0.8,
        },
        ("c0", "c1", "inst-01"): {
            "case_targeted": True,
            "target_vs_foil_verified": True,
            "negative_control_verified": True,
            "verification_scope": "case_and_intervention_site",
            "prompt_fingerprint": inst_fp,
            "failed_case_id": "inst-01",
            "layer_index": 1,
            "intervention_type": "residual_stream_patch",
            "target_token": "-",
            "foil_token": "1",
            "negative_control_result": {"control_layer_index": 0, "control_recovery_fraction": 0.03},
            "recovery_measurement": 0.79,
            "clean_logit_diff": 3.5,
            "corrupted_logit_diff": -0.4,
        },
    }

    evs = RegressionOriginEngine().detect(
        dag=dag,
        behavioral_evolution=beh_payload,
        tensor_transitions=[],
        representation_evolution=rep_payload,
        case_interventions=interventions,
    )
    by_cat = {e["affected_category"]: e for e in evs}

    # 1. `json_reliability` (`66.7% -> 66.7%`, `score_only`):
    #    `behavioral_regression_observed` is False -> MUST NEVER claim CAUSALLY VERIFIED or CAUSAL SUPPORT!
    j_ev = by_cat["json_reliability"]
    assert j_ev["regression_kind"] == "score_only"
    assert j_ev["behavioral_regression_observed"] is False
    assert j_ev["score_regression_observed"] is True
    assert j_ev["causal_intervention_verified"] is False
    assert j_ev["epistemic_status"] == EpistemicStatus.CORRELATED.value
    assert j_ev["causal_verification"]["causal_verification_exists"] is False
    assert j_ev["causal_verification"]["causal_support_exists"] is False

    # 2. `instruction_following` (`66.7% -> 0.0%`, `pass_rate_only`):
    #    `behavioral_regression_observed` is True, `score_regression_observed` is False, `causal_intervention_verified` is True
    i_ev = by_cat["instruction_following"]
    assert i_ev["regression_kind"] == "pass_rate_only"
    assert i_ev["behavioral_regression_observed"] is True
    assert i_ev["score_regression_observed"] is False
    assert i_ev["causal_intervention_verified"] is True
    assert i_ev["epistemic_status"] == EpistemicStatus.CAUSALLY_VERIFIED.value

    # 3. `overall` (`66.7% -> 33.3%`, `0.8675 -> 0.7108`, `pass_rate_and_score`):
    #    All 3 flags True, AND disclaimer explicitly warns against equating logit patching with rubric score causation!
    o_ev = by_cat["overall"]
    assert o_ev["regression_kind"] == "pass_rate_and_score"
    assert o_ev["behavioral_regression_observed"] is True
    assert o_ev["score_regression_observed"] is True
    assert o_ev["causal_intervention_verified"] is True
    assert "continuous multi-token rubric score" in o_ev["causal_verification"]["scope_disclaimer"]


@pytest.mark.unit
def test_maximal_common_ancestors_vs_tie_broken_lca_in_bowtie_merge_dag() -> None:
    """Section 3 DAG note: Bowtie multi-parent merge DAG distinguishes full MCA set from tie-broken LCA."""
    r0 = _dummy_ckpt("r0", "Root", 0)
    p1 = _dummy_ckpt("p1", "Branch-P1", 1)
    p2 = _dummy_ckpt("p2", "Branch-P2", 2)
    m1 = _dummy_ckpt("m1", "Merge-1", 3)
    m2 = _dummy_ckpt("m2", "Merge-2", 4)
    rels = [
        EvolutionRelationship("e1", "p", "r0", "p1", TransformationType.SFT),
        EvolutionRelationship("e2", "p", "r0", "p2", TransformationType.DPO),
        EvolutionRelationship("e3", "p", "p1", "m1", TransformationType.MODEL_MERGING),
        EvolutionRelationship("e4", "p", "p2", "m1", TransformationType.MODEL_MERGING),
        EvolutionRelationship("e5", "p", "p1", "m2", TransformationType.MODEL_MERGING),
        EvolutionRelationship("e6", "p", "p2", "m2", TransformationType.MODEL_MERGING),
    ]
    dag = EvolutionDAG([r0, p1, p2, m1, m2], rels)
    # Both `p1` and `p2` (depth 1) are incomparable maximal common ancestors of `m1` and `m2`; `r0` is dominated!
    assert dag.maximal_common_ancestors("m1", "m2") == ["p1", "p2"]
    assert dag.lowest_common_ancestor("m1", "m2") == "p1"
