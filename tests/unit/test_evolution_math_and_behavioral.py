"""Unit tests for Tensor Evolution, Direct Cumulative Distance, Layer Evolution, Behavioral & Regression Origin."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from benchmarks.datasets.loader import BenchmarkSuite
from benchmarks.evaluators.base import EvalCase
from core.evolution.behavioral_evolution import BehavioralEvolutionEngine, wilson_ci_95
from core.evolution.dag import EvolutionDAG
from core.evolution.layer_evolution import LayerEvolutionEngine
from core.evolution.registry import inspect_and_build_checkpoint_record
from core.evolution.regression_origin import RegressionOriginEngine
from core.evolution.store import EvolutionStore
from core.evolution.tensor_evolution import MultiGenTensorEvolutionEngine
from core.evolution.types import RegisteredCheckpoint
from core.security import SecurityPolicy
from tests.fixtures.evolution_factory import create_known_evolution_checkpoints


@pytest.mark.unit
def test_exact_tensor_evolution_cumulative_distance_and_layer_engine(tmp_path: Path) -> None:
    """Verify exact mathematical deltas on A -> B -> C -> D -> E and prove Base->C != L2(A->B)+L2(B->C)."""
    ckpts = create_known_evolution_checkpoints(tmp_path)
    policy = SecurityPolicy(allowed_roots=[tmp_path])
    store = EvolutionStore(db_path=tmp_path / "math.db")
    proj = store.create_project(name="Exact Math Chain")

    ca = store.add_checkpoint(
        inspect_and_build_checkpoint_record(
            project_id=proj.project_id,
            source=ckpts["a"],
            version_label="A-Base",
            transformation_type="Base Model",
            policy=policy,
            local_files_only=True,
        )
    )
    cb = store.add_checkpoint(
        inspect_and_build_checkpoint_record(
            project_id=proj.project_id,
            source=ckpts["b"],
            version_label="B-SFT",
            parent_checkpoint_ids=[ca.checkpoint_id],
            transformation_type="SFT",
            policy=policy,
            local_files_only=True,
        )
    )
    cc = store.add_checkpoint(
        inspect_and_build_checkpoint_record(
            project_id=proj.project_id,
            source=ckpts["c"],
            version_label="C-DPO",
            parent_checkpoint_ids=[cb.checkpoint_id],
            transformation_type="DPO",
            policy=policy,
            local_files_only=True,
        )
    )
    cd = store.add_checkpoint(
        inspect_and_build_checkpoint_record(
            project_id=proj.project_id,
            source=ckpts["d"],
            version_label="D-Quant",
            parent_checkpoint_ids=[cc.checkpoint_id],
            transformation_type="Quantization",
            policy=policy,
            local_files_only=True,
        )
    )
    ce = store.add_checkpoint(
        inspect_and_build_checkpoint_record(
            project_id=proj.project_id,
            source=ckpts["e"],
            version_label="E-Pruned",
            parent_checkpoint_ids=[cd.checkpoint_id],
            transformation_type="Pruning",
            policy=policy,
            local_files_only=True,
        )
    )

    dag = store.get_dag(proj.project_id)
    t_engine = MultiGenTensorEvolutionEngine(policy=policy, local_files_only=True)
    transitions, raw_map, warnings = t_engine.run_all_transitions(
        dag, base_checkpoint_id=ca.checkpoint_id
    )
    assert warnings == []

    by_key = {
        (t.scope, t.source_checkpoint_id, t.target_checkpoint_id): t
        for t in transitions
    }

    # 1. Transition A -> B: ONLY `model.layers.0.self_attn.q_proj.weight` (16x16 = 256 elems) shifted +0.25
    ab = by_key[("consecutive", ca.checkpoint_id, cb.checkpoint_id)]
    assert ab.modified_tensors == 1
    assert ab.unchanged_tensors == ca.tensor_count - 1
    assert ab.max_abs_diff is not None and abs(ab.max_abs_diff - 0.25) < 1e-6
    assert ab.l2_distance is not None and abs(ab.l2_distance - 4.0) < 1e-5
    assert ab.top_k_modified_tensors[0]["name"] == "model.layers.0.self_attn.q_proj.weight"

    # 2. Direct Cumulative Distance Base (A) -> C:
    # A->B L2 = 4.0
    # B->C delta: -0.125 on 256 elems (sq=4.0) + +0.50 on 512 elems (sq=128.0) => L2(B->C) = sqrt(132)
    # Direct A->C delta: +0.125 on 256 elems (sq=4.0) + +0.50 on 512 elems (sq=128.0) => Direct L2(A->C) = sqrt(132)
    bc = by_key[("consecutive", cb.checkpoint_id, cc.checkpoint_id)]
    ac_direct = by_key[("base_to_current", ca.checkpoint_id, cc.checkpoint_id)]
    expected_ac_l2 = math.sqrt(256 * (0.125**2) + 512 * (0.50**2))
    naive_sum = 4.0 + math.sqrt(132.0)

    assert ac_direct.l2_distance is not None
    assert abs(ac_direct.l2_distance - expected_ac_l2) < 1e-4
    assert abs(ac_direct.l2_distance - naive_sum) > 3.9  # Strictly NOT naive step sum!
    assert ac_direct.sum_of_step_l2_along_path is not None
    assert abs(ac_direct.sum_of_step_l2_along_path - naive_sum) < 1e-4
    assert ac_direct.path_cancellation_ratio is not None
    assert ac_direct.path_cancellation_ratio < 0.80
    assert bc.l2_distance is not None and abs(bc.l2_distance - math.sqrt(132.0)) < 1e-4

    # 3. Pruning D -> E increases sparsity and nonzero_to_zero transitions
    de = by_key[("consecutive", cd.checkpoint_id, ce.checkpoint_id)]
    assert de.nonzero_to_zero_transitions > 0
    assert de.sparsity_delta is not None and de.sparsity_delta > 0.05

    # 4. Layer Evolution Engine verifies Layer 2 (modified +0.50 on MLP + pruned) is most modified
    layer_res = LayerEvolutionEngine().analyze(
        raw_records_map=raw_map,
        checkpoint_labels={c.checkpoint_id: c.version_label for c in store.list_checkpoints(proj.project_id)},
    )
    assert layer_res["layer_indices"] == [0, 1, 2]
    assert 2 in layer_res["most_modified_layers"]
    assert len(layer_res["heatmap"]["rows"]) == 4


@pytest.mark.unit
def test_behavioral_evolution_and_regression_origin_pinpointing(tmp_path: Path) -> None:
    """Verify Base(92%) -> SFT(95%) -> DPO(91%) -> Quantized(78%) -> Pruned(76%) pinpoints Quantized!"""
    # Wilson CI sanity check
    low, high = wilson_ci_95(92, 100)
    assert 0.84 < low < 0.92 < high < 0.97

    ckpts = create_known_evolution_checkpoints(tmp_path)
    policy = SecurityPolicy(allowed_roots=[tmp_path])
    store = EvolutionStore(db_path=tmp_path / "reg_origin.db")
    proj = store.create_project(name="Regression Origin Verification")

    stages = [
        ("a", "Base", "Base Model", None, 92),
        ("b", "SFT", "SFT", "prev", 95),
        ("c", "DPO", "DPO", "prev", 91),
        ("d", "Quantized", "Quantization", "prev", 78),
        ("e", "Pruned", "Pruning", "prev", 76),
    ]
    registered: list[RegisteredCheckpoint] = []
    pass_quota_by_id: dict[str, int] = {}

    for key, label, t_type, parent_flag, pass_cnt in stages:
        parents = [registered[-1].checkpoint_id] if parent_flag == "prev" else []
        rec = store.add_checkpoint(
            inspect_and_build_checkpoint_record(
                project_id=proj.project_id,
                source=ckpts[key],
                version_label=label,
                parent_checkpoint_ids=parents,
                transformation_type=t_type,
                policy=policy,
                local_files_only=True,
            )
        )
        registered.append(rec)
        pass_quota_by_id[rec.checkpoint_id] = pass_cnt

    # Build a 100-case deterministic math_reasoning suite so pass rates equal exact 0.92, 0.95, 0.91, 0.78, 0.76
    cases = [
        EvalCase(
            case_id=f"math-{idx:03d}",
            category="math_reasoning",
            prompt=f"Compute {idx} + 1 =",
            description=f"Arithmetic case {idx}",
            expected={"answer": str(idx + 1)},
        )
        for idx in range(100)
    ]
    suite_100 = BenchmarkSuite(
        suite_id="controlled-100",
        version="1.0.0",
        description="100-case controlled regression origin suite",
        cases=cases,
    )

    def _controlled_gen(ckpt: RegisteredCheckpoint, case: EvalCase) -> tuple[str, str, float]:
        idx = int(case.case_id.split("-")[1])
        quota = pass_quota_by_id[ckpt.checkpoint_id]
        ans = str(idx + 1) if idx < quota else "999999"
        return (case.prompt, ans, 1.0)

    dag: EvolutionDAG = store.get_dag(proj.project_id)
    t_objs, _, _ = MultiGenTensorEvolutionEngine(policy=policy, local_files_only=True).run_all_transitions(
        dag, base_checkpoint_id=registered[0].checkpoint_id
    )
    tensor_dicts = [t.to_dict() for t in t_objs]

    beh_engine = BehavioralEvolutionEngine(store=store, policy=policy, use_cache=False)
    beh_payload, beh_warns = beh_engine.run(
        dag,
        base_checkpoint_id=registered[0].checkpoint_id,
        suite=suite_100,
        custom_generator=_controlled_gen,
    )
    assert beh_warns == []
    pass_rates = [row["pass_rate"] for row in beh_payload["timeline"]]
    assert pass_rates == [0.92, 0.95, 0.91, 0.78, 0.76]

    # Configure major regression threshold = 0.05 (so 95%->91% = 4% is minor alignment tradeoff,
    # and 91%->78% = 13% Quantization drop is identified as the FIRST major regression!)
    reg_engine = RegressionOriginEngine(
        overall_pass_rate_drop_threshold=0.05,
        category_score_drop_threshold=0.05,
    )
    events = reg_engine.detect(
        dag=dag,
        behavioral_evolution=beh_payload,
        tensor_transitions=tensor_dicts,
        representation_evolution=None,
    )
    assert len(events) >= 1
    primary = events[0]
    assert primary["previous_label"] == "DPO"
    assert primary["first_affected_label"] == "Quantized"
    assert primary["transformation_type"] == "Quantization"
    assert abs(primary["pass_rate_drop"] - 0.13) < 1e-6
    assert primary["survives_to_leaf"] is True
    assert len(primary["failed_test_case_ids"]) == 13
    # Crucially: without causal patching, epistemic status MUST be CORRELATED, never CAUSALLY VERIFIED!
    assert primary["epistemic_status"] == "CORRELATED"
    assert primary["causal_verification"]["causal_verification_exists"] is False
