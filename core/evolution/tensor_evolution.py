"""Multi-Generation Tensor & Cumulative Weight Evolution Engine (Section 3).

Strictly separates:
1. `consecutive`: Direct `parent -> child` DAG edge comparisons.
2. `base_to_current`: Direct `base -> current` comparison against the original root checkpoint
   (NEVER summing consecutive step differences!).
3. `branch_to_branch`: Cross-lineage comparisons between branch leaves or arbitrary version pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from core.evolution.dag import EvolutionDAG
from core.evolution.types import RegisteredCheckpoint
from core.loaders.model_loader import ModelLoader
from core.resource_manager import CancellationToken
from core.security import SecurityPolicy
from core.structural_analysis.compatibility import validate_compatibility
from core.tensor_engine.comparator import (
    TensorComparisonRecord,
    TensorDiffEngine,
    TensorDiffReport,
    TensorDiffStatus,
)
from core.tensor_engine.metrics import DiffTolerances
from core.types import EpistemicStatus


@dataclass
class TransitionTensorAnalysis:
    """Complete multi-generation tensor & weight-shift measurements for one pair `(source -> target)`."""

    scope: str  # "consecutive" | "base_to_current" | "branch_to_branch"
    source_checkpoint_id: str
    source_label: str
    target_checkpoint_id: str
    target_label: str
    transformation_type: str
    epistemic_status: str
    compatible_for_direct_tensor_diff: bool
    compatibility_warnings: list[str]
    compared_tensors: int
    modified_tensors: int
    unchanged_tensors: int
    modified_tensor_pct: float
    unchanged_tensor_pct: float
    added_tensors: list[str]
    removed_tensors: list[str]
    shape_mismatched_tensors: list[dict[str, Any]]
    total_compared_parameters: int
    modified_parameters: int
    modified_parameter_pct: float
    max_abs_diff: float | None
    mean_abs_diff: float | None
    rmse: float | None
    l2_distance: float | None
    relative_l2_distance: float | None
    global_cosine_similarity: float | None
    mean_tensor_cosine_similarity: float | None
    zero_to_nonzero_transitions: int
    nonzero_to_zero_transitions: int
    sparsity_ratio_source: float | None
    sparsity_ratio_target: float | None
    sparsity_delta: float | None
    distribution_shift_summary: dict[str, Any]
    per_layer_changes: list[dict[str, Any]]
    top_k_modified_tensors: list[dict[str, Any]]
    # Populated on `base_to_current` to demonstrate direct comparison vs naive step summation
    sum_of_step_l2_along_path: float | None = None
    path_cancellation_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "source_checkpoint_id": self.source_checkpoint_id,
            "source_label": self.source_label,
            "target_checkpoint_id": self.target_checkpoint_id,
            "target_label": self.target_label,
            "transformation_type": self.transformation_type,
            "epistemic_status": self.epistemic_status,
            "compatible_for_direct_tensor_diff": self.compatible_for_direct_tensor_diff,
            "compatibility_warnings": self.compatibility_warnings,
            "compared_tensors": self.compared_tensors,
            "modified_tensors": self.modified_tensors,
            "unchanged_tensors": self.unchanged_tensors,
            "modified_tensor_pct": self.modified_tensor_pct,
            "unchanged_tensor_pct": self.unchanged_tensor_pct,
            "added_tensors": self.added_tensors,
            "removed_tensors": self.removed_tensors,
            "shape_mismatched_tensors": self.shape_mismatched_tensors,
            "total_compared_parameters": self.total_compared_parameters,
            "modified_parameters": self.modified_parameters,
            "modified_parameter_pct": self.modified_parameter_pct,
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "rmse": self.rmse,
            "l2_distance": self.l2_distance,
            "relative_l2_distance": self.relative_l2_distance,
            "global_cosine_similarity": self.global_cosine_similarity,
            "mean_tensor_cosine_similarity": self.mean_tensor_cosine_similarity,
            "zero_to_nonzero_transitions": self.zero_to_nonzero_transitions,
            "nonzero_to_zero_transitions": self.nonzero_to_zero_transitions,
            "sparsity_ratio_source": self.sparsity_ratio_source,
            "sparsity_ratio_target": self.sparsity_ratio_target,
            "sparsity_delta": self.sparsity_delta,
            "distribution_shift_summary": self.distribution_shift_summary,
            "per_layer_changes": self.per_layer_changes,
            "top_k_modified_tensors": self.top_k_modified_tensors,
            "sum_of_step_l2_along_path": self.sum_of_step_l2_along_path,
            "path_cancellation_ratio": self.path_cancellation_ratio,
        }


def summarize_tensor_diff_for_evolution(
    *,
    scope: str,
    ckpt_a: RegisteredCheckpoint,
    ckpt_b: RegisteredCheckpoint,
    transformation_type: str,
    diff_report: TensorDiffReport,
    compat_ok: bool,
    compat_warnings: list[str],
    top_k: int = 10,
) -> tuple[TransitionTensorAnalysis, list[TensorComparisonRecord]]:
    """Aggregate exact parameter-weighted global metrics from `TensorDiffReport`."""
    added = [r.name for r in diff_report.records if r.status == TensorDiffStatus.ADDED_IN_B]
    removed = [r.name for r in diff_report.records if r.status == TensorDiffStatus.MISSING_IN_B]
    shape_mismatches = [
        {
            "name": r.name,
            "shape_source": list(r.shape_a) if r.shape_a else None,
            "shape_target": list(r.shape_b) if r.shape_b else None,
        }
        for r in diff_report.records
        if r.status == TensorDiffStatus.SHAPE_MISMATCH
    ]

    compared = diff_report.compared_tensors
    modified_t = diff_report.modified_tensors
    unchanged_t = diff_report.identical_tensors + diff_report.within_tolerance_tensors
    mod_t_pct = round((modified_t / compared) * 100.0, 4) if compared > 0 else 0.0
    unch_t_pct = round((unchanged_t / compared) * 100.0, 4) if compared > 0 else 0.0

    # Exact global aggregations over all finite aligned elements
    sum_abs = 0.0
    sum_sq_diff = 0.0
    sum_sq_a = 0.0
    sum_sq_b = 0.0
    sum_dot = 0.0
    finite_total = 0
    z_to_nz = 0
    nz_to_z = 0

    zero_a_total = 0
    zero_b_total = 0
    dist_elem_total = 0
    wasserstein_weighted = 0.0
    mean_delta_weighted = 0.0
    std_delta_weighted = 0.0
    w_elem_total = 0

    aligned_records: list[TensorComparisonRecord] = []
    for rec in diff_report.records:
        if rec.metrics is None:
            continue
        aligned_records.append(rec)
        m = rec.metrics
        z_to_nz += m.zero_to_nonzero
        nz_to_z += m.nonzero_to_zero
        fn = m.finite_pair_elements
        if fn > 0 and m.mean_abs_diff is not None and m.l2_diff is not None:
            finite_total += fn
            sum_abs += m.mean_abs_diff * fn
            sum_sq_diff += m.l2_diff * m.l2_diff
            if m.l2_norm_a is not None:
                sum_sq_a += m.l2_norm_a * m.l2_norm_a
            if m.l2_norm_b is not None:
                sum_sq_b += m.l2_norm_b * m.l2_norm_b
            if (
                m.cosine_similarity is not None
                and m.l2_norm_a is not None
                and m.l2_norm_b is not None
            ):
                sum_dot += m.cosine_similarity * m.l2_norm_a * m.l2_norm_b

        if rec.distribution_a is not None and rec.distribution_b is not None:
            zero_a_total += rec.distribution_a.zero_count
            zero_b_total += rec.distribution_b.zero_count
            dist_elem_total += rec.distribution_a.total_elements
        if rec.distribution_shift is not None and fn > 0:
            sh = rec.distribution_shift
            if sh.wasserstein_1d_approx is not None:
                wasserstein_weighted += sh.wasserstein_1d_approx * fn
                w_elem_total += fn
            if sh.mean_delta is not None:
                mean_delta_weighted += sh.mean_delta * fn
            if sh.std_delta is not None:
                std_delta_weighted += sh.std_delta * fn

    if finite_total > 0:
        mean_abs_val = sum_abs / finite_total
        rmse_val = math.sqrt(max(0.0, sum_sq_diff / finite_total))
        l2_dist_val = math.sqrt(max(0.0, sum_sq_diff))
        norm_a = math.sqrt(max(0.0, sum_sq_a))
        norm_b = math.sqrt(max(0.0, sum_sq_b))
        rel_l2: float | None = (
            0.0
            if l2_dist_val == 0.0
            else (l2_dist_val / max(norm_a, diff_report.tolerances.eps) if norm_a > 0.0 else None)
        )
        mean_abs: float | None = mean_abs_val
        rmse: float | None = rmse_val
        l2_dist: float | None = l2_dist_val
        if norm_a == 0.0 and norm_b == 0.0:
            g_cos: float | None = 1.0
        elif norm_a == 0.0 or norm_b == 0.0:
            g_cos = 0.0
        else:
            g_cos = max(-1.0, min(1.0, sum_dot / (norm_a * norm_b)))
    else:
        mean_abs = None
        rmse = None
        l2_dist = None
        rel_l2 = None
        g_cos = None

    sparsity_a = round(zero_a_total / dist_elem_total, 8) if dist_elem_total > 0 else None
    sparsity_b = round(zero_b_total / dist_elem_total, 8) if dist_elem_total > 0 else None
    sparsity_delta = (
        round(sparsity_b - sparsity_a, 8)
        if (sparsity_a is not None and sparsity_b is not None)
        else None
    )

    dist_summary = {
        "mean_wasserstein_1d": (
            round(wasserstein_weighted / w_elem_total, 8) if w_elem_total > 0 else None
        ),
        "weighted_mean_shift": (
            round(mean_delta_weighted / finite_total, 8) if finite_total > 0 else None
        ),
        "weighted_std_shift": (
            round(std_delta_weighted / finite_total, 8) if finite_total > 0 else None
        ),
        "max_tensor_wasserstein_1d": max(
            (
                r.distribution_shift.wasserstein_1d_approx
                for r in aligned_records
                if r.distribution_shift and r.distribution_shift.wasserstein_1d_approx is not None
            ),
            default=None,
        ),
    }

    # Rank Top-K most modified tensors by relative_l2 then max_abs_diff
    ranked = sorted(
        aligned_records,
        key=lambda r: (
            r.metrics.relative_l2 if (r.metrics and r.metrics.relative_l2 is not None) else -1.0,
            r.metrics.max_abs_diff if (r.metrics and r.metrics.max_abs_diff is not None) else -1.0,
        ),
        reverse=True,
    )
    top_tensors = [
        {
            "name": r.name,
            "layer_index": r.layer_index,
            "canonical_role": r.canonical_role.value,
            "shape": list(r.shape_a) if r.shape_a else [],
            "status": r.status.value,
            "relative_l2": r.metrics.relative_l2 if r.metrics else None,
            "l2_diff": r.metrics.l2_diff if r.metrics else None,
            "max_abs_diff": r.metrics.max_abs_diff if r.metrics else None,
            "mean_abs_diff": r.metrics.mean_abs_diff if r.metrics else None,
            "rmse": r.metrics.rmse if r.metrics else None,
            "cosine_similarity": r.metrics.cosine_similarity if r.metrics else None,
            "changed_pct": r.metrics.changed_pct if r.metrics else 0.0,
            "zero_to_nonzero": r.metrics.zero_to_nonzero if r.metrics else 0,
            "nonzero_to_zero": r.metrics.nonzero_to_zero if r.metrics else 0,
            "sparsity_delta": (
                r.distribution_shift.sparsity_delta if r.distribution_shift else None
            ),
        }
        for r in ranked[: max(1, top_k)]
    ]

    per_layer = [ls.to_dict() for ls in diff_report.layer_summaries]

    analysis = TransitionTensorAnalysis(
        scope=scope,
        source_checkpoint_id=ckpt_a.checkpoint_id,
        source_label=ckpt_a.version_label,
        target_checkpoint_id=ckpt_b.checkpoint_id,
        target_label=ckpt_b.version_label,
        transformation_type=transformation_type,
        epistemic_status=EpistemicStatus.OBSERVED.value,
        compatible_for_direct_tensor_diff=compat_ok,
        compatibility_warnings=compat_warnings,
        compared_tensors=compared,
        modified_tensors=modified_t,
        unchanged_tensors=unchanged_t,
        modified_tensor_pct=mod_t_pct,
        unchanged_tensor_pct=unch_t_pct,
        added_tensors=added,
        removed_tensors=removed,
        shape_mismatched_tensors=shape_mismatches,
        total_compared_parameters=diff_report.total_compared_parameters,
        modified_parameters=diff_report.total_changed_parameters,
        modified_parameter_pct=diff_report.global_changed_pct,
        max_abs_diff=diff_report.global_max_abs_diff,
        mean_abs_diff=mean_abs,
        rmse=rmse,
        l2_distance=l2_dist,
        relative_l2_distance=rel_l2,
        global_cosine_similarity=g_cos,
        mean_tensor_cosine_similarity=diff_report.global_mean_cosine,
        zero_to_nonzero_transitions=z_to_nz,
        nonzero_to_zero_transitions=nz_to_z,
        sparsity_ratio_source=sparsity_a,
        sparsity_ratio_target=sparsity_b,
        sparsity_delta=sparsity_delta,
        distribution_shift_summary=dist_summary,
        per_layer_changes=per_layer,
        top_k_modified_tensors=top_tensors,
    )
    return analysis, list(diff_report.records)


class MultiGenTensorEvolutionEngine:
    """Memory-bounded multi-generation tensor & cumulative distance engine."""

    def __init__(
        self,
        *,
        policy: SecurityPolicy | None = None,
        tolerances: DiffTolerances | None = None,
        chunk_elements: int = 1_048_576,
        top_k: int = 10,
        local_files_only: bool = False,
    ) -> None:
        self.policy = policy or SecurityPolicy()
        self.tolerances = tolerances or DiffTolerances()
        self.chunk_elements = max(4096, int(chunk_elements))
        self.top_k = max(1, int(top_k))
        self.local_files_only = local_files_only

    def compare_pair(
        self,
        ckpt_a: RegisteredCheckpoint,
        ckpt_b: RegisteredCheckpoint,
        *,
        scope: str,
        transformation_type: str | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> tuple[TransitionTensorAnalysis, list[TensorComparisonRecord]]:
        """Compare exactly two checkpoints via streaming mmap handles and release handles immediately."""
        loader = ModelLoader(policy=self.policy)
        engine = TensorDiffEngine(
            tolerances=self.tolerances,
            chunk_elements=self.chunk_elements,
            include_distributions=True,
        )
        eff_trans = transformation_type or ckpt_b.transformation_type.value

        with (
            loader.load(ckpt_a.source_path, local_files_only=self.local_files_only) as ha,
            loader.load(ckpt_b.source_path, local_files_only=self.local_files_only) as hb,
        ):
            compat = validate_compatibility(ha, hb)
            warnings = [
                f"[{issue.severity.value}] {issue.code}: {issue.message}"
                for issue in compat.issues
            ]
            diff_report = engine.compare(ha, hb, cancellation_token=cancellation_token)

        return summarize_tensor_diff_for_evolution(
            scope=scope,
            ckpt_a=ckpt_a,
            ckpt_b=ckpt_b,
            transformation_type=eff_trans,
            diff_report=diff_report,
            compat_ok=compat.compatible_for_direct_tensor_diff,
            compat_warnings=warnings,
            top_k=self.top_k,
        )

    def run_all_transitions(
        self,
        dag: EvolutionDAG,
        *,
        base_checkpoint_id: str | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> tuple[list[TransitionTensorAnalysis], dict[tuple[str, str, str], list[TensorComparisonRecord]], list[str]]:
        """Compute consecutive, direct base-to-current, and branch-to-branch tensor transitions."""
        results: list[TransitionTensorAnalysis] = []
        raw_records_map: dict[tuple[str, str, str], list[TensorComparisonRecord]] = {}
        warnings: list[str] = []

        roots = dag.roots()
        effective_base_id = base_checkpoint_id or (roots[0] if roots else None)
        step_l2_by_edge: dict[tuple[str, str], float] = {}

        # 1. Consecutive parent -> child transitions
        for rel in dag.relationships:
            if cancellation_token is not None:
                cancellation_token.check()
            pa = dag.checkpoints_by_id[rel.parent_checkpoint_id]
            ch = dag.checkpoints_by_id[rel.child_checkpoint_id]
            try:
                tr, recs = self.compare_pair(
                    pa,
                    ch,
                    scope="consecutive",
                    transformation_type=rel.transformation_type.value,
                    cancellation_token=cancellation_token,
                )
                results.append(tr)
                raw_records_map[("consecutive", pa.checkpoint_id, ch.checkpoint_id)] = recs
                if tr.l2_distance is not None:
                    step_l2_by_edge[(pa.checkpoint_id, ch.checkpoint_id)] = tr.l2_distance
            except Exception as exc:
                warnings.append(
                    f"Consecutive tensor diff skipped for '{pa.version_label} -> {ch.version_label}': {exc}"
                )

        # 2. Direct Base-to-Current cumulative transitions (strictly comparing Base directly against Current!)
        if effective_base_id and effective_base_id in dag.checkpoints_by_id:
            base_ckpt = dag.checkpoints_by_id[effective_base_id]
            for cid in dag.topological_order():
                if cancellation_token is not None:
                    cancellation_token.check()
                target_ckpt = dag.checkpoints_by_id[cid]
                if cid == effective_base_id:
                    # Identity baseline entry (distance = 0.0)
                    zero_entry = TransitionTensorAnalysis(
                        scope="base_to_current",
                        source_checkpoint_id=base_ckpt.checkpoint_id,
                        source_label=base_ckpt.version_label,
                        target_checkpoint_id=base_ckpt.checkpoint_id,
                        target_label=base_ckpt.version_label,
                        transformation_type=base_ckpt.transformation_type.value,
                        epistemic_status=EpistemicStatus.OBSERVED.value,
                        compatible_for_direct_tensor_diff=True,
                        compatibility_warnings=[],
                        compared_tensors=base_ckpt.tensor_count,
                        modified_tensors=0,
                        unchanged_tensors=base_ckpt.tensor_count,
                        modified_tensor_pct=0.0,
                        unchanged_tensor_pct=100.0,
                        added_tensors=[],
                        removed_tensors=[],
                        shape_mismatched_tensors=[],
                        total_compared_parameters=base_ckpt.parameter_count,
                        modified_parameters=0,
                        modified_parameter_pct=0.0,
                        max_abs_diff=0.0,
                        mean_abs_diff=0.0,
                        rmse=0.0,
                        l2_distance=0.0,
                        relative_l2_distance=0.0,
                        global_cosine_similarity=1.0,
                        mean_tensor_cosine_similarity=1.0,
                        zero_to_nonzero_transitions=0,
                        nonzero_to_zero_transitions=0,
                        sparsity_ratio_source=None,
                        sparsity_ratio_target=None,
                        sparsity_delta=0.0,
                        distribution_shift_summary={
                            "mean_wasserstein_1d": 0.0,
                            "weighted_mean_shift": 0.0,
                            "weighted_std_shift": 0.0,
                            "max_tensor_wasserstein_1d": 0.0,
                        },
                        per_layer_changes=[],
                        top_k_modified_tensors=[],
                        sum_of_step_l2_along_path=0.0,
                        path_cancellation_ratio=1.0,
                    )
                    results.append(zero_entry)
                    continue

                # Reuse already-computed direct transition ONLY when `base_ckpt -> target_ckpt` is a direct 1-hop edge;
                # for >=2 hops, ALWAYS compare `base_ckpt` directly against `target_ckpt` on disk!
                direct_path = dag.shortest_directed_path(effective_base_id, cid)
                sum_step_l2: float | None = None
                if direct_path and len(direct_path) >= 2:
                    edge_vals = [
                        step_l2_by_edge.get((direct_path[i], direct_path[i + 1]))
                        for i in range(len(direct_path) - 1)
                    ]
                    if all(v is not None for v in edge_vals):
                        sum_step_l2 = sum(float(v) for v in edge_vals if v is not None)

                try:
                    tr_base, recs_base = self.compare_pair(
                        base_ckpt,
                        target_ckpt,
                        scope="base_to_current",
                        transformation_type=target_ckpt.transformation_type.value,
                        cancellation_token=cancellation_token,
                    )
                    tr_base.sum_of_step_l2_along_path = sum_step_l2
                    if (
                        sum_step_l2 is not None
                        and sum_step_l2 > 1e-12
                        and tr_base.l2_distance is not None
                    ):
                        tr_base.path_cancellation_ratio = round(
                            tr_base.l2_distance / sum_step_l2, 6
                        )
                    results.append(tr_base)
                    raw_records_map[("base_to_current", base_ckpt.checkpoint_id, cid)] = recs_base
                except Exception as exc:
                    warnings.append(
                        f"Base-to-current tensor diff skipped for '{base_ckpt.version_label} -> {target_ckpt.version_label}': {exc}"
                    )

        # 3. Branch-to-Branch leaf comparisons when multiple distinct leaf checkpoints exist
        leaves = dag.leaves()
        if len(leaves) >= 2:
            for idx_a in range(len(leaves)):
                for idx_b in range(idx_a + 1, min(len(leaves), idx_a + 4)):
                    if cancellation_token is not None:
                        cancellation_token.check()
                    leaf_a = dag.checkpoints_by_id[leaves[idx_a]]
                    leaf_b = dag.checkpoints_by_id[leaves[idx_b]]
                    try:
                        tr_br, recs_br = self.compare_pair(
                            leaf_a,
                            leaf_b,
                            scope="branch_to_branch",
                            transformation_type="Branch Divergence",
                            cancellation_token=cancellation_token,
                        )
                        results.append(tr_br)
                        raw_records_map[
                            ("branch_to_branch", leaf_a.checkpoint_id, leaf_b.checkpoint_id)
                        ] = recs_br
                    except Exception as exc:
                        warnings.append(
                            f"Branch-to-branch tensor diff skipped for '{leaf_a.version_label} <-> {leaf_b.version_label}': {exc}"
                        )

        return results, raw_records_map, warnings
