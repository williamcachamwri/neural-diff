"""Dedicated Layer-Level Evolution Engine (Section 4).

Tracks historical weight, sparsity, cosine similarity, relative L2, and canonical role-group
modifications (attention, MLP, norm, embedding, output head) across every checkpoint transition.
"""

from __future__ import annotations

import statistics
from typing import Any

from core.tensor_engine.comparator import TensorComparisonRecord
from core.types import CanonicalRole, EpistemicStatus

_ATTN_ROLES = {
    CanonicalRole.ATTN_Q,
    CanonicalRole.ATTN_K,
    CanonicalRole.ATTN_V,
    CanonicalRole.ATTN_O,
    CanonicalRole.ATTN_QKV_FUSED,
}
_MLP_ROLES = {
    CanonicalRole.MLP_GATE,
    CanonicalRole.MLP_UP,
    CanonicalRole.MLP_DOWN,
    CanonicalRole.MLP_FC,
}


def _role_group(role: CanonicalRole) -> str:
    if role in _ATTN_ROLES:
        return "attention"
    if role in _MLP_ROLES:
        return "mlp"
    if role == CanonicalRole.NORM:
        return "normalization"
    if role == CanonicalRole.EMBEDDING:
        return "embedding"
    if role == CanonicalRole.LM_HEAD:
        return "output_head"
    return "other"


def _summarize_record_group(records: list[TensorComparisonRecord]) -> dict[str, Any]:
    aligned = [r for r in records if r.metrics is not None]
    if not aligned:
        return {
            "tensor_count": 0,
            "total_parameters": 0,
            "modified_parameters": 0,
            "modified_pct": 0.0,
            "mean_relative_l2": None,
            "max_abs_diff": None,
            "mean_cosine_similarity": None,
            "sparsity_target": None,
            "sparsity_delta": None,
        }

    total_params = sum(r.metrics.num_elements for r in aligned if r.metrics is not None)
    mod_params = sum(r.metrics.changed_elements for r in aligned if r.metrics is not None)
    rel_l2s = [
        r.metrics.relative_l2
        for r in aligned
        if r.metrics is not None and r.metrics.relative_l2 is not None
    ]
    cosines = [
        r.metrics.cosine_similarity
        for r in aligned
        if r.metrics is not None and r.metrics.cosine_similarity is not None
    ]
    max_abs = max(
        (
            r.metrics.max_abs_diff
            for r in aligned
            if r.metrics is not None and r.metrics.max_abs_diff is not None
        ),
        default=None,
    )

    zero_b = sum(
        r.distribution_b.zero_count for r in aligned if r.distribution_b is not None
    )
    zero_a = sum(
        r.distribution_a.zero_count for r in aligned if r.distribution_a is not None
    )
    dist_elems = sum(
        r.distribution_b.total_elements for r in aligned if r.distribution_b is not None
    )
    sp_b = round(zero_b / dist_elems, 8) if dist_elems > 0 else None
    sp_a = round(zero_a / dist_elems, 8) if dist_elems > 0 else None
    sp_delta = round(sp_b - sp_a, 8) if (sp_a is not None and sp_b is not None) else None

    return {
        "tensor_count": len(aligned),
        "total_parameters": total_params,
        "modified_parameters": mod_params,
        "modified_pct": round((mod_params / total_params) * 100.0, 4) if total_params > 0 else 0.0,
        "mean_relative_l2": round(sum(rel_l2s) / len(rel_l2s), 8) if rel_l2s else None,
        "max_abs_diff": max_abs,
        "mean_cosine_similarity": round(sum(cosines) / len(cosines), 8) if cosines else None,
        "sparsity_target": sp_b,
        "sparsity_delta": sp_delta,
    }


class LayerEvolutionEngine:
    """Computes historical per-layer and role-group evolution across transitions."""

    def analyze(
        self,
        *,
        raw_records_map: dict[tuple[str, str, str], list[TensorComparisonRecord]],
        checkpoint_labels: dict[str, str],
        representation_evolution: dict[str, Any] | None = None,
        regression_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        transition_profiles: list[dict[str, Any]] = []
        cumulative_layer_rel_l2: dict[int, list[float]] = {}
        consecutive_modified_counts: dict[int, int] = {}
        total_consecutive_transitions = 0

        all_layers_seen: set[int] = set()

        for (scope, src_id, tgt_id), records in raw_records_map.items():
            if scope == "branch_to_branch":
                continue

            by_layer: dict[int, list[TensorComparisonRecord]] = {}
            global_roles: dict[str, list[TensorComparisonRecord]] = {
                "embedding": [],
                "output_head": [],
                "attention": [],
                "mlp": [],
                "normalization": [],
            }

            for rec in records:
                rg = _role_group(rec.canonical_role)
                if rg in global_roles:
                    global_roles[rg].append(rec)
                if rec.layer_index is not None:
                    by_layer.setdefault(rec.layer_index, []).append(rec)
                    all_layers_seen.add(rec.layer_index)

            if scope == "consecutive":
                total_consecutive_transitions += 1

            layer_entries: list[dict[str, Any]] = []
            for l_idx in sorted(by_layer):
                l_recs = by_layer[l_idx]
                overall = _summarize_record_group(l_recs)
                attn = _summarize_record_group([r for r in l_recs if _role_group(r.canonical_role) == "attention"])
                mlp = _summarize_record_group([r for r in l_recs if _role_group(r.canonical_role) == "mlp"])
                norm = _summarize_record_group([r for r in l_recs if _role_group(r.canonical_role) == "normalization"])

                rel_l2 = overall["mean_relative_l2"]
                if scope == "consecutive":
                    if rel_l2 is not None:
                        cumulative_layer_rel_l2.setdefault(l_idx, []).append(float(rel_l2))
                    if overall["modified_parameters"] > 0:
                        consecutive_modified_counts[l_idx] = consecutive_modified_counts.get(l_idx, 0) + 1

                layer_entries.append(
                    {
                        "layer_index": l_idx,
                        **overall,
                        "attention_changes": attn,
                        "mlp_changes": mlp,
                        "normalization_changes": norm,
                    }
                )

            transition_profiles.append(
                {
                    "scope": scope,
                    "source_checkpoint_id": src_id,
                    "source_label": checkpoint_labels.get(src_id, src_id),
                    "target_checkpoint_id": tgt_id,
                    "target_label": checkpoint_labels.get(tgt_id, tgt_id),
                    "embedding_changes": _summarize_record_group(global_roles["embedding"]),
                    "output_head_changes": _summarize_record_group(global_roles["output_head"]),
                    "attention_projection_changes": _summarize_record_group(global_roles["attention"]),
                    "mlp_changes": _summarize_record_group(global_roles["mlp"]),
                    "normalization_changes": _summarize_record_group(global_roles["normalization"]),
                    "layers": layer_entries,
                }
            )

        # Aggregate historical layer rankings across consecutive transitions
        layer_aggregates: list[dict[str, Any]] = []
        for l_idx in sorted(all_layers_seen):
            vals = cumulative_layer_rel_l2.get(l_idx, [])
            mean_val = sum(vals) / len(vals) if vals else 0.0
            max_val = max(vals) if vals else 0.0
            mod_steps = consecutive_modified_counts.get(l_idx, 0)
            mod_ratio = (
                mod_steps / total_consecutive_transitions
                if total_consecutive_transitions > 0
                else 0.0
            )
            layer_aggregates.append(
                {
                    "layer_index": l_idx,
                    "mean_transition_relative_l2": round(mean_val, 8),
                    "peak_transition_relative_l2": round(max_val, 8),
                    "transitions_modified": mod_steps,
                    "total_transitions": total_consecutive_transitions,
                    "consistency_ratio": round(mod_ratio, 4),
                }
            )

        sorted_desc = sorted(
            layer_aggregates,
            key=lambda x: (x["peak_transition_relative_l2"], x["mean_transition_relative_l2"]),
            reverse=True,
        )
        sorted_asc = sorted(
            layer_aggregates,
            key=lambda x: (x["mean_transition_relative_l2"], x["peak_transition_relative_l2"]),
        )

        most_modified = [x["layer_index"] for x in sorted_desc[:3] if x["peak_transition_relative_l2"] > 0.0]
        least_modified = [x["layer_index"] for x in sorted_asc[:3]]

        # Detect layers with unusually large changes (robust statistical outlier > mean + 1.25 * std & > 1.5x median)
        peaks = [x["peak_transition_relative_l2"] for x in layer_aggregates]
        anomalous_layers: list[dict[str, Any]] = []
        if len(peaks) >= 2 and max(peaks) > 0.0:
            mean_p = statistics.fmean(peaks)
            std_p = statistics.pstdev(peaks)
            med_p = statistics.median(peaks)
            threshold = max(mean_p + 1.25 * std_p, med_p * 1.5, 1e-6)
            for item in sorted_desc:
                if item["peak_transition_relative_l2"] >= threshold and std_p > 1e-9:
                    z_val = (item["peak_transition_relative_l2"] - mean_p) / std_p
                    anomalous_layers.append(
                        {
                            "layer_index": item["layer_index"],
                            "peak_relative_l2": item["peak_transition_relative_l2"],
                            "z_score": round(z_val, 3),
                            "epistemic_status": EpistemicStatus.OBSERVED.value,
                        }
                    )

        # Consistently changing layers (modified in >= 75% of consecutive transitions AND CKA drift if available)
        consistent_layers = [
            x["layer_index"]
            for x in layer_aggregates
            if total_consecutive_transitions >= 2
            and x["consistency_ratio"] >= 0.75
            and x["mean_transition_relative_l2"] > 0.0
        ]

        consistently_drifting_representations: list[int] = []
        if representation_evolution and representation_evolution.get("consistent_drift_layers"):
            consistently_drifting_representations = list(
                representation_evolution["consistent_drift_layers"]
            )

        # Correlated layers on regression transitions (explicitly labeled CORRELATED unless causally patched)
        regression_correlated: list[dict[str, Any]] = []
        for ev in regression_events or []:
            for cl in ev.get("correlated_layers", []):
                regression_correlated.append(
                    {
                        "event_id": ev["event_id"],
                        "transition": f"{ev['previous_label']} -> {ev['first_affected_label']}",
                        "affected_category": ev["affected_category"],
                        **cl,
                    }
                )

        # Build compact heatmap matrix (`transitions × layers`) for UI & HTML/PDF reports
        sorted_layer_ids = sorted(all_layers_seen)
        consecutive_profiles = [p for p in transition_profiles if p["scope"] == "consecutive"]
        heatmap_rows: list[dict[str, Any]] = []
        for prof in consecutive_profiles:
            val_by_layer = {
                int(le["layer_index"]): le["mean_relative_l2"] for le in prof["layers"]
            }
            cos_by_layer = {
                int(le["layer_index"]): le["mean_cosine_similarity"] for le in prof["layers"]
            }
            heatmap_rows.append(
                {
                    "transition_label": f"{prof['source_label']} → {prof['target_label']}",
                    "source_checkpoint_id": prof["source_checkpoint_id"],
                    "target_checkpoint_id": prof["target_checkpoint_id"],
                    "relative_l2_by_layer": [val_by_layer.get(lid, 0.0) for lid in sorted_layer_ids],
                    "cosine_by_layer": [cos_by_layer.get(lid, 1.0) for lid in sorted_layer_ids],
                }
            )

        return {
            "epistemic_status": EpistemicStatus.OBSERVED.value,
            "layer_indices": sorted_layer_ids,
            "layer_aggregates": layer_aggregates,
            "most_modified_layers": most_modified,
            "least_modified_layers": least_modified,
            "unusually_large_change_layers": anomalous_layers,
            "consistently_changing_weight_layers": consistent_layers,
            "consistently_drifting_representation_layers": consistently_drifting_representations,
            "regression_associated_layers": regression_correlated,
            "heatmap": {
                "layer_indices": sorted_layer_ids,
                "rows": heatmap_rows,
            },
            "transition_profiles": transition_profiles,
        }
