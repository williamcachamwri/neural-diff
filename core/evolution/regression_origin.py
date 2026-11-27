"""Automatic Regression Origin Detection Engine (Section 6).

Identifies the FIRST observed checkpoint transition along each lineage associated with a
behavioral capability regression, correlating weight/representation shifts (`CORRELATED`)
and strictly distinguishing them from interventional activation patching (`CAUSALLY VERIFIED`).
"""

from __future__ import annotations

import uuid
from typing import Any

from core.evolution.dag import EvolutionDAG
from core.types import EpistemicStatus


def _confidence_tier(sample_size: int, p_value: float, score_drop: float) -> str:
    if sample_size >= 20 and p_value < 0.05 and score_drop >= 0.10:
        return "high_statistical_confidence"
    if sample_size >= 8 and score_drop >= 0.08:
        return "moderate_observed_effect"
    return "exploratory_small_sample"


class RegressionOriginEngine:
    """Detects first-occurrence behavioral regressions across multi-generation DAG lineages."""

    def __init__(
        self,
        *,
        overall_pass_rate_drop_threshold: float = 0.05,
        category_score_drop_threshold: float = 0.10,
    ) -> None:
        self.overall_pass_rate_drop_threshold = overall_pass_rate_drop_threshold
        self.category_score_drop_threshold = category_score_drop_threshold

    def detect(
        self,
        *,
        dag: EvolutionDAG,
        behavioral_evolution: dict[str, Any] | None,
        tensor_transitions: list[dict[str, Any]],
        representation_evolution: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not behavioral_evolution:
            return []

        eval_map: dict[str, dict[str, Any]] = {
            ev["checkpoint_id"]: ev
            for ev in behavioral_evolution.get("checkpoint_evaluations", [])
        }
        beh_edge_map: dict[tuple[str, str], dict[str, Any]] = {
            (tr["source_checkpoint_id"], tr["target_checkpoint_id"]): tr
            for tr in behavioral_evolution.get("consecutive_transitions", [])
        }
        tensor_edge_map: dict[tuple[str, str], dict[str, Any]] = {
            (tr["source_checkpoint_id"], tr["target_checkpoint_id"]): tr
            for tr in tensor_transitions
            if tr.get("scope") == "consecutive"
        }
        rep_edge_map: dict[tuple[str, str], dict[str, Any]] = {}
        if representation_evolution:
            rep_edge_map = {
                (tr["source_checkpoint_id"], tr["target_checkpoint_id"]): tr
                for tr in representation_evolution.get("consecutive_transitions", [])
            }

        events: list[dict[str, Any]] = []
        seen_origin_keys: set[tuple[str, str, str]] = set()

        for branch in dag.enumerate_branches():
            chain = [cid for cid in branch.checkpoint_ids if cid in eval_map]
            if len(chain) < 2:
                continue

            # Track which categories have ALREADY regressed earlier on this branch so we
            # pinpoint the FIRST transition where a regression originated!
            first_seen_on_branch: set[str] = set()
            leaf_eval = eval_map[chain[-1]]

            for idx in range(len(chain) - 1):
                prev_id = chain[idx]
                curr_id = chain[idx + 1]
                prev_ev = eval_map[prev_id]
                curr_ev = eval_map[curr_id]
                beh_tr = beh_edge_map.get((prev_id, curr_id))
                if beh_tr is None:
                    continue

                t_edge = tensor_edge_map.get((prev_id, curr_id))
                r_edge = rep_edge_map.get((prev_id, curr_id))
                curr_ckpt = dag.checkpoints_by_id[curr_id]
                prev_ckpt = dag.checkpoints_by_id[prev_id]

                # Gather correlated weight/CKA evidence & check whether interventional patching exists
                correlated_layers, weight_summary, act_summary, causal_info = (
                    self._build_mechanistic_context(t_edge, r_edge)
                )

                # 1. Check overall pass-rate / score regression
                pr_drop = round(float(prev_ev["pass_rate"]) - float(curr_ev["pass_rate"]), 6)
                sc_drop = round(float(prev_ev["mean_score"]) - float(curr_ev["mean_score"]), 6)
                if (
                    pr_drop >= self.overall_pass_rate_drop_threshold
                    or sc_drop >= self.category_score_drop_threshold
                ) and "overall" not in first_seen_on_branch:
                    first_seen_on_branch.add("overall")
                    origin_key = (prev_id, curr_id, "overall")
                    if origin_key not in seen_origin_keys:
                        seen_origin_keys.add(origin_key)
                        pstats = beh_tr["paired_statistics"]
                        n_cases = int(beh_tr["sample_size"])
                        pval = float(pstats["mcnemar_p_value"])
                        downstream_ids = [
                            d_id
                            for d_id in chain[idx + 1 :]
                            if float(eval_map[d_id]["pass_rate"])
                            <= float(prev_ev["pass_rate"]) - self.overall_pass_rate_drop_threshold
                        ]
                        events.append(
                            {
                                "event_id": f"reg-orig-{uuid.uuid4().hex[:8]}",
                                "branch_id": branch.branch_id,
                                "first_affected_checkpoint_id": curr_id,
                                "first_affected_label": curr_ckpt.version_label,
                                "previous_checkpoint_id": prev_id,
                                "previous_label": prev_ckpt.version_label,
                                "transformation_type": beh_tr["transformation_type"],
                                "affected_category": "overall",
                                "pass_rate_before": prev_ev["pass_rate"],
                                "pass_rate_after": curr_ev["pass_rate"],
                                "pass_rate_drop": pr_drop,
                                "score_before": prev_ev["mean_score"],
                                "score_after": curr_ev["mean_score"],
                                "score_drop": max(pr_drop, sc_drop),
                                "failed_test_case_ids": list(beh_tr["newly_introduced_failures"]),
                                "survives_to_leaf": (
                                    float(leaf_eval["pass_rate"])
                                    <= float(prev_ev["pass_rate"]) - self.overall_pass_rate_drop_threshold
                                ),
                                "downstream_affected_checkpoints": downstream_ids,
                                "correlated_layers": correlated_layers,
                                "weight_changes": weight_summary,
                                "activation_changes": act_summary,
                                "statistical_confidence": {
                                    "sample_size": n_cases,
                                    "mcnemar_p_value": pval,
                                    "bootstrap_ci95": [
                                        pstats["bootstrap_ci95_low"],
                                        pstats["bootstrap_ci95_high"],
                                    ],
                                    "confidence_level": _confidence_tier(n_cases, pval, max(pr_drop, sc_drop)),
                                },
                                "causal_verification": causal_info,
                                "epistemic_status": (
                                    EpistemicStatus.CAUSALLY_VERIFIED.value
                                    if causal_info["causal_verification_exists"]
                                    else EpistemicStatus.CORRELATED.value
                                ),
                                "interpretation_note": (
                                    "Behavioral drop is directly OBSERVED; associated weight and CKA shifts are CORRELATED unless interventional patching recovers the clean logit margin."
                                ),
                            }
                        )

                # 2. Check per-category first-origin regressions
                for c_delta in beh_tr.get("category_deltas", []):
                    cat = str(c_delta["category"])
                    c_sc_drop = round(
                        float(c_delta["mean_score_before"]) - float(c_delta["mean_score_after"]), 6
                    )
                    c_pr_drop = round(
                        float(c_delta["pass_rate_before"]) - float(c_delta["pass_rate_after"]), 6
                    )
                    if (
                        c_sc_drop >= self.category_score_drop_threshold
                        or c_pr_drop >= self.category_score_drop_threshold
                    ) and cat not in first_seen_on_branch:
                        first_seen_on_branch.add(cat)
                        origin_key = (prev_id, curr_id, cat)
                        if origin_key in seen_origin_keys:
                            continue
                        seen_origin_keys.add(origin_key)

                        cat_failed_ids = [
                            r["case_id"]
                            for r in curr_ev["case_results"]
                            if r["category"] == cat
                            and r["case_id"] in set(beh_tr["newly_introduced_failures"])
                        ]
                        leaf_cat = leaf_eval["categories"].get(cat, {})
                        survives = float(leaf_cat.get("mean_score", 1.0)) <= (
                            float(c_delta["mean_score_before"]) - self.category_score_drop_threshold
                        )
                        n_cat = int(c_delta["sample_size"])

                        events.append(
                            {
                                "event_id": f"reg-orig-{uuid.uuid4().hex[:8]}",
                                "branch_id": branch.branch_id,
                                "first_affected_checkpoint_id": curr_id,
                                "first_affected_label": curr_ckpt.version_label,
                                "previous_checkpoint_id": prev_id,
                                "previous_label": prev_ckpt.version_label,
                                "transformation_type": beh_tr["transformation_type"],
                                "affected_category": cat,
                                "pass_rate_before": c_delta["pass_rate_before"],
                                "pass_rate_after": c_delta["pass_rate_after"],
                                "pass_rate_drop": c_pr_drop,
                                "score_before": c_delta["mean_score_before"],
                                "score_after": c_delta["mean_score_after"],
                                "score_drop": max(c_sc_drop, c_pr_drop),
                                "failed_test_case_ids": cat_failed_ids,
                                "survives_to_leaf": survives,
                                "downstream_affected_checkpoints": [
                                    d_id
                                    for d_id in chain[idx + 1 :]
                                    if float(
                                        eval_map[d_id]["categories"].get(cat, {}).get("mean_score", 1.0)
                                    )
                                    <= float(c_delta["mean_score_before"]) - self.category_score_drop_threshold
                                ],
                                "correlated_layers": correlated_layers,
                                "weight_changes": weight_summary,
                                "activation_changes": act_summary,
                                "statistical_confidence": {
                                    "sample_size": n_cat,
                                    "mcnemar_p_value": beh_tr["paired_statistics"]["mcnemar_p_value"],
                                    "wilson_ci95_after": c_delta["wilson_ci95_after"],
                                    "confidence_level": _confidence_tier(n_cat, 1.0, max(c_sc_drop, c_pr_drop)),
                                },
                                "causal_verification": causal_info,
                                "epistemic_status": (
                                    EpistemicStatus.CAUSALLY_VERIFIED.value
                                    if causal_info["causal_verification_exists"]
                                    else EpistemicStatus.CORRELATED.value
                                ),
                                "interpretation_note": (
                                    "Category drop is OBSERVED on deterministic rubric cases; co-occurring layer weight/CKA changes represent correlation, not standalone proof of causation."
                                ),
                            }
                        )

        events.sort(
            key=lambda e: (
                0 if e["affected_category"] == "overall" else 1,
                -float(e["score_drop"]),
            )
        )
        return events

    @staticmethod
    def _build_mechanistic_context(
        t_edge: dict[str, Any] | None,
        r_edge: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
        correlated_layers: list[dict[str, Any]] = []
        weight_summary: dict[str, Any] = {}
        act_summary: dict[str, Any] = {}

        cka_map: dict[int, float | None] = {}
        if r_edge:
            act_summary = {
                "mean_linear_cka": r_edge.get("mean_cka"),
                "mean_representation_drift": r_edge.get("mean_drift"),
                "mean_hidden_cosine": r_edge.get("mean_hidden_cosine"),
                "mean_logit_js_divergence": r_edge.get("mean_logit_js_divergence"),
            }
            for k, v in (r_edge.get("cka_by_layer") or {}).items():
                cka_map[int(k)] = v

        if t_edge:
            weight_summary = {
                "relative_l2_distance": t_edge.get("relative_l2_distance"),
                "l2_distance": t_edge.get("l2_distance"),
                "max_abs_diff": t_edge.get("max_abs_diff"),
                "modified_parameter_pct": t_edge.get("modified_parameter_pct"),
                "sparsity_delta": t_edge.get("sparsity_delta"),
            }
            per_layer = sorted(
                t_edge.get("per_layer_changes", []),
                key=lambda x: x.get("mean_relative_l2") or 0.0,
                reverse=True,
            )
            for ls in per_layer[:3]:
                lid = int(ls["layer_index"])
                correlated_layers.append(
                    {
                        "layer_index": lid,
                        "weight_mean_relative_l2": ls.get("mean_relative_l2"),
                        "weight_changed_pct": ls.get("changed_pct"),
                        "linear_cka": cka_map.get(lid),
                        "evidence_relationship": EpistemicStatus.CORRELATED.value,
                    }
                )

        patching = r_edge.get("causal_patching") if r_edge else None
        has_verified_patch = bool(
            isinstance(patching, dict)
            and abs(float(patching.get("peak_recovery_fraction", 0.0))) >= 0.20
        )
        causal_info = {
            "causal_verification_exists": has_verified_patch,
            "epistemic_status": (
                EpistemicStatus.CAUSALLY_VERIFIED.value
                if has_verified_patch
                else EpistemicStatus.CORRELATED.value
            ),
            "most_causal_layer": patching.get("most_causal_layer") if isinstance(patching, dict) else None,
            "peak_recovery_fraction": (
                patching.get("peak_recovery_fraction") if isinstance(patching, dict) else None
            ),
            "scope_disclaimer": (
                patching.get("scope")
                if has_verified_patch and isinstance(patching, dict)
                else "No interventional recovery >= 20% verified; layer associations are strictly observational/correlational."
            ),
        }
        return correlated_layers, weight_summary, act_summary, causal_info
