"""Evidence-Backed Evolution Insights Generator (Section 9).

Synthesizes strictly measurement-backed findings from weight, layer, behavioral,
representation, and regression-origin results. Zero unsupported prose.
"""

from __future__ import annotations

from typing import Any

from core.types import EpistemicStatus


def generate_evolution_insights(
    *,
    tensor_transitions: list[dict[str, Any]],
    layer_evolution: dict[str, Any],
    behavioral_evolution: dict[str, Any] | None,
    representation_evolution: dict[str, Any] | None,
    regression_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build structured, verifiable insights backed by exact metrics, checkpoint IDs, and case IDs."""
    insights: list[dict[str, Any]] = []

    # 1. Largest weight modification / reconstruction shift across consecutive transitions
    consec_tensors = [
        t
        for t in tensor_transitions
        if t.get("scope") == "consecutive" and t.get("relative_l2_distance") is not None
    ]
    if consec_tensors:
        peak_wt = max(consec_tensors, key=lambda t: float(t["relative_l2_distance"] or 0.0))
        rel_l2 = float(peak_wt["relative_l2_distance"] or 0.0)
        if rel_l2 > 0.0:
            insights.append(
                {
                    "insight_id": "insight-peak-weight-shift",
                    "domain": "weight_evolution",
                    "statement": (
                        f"{peak_wt['transformation_type']} ({peak_wt['source_label']} → {peak_wt['target_label']}) "
                        f"introduced the largest observed weight perturbation (Relative L2 = {rel_l2:.6f}, "
                        f"RMSE = {float(peak_wt['rmse'] or 0.0):.6f}, {float(peak_wt['modified_parameter_pct']):.2f}% parameters modified)."
                    ),
                    "epistemic_status": EpistemicStatus.OBSERVED.value,
                    "checkpoints": [
                        peak_wt["source_checkpoint_id"],
                        peak_wt["target_checkpoint_id"],
                    ],
                    "evaluation_cases": [],
                    "supporting_measurements": {
                        "relative_l2_distance": rel_l2,
                        "l2_distance": peak_wt["l2_distance"],
                        "rmse": peak_wt["rmse"],
                        "max_abs_diff": peak_wt["max_abs_diff"],
                        "modified_parameter_pct": peak_wt["modified_parameter_pct"],
                    },
                }
            )

    # 2. Direct Base-to-Current cumulative distance vs sum of consecutive step distances
    base_transitions = [
        t
        for t in tensor_transitions
        if t.get("scope") == "base_to_current"
        and t.get("path_cancellation_ratio") is not None
        and float(t.get("l2_distance") or 0.0) > 0.0
        and t["source_checkpoint_id"] != t["target_checkpoint_id"]
    ]
    if base_transitions:
        deepest = max(
            base_transitions,
            key=lambda t: float(t.get("sum_of_step_l2_along_path") or 0.0),
        )
        ratio = float(deepest["path_cancellation_ratio"])
        direct_l2 = float(deepest["l2_distance"])
        sum_l2 = float(deepest["sum_of_step_l2_along_path"])
        if abs(sum_l2 - direct_l2) > 1e-6:
            insights.append(
                {
                    "insight_id": "insight-cumulative-geometry",
                    "domain": "weight_evolution",
                    "statement": (
                        f"Direct cumulative L2 distance from {deepest['source_label']} to {deepest['target_label']} "
                        f"is {direct_l2:.6f} versus {sum_l2:.6f} summed across individual transitions "
                        f"(direct/sum ratio = {ratio:.4f}), reflecting non-collinear multi-stage parameter updates."
                    ),
                    "epistemic_status": EpistemicStatus.OBSERVED.value,
                    "checkpoints": [
                        deepest["source_checkpoint_id"],
                        deepest["target_checkpoint_id"],
                    ],
                    "evaluation_cases": [],
                    "supporting_measurements": {
                        "direct_base_to_target_l2": direct_l2,
                        "sum_of_consecutive_step_l2": sum_l2,
                        "path_cancellation_ratio": ratio,
                    },
                }
            )

    # 3. Peak representation drift transition
    if representation_evolution:
        rep_edges = [
            e
            for e in representation_evolution.get("consecutive_transitions", [])
            if e.get("mean_drift") is not None
        ]
        if rep_edges:
            peak_rep = max(rep_edges, key=lambda e: float(e["mean_drift"] or 0.0))
            drift = float(peak_rep["mean_drift"] or 0.0)
            if drift > 1e-6:
                insights.append(
                    {
                        "insight_id": "insight-peak-representation-drift",
                        "domain": "representation_evolution",
                        "statement": (
                            f"Most representation drift occurred between {peak_rep['source_label']} and "
                            f"{peak_rep['target_label']} ({peak_rep['transformation_type']}: "
                            f"mean Linear CKA = {float(peak_rep['mean_cka']):.6f}, drift = {drift:.6f})."
                        ),
                        "epistemic_status": EpistemicStatus.OBSERVED.value,
                        "checkpoints": [
                            peak_rep["source_checkpoint_id"],
                            peak_rep["target_checkpoint_id"],
                        ],
                        "evaluation_cases": [],
                        "supporting_measurements": {
                            "mean_linear_cka": peak_rep["mean_cka"],
                            "mean_representation_drift": drift,
                            "mean_logit_js_divergence": peak_rep.get("mean_logit_js_divergence"),
                        },
                    }
                )

    # 4. Behavioral trajectory reversal ("Capability improved after X but declined after Y" along a sequential chain)
    if behavioral_evolution:
        beh_edges = behavioral_evolution.get("consecutive_transitions", [])
        if len(beh_edges) >= 2:
            by_src: dict[str, list[dict[str, Any]]] = {}
            for edge in beh_edges:
                by_src.setdefault(str(edge["source_checkpoint_id"]), []).append(edge)

            reversal_found = False
            for first_edge in beh_edges:
                if reversal_found:
                    break
                mid_id = str(first_edge["target_checkpoint_id"])
                # Walk downstream reachable edges from `mid_id`
                downstream: list[dict[str, Any]] = []
                stack = list(by_src.get(mid_id, []))
                visited_e: set[tuple[str, str]] = set()
                while stack:
                    cur_e = stack.pop(0)
                    ekey = (str(cur_e["source_checkpoint_id"]), str(cur_e["target_checkpoint_id"]))
                    if ekey in visited_e:
                        continue
                    visited_e.add(ekey)
                    downstream.append(cur_e)
                    stack.extend(by_src.get(str(cur_e["target_checkpoint_id"]), []))

                first_gains = {
                    str(cd["category"]): float(cd["mean_score_delta"])
                    for cd in first_edge.get("category_deltas", [])
                    if float(cd["mean_score_delta"]) >= 0.05
                }
                for later_edge in downstream:
                    if reversal_found:
                        break
                    for cd in later_edge.get("category_deltas", []):
                        cat = str(cd["category"])
                        drop_delta = float(cd["mean_score_delta"])
                        if cat in first_gains and drop_delta <= -0.05:
                            gain_val = first_gains[cat]
                            insights.append(
                                {
                                    "insight_id": f"insight-trajectory-reversal-{cat}",
                                    "domain": "behavioral_evolution",
                                    "statement": (
                                        f"Category '{cat}' improved after {first_edge['transformation_type']} "
                                        f"({first_edge['source_label']} → {first_edge['target_label']}, Δ={gain_val:+.3f}) "
                                        f"but declined downstream after {later_edge['transformation_type']} "
                                        f"({later_edge['source_label']} → {later_edge['target_label']}, Δ={drop_delta:+.3f}, n={cd['sample_size']})."
                                    ),
                                    "epistemic_status": EpistemicStatus.OBSERVED.value,
                                    "checkpoints": [
                                        first_edge["source_checkpoint_id"],
                                        first_edge["target_checkpoint_id"],
                                        later_edge["target_checkpoint_id"],
                                    ],
                                    "evaluation_cases": list(
                                        later_edge.get("newly_introduced_failures", [])
                                    ),
                                    "supporting_measurements": {
                                        "category": cat,
                                        "sample_size": cd["sample_size"],
                                        "initial_gain_delta": gain_val,
                                        "subsequent_drop_delta": drop_delta,
                                    },
                                }
                            )
                            reversal_found = True
                            break

    # 5. Primary regression origin finding
    if regression_events:
        top_reg = regression_events[0]
        insights.append(
            {
                "insight_id": f"insight-regression-origin-{top_reg['event_id']}",
                "domain": "regression_origin",
                "statement": (
                    f"{top_reg['transformation_type']} ({top_reg['previous_label']} → {top_reg['first_affected_label']}) "
                    f"is the first observed transition associated with regression in '{top_reg['affected_category']}' "
                    f"(pass rate {float(top_reg['pass_rate_before']) * 100:.1f}% → {float(top_reg['pass_rate_after']) * 100:.1f}%, "
                    f"failed cases: {', '.join(top_reg['failed_test_case_ids']) or 'score drop'})."
                ),
                "epistemic_status": top_reg["epistemic_status"],
                "checkpoints": [
                    top_reg["previous_checkpoint_id"],
                    top_reg["first_affected_checkpoint_id"],
                ],
                "evaluation_cases": list(top_reg["failed_test_case_ids"]),
                "supporting_measurements": {
                    "pass_rate_before": top_reg["pass_rate_before"],
                    "pass_rate_after": top_reg["pass_rate_after"],
                    "score_drop": top_reg["score_drop"],
                    "confidence_level": top_reg["statistical_confidence"]["confidence_level"],
                    "causal_verification_exists": top_reg["causal_verification"]["causal_verification_exists"],
                },
            }
        )

    # 6. Anomalous layer concentration insight
    anomalous = layer_evolution.get("unusually_large_change_layers", [])
    if anomalous:
        top_anom = anomalous[0]
        insights.append(
            {
                "insight_id": "insight-anomalous-layer-shift",
                "domain": "layer_evolution",
                "statement": (
                    f"Layer {top_anom['layer_index']} exhibited an unusually concentrated weight modification "
                    f"(peak Relative L2 = {float(top_anom['peak_relative_l2']):.6f}, z-score = {top_anom['z_score']:+.2f})."
                ),
                "epistemic_status": EpistemicStatus.OBSERVED.value,
                "checkpoints": [],
                "evaluation_cases": [],
                "supporting_measurements": top_anom,
            }
        )

    return insights
