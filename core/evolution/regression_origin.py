"""Automatic Regression Origin Detection Engine (Section 6).

Identifies the FIRST observed checkpoint transition along each lineage associated with a
behavioral capability regression, enforcing strict scientific epistemic tiers:
* `OBSERVED`: Behavioral regression observed without aligned mechanistic covariates.
* `CORRELATED`: Behavioral regression co-occurring with measured weight/CKA shifts.
* `CAUSAL SUPPORT`: Transition-level diagnostic-prompt patching recovers >= 20% logit margin
  (supports a mechanistic locus on the transition without proving category-wide causation).
* `CAUSALLY VERIFIED`: Reserved strictly for case-targeted counterfactual interventions executed
  directly on the failed test case(s) with target/foil margin and control-layer checks.

Also enforces sample-size floors (`is_exploratory` when `n < min_sample_size`), separates
`score_only`, `pass_rate_only`, and `pass_rate_and_score` drops, applies Holm-Bonferroni
step-down multiple-comparison correction, and guarantees full DAG edge coverage even when
root-to-leaf branch enumeration exceeds `max_branches`.
"""

from __future__ import annotations

import uuid
from typing import Any

from core.evolution.behavioral_evolution import holm_bonferroni_adjust
from core.evolution.dag import BranchEnumerationResult, EvolutionDAG
from core.types import EpistemicStatus


def _classify_regression_kind(
    pr_drop: float,
    sc_drop: float,
    *,
    pr_threshold: float,
    sc_threshold: float,
) -> str:
    pr_hit = pr_drop >= pr_threshold
    sc_hit = sc_drop >= sc_threshold
    if pr_hit and sc_hit:
        return "pass_rate_and_score"
    if pr_hit:
        return "pass_rate_only"
    return "score_only"


def _confidence_tier(
    *,
    sample_size: int,
    min_required_sample_size: int,
    adjusted_p_value: float,
    alpha: float,
    effect_drop: float,
) -> tuple[str, bool, bool]:
    """Return `(confidence_level, is_exploratory, statistically_significant)`."""
    is_exploratory = sample_size < min_required_sample_size
    if is_exploratory:
        return ("exploratory_small_sample", True, False)
    if adjusted_p_value < alpha and effect_drop >= 0.08:
        return ("statistically_significant", False, True)
    if effect_drop >= 0.08:
        return ("moderate_observed_effect", False, False)
    return ("low_effect_size", False, False)


class RegressionOriginEngine:
    """Detects first-occurrence behavioral regressions across multi-generation DAG lineages."""

    def __init__(
        self,
        *,
        overall_pass_rate_drop_threshold: float = 0.05,
        category_score_drop_threshold: float = 0.10,
        min_category_sample_size: int = 8,
        min_overall_sample_size: int = 20,
        significance_alpha: float = 0.05,
        max_branches: int = 256,
    ) -> None:
        self.overall_pass_rate_drop_threshold = overall_pass_rate_drop_threshold
        self.category_score_drop_threshold = category_score_drop_threshold
        self.min_category_sample_size = max(1, int(min_category_sample_size))
        self.min_overall_sample_size = max(1, int(min_overall_sample_size))
        self.significance_alpha = significance_alpha
        self.max_branches = max(1, int(max_branches))
        self.last_branch_enumeration: BranchEnumerationResult | None = None

    def detect(
        self,
        *,
        dag: EvolutionDAG,
        behavioral_evolution: dict[str, Any] | None,
        tensor_transitions: list[dict[str, Any]],
        representation_evolution: dict[str, Any] | None,
        case_interventions: dict[tuple[str, str, str], dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if not behavioral_evolution:
            self.last_branch_enumeration = dag.enumerate_branches_with_meta(
                max_paths=self.max_branches
            )
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

        branch_enum = dag.enumerate_branches_with_meta(max_paths=self.max_branches)
        self.last_branch_enumeration = branch_enum

        events: list[dict[str, Any]] = []
        seen_origin_keys: set[tuple[str, str, str]] = set()
        covered_edges: set[tuple[str, str]] = set()

        def _evaluate_transition_step(
            *,
            prev_id: str,
            curr_id: str,
            branch_id: str,
            downstream_chain: list[str],
            leaf_ids_for_edge: list[str],
            already_regressed_cats: set[str],
        ) -> None:
            covered_edges.add((prev_id, curr_id))
            prev_ev = eval_map.get(prev_id)
            curr_ev = eval_map.get(curr_id)
            beh_tr = beh_edge_map.get((prev_id, curr_id))
            if prev_ev is None or curr_ev is None or beh_tr is None:
                return

            t_edge = tensor_edge_map.get((prev_id, curr_id))
            r_edge = rep_edge_map.get((prev_id, curr_id))
            curr_ckpt = dag.checkpoints_by_id[curr_id]
            prev_ckpt = dag.checkpoints_by_id[prev_id]

            correlated_layers, weight_summary, act_summary, base_causal = (
                self._build_mechanistic_context(t_edge, r_edge)
            )

            # Build family of tested p-values on this transition (overall + all categories)
            pstats = beh_tr["paired_statistics"]
            cat_deltas = list(beh_tr.get("category_deltas", []))
            raw_family_ps: list[float] = [float(pstats["mcnemar_p_value"])] + [
                float(cd.get("mcnemar_p_value", pstats["mcnemar_p_value"])) for cd in cat_deltas
            ]
            adj_family_ps = holm_bonferroni_adjust(raw_family_ps)
            overall_adj_p = adj_family_ps[0]
            cat_adj_p_map = {
                str(cd["category"]): adj_family_ps[i + 1] for i, cd in enumerate(cat_deltas)
            }
            num_hypotheses = len(raw_family_ps)

            # 1. Check overall pass-rate / score regression
            pr_drop = round(float(prev_ev["pass_rate"]) - float(curr_ev["pass_rate"]), 6)
            sc_drop = round(float(prev_ev["mean_score"]) - float(curr_ev["mean_score"]), 6)
            if (
                pr_drop >= self.overall_pass_rate_drop_threshold
                or sc_drop >= self.category_score_drop_threshold
            ) and "overall" not in already_regressed_cats:
                already_regressed_cats.add("overall")
                origin_key = (prev_id, curr_id, "overall")
                if origin_key not in seen_origin_keys:
                    seen_origin_keys.add(origin_key)
                    n_cases = int(beh_tr["sample_size"])
                    raw_p = float(pstats["mcnemar_p_value"])
                    reg_kind = _classify_regression_kind(
                        pr_drop,
                        sc_drop,
                        pr_threshold=self.overall_pass_rate_drop_threshold,
                        sc_threshold=self.category_score_drop_threshold,
                    )
                    conf_level, is_expl, is_sig = _confidence_tier(
                        sample_size=n_cases,
                        min_required_sample_size=self.min_overall_sample_size,
                        adjusted_p_value=overall_adj_p,
                        alpha=self.significance_alpha,
                        effect_drop=max(pr_drop, sc_drop),
                    )
                    failed_ids = list(beh_tr["newly_introduced_failures"])
                    ev_causal, ev_status, interp = self._resolve_event_epistemic_status(
                        prev_id=prev_id,
                        curr_id=curr_id,
                        failed_case_ids=failed_ids,
                        correlated_layers=correlated_layers,
                        weight_summary=weight_summary,
                        act_summary=act_summary,
                        base_causal=base_causal,
                        case_interventions=case_interventions,
                        is_exploratory=is_expl,
                        sample_size=n_cases,
                        min_required=self.min_overall_sample_size,
                    )
                    downstream_ids = [
                        d_id
                        for d_id in downstream_chain
                        if d_id in eval_map
                        and (
                            float(eval_map[d_id]["pass_rate"])
                            <= float(prev_ev["pass_rate"]) - self.overall_pass_rate_drop_threshold
                            or float(eval_map[d_id]["mean_score"])
                            <= float(prev_ev["mean_score"]) - self.category_score_drop_threshold
                        )
                    ]
                    survives = any(
                        lid in eval_map
                        and (
                            float(eval_map[lid]["pass_rate"])
                            <= float(prev_ev["pass_rate"]) - self.overall_pass_rate_drop_threshold
                            or float(eval_map[lid]["mean_score"])
                            <= float(prev_ev["mean_score"]) - self.category_score_drop_threshold
                        )
                        for lid in leaf_ids_for_edge
                    )
                    events.append(
                        {
                            "event_id": f"reg-orig-{uuid.uuid4().hex[:8]}",
                            "branch_id": branch_id,
                            "first_affected_checkpoint_id": curr_id,
                            "first_affected_label": curr_ckpt.version_label,
                            "previous_checkpoint_id": prev_id,
                            "previous_label": prev_ckpt.version_label,
                            "transformation_type": beh_tr["transformation_type"],
                            "affected_category": "overall",
                            "regression_kind": reg_kind,
                            "pass_rate_before": prev_ev["pass_rate"],
                            "pass_rate_after": curr_ev["pass_rate"],
                            "pass_rate_drop": pr_drop,
                            "score_before": prev_ev["mean_score"],
                            "score_after": curr_ev["mean_score"],
                            "score_drop": max(pr_drop, sc_drop),
                            "raw_score_drop": sc_drop,
                            "failed_test_case_ids": failed_ids,
                            "survives_to_leaf": survives,
                            "downstream_affected_checkpoints": downstream_ids,
                            "correlated_layers": correlated_layers,
                            "weight_changes": weight_summary,
                            "activation_changes": act_summary,
                            "statistical_confidence": {
                                "sample_size": n_cases,
                                "min_required_sample_size": self.min_overall_sample_size,
                                "is_exploratory": is_expl,
                                "statistically_significant": is_sig,
                                "mcnemar_p_value": raw_p,
                                "holm_bonferroni_p_value": overall_adj_p,
                                "multiple_comparison_method": "holm-bonferroni",
                                "hypotheses_tested": num_hypotheses,
                                "bootstrap_ci95": [
                                    pstats["bootstrap_ci95_low"],
                                    pstats["bootstrap_ci95_high"],
                                ],
                                "confidence_level": conf_level,
                            },
                            "causal_verification": ev_causal,
                            "epistemic_status": ev_status,
                            "interpretation_note": interp,
                        }
                    )

            # 2. Check per-category first-origin regressions
            for c_delta in cat_deltas:
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
                ) and cat not in already_regressed_cats:
                    already_regressed_cats.add(cat)
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
                    n_cat = int(c_delta["sample_size"])
                    c_raw_p = float(
                        c_delta.get("mcnemar_p_value", pstats["mcnemar_p_value"])
                    )
                    c_adj_p = float(cat_adj_p_map.get(cat, c_raw_p))
                    c_reg_kind = _classify_regression_kind(
                        c_pr_drop,
                        c_sc_drop,
                        pr_threshold=self.category_score_drop_threshold,
                        sc_threshold=self.category_score_drop_threshold,
                    )
                    c_conf_level, c_is_expl, c_is_sig = _confidence_tier(
                        sample_size=n_cat,
                        min_required_sample_size=self.min_category_sample_size,
                        adjusted_p_value=c_adj_p,
                        alpha=self.significance_alpha,
                        effect_drop=max(c_sc_drop, c_pr_drop),
                    )
                    c_causal, c_status, c_interp = self._resolve_event_epistemic_status(
                        prev_id=prev_id,
                        curr_id=curr_id,
                        failed_case_ids=cat_failed_ids,
                        correlated_layers=correlated_layers,
                        weight_summary=weight_summary,
                        act_summary=act_summary,
                        base_causal=base_causal,
                        case_interventions=case_interventions,
                        is_exploratory=c_is_expl,
                        sample_size=n_cat,
                        min_required=self.min_category_sample_size,
                    )
                    survives = any(
                        lid in eval_map
                        and float(eval_map[lid]["categories"].get(cat, {}).get("mean_score", 1.0))
                        <= float(c_delta["mean_score_before"]) - self.category_score_drop_threshold
                        for lid in leaf_ids_for_edge
                    )

                    events.append(
                        {
                            "event_id": f"reg-orig-{uuid.uuid4().hex[:8]}",
                            "branch_id": branch_id,
                            "first_affected_checkpoint_id": curr_id,
                            "first_affected_label": curr_ckpt.version_label,
                            "previous_checkpoint_id": prev_id,
                            "previous_label": prev_ckpt.version_label,
                            "transformation_type": beh_tr["transformation_type"],
                            "affected_category": cat,
                            "regression_kind": c_reg_kind,
                            "pass_rate_before": c_delta["pass_rate_before"],
                            "pass_rate_after": c_delta["pass_rate_after"],
                            "pass_rate_drop": c_pr_drop,
                            "score_before": c_delta["mean_score_before"],
                            "score_after": c_delta["mean_score_after"],
                            "score_drop": max(c_sc_drop, c_pr_drop),
                            "raw_score_drop": c_sc_drop,
                            "failed_test_case_ids": cat_failed_ids,
                            "survives_to_leaf": survives,
                            "downstream_affected_checkpoints": [
                                d_id
                                for d_id in downstream_chain
                                if d_id in eval_map
                                and float(
                                    eval_map[d_id]["categories"].get(cat, {}).get("mean_score", 1.0)
                                )
                                <= float(c_delta["mean_score_before"])
                                - self.category_score_drop_threshold
                            ],
                            "correlated_layers": correlated_layers,
                            "weight_changes": weight_summary,
                            "activation_changes": act_summary,
                            "statistical_confidence": {
                                "sample_size": n_cat,
                                "min_required_sample_size": self.min_category_sample_size,
                                "is_exploratory": c_is_expl,
                                "statistically_significant": c_is_sig,
                                "mcnemar_p_value": c_raw_p,
                                "holm_bonferroni_p_value": c_adj_p,
                                "multiple_comparison_method": "holm-bonferroni",
                                "hypotheses_tested": num_hypotheses,
                                "wilson_ci95_after": c_delta["wilson_ci95_after"],
                                "confidence_level": c_conf_level,
                            },
                            "causal_verification": c_causal,
                            "epistemic_status": c_status,
                            "interpretation_note": c_interp,
                        }
                    )

        # Primary pass across enumerated branches
        for branch in branch_enum.branches:
            chain = [cid for cid in branch.checkpoint_ids if cid in eval_map]
            if len(chain) < 2:
                continue
            first_seen_on_branch: set[str] = set()
            leaf_id = chain[-1]
            for idx in range(len(chain) - 1):
                _evaluate_transition_step(
                    prev_id=chain[idx],
                    curr_id=chain[idx + 1],
                    branch_id=branch.branch_id,
                    downstream_chain=chain[idx + 1 :],
                    leaf_ids_for_edge=[leaf_id],
                    already_regressed_cats=first_seen_on_branch,
                )

        # Guaranteed complete-DAG fallback: if branch paths were truncated (`total > max_branches`),
        # inspect every remaining DAG edge in topological order so zero edges are ever missed!
        if branch_enum.truncated:
            all_leaves = set(dag.leaves())
            topo_order = dag.topological_order()
            for u in topo_order:
                for v in dag.children_of.get(u, []):
                    if (u, v) in covered_edges:
                        continue
                    desc = dag.descendants_of(v) | {v}
                    reachable_leaves = sorted(desc & all_leaves)
                    anc_regressed: set[str] = {
                        ev["affected_category"]
                        for ev in events
                        if ev["first_affected_checkpoint_id"] in (dag.ancestors_of(u) | {u})
                    }
                    _evaluate_transition_step(
                        prev_id=u,
                        curr_id=v,
                        branch_id="dag-edge-scan",
                        downstream_chain=[x for x in topo_order if x in desc],
                        leaf_ids_for_edge=reachable_leaves,
                        already_regressed_cats=anc_regressed,
                    )

        events.sort(
            key=lambda e: (
                0 if e["affected_category"] == "overall" else 1,
                -float(e["score_drop"]),
            )
        )
        return events

    @staticmethod
    def _resolve_event_epistemic_status(
        *,
        prev_id: str,
        curr_id: str,
        failed_case_ids: list[str],
        correlated_layers: list[dict[str, Any]],
        weight_summary: dict[str, Any],
        act_summary: dict[str, Any],
        base_causal: dict[str, Any],
        case_interventions: dict[tuple[str, str, str], dict[str, Any]] | None,
        is_exploratory: bool,
        sample_size: int,
        min_required: int,
    ) -> tuple[dict[str, Any], str, str]:
        """Classify event into OBSERVED, CORRELATED, CAUSAL SUPPORT, or CAUSALLY VERIFIED."""
        verified_cases: list[str] = []
        best_case_patch: dict[str, Any] | None = None

        if case_interventions and failed_case_ids:
            for cid in failed_case_ids:
                cp = case_interventions.get((prev_id, curr_id, cid))
                if not isinstance(cp, dict):
                    continue
                rec_frac = abs(float(cp.get("peak_recovery_fraction", 0.0)))
                margin_ok = abs(
                    float(cp.get("clean_logit_diff", 0.0))
                    - float(cp.get("corrupted_logit_diff", 0.0))
                ) > 1e-6
                control_ok = bool(cp.get("control_verified", False))
                if rec_frac >= 0.20 and margin_ok and control_ok:
                    verified_cases.append(cid)
                    if best_case_patch is None or rec_frac > abs(
                        float(best_case_patch.get("peak_recovery_fraction", 0.0))
                    ):
                        best_case_patch = cp

        expl_prefix = (
            f"Exploratory sample (n={sample_size} < {min_required} min threshold; insufficient for capability-loss claims). "
            if is_exploratory
            else ""
        )

        # Tier 4: CAUSALLY VERIFIED (only when case-targeted patching verified the actual failed case(s))
        if verified_cases and best_case_patch is not None:
            layer_idx = best_case_patch.get("most_causal_layer")
            rec_val = best_case_patch.get("peak_recovery_fraction")
            scope_msg = (
                f"Causally verified strictly on failed prompt(s) [{', '.join(verified_cases)}] "
                f"at layer L{layer_idx} with target/foil and negative-control checks."
            )
            return (
                {
                    "causal_verification_exists": True,
                    "causal_support_exists": True,
                    "verified_case_ids": verified_cases,
                    "epistemic_status": EpistemicStatus.CAUSALLY_VERIFIED.value,
                    "most_causal_layer": layer_idx,
                    "peak_recovery_fraction": rec_val,
                    "scope_disclaimer": scope_msg,
                },
                EpistemicStatus.CAUSALLY_VERIFIED.value,
                expl_prefix + scope_msg,
            )

        # Tier 3: CAUSAL SUPPORT (generic diagnostic-prompt transition patching >= 20%)
        if base_causal["transition_patching_support"]:
            layer_idx = base_causal["most_causal_layer"]
            rec_val = base_causal["peak_recovery_fraction"]
            scope_msg = (
                f"Diagnostic-prompt residual patching at L{layer_idx} recovered {float(rec_val or 0.0) * 100:.1f}% "
                "logit margin (CAUSAL SUPPORT for transition locus; not case-specific causal proof for this rubric)."
            )
            return (
                {
                    "causal_verification_exists": False,
                    "causal_support_exists": True,
                    "verified_case_ids": [],
                    "epistemic_status": EpistemicStatus.CAUSAL_SUPPORT.value,
                    "most_causal_layer": layer_idx,
                    "peak_recovery_fraction": rec_val,
                    "scope_disclaimer": scope_msg,
                },
                EpistemicStatus.CAUSAL_SUPPORT.value,
                expl_prefix + scope_msg,
            )

        # Tier 2: CORRELATED (co-occurring weight or representation shifts without patching support)
        has_covariates = bool(correlated_layers or weight_summary or act_summary)
        if has_covariates:
            scope_msg = (
                "Behavioral drop is OBSERVED; co-occurring layer weight and Linear CKA shifts are strictly CORRELATED "
                "(no counterfactual recovery >= 20% verified)."
            )
            return (
                {
                    "causal_verification_exists": False,
                    "causal_support_exists": False,
                    "verified_case_ids": [],
                    "epistemic_status": EpistemicStatus.CORRELATED.value,
                    "most_causal_layer": base_causal["most_causal_layer"],
                    "peak_recovery_fraction": base_causal["peak_recovery_fraction"],
                    "scope_disclaimer": scope_msg,
                },
                EpistemicStatus.CORRELATED.value,
                expl_prefix + scope_msg,
            )

        # Tier 1: OBSERVED (pure behavioral evaluation delta, zero mechanistic covariates)
        scope_msg = "Behavioral regression directly OBSERVED on deterministic rubric evaluation; no weight or activation covariates recorded."
        return (
            {
                "causal_verification_exists": False,
                "causal_support_exists": False,
                "verified_case_ids": [],
                "epistemic_status": EpistemicStatus.OBSERVED.value,
                "most_causal_layer": None,
                "peak_recovery_fraction": None,
                "scope_disclaimer": scope_msg,
            },
            EpistemicStatus.OBSERVED.value,
            expl_prefix + scope_msg,
        )

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
        has_transition_patch_support = bool(
            isinstance(patching, dict)
            and abs(float(patching.get("peak_recovery_fraction", 0.0))) >= 0.20
        )
        base_causal = {
            "transition_patching_support": has_transition_patch_support,
            "most_causal_layer": (
                patching.get("most_causal_layer") if isinstance(patching, dict) else None
            ),
            "peak_recovery_fraction": (
                patching.get("peak_recovery_fraction") if isinstance(patching, dict) else None
            ),
        }
        return correlated_layers, weight_summary, act_summary, base_causal
