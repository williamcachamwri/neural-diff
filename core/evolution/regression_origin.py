"""Automatic Regression Origin Detection Engine (Section 6).

Identifies first-occurrence behavioral capability regressions across multi-generation
checkpoint DAGs while enforcing strict epistemic, topological, and statistical guarantees:

1. Epistemic Tiers & Causal Provenance (Audit Item 1):
   * `OBSERVED`: Behavioral regression observed without aligned mechanistic covariates.
   * `CORRELATED`: Behavioral regression co-occurring with measured weight/CKA shifts.
   * `CAUSAL SUPPORT`: Transition-level diagnostic-prompt patching recovers >= 20% logit margin
     (supports a mechanistic locus on the transition without proving category-wide causation).
   * `CAUSALLY VERIFIED`: Requires explicit, independently validated provenance for EVERY failed
     test case in the event (`case_targeted=True`, `target_vs_foil_verified=True`,
     `negative_control_verified=True`, and valid case+site `verification_scope`, backed by
     `prompt_fingerprint`, `failed_case_id`, `layer_index`, `intervention_type`, `target_token`,
     `foil_token`, `negative_control_result`, and `recovery_measurement >= 0.20`). Missing
     evidence never defaults to `True`.

2. Complete-DAG Lineage-State DP & Truncation-Safe Fallback (Audit Item 2):
   * Tracks clean vs. active-regression lineage states in exact O(|V| * |E|) topological DP
     across multiple roots, branch points, merge nodes, and recovery transitions.
   * Never misses transitions when `branch_count > max_branches`, never duplicates shared-edge
     events, and never contaminates unrelated branch histories across merges.

3. Statistical Rigor & Limitations (Audit Item 4):
   * Enforces minimum sample floors (`is_exploratory=True`, `statistically_significant=False`
     when `n < min_sample_size`), step-down Holm-Bonferroni correction across category families,
     and `pass_rate_and_score` / `pass_rate_only` / `score_only` classification.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from typing import Any, TypeGuard

from core.evolution.behavioral_evolution import holm_bonferroni_adjust
from core.evolution.dag import BranchEnumerationResult, EvolutionDAG
from core.evolution.types import CausalInterventionProvenance
from core.types import EpistemicStatus

_DISALLOWED_BROAD_SCOPES = frozenset(
    {
        "diagnostic_prompt",
        "diagnostic_prompt_transition_only",
        "transition",
        "category",
        "global",
        "model",
        "all_cases",
        "unscoped",
        "none",
    }
)


def _is_real_number(val: Any) -> TypeGuard[int | float]:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def _is_nonempty_token(val: Any) -> bool:
    if val is None or isinstance(val, bool):
        return False
    if isinstance(val, int):
        return True
    if isinstance(val, str):
        return len(val.strip()) > 0
    return False


def validate_causal_verification_provenance(
    raw: dict[str, Any] | CausalInterventionProvenance | None,
    *,
    expected_case_id: str,
    expected_prompt_fingerprints: set[str] | None = None,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Independently validate all causal verification conditions and provenance measurements.

    Missing fields strictly evaluate to `False` / `None` and NEVER default to `True`.
    """
    data: dict[str, Any]
    if isinstance(raw, CausalInterventionProvenance):
        data = raw.to_dict()
    elif isinstance(raw, dict):
        data = dict(raw)
    else:
        empty_prov = CausalInterventionProvenance().to_dict()
        return (
            False,
            {
                **empty_prov,
                "verification_scope_valid": False,
                "validation_failures": ["missing_intervention_record"],
            },
            ["missing_intervention_record"],
        )

    failures: list[str] = []

    # 1. Explicit boolean flags — strictly `is True` (never truthy coercion, never default True)
    flag_case_targeted = data.get("case_targeted") is True
    flag_target_vs_foil = data.get("target_vs_foil_verified") is True
    flag_negative_control = data.get("negative_control_verified") is True

    if not flag_case_targeted:
        failures.append("case_targeted_not_explicitly_true")
    if not flag_target_vs_foil:
        failures.append("target_vs_foil_verified_not_explicitly_true")
    if not flag_negative_control:
        failures.append("negative_control_verified_not_explicitly_true")

    # 2. Failed case ID & prompt fingerprint validation
    raw_case_id = data.get("failed_case_id")
    failed_case_id = raw_case_id.strip() if isinstance(raw_case_id, str) and raw_case_id.strip() else None
    if failed_case_id != expected_case_id:
        failures.append("failed_case_id_missing_or_mismatched")

    raw_prompt_fp = data.get("prompt_fingerprint")
    prompt_fp = (
        raw_prompt_fp.strip()
        if isinstance(raw_prompt_fp, str) and len(raw_prompt_fp.strip()) >= 8
        else None
    )
    if prompt_fp is None:
        failures.append("prompt_fingerprint_missing_or_invalid")
    elif expected_prompt_fingerprints and prompt_fp not in expected_prompt_fingerprints:
        failures.append("prompt_fingerprint_does_not_match_case_prompt")

    # 3. Intervention site (`layer_index` & `intervention_type`)
    raw_layer = data.get("layer_index")
    layer_idx: int | None = (
        int(raw_layer)
        if isinstance(raw_layer, int) and not isinstance(raw_layer, bool) and raw_layer >= 0
        else None
    )
    if layer_idx is None:
        failures.append("layer_index_missing_or_invalid")

    raw_itype = data.get("intervention_type")
    intervention_type = (
        raw_itype.strip() if isinstance(raw_itype, str) and raw_itype.strip() else None
    )
    if intervention_type is None:
        failures.append("intervention_type_missing")

    # 4. Target vs. foil tokens and recovery measurement
    target_token = data.get("target_token") if _is_nonempty_token(data.get("target_token")) else None
    foil_token = data.get("foil_token") if _is_nonempty_token(data.get("foil_token")) else None
    if target_token is None or foil_token is None:
        failures.append("target_or_foil_token_missing")
    elif target_token == foil_token:
        failures.append("target_and_foil_tokens_identical")

    raw_rec = data.get("recovery_measurement")
    recovery_val: float | None = round(float(raw_rec), 6) if _is_real_number(raw_rec) else None
    if recovery_val is None:
        failures.append("recovery_measurement_missing")
    elif recovery_val < 0.20:
        failures.append("recovery_measurement_below_0.20_threshold")

    clean_diff_raw = data.get("clean_logit_diff")
    corr_diff_raw = data.get("corrupted_logit_diff")
    clean_diff_val = round(float(clean_diff_raw), 6) if _is_real_number(clean_diff_raw) else None
    corr_diff_val = round(float(corr_diff_raw), 6) if _is_real_number(corr_diff_raw) else None
    if (
        clean_diff_val is not None
        and corr_diff_val is not None
        and (clean_diff_val - corr_diff_val) <= 1e-6
    ):
        failures.append("clean_minus_corrupted_logit_margin_non_positive")

    # 5. Negative-control result validation
    raw_neg = data.get("negative_control_result")
    neg_ctrl_stored: dict[str, Any] | float | None = None
    neg_ctrl_numeric_ok = False
    if _is_real_number(raw_neg):
        neg_val = round(float(raw_neg), 6)
        neg_ctrl_stored = neg_val
        if recovery_val is not None and abs(neg_val) < 0.15 and abs(neg_val) < recovery_val:
            neg_ctrl_numeric_ok = True
    elif isinstance(raw_neg, dict) and len(raw_neg) > 0:
        neg_ctrl_stored = dict(raw_neg)
        if raw_neg.get("passed") is not False:
            ctrl_candidates = [
                raw_neg.get("control_recovery_fraction"),
                raw_neg.get("max_abs_control_recovery"),
            ]
            numeric_ctrls = [abs(float(v)) for v in ctrl_candidates if _is_real_number(v)]
            if (
                numeric_ctrls
                and recovery_val is not None
                and max(numeric_ctrls) < 0.15
                and max(numeric_ctrls) < recovery_val
            ):
                neg_ctrl_numeric_ok = True
    if not neg_ctrl_numeric_ok:
        failures.append("negative_control_result_missing_or_exceeds_bound")

    # 6. Strict case-and-intervention-site verification_scope validation
    raw_scope = data.get("verification_scope")
    scope_str = raw_scope.strip() if isinstance(raw_scope, str) and raw_scope.strip() else None
    scope_valid = False
    if scope_str is None:
        failures.append("verification_scope_missing")
    elif scope_str.lower() in _DISALLOWED_BROAD_SCOPES:
        failures.append("verification_scope_unscoped_or_too_broad")
    else:
        # Reject any scope string that names a different case or a different layer index
        case_mismatch = ("case:" in scope_str and f"case:{expected_case_id}" not in scope_str)
        layer_mismatch = (
            layer_idx is not None
            and "layer:" in scope_str
            and f"layer:{layer_idx}" not in scope_str
        )
        if case_mismatch or layer_mismatch:
            failures.append("verification_scope_mismatches_case_or_layer")
        elif scope_str == "case_and_intervention_site" or expected_case_id in scope_str:
            scope_valid = True
        else:
            failures.append("verification_scope_must_reference_tested_case_and_site")

    case_targeted_valid = (
        flag_case_targeted
        and failed_case_id == expected_case_id
        and prompt_fp is not None
        and "prompt_fingerprint_does_not_match_case_prompt" not in failures
    )
    target_foil_valid = (
        flag_target_vs_foil
        and target_token is not None
        and foil_token is not None
        and target_token != foil_token
        and recovery_val is not None
        and recovery_val >= 0.20
        and "clean_minus_corrupted_logit_margin_non_positive" not in failures
    )
    neg_ctrl_valid = flag_negative_control and neg_ctrl_numeric_ok
    site_valid = layer_idx is not None and intervention_type is not None and scope_valid

    all_valid = bool(
        case_targeted_valid and target_foil_valid and neg_ctrl_valid and site_valid and not failures
    )

    canonical_scope = (
        f"case:{expected_case_id}|prompt_sha256:{prompt_fp[:16]}|layer:{layer_idx}|intervention:{intervention_type}"
        if all_valid and prompt_fp is not None and layer_idx is not None and intervention_type is not None
        else scope_str
    )

    normalized = {
        "case_targeted": bool(case_targeted_valid),
        "target_vs_foil_verified": bool(target_foil_valid),
        "negative_control_verified": bool(neg_ctrl_valid),
        "verification_scope": canonical_scope,
        "verification_scope_valid": bool(site_valid),
        "prompt_fingerprint": prompt_fp,
        "failed_case_id": failed_case_id,
        "layer_index": layer_idx,
        "intervention_type": intervention_type,
        "target_token": target_token,
        "foil_token": foil_token,
        "negative_control_result": neg_ctrl_stored,
        "recovery_measurement": recovery_val,
        "clean_logit_diff": clean_diff_val,
        "corrupted_logit_diff": corr_diff_val,
        "is_causally_verified": all_valid,
        "validation_failures": failures,
    }
    return all_valid, normalized, failures


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


def _statistical_limitations_note(
    *,
    sample_size: int,
    min_required_sample_size: int,
    hypotheses_tested: int,
    regression_kind: str,
) -> str:
    min_p_at_n = 2.0 ** (1 - max(1, sample_size))
    min_adj_p_at_n = min(1.0, hypotheses_tested * min_p_at_n)
    parts = [
        f"Minimum sample floor n>={min_required_sample_size} prevents false significance on tiny buckets; "
        f"exact two-sided McNemar has discrete floor p_min=2^(1-n)={min_p_at_n:.6f} "
        f"(rank-1 Holm-Bonferroni floor={min_adj_p_at_n:.6f} across m={hypotheses_tested} hypothesis test(s)), "
        "requiring n>=9 for non-zero rejection power at alpha=0.05 when m>=7 and n>=30 for >=80% power on moderate drops."
    ]
    if regression_kind == "score_only":
        parts.append(
            "McNemar tests binary pass/fail discordance only; continuous score_only drops without binary flips yield McNemar p=1.0 and are quantified via paired bootstrap CIs."
        )
    return " ".join(parts)


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

    def _metric_pair(
        self, eval_map: dict[str, dict[str, Any]], ckpt_id: str, metric_key: str
    ) -> tuple[float, float] | None:
        ev = eval_map.get(ckpt_id)
        if ev is None:
            return None
        if metric_key == "overall":
            return (float(ev["pass_rate"]), float(ev["mean_score"]))
        cat_info = ev.get("categories", {}).get(metric_key)
        if not isinstance(cat_info, dict):
            return None
        return (float(cat_info["pass_rate"]), float(cat_info["mean_score"]))

    def _thresholds_for(self, metric_key: str) -> tuple[float, float]:
        if metric_key == "overall":
            return (self.overall_pass_rate_drop_threshold, self.category_score_drop_threshold)
        return (self.category_score_drop_threshold, self.category_score_drop_threshold)

    def _is_regressed_relative_to(
        self,
        eval_map: dict[str, dict[str, Any]],
        ref_id: str,
        target_id: str,
        metric_key: str,
    ) -> bool:
        ref_pair = self._metric_pair(eval_map, ref_id, metric_key)
        tgt_pair = self._metric_pair(eval_map, target_id, metric_key)
        if ref_pair is None or tgt_pair is None:
            return False
        pr_thresh, sc_thresh = self._thresholds_for(metric_key)
        pr_drop = round(ref_pair[0] - tgt_pair[0], 6)
        sc_drop = round(ref_pair[1] - tgt_pair[1], 6)
        return pr_drop >= pr_thresh or sc_drop >= sc_thresh

    def detect(
        self,
        *,
        dag: EvolutionDAG,
        behavioral_evolution: dict[str, Any] | None,
        tensor_transitions: list[dict[str, Any]],
        representation_evolution: dict[str, Any] | None,
        case_interventions: dict[tuple[str, str, str], dict[str, Any] | CausalInterventionProvenance]
        | None = None,
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

        topo_order = dag.topological_order()
        all_leaves = set(dag.leaves())
        descendants_incl: dict[str, list[str]] = {}
        reachable_leaves: dict[str, list[str]] = {}
        for nid in topo_order:
            desc_set = dag.descendants_of(nid) | {nid}
            descendants_incl[nid] = [x for x in topo_order if x in desc_set]
            reachable_leaves[nid] = [x for x in topo_order if x in (desc_set & all_leaves)]

        # Collect all evaluated categories deterministically
        all_categories: set[str] = set()
        for tr in beh_edge_map.values():
            for cd in tr.get("category_deltas", []):
                all_categories.add(str(cd["category"]))
        metric_keys = ["overall", *sorted(all_categories)]

        # Step 1: Witness branch attribution on bounded enumerated branches (with exact recovery tracking)
        witness_branches: dict[tuple[str, str, str], list[str]] = {}
        for branch in branch_enum.branches:
            chain = [cid for cid in branch.checkpoint_ids if cid in eval_map]
            if len(chain) < 2:
                continue
            for m_key in metric_keys:
                active_ref: str | None = None
                for idx in range(len(chain) - 1):
                    u_id = chain[idx]
                    v_id = chain[idx + 1]
                    if (u_id, v_id) not in beh_edge_map:
                        continue
                    if active_ref is None:
                        if self._is_regressed_relative_to(eval_map, u_id, v_id, m_key):
                            b_list = witness_branches.setdefault((u_id, v_id, m_key), [])
                            if branch.branch_id not in b_list:
                                b_list.append(branch.branch_id)
                            active_ref = u_id
                    else:
                        if not self._is_regressed_relative_to(eval_map, active_ref, v_id, m_key):
                            # Capability recovered at v_id along this branch
                            active_ref = None

        # Step 2: Complete-DAG O(|V| * |E|) topological lineage-state DP.
        # Guarantees zero missed transitions when `branch_enum.truncated` is True, zero duplicate
        # events on shared edges, and zero cross-branch history mixing across merges or multiple roots.
        valid_origins_ordered: list[tuple[str, str, str]] = []
        seen_origin_keys: set[tuple[str, str, str]] = set()

        for m_key in metric_keys:
            # `states[node]` holds `None` if >=1 root-to-node path is currently unregressed/recovered,
            # plus `ref_id` for each active unrecovered regression reference reaching `node`.
            states: dict[str, set[str | None]] = {cid: set() for cid in topo_order if cid in eval_map}
            for cid in topo_order:
                if cid not in eval_map:
                    continue
                ev_parents = [
                    p for p in dag.parents_of.get(cid, []) if p in eval_map and (p, cid) in beh_edge_map
                ]
                if not ev_parents or not states[cid]:
                    states[cid].add(None)

                for ch_id in dag.children_of.get(cid, []):
                    if ch_id not in eval_map or (cid, ch_id) not in beh_edge_map:
                        continue
                    edge_regresses = self._is_regressed_relative_to(eval_map, cid, ch_id, m_key)
                    if None in states[cid]:
                        if edge_regresses:
                            okey = (cid, ch_id, m_key)
                            if okey not in seen_origin_keys:
                                seen_origin_keys.add(okey)
                                valid_origins_ordered.append(okey)
                            states[ch_id].add(cid)
                        else:
                            states[ch_id].add(None)

                    for ref_id in states[cid]:
                        if ref_id is None:
                            continue
                        if self._is_regressed_relative_to(eval_map, ref_id, ch_id, m_key):
                            states[ch_id].add(ref_id)
                        else:
                            states[ch_id].add(None)

        # Precompute edge-level mechanistic & Holm-Bonferroni payloads once per `(u, v)`
        edge_context_cache: dict[tuple[str, str], dict[str, Any]] = {}
        for (u_id, v_id), beh_tr in beh_edge_map.items():
            t_edge = tensor_edge_map.get((u_id, v_id))
            r_edge = rep_edge_map.get((u_id, v_id))
            corr_layers, w_sum, a_sum, base_causal = self._build_mechanistic_context(t_edge, r_edge)
            cat_deltas = list(beh_tr.get("category_deltas", []))
            cat_raw_ps = [
                float(cd.get("mcnemar_p_value", beh_tr["paired_statistics"]["mcnemar_p_value"]))
                for cd in cat_deltas
            ]
            cat_adj_ps = holm_bonferroni_adjust(cat_raw_ps)
            cat_delta_map = {str(cd["category"]): cd for cd in cat_deltas}
            cat_adj_map = {
                str(cd["category"]): cat_adj_ps[idx] for idx, cd in enumerate(cat_deltas)
            }
            edge_context_cache[(u_id, v_id)] = {
                "correlated_layers": corr_layers,
                "weight_summary": w_sum,
                "act_summary": a_sum,
                "base_causal": base_causal,
                "cat_delta_map": cat_delta_map,
                "cat_adj_map": cat_adj_map,
                "num_categories": len(cat_deltas),
            }

        events: list[dict[str, Any]] = []
        for prev_id, curr_id, m_key in valid_origins_ordered:
            prev_ev = eval_map[prev_id]
            curr_ev = eval_map[curr_id]
            beh_tr = beh_edge_map[(prev_id, curr_id)]
            ctx = edge_context_cache[(prev_id, curr_id)]
            prev_ckpt = dag.checkpoints_by_id[prev_id]
            curr_ckpt = dag.checkpoints_by_id[curr_id]

            w_branches = witness_branches.get((prev_id, curr_id, m_key), [])
            primary_branch_id = w_branches[0] if w_branches else "dag-edge-scan"

            # Compute exact downstream persistence, recovery nodes, and per-leaf outcomes across DAG
            downstream_regressed_ids: list[str] = []
            recovered_at_ids: list[str] = []
            for d_id in descendants_incl[curr_id]:
                if self._metric_pair(eval_map, d_id, m_key) is None:
                    continue
                if self._is_regressed_relative_to(eval_map, prev_id, d_id, m_key):
                    downstream_regressed_ids.append(d_id)
                elif d_id != curr_id:
                    recovered_at_ids.append(d_id)

            leaf_outcomes: dict[str, str] = {}
            for lid in reachable_leaves[curr_id]:
                if self._metric_pair(eval_map, lid, m_key) is None:
                    continue
                leaf_outcomes[lid] = (
                    "regressed"
                    if self._is_regressed_relative_to(eval_map, prev_id, lid, m_key)
                    else "recovered"
                )
            surviving_leaf_ids = [lid for lid, st in leaf_outcomes.items() if st == "regressed"]
            recovered_leaf_ids = [lid for lid, st in leaf_outcomes.items() if st == "recovered"]
            survives = len(surviving_leaf_ids) > 0

            # Build prompt fingerprint lookup from evaluated case results when available
            case_prompt_hashes: dict[str, set[str]] = {}
            for cr in curr_ev.get("case_results", []):
                cid_str = str(cr.get("case_id", ""))
                if not cid_str:
                    continue
                hashes: set[str] = set()
                for field_name in ("prompt", "formatted_input"):
                    txt = cr.get(field_name)
                    if isinstance(txt, str) and txt:
                        hashes.add(hashlib.sha256(txt.encode("utf-8")).hexdigest())
                if hashes:
                    case_prompt_hashes[cid_str] = hashes

            pstats = beh_tr["paired_statistics"]
            if m_key == "overall":
                pr_drop = round(float(prev_ev["pass_rate"]) - float(curr_ev["pass_rate"]), 6)
                sc_drop = round(float(prev_ev["mean_score"]) - float(curr_ev["mean_score"]), 6)
                reg_kind = _classify_regression_kind(
                    pr_drop,
                    sc_drop,
                    pr_threshold=self.overall_pass_rate_drop_threshold,
                    sc_threshold=self.category_score_drop_threshold,
                )
                n_cases = int(beh_tr["sample_size"])
                raw_p = round(min(1.0, max(0.0, float(pstats["mcnemar_p_value"]))), 6)
                adj_p = raw_p
                conf_level, is_expl, is_sig = _confidence_tier(
                    sample_size=n_cases,
                    min_required_sample_size=self.min_overall_sample_size,
                    adjusted_p_value=adj_p,
                    alpha=self.significance_alpha,
                    effect_drop=max(pr_drop, sc_drop),
                )
                failed_ids = list(beh_tr["newly_introduced_failures"])
                ev_causal, ev_status, interp = self._resolve_event_epistemic_status(
                    prev_id=prev_id,
                    curr_id=curr_id,
                    failed_case_ids=failed_ids,
                    case_prompt_hashes=case_prompt_hashes,
                    correlated_layers=ctx["correlated_layers"],
                    weight_summary=ctx["weight_summary"],
                    act_summary=ctx["act_summary"],
                    base_causal=ctx["base_causal"],
                    case_interventions=case_interventions,
                    is_exploratory=is_expl,
                    sample_size=n_cases,
                    min_required=self.min_overall_sample_size,
                )
                events.append(
                    {
                        "event_id": f"reg-orig-{uuid.uuid4().hex[:8]}",
                        "branch_id": primary_branch_id,
                        "affected_branch_ids": w_branches,
                        "discovered_via_fallback": len(w_branches) == 0,
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
                        "surviving_leaf_ids": surviving_leaf_ids,
                        "recovered_leaf_ids": recovered_leaf_ids,
                        "recovered_at_checkpoint_ids": recovered_at_ids,
                        "leaf_outcomes": leaf_outcomes,
                        "downstream_affected_checkpoints": downstream_regressed_ids,
                        "correlated_layers": ctx["correlated_layers"],
                        "weight_changes": ctx["weight_summary"],
                        "activation_changes": ctx["act_summary"],
                        "statistical_confidence": {
                            "sample_size": n_cases,
                            "min_required_sample_size": self.min_overall_sample_size,
                            "is_exploratory": is_expl,
                            "statistically_significant": is_sig,
                            "raw_p_value": raw_p,
                            "mcnemar_p_value": raw_p,
                            "adjusted_p_value": adj_p,
                            "holm_bonferroni_p_value": adj_p,
                            "multiple_comparison_method": "holm-bonferroni",
                            "hypotheses_tested": 1,
                            "bootstrap_ci95": [
                                pstats["bootstrap_ci95_low"],
                                pstats["bootstrap_ci95_high"],
                            ],
                            "confidence_level": conf_level,
                            "statistical_limitations": _statistical_limitations_note(
                                sample_size=n_cases,
                                min_required_sample_size=self.min_overall_sample_size,
                                hypotheses_tested=1,
                                regression_kind=reg_kind,
                            ),
                        },
                        "causal_verification": ev_causal,
                        "epistemic_status": ev_status,
                        "interpretation_note": interp,
                    }
                )
            else:
                c_delta = ctx["cat_delta_map"].get(m_key)
                if c_delta is None:
                    continue
                c_sc_drop = round(
                    float(c_delta["mean_score_before"]) - float(c_delta["mean_score_after"]), 6
                )
                c_pr_drop = round(
                    float(c_delta["pass_rate_before"]) - float(c_delta["pass_rate_after"]), 6
                )
                c_reg_kind = _classify_regression_kind(
                    c_pr_drop,
                    c_sc_drop,
                    pr_threshold=self.category_score_drop_threshold,
                    sc_threshold=self.category_score_drop_threshold,
                )
                cat_failed_ids = [
                    r["case_id"]
                    for r in curr_ev.get("case_results", [])
                    if r["category"] == m_key
                    and r["case_id"] in set(beh_tr["newly_introduced_failures"])
                ]
                n_cat = int(c_delta["sample_size"])
                c_raw_p = round(
                    min(
                        1.0,
                        max(0.0, float(c_delta.get("mcnemar_p_value", pstats["mcnemar_p_value"]))),
                    ),
                    6,
                )
                c_adj_p = float(ctx["cat_adj_map"].get(m_key, c_raw_p))
                m_hyps = max(1, int(ctx["num_categories"]))
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
                    case_prompt_hashes=case_prompt_hashes,
                    correlated_layers=ctx["correlated_layers"],
                    weight_summary=ctx["weight_summary"],
                    act_summary=ctx["act_summary"],
                    base_causal=ctx["base_causal"],
                    case_interventions=case_interventions,
                    is_exploratory=c_is_expl,
                    sample_size=n_cat,
                    min_required=self.min_category_sample_size,
                )
                events.append(
                    {
                        "event_id": f"reg-orig-{uuid.uuid4().hex[:8]}",
                        "branch_id": primary_branch_id,
                        "affected_branch_ids": w_branches,
                        "discovered_via_fallback": len(w_branches) == 0,
                        "first_affected_checkpoint_id": curr_id,
                        "first_affected_label": curr_ckpt.version_label,
                        "previous_checkpoint_id": prev_id,
                        "previous_label": prev_ckpt.version_label,
                        "transformation_type": beh_tr["transformation_type"],
                        "affected_category": m_key,
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
                        "surviving_leaf_ids": surviving_leaf_ids,
                        "recovered_leaf_ids": recovered_leaf_ids,
                        "recovered_at_checkpoint_ids": recovered_at_ids,
                        "leaf_outcomes": leaf_outcomes,
                        "downstream_affected_checkpoints": downstream_regressed_ids,
                        "correlated_layers": ctx["correlated_layers"],
                        "weight_changes": ctx["weight_summary"],
                        "activation_changes": ctx["act_summary"],
                        "statistical_confidence": {
                            "sample_size": n_cat,
                            "min_required_sample_size": self.min_category_sample_size,
                            "is_exploratory": c_is_expl,
                            "statistically_significant": c_is_sig,
                            "raw_p_value": c_raw_p,
                            "mcnemar_p_value": c_raw_p,
                            "adjusted_p_value": c_adj_p,
                            "holm_bonferroni_p_value": c_adj_p,
                            "multiple_comparison_method": "holm-bonferroni",
                            "hypotheses_tested": m_hyps,
                            "wilson_ci95_after": c_delta["wilson_ci95_after"],
                            "confidence_level": c_conf_level,
                            "statistical_limitations": _statistical_limitations_note(
                                sample_size=n_cat,
                                min_required_sample_size=self.min_category_sample_size,
                                hypotheses_tested=m_hyps,
                                regression_kind=c_reg_kind,
                            ),
                        },
                        "causal_verification": c_causal,
                        "epistemic_status": c_status,
                        "interpretation_note": c_interp,
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
    def _resolve_event_epistemic_status(
        *,
        prev_id: str,
        curr_id: str,
        failed_case_ids: list[str],
        case_prompt_hashes: dict[str, set[str]],
        correlated_layers: list[dict[str, Any]],
        weight_summary: dict[str, Any],
        act_summary: dict[str, Any],
        base_causal: dict[str, Any],
        case_interventions: dict[tuple[str, str, str], dict[str, Any] | CausalInterventionProvenance]
        | None,
        is_exploratory: bool,
        sample_size: int,
        min_required: int,
    ) -> tuple[dict[str, Any], str, str]:
        """Classify event into OBSERVED, CORRELATED, CAUSAL SUPPORT, or CAUSALLY VERIFIED."""
        verified_cases: list[str] = []
        unverified_cases: list[str] = []
        case_provenance_records: list[dict[str, Any]] = []
        best_verified_record: dict[str, Any] | None = None

        for cid in failed_case_ids:
            raw_cp = case_interventions.get((prev_id, curr_id, cid)) if case_interventions else None
            if raw_cp is None:
                unverified_cases.append(cid)
                continue
            is_valid, norm_prov, _ = validate_causal_verification_provenance(
                raw_cp,
                expected_case_id=cid,
                expected_prompt_fingerprints=case_prompt_hashes.get(cid),
            )
            case_provenance_records.append(norm_prov)
            if is_valid:
                verified_cases.append(cid)
                if best_verified_record is None or float(
                    norm_prov["recovery_measurement"] or 0.0
                ) > float(best_verified_record["recovery_measurement"] or 0.0):
                    best_verified_record = norm_prov
            else:
                unverified_cases.append(cid)

        expl_prefix = (
            f"Exploratory sample (n={sample_size} < {min_required} min threshold; insufficient for capability-loss claims). "
            if is_exploratory
            else ""
        )

        # Tier 4: CAUSALLY VERIFIED — requires non-empty `failed_case_ids` AND 100% of failed cases
        # in this event passing all 4 independent provenance checks (strictly scoped to tested case & site).
        all_failed_cases_verified = (
            len(failed_case_ids) > 0
            and len(verified_cases) == len(failed_case_ids)
            and len(unverified_cases) == 0
            and best_verified_record is not None
        )
        partially_verified = len(verified_cases) > 0 and not all_failed_cases_verified

        if all_failed_cases_verified and best_verified_record is not None:
            layer_idx = best_verified_record["layer_index"]
            rec_val = best_verified_record["recovery_measurement"]
            itype = best_verified_record["intervention_type"]
            combined_scope = ";".join(
                str(r["verification_scope"])
                for r in case_provenance_records
                if r.get("is_causally_verified")
            )
            scope_msg = (
                f"Causally verified strictly on failed case(s) [{', '.join(verified_cases)}] "
                f"at layer L{layer_idx} ({itype}) with target/foil and negative-control provenance; "
                "does not generalize to untested prompts."
            )
            return (
                {
                    "causal_verification_exists": True,
                    "causal_support_exists": True,
                    "partially_verified": False,
                    "case_targeted": True,
                    "target_vs_foil_verified": True,
                    "negative_control_verified": True,
                    "verification_scope": combined_scope,
                    "verification_scope_valid": True,
                    "prompt_fingerprint": best_verified_record["prompt_fingerprint"],
                    "failed_case_id": (
                        best_verified_record["failed_case_id"]
                        if len(verified_cases) == 1
                        else ",".join(verified_cases)
                    ),
                    "layer_index": layer_idx,
                    "intervention_type": best_verified_record["intervention_type"],
                    "target_token": best_verified_record["target_token"],
                    "foil_token": best_verified_record["foil_token"],
                    "negative_control_result": best_verified_record["negative_control_result"],
                    "recovery_measurement": rec_val,
                    "verified_case_ids": verified_cases,
                    "unverified_case_ids": [],
                    "case_provenance_records": case_provenance_records,
                    "epistemic_status": EpistemicStatus.CAUSALLY_VERIFIED.value,
                    "most_causal_layer": layer_idx,
                    "peak_recovery_fraction": rec_val,
                    "scope_disclaimer": scope_msg,
                },
                EpistemicStatus.CAUSALLY_VERIFIED.value,
                expl_prefix + scope_msg,
            )

        partial_note = (
            f" Case-specific causal verification exists ONLY for [{', '.join(verified_cases)}] "
            f"and does NOT cover unverified case(s) [{', '.join(unverified_cases)}]."
            if partially_verified
            else ""
        )

        # Tier 3: CAUSAL SUPPORT (diagnostic transition patching >= 20%, or partial case verification)
        if base_causal["transition_patching_support"] or partially_verified:
            layer_idx = (
                best_verified_record["layer_index"]
                if best_verified_record is not None
                else base_causal["most_causal_layer"]
            )
            rec_val = (
                best_verified_record["recovery_measurement"]
                if best_verified_record is not None
                else base_causal["peak_recovery_fraction"]
            )
            scope_msg = (
                f"Residual patching at L{layer_idx} recovered {float(rec_val or 0.0) * 100:.1f}% "
                "logit margin (CAUSAL SUPPORT for transition locus; not complete case-specific causal proof for all failed cases in this event)."
                + partial_note
            )
            return (
                {
                    "causal_verification_exists": False,
                    "causal_support_exists": True,
                    "partially_verified": partially_verified,
                    "case_targeted": False,
                    "target_vs_foil_verified": bool(base_causal.get("target_vs_foil_verified", False)),
                    "negative_control_verified": bool(
                        base_causal.get("negative_control_verified", False)
                    ),
                    "verification_scope": (
                        "partial_case_subset_only"
                        if partially_verified
                        else base_causal.get("verification_scope")
                    ),
                    "verification_scope_valid": False,
                    "prompt_fingerprint": base_causal.get("prompt_fingerprint"),
                    "failed_case_id": None,
                    "layer_index": layer_idx,
                    "intervention_type": base_causal.get("intervention_type"),
                    "target_token": base_causal.get("target_token"),
                    "foil_token": base_causal.get("foil_token"),
                    "negative_control_result": base_causal.get("negative_control_result"),
                    "recovery_measurement": rec_val,
                    "verified_case_ids": verified_cases,
                    "unverified_case_ids": unverified_cases,
                    "case_provenance_records": case_provenance_records,
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
                    "partially_verified": False,
                    "case_targeted": False,
                    "target_vs_foil_verified": False,
                    "negative_control_verified": False,
                    "verification_scope": None,
                    "verification_scope_valid": False,
                    "prompt_fingerprint": base_causal.get("prompt_fingerprint"),
                    "failed_case_id": None,
                    "layer_index": base_causal["most_causal_layer"],
                    "intervention_type": base_causal.get("intervention_type"),
                    "target_token": base_causal.get("target_token"),
                    "foil_token": base_causal.get("foil_token"),
                    "negative_control_result": base_causal.get("negative_control_result"),
                    "recovery_measurement": base_causal["peak_recovery_fraction"],
                    "verified_case_ids": [],
                    "unverified_case_ids": unverified_cases,
                    "case_provenance_records": case_provenance_records,
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
                "partially_verified": False,
                "case_targeted": False,
                "target_vs_foil_verified": False,
                "negative_control_verified": False,
                "verification_scope": None,
                "verification_scope_valid": False,
                "prompt_fingerprint": None,
                "failed_case_id": None,
                "layer_index": None,
                "intervention_type": None,
                "target_token": None,
                "foil_token": None,
                "negative_control_result": None,
                "recovery_measurement": None,
                "verified_case_ids": [],
                "unverified_case_ids": unverified_cases,
                "case_provenance_records": case_provenance_records,
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
        p_dict = patching if isinstance(patching, dict) else {}
        rec_raw = p_dict.get("recovery_measurement", p_dict.get("peak_recovery_fraction", 0.0))
        rec_num = float(rec_raw) if _is_real_number(rec_raw) else 0.0
        has_transition_patch_support = bool(p_dict and rec_num >= 0.20)
        base_causal = {
            "transition_patching_support": has_transition_patch_support,
            "most_causal_layer": p_dict.get("layer_index", p_dict.get("most_causal_layer")),
            "peak_recovery_fraction": p_dict.get("peak_recovery_fraction"),
            "target_vs_foil_verified": p_dict.get("target_vs_foil_verified") is True,
            "negative_control_verified": p_dict.get("negative_control_verified") is True,
            "verification_scope": p_dict.get("verification_scope", "diagnostic_prompt_transition_only")
            if p_dict
            else None,
            "prompt_fingerprint": p_dict.get("prompt_fingerprint"),
            "intervention_type": p_dict.get("intervention_type"),
            "target_token": p_dict.get("target_token", p_dict.get("target_token_id")),
            "foil_token": p_dict.get("foil_token", p_dict.get("foil_token_id")),
            "negative_control_result": p_dict.get("negative_control_result"),
        }
        return correlated_layers, weight_summary, act_summary, base_causal
