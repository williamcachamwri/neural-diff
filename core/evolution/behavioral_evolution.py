"""Multi-Generation Behavioral Evolution Tracking Engine (Section 5).

Reuses the existing deterministic zero-LLM-judge `BenchmarkSuite` & evaluators.
Runs the identical dataset across every checkpoint (one model in memory at a time,
backed by SHA-256 content-addressed caching) and tracks capability deltas,
Wilson/Bootstrap confidence intervals, output consistency, and surviving regressions.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmarks.datasets.loader import BenchmarkSuite, load_benchmark_suite
from benchmarks.evaluators.base import EvalCase
from benchmarks.evaluators.registry import evaluate_case
from core.behavioral_analysis.runner import BehavioralDiffEngine
from core.evolution.dag import EvolutionDAG
from core.evolution.store import EvolutionStore
from core.evolution.types import RegisteredCheckpoint
from core.resource_manager import CancellationToken, ResourceManager
from core.security import SecurityPolicy
from core.statistics.significance import compute_paired_significance
from core.types import EpistemicStatus

CheckpointGeneratorFn = Callable[[RegisteredCheckpoint, EvalCase], tuple[str, str, float]]


def compute_suite_fingerprint(suite: BenchmarkSuite) -> str:
    """Compute canonical SHA-256 fingerprint over all evaluation case IDs, prompts, and rubrics."""
    payload = json.dumps(
        {
            "suite_id": suite.suite_id,
            "version": suite.version,
            "cases": [c.to_dict() for c in suite.cases],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def wilson_ci_95(successes: int, total: int) -> tuple[float, float]:
    """Compute exact 95% Wilson score binomial confidence interval `[low, high]`."""
    if total <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    phat = successes / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denom
    half = (
        z
        * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * total)) / total)
        / denom
    )
    return (round(max(0.0, center - half), 6), round(min(1.0, center + half), 6))


def holm_bonferroni_adjust(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni step-down family-wise error correction to a list of raw p-values.

    Uses deterministic `(clamped_p, original_index)` tie-breaking so tied raw p-values
    always receive identical monotonic adjusted p-values in original input order.
    """
    m = len(p_values)
    if m == 0:
        return []
    clamped = [min(1.0, max(0.0, float(p))) for p in p_values]
    indexed = sorted(enumerate(clamped), key=lambda kv: (kv[1], kv[0]))
    adjusted = [1.0] * m
    running_max = 0.0
    for rank, (orig_idx, raw_p) in enumerate(indexed):
        scaled = min(1.0, raw_p * (m - rank))
        running_max = max(running_max, scaled)
        adjusted[orig_idx] = round(running_max, 6)
    return adjusted


def build_behavioral_checkpoint_cache_key(ckpt: RegisteredCheckpoint) -> str:
    """Build cache key spanning all 5 component fingerprints (`weights`, `architecture_config`, `tokenizer`, `adapter`, `metadata`)."""
    comp = ckpt.component_fingerprints or {}
    if comp:
        payload = json.dumps(
            {
                "composite": ckpt.sha256_fingerprint,
                "tensor_metadata_hash": ckpt.tensor_metadata_hash,
                "weights_sha256": comp.get("weights_sha256"),
                "architecture_config_sha256": comp.get("architecture_config_sha256"),
                "tokenizer_sha256": comp.get("tokenizer_sha256"),
                "adapter_sha256": comp.get("adapter_sha256"),
                "metadata_sha256": comp.get("metadata_sha256"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    return ckpt.sha256_fingerprint


class BehavioralEvolutionEngine:
    """Executes deterministic multi-checkpoint behavioral evaluation and lineage comparison."""

    def __init__(
        self,
        *,
        store: EvolutionStore | None = None,
        resource_manager: ResourceManager | None = None,
        policy: SecurityPolicy | None = None,
        max_new_tokens: int = 96,
        seed: int = 42,
        prompt_mode: str = "chat_if_available",
        dtype: str = "bfloat16",
        local_files_only: bool = False,
        use_cache: bool = True,
    ) -> None:
        self.store = store
        self.resource_manager = resource_manager or ResourceManager()
        self.policy = policy or SecurityPolicy()
        self.max_new_tokens = max(8, int(max_new_tokens))
        self.seed = seed
        self.prompt_mode = prompt_mode
        self.dtype = dtype
        self.local_files_only = local_files_only
        self.use_cache = use_cache

    def evaluate_single_checkpoint(
        self,
        ckpt: RegisteredCheckpoint,
        suite: BenchmarkSuite,
        *,
        suite_fingerprint: str,
        custom_generator: CheckpointGeneratorFn | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """Run `suite` on one checkpoint (releasing VRAM/RAM immediately) or return valid content cache."""
        inf_cfg = {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "prompt_mode": self.prompt_mode if custom_generator is None else "custom_generator",
            "dtype": self.dtype,
        }
        inf_hash = hashlib.sha256(
            json.dumps(inf_cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        eval_cache_fp = build_behavioral_checkpoint_cache_key(ckpt)
        if self.use_cache and self.store is not None and custom_generator is None:
            cached = self.store.get_cached_checkpoint_eval(
                eval_cache_fp, suite_fingerprint, inf_hash
            )
            if cached is not None:
                return {**cached, "from_cache": True}

        raw_outputs: list[tuple[str, str, float]] = []
        if custom_generator is not None:
            for case in suite.cases:
                if cancellation_token is not None:
                    cancellation_token.check()
                raw_outputs.append(custom_generator(ckpt, case))
        else:
            resolved = Path(ckpt.resolved_path)
            model_dir = resolved if resolved.is_dir() else resolved.parent
            if not (model_dir / "config.json").is_file():
                raise ValueError(
                    f"Checkpoint '{ckpt.version_label}' does not have `config.json` / tokenizer files required for text generation."
                )
            runner = BehavioralDiffEngine(
                resource_manager=self.resource_manager,
                policy=self.policy,
                max_new_tokens=self.max_new_tokens,
                seed=self.seed,
                prompt_mode=self.prompt_mode,
                dtype=self.dtype,
            )
            device = self.resource_manager.resolve_device()
            raw_outputs = runner._generate_suite_outputs(
                model_dir,
                suite.cases,
                device=device,
                cancellation_token=cancellation_token,
            )
            self.resource_manager.cleanup()

        case_results: list[dict[str, Any]] = []
        by_cat: dict[str, list[dict[str, Any]]] = {}
        failure_categories: dict[str, int] = {}
        contradiction_failures = 0
        factual_total = 0

        for case, (fmt_prompt, output_txt, latency_ms) in zip(
            suite.cases, raw_outputs, strict=True
        ):
            ev = evaluate_case(case, output_txt)
            rec = {
                "case_id": case.case_id,
                "category": case.category,
                "description": case.description,
                "prompt": case.prompt,
                "formatted_input": fmt_prompt,
                "output": output_txt,
                "latency_ms": latency_ms,
                "passed": ev.passed,
                "score": ev.score,
                "evaluator_type": ev.evaluator_type,
                "explanation": ev.rationale,
                "metrics": ev.submetrics,
            }
            case_results.append(rec)
            by_cat.setdefault(case.category, []).append(rec)
            if not ev.passed:
                failure_categories[case.category] = failure_categories.get(case.category, 0) + 1
            if case.category == "factual_accuracy":
                factual_total += 1
                if ev.submetrics.get("contradiction_triggered") or ev.submetrics.get("forbidden_contradictions_found"):
                    contradiction_failures += 1

        total_n = len(case_results)
        passed_n = sum(1 for r in case_results if r["passed"])
        pass_rate = round(passed_n / total_n, 6) if total_n > 0 else 0.0
        mean_score = round(sum(r["score"] for r in case_results) / total_n, 6) if total_n > 0 else 0.0
        ci_low, ci_high = wilson_ci_95(passed_n, total_n)

        category_scores: dict[str, dict[str, Any]] = {}
        for cat, items in by_cat.items():
            cn = len(items)
            cp = sum(1 for r in items if r["passed"])
            c_low, c_high = wilson_ci_95(cp, cn)
            category_scores[cat] = {
                "category": cat,
                "sample_size": cn,
                "passed_count": cp,
                "pass_rate": round(cp / cn, 6) if cn > 0 else 0.0,
                "pass_rate_wilson_ci95": [c_low, c_high],
                "mean_score": round(sum(r["score"] for r in items) / cn, 6) if cn > 0 else 0.0,
            }

        hallucination_proxy_rate = (
            round(contradiction_failures / factual_total, 6) if factual_total > 0 else None
        )

        payload = {
            "checkpoint_id": ckpt.checkpoint_id,
            "version_label": ckpt.version_label,
            "transformation_type": ckpt.transformation_type.value,
            "checkpoint_sha256": ckpt.sha256_fingerprint,
            "component_fingerprints": dict(ckpt.component_fingerprints or {}),
            "suite_id": suite.suite_id,
            "suite_version": suite.version,
            "suite_fingerprint": suite_fingerprint,
            "evaluated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "random_seed": self.seed,
            "inference_config": inf_cfg,
            "model_config": {
                "architecture": ckpt.architecture,
                "model_family": ckpt.model_family,
                "parameter_count": ckpt.parameter_count,
                "dtypes_present": ckpt.dtypes_present,
            },
            "sample_size": total_n,
            "test_case_ids": [c.case_id for c in suite.cases],
            "passed_count": passed_n,
            "pass_rate": pass_rate,
            "pass_rate_wilson_ci95": [ci_low, ci_high],
            "mean_score": mean_score,
            "failure_categories": failure_categories,
            "factual_contradiction_rate": hallucination_proxy_rate,
            "categories": category_scores,
            "case_results": case_results,
            "from_cache": False,
        }

        if self.use_cache and self.store is not None and custom_generator is None:
            self.store.save_cached_checkpoint_eval(
                eval_cache_fp, suite_fingerprint, inf_hash, payload
            )
        return payload

    def compare_checkpoint_evals(
        self,
        eval_a: dict[str, Any],
        eval_b: dict[str, Any],
        *,
        scope: str,
        transformation_type: str,
    ) -> dict[str, Any]:
        """Compute paired transition metrics, statistical confidence, consistency & case transitions."""
        cases_a = {r["case_id"]: r for r in eval_a["case_results"]}
        cases_b = {r["case_id"]: r for r in eval_b["case_results"]}
        ordered_ids = [cid for cid in eval_a["test_case_ids"] if cid in cases_b]

        scores_a = [float(cases_a[cid]["score"]) for cid in ordered_ids]
        scores_b = [float(cases_b[cid]["score"]) for cid in ordered_ids]
        passed_a = [bool(cases_a[cid]["passed"]) for cid in ordered_ids]
        passed_b = [bool(cases_b[cid]["passed"]) for cid in ordered_ids]

        sig = compute_paired_significance(scores_a, scores_b, passed_a, passed_b, seed=self.seed)

        newly_failed_ids: list[str] = []
        newly_passed_ids: list[str] = []
        persistent_failed_ids: list[str] = []
        exact_output_matches = 0

        for cid in ordered_ids:
            ra = cases_a[cid]
            rb = cases_b[cid]
            if str(ra["output"]).strip() == str(rb["output"]).strip():
                exact_output_matches += 1
            if ra["passed"] and not rb["passed"]:
                newly_failed_ids.append(cid)
            elif (not ra["passed"]) and rb["passed"]:
                newly_passed_ids.append(cid)
            elif (not ra["passed"]) and (not rb["passed"]):
                persistent_failed_ids.append(cid)

        consistency_rate = (
            round(exact_output_matches / len(ordered_ids), 6) if ordered_ids else 1.0
        )

        cat_deltas: list[dict[str, Any]] = []
        improved_cats: list[dict[str, Any]] = []
        degraded_cats: list[dict[str, Any]] = []
        newly_observed_pass_cats: list[dict[str, Any]] = []

        all_cats = sorted(set(eval_a["categories"]) | set(eval_b["categories"]))
        raw_cat_ps: list[float] = []
        for cat in all_cats:
            ca = eval_a["categories"].get(cat)
            cb = eval_b["categories"].get(cat)
            if not ca or not cb:
                continue
            cat_ids = [
                cid
                for cid in ordered_ids
                if cases_a[cid]["category"] == cat and cases_b[cid]["category"] == cat
            ]
            c_scores_a = [float(cases_a[cid]["score"]) for cid in cat_ids]
            c_scores_b = [float(cases_b[cid]["score"]) for cid in cat_ids]
            c_passed_a = [bool(cases_a[cid]["passed"]) for cid in cat_ids]
            c_passed_b = [bool(cases_b[cid]["passed"]) for cid in cat_ids]
            cat_sig = compute_paired_significance(
                c_scores_a, c_scores_b, c_passed_a, c_passed_b, seed=self.seed
            )

            s_delta = round(cb["mean_score"] - ca["mean_score"], 6)
            pr_delta = round(cb["pass_rate"] - ca["pass_rate"], 6)
            n_cat = int(cb["sample_size"])
            raw_cat_ps.append(cat_sig.mcnemar_p_value)
            entry = {
                "category": cat,
                "sample_size": n_cat,
                "mean_score_before": ca["mean_score"],
                "mean_score_after": cb["mean_score"],
                "mean_score_delta": s_delta,
                "pass_rate_before": ca["pass_rate"],
                "pass_rate_after": cb["pass_rate"],
                "pass_rate_delta": pr_delta,
                "wilson_ci95_after": cb["pass_rate_wilson_ci95"],
                "raw_p_value": cat_sig.mcnemar_p_value,
                "mcnemar_p_value": cat_sig.mcnemar_p_value,
                "bootstrap_ci95": [cat_sig.bootstrap_ci95_low, cat_sig.bootstrap_ci95_high],
                "is_exploratory": n_cat < 8,
                "statistical_note": (
                    f"Exploratory sample (n={n_cat} < 8); insufficient power for standalone capability claims without an expanded suite."
                    if n_cat < 8
                    else (
                        f"Observed on n={n_cat} deterministic test cases; exact two-sided McNemar minimum attainable p=2^(1-n)={2 ** (1 - n_cat):.6f} limits Holm-Bonferroni power across multi-category families unless n >= 30."
                        if n_cat < 30
                        else f"Observed on n={n_cat} test cases."
                    )
                ),
            }
            cat_deltas.append(entry)
            if ca["pass_rate"] == 0.0 and cb["pass_rate"] > 0.0:
                newly_observed_pass_cats.append(entry)
            if s_delta > 0.02 or pr_delta > 0.0:
                improved_cats.append(entry)
            elif s_delta < -0.02 or pr_delta < 0.0:
                degraded_cats.append(entry)

        holm_ps = holm_bonferroni_adjust(raw_cat_ps)
        for entry, adj_p in zip(cat_deltas, holm_ps, strict=True):
            entry["holm_bonferroni_p_value"] = adj_p
            entry["adjusted_p_value"] = adj_p

        return {
            "scope": scope,
            "source_checkpoint_id": eval_a["checkpoint_id"],
            "source_label": eval_a["version_label"],
            "target_checkpoint_id": eval_b["checkpoint_id"],
            "target_label": eval_b["version_label"],
            "transformation_type": transformation_type,
            "epistemic_status": EpistemicStatus.OBSERVED.value,
            "sample_size": len(ordered_ids),
            "paired_statistics": sig.to_dict(),
            "output_consistency_rate": consistency_rate,
            "newly_introduced_failures": newly_failed_ids,
            "newly_passed_cases": newly_passed_ids,
            "persistent_failures": persistent_failed_ids,
            "observed_zero_to_positive_categories": newly_observed_pass_cats,
            "improved_categories": improved_cats,
            "degraded_categories": degraded_cats,
            "category_deltas": cat_deltas,
        }

    def run(
        self,
        dag: EvolutionDAG,
        *,
        base_checkpoint_id: str | None = None,
        suite: BenchmarkSuite | None = None,
        categories: list[str] | None = None,
        max_cases: int | None = None,
        custom_generator: CheckpointGeneratorFn | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> tuple[dict[str, Any], list[str]]:
        """Run deterministic behavioral tracking across all compatible checkpoints in `dag`."""
        active_suite = suite or load_benchmark_suite(categories=categories, policy=self.policy)
        if max_cases is not None and max_cases > 0:
            active_suite = BenchmarkSuite(
                suite_id=active_suite.suite_id,
                version=active_suite.version,
                description=active_suite.description,
                cases=active_suite.cases[:max_cases],
            )
        suite_fp = compute_suite_fingerprint(active_suite)

        evals_by_id: dict[str, dict[str, Any]] = {}
        warnings: list[str] = []
        topo = dag.topological_order()

        for cid in topo:
            if cancellation_token is not None:
                cancellation_token.check()
            ckpt = dag.checkpoints_by_id[cid]
            try:
                ev = self.evaluate_single_checkpoint(
                    ckpt,
                    active_suite,
                    suite_fingerprint=suite_fp,
                    custom_generator=custom_generator,
                    cancellation_token=cancellation_token,
                )
                evals_by_id[cid] = ev
            except Exception as exc:
                warnings.append(f"Behavioral evaluation skipped for '{ckpt.version_label}': {exc}")

        consecutive_transitions: list[dict[str, Any]] = []
        for rel in dag.relationships:
            pa_id = rel.parent_checkpoint_id
            ch_id = rel.child_checkpoint_id
            if pa_id in evals_by_id and ch_id in evals_by_id:
                consecutive_transitions.append(
                    self.compare_checkpoint_evals(
                        evals_by_id[pa_id],
                        evals_by_id[ch_id],
                        scope="consecutive",
                        transformation_type=rel.transformation_type.value,
                    )
                )

        roots = dag.roots()
        eff_base = base_checkpoint_id or (roots[0] if roots else None)
        base_to_current: list[dict[str, Any]] = []
        if eff_base and eff_base in evals_by_id:
            for cid in topo:
                if cid != eff_base and cid in evals_by_id:
                    base_to_current.append(
                        self.compare_checkpoint_evals(
                            evals_by_id[eff_base],
                            evals_by_id[cid],
                            scope="base_to_current",
                            transformation_type=dag.checkpoints_by_id[cid].transformation_type.value,
                        )
                    )

        # Track persistent failures across entire lineage & regressions that survive to leaves!
        surviving_regressions: list[dict[str, Any]] = []
        lineage_persistent_failures: dict[str, list[str]] = {}
        for branch in dag.enumerate_branches():
            eval_chain = [evals_by_id[cid] for cid in branch.checkpoint_ids if cid in evals_by_id]
            if len(eval_chain) < 2:
                continue

            # Persistent failures: failed at root AND failed at every subsequent checkpoint on branch
            case_ids = eval_chain[0]["test_case_ids"]
            always_failed: list[str] = []
            for case_id in case_ids:
                passed_flags = [
                    next((r["passed"] for r in ev["case_results"] if r["case_id"] == case_id), True)
                    for ev in eval_chain
                ]
                if all(not flag for flag in passed_flags):
                    always_failed.append(case_id)
            lineage_persistent_failures[branch.branch_id] = always_failed

            # Surviving regressions: passed at step k, failed at step k+1, AND still failed at leaf!
            leaf_ev = eval_chain[-1]
            leaf_pass_map = {r["case_id"]: bool(r["passed"]) for r in leaf_ev["case_results"]}
            leaf_cat_map = {r["case_id"]: str(r["category"]) for r in leaf_ev["case_results"]}
            for step_i in range(len(eval_chain) - 1):
                prev_ev = eval_chain[step_i]
                curr_ev = eval_chain[step_i + 1]
                prev_pass = {r["case_id"]: bool(r["passed"]) for r in prev_ev["case_results"]}
                curr_pass = {r["case_id"]: bool(r["passed"]) for r in curr_ev["case_results"]}
                survived_ids = [
                    cid
                    for cid in case_ids
                    if prev_pass.get(cid, False)
                    and (not curr_pass.get(cid, True))
                    and (not leaf_pass_map.get(cid, True))
                ]
                if survived_ids:
                    surviving_regressions.append(
                        {
                            "branch_id": branch.branch_id,
                            "introduced_from_checkpoint_id": prev_ev["checkpoint_id"],
                            "introduced_from_label": prev_ev["version_label"],
                            "introduced_at_checkpoint_id": curr_ev["checkpoint_id"],
                            "introduced_at_label": curr_ev["version_label"],
                            "transformation_type": curr_ev["transformation_type"],
                            "leaf_checkpoint_id": leaf_ev["checkpoint_id"],
                            "leaf_label": leaf_ev["version_label"],
                            "surviving_failed_case_ids": survived_ids,
                            "affected_categories": sorted({leaf_cat_map[cid] for cid in survived_ids}),
                        }
                    )

        timeline = [
            {
                "checkpoint_id": cid,
                "version_label": evals_by_id[cid]["version_label"],
                "transformation_type": evals_by_id[cid]["transformation_type"],
                "sample_size": evals_by_id[cid]["sample_size"],
                "pass_rate": evals_by_id[cid]["pass_rate"],
                "pass_rate_wilson_ci95": evals_by_id[cid]["pass_rate_wilson_ci95"],
                "mean_score": evals_by_id[cid]["mean_score"],
                "factual_contradiction_rate": evals_by_id[cid]["factual_contradiction_rate"],
                "category_scores": {
                    cat: info["mean_score"]
                    for cat, info in evals_by_id[cid]["categories"].items()
                },
                "category_pass_rates": {
                    cat: info["pass_rate"]
                    for cat, info in evals_by_id[cid]["categories"].items()
                },
            }
            for cid in topo
            if cid in evals_by_id
        ]

        return (
            {
                "epistemic_status": EpistemicStatus.OBSERVED.value,
                "suite_id": active_suite.suite_id,
                "suite_version": active_suite.version,
                "suite_fingerprint": suite_fp,
                "sample_size": len(active_suite.cases),
                "statistical_disclosure": (
                    f"Deterministic evaluation over n={len(active_suite.cases)} cases ({len({c.category for c in active_suite.cases})} categories). "
                    "Pass rates include 95% Wilson binomial CIs; transition deltas include 95% bootstrap CIs, exact two-sided McNemar raw p-values, and step-down Holm-Bonferroni adjusted p-values across category families. "
                    "Categories with n < 8 are strictly gated as exploratory (`statistically_significant=False`); note that exact two-sided McNemar has a discrete floor p_min = 2^(1-n) (0.0078125 at n=8), requiring n >= 9 to reject at alpha=0.05 across m >= 7 categories and n >= 30 for >=80% power on moderate drops, while continuous score-only drops with zero binary flips yield McNemar p=1.0."
                ),
                "checkpoint_evaluations": [evals_by_id[cid] for cid in topo if cid in evals_by_id],
                "timeline": timeline,
                "consecutive_transitions": consecutive_transitions,
                "base_to_current_transitions": base_to_current,
                "lineage_persistent_failures": lineage_persistent_failures,
                "surviving_regressions": surviving_regressions,
            },
            warnings,
        )
