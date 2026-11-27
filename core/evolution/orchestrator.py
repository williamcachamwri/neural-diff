"""Reproducible Multi-Generation Evolution Analysis Orchestrator (Sections 10 & 13).

Coordinates Tensor Evolution, Layer Evolution, Behavioral Evolution, Representation Evolution,
Regression Origin Detection, Evidence-Backed Insights, and Content-Addressed Caching.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import transformers

import safetensors
from benchmarks.datasets.loader import BenchmarkSuite, load_benchmark_suite
from core import __version__
from core.evolution.behavioral_evolution import (
    BehavioralEvolutionEngine,
    CheckpointGeneratorFn,
    compute_suite_fingerprint,
)
from core.evolution.insights import generate_evolution_insights
from core.evolution.layer_evolution import LayerEvolutionEngine
from core.evolution.registry import revalidate_checkpoint_health
from core.evolution.regression_origin import RegressionOriginEngine
from core.evolution.representation_evolution import RepresentationEvolutionEngine
from core.evolution.store import EvolutionStore
from core.evolution.tensor_evolution import MultiGenTensorEvolutionEngine
from core.evolution.types import CheckpointValidationStatus
from core.exceptions import EvolutionGraphError
from core.resource_manager import CancellationToken, ResourceManager
from core.security import SecurityPolicy
from core.tensor_engine.metrics import DiffTolerances


@dataclass
class EvolutionAnalysisConfig:
    """Complete reproducible configuration for a multi-generation evolution analysis run."""

    include_tensor_evolution: bool = True
    include_layer_evolution: bool = True
    include_behavioral_evolution: bool = True
    include_representation_evolution: bool = True
    enable_causal_patching: bool = True
    preferred_device: str = "auto"
    tolerances: DiffTolerances = field(default_factory=DiffTolerances)
    chunk_elements: int = 1_048_576
    top_k_tensors: int = 10
    behavioral_categories: list[str] | None = None
    behavioral_max_cases: int | None = None
    max_new_tokens: int = 64
    seed: int = 42
    prompt_mode: str = "chat_if_available"
    overall_pass_rate_drop_threshold: float = 0.05
    category_score_drop_threshold: float = 0.10
    local_files_only: bool = False
    allow_cache: bool = True
    force_rerun: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "include_tensor_evolution": self.include_tensor_evolution,
            "include_layer_evolution": self.include_layer_evolution,
            "include_behavioral_evolution": self.include_behavioral_evolution,
            "include_representation_evolution": self.include_representation_evolution,
            "enable_causal_patching": self.enable_causal_patching,
            "preferred_device": self.preferred_device,
            "tolerances": self.tolerances.to_dict(),
            "chunk_elements": self.chunk_elements,
            "top_k_tensors": self.top_k_tensors,
            "behavioral_categories": self.behavioral_categories,
            "behavioral_max_cases": self.behavioral_max_cases,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "prompt_mode": self.prompt_mode,
            "overall_pass_rate_drop_threshold": self.overall_pass_rate_drop_threshold,
            "category_score_drop_threshold": self.category_score_drop_threshold,
            "local_files_only": self.local_files_only,
            "allow_cache": self.allow_cache,
            "force_rerun": self.force_rerun,
        }


EvolutionProgressFn = Callable[[str, int, int], None]


class EvolutionOrchestrator:
    """Executes reproducible, memory-bounded evolution tracking across a project's checkpoint DAG."""

    def __init__(
        self,
        *,
        store: EvolutionStore,
        policy: SecurityPolicy | None = None,
        resource_manager: ResourceManager | None = None,
    ) -> None:
        self.store = store
        self.policy = policy or SecurityPolicy()
        self.resource_manager = resource_manager

    def compute_content_cache_key(
        self,
        project_id: str,
        config: EvolutionAnalysisConfig,
        *,
        suite_fingerprint: str | None = None,
    ) -> str:
        """Compute deterministic SHA-256 content cache key over checkpoint hashes, DAG edges, dataset & config."""
        proj = self.store.get_project(project_id)
        ckpts = self.store.list_checkpoints(project_id)
        rels = self.store.list_relationships(project_id)

        cfg_for_hash = config.to_dict()
        cfg_for_hash.pop("allow_cache", None)
        cfg_for_hash.pop("force_rerun", None)

        canonical_payload = {
            "neural_diff_version": __version__,
            "base_checkpoint_id": proj.base_checkpoint_id,
            "checkpoints": [
                {
                    "id": c.checkpoint_id,
                    "sha256": c.sha256_fingerprint,
                    "tensor_meta_hash": c.tensor_metadata_hash,
                    "transformation": c.transformation_type.value,
                }
                for c in sorted(ckpts, key=lambda x: x.checkpoint_id)
            ],
            "relationships": [
                {
                    "parent": r.parent_checkpoint_id,
                    "child": r.child_checkpoint_id,
                    "transformation": r.transformation_type.value,
                    "merge_weight": r.merge_weight,
                }
                for r in sorted(rels, key=lambda x: (x.parent_checkpoint_id, x.child_checkpoint_id))
            ],
            "suite_fingerprint": suite_fingerprint,
            "config": cfg_for_hash,
        }
        encoded = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def run_analysis(
        self,
        project_id: str,
        config: EvolutionAnalysisConfig | None = None,
        *,
        run_id: str | None = None,
        suite: BenchmarkSuite | None = None,
        custom_generator: CheckpointGeneratorFn | None = None,
        cancellation_token: CancellationToken | None = None,
        on_progress: EvolutionProgressFn | None = None,
    ) -> dict[str, Any]:
        cfg = config or EvolutionAnalysisConfig()
        proj = self.store.get_project(project_id)
        ckpts = self.store.list_checkpoints(project_id)
        if not ckpts:
            raise EvolutionGraphError(
                f"Evolution project '{proj.name}' has no registered checkpoints to analyze."
            )

        dag = self.store.get_dag(project_id)
        active_suite: BenchmarkSuite | None = None
        suite_fp: str | None = None
        if cfg.include_behavioral_evolution:
            active_suite = suite or load_benchmark_suite(
                categories=cfg.behavioral_categories, policy=self.policy
            )
            if cfg.behavioral_max_cases is not None and cfg.behavioral_max_cases > 0:
                active_suite = BenchmarkSuite(
                    suite_id=active_suite.suite_id,
                    version=active_suite.version,
                    description=active_suite.description,
                    cases=active_suite.cases[: cfg.behavioral_max_cases],
                )
            suite_fp = compute_suite_fingerprint(active_suite)

        cache_key = self.compute_content_cache_key(
            project_id, cfg, suite_fingerprint=suite_fp
        )
        if (
            cfg.allow_cache
            and not cfg.force_rerun
            and custom_generator is None
        ):
            cached_run = self.store.find_cached_run(project_id, cache_key)
            if cached_run is not None:
                return {**cached_run, "cache_hit": True}

        rm = self.resource_manager or ResourceManager(preferred_device=cfg.preferred_device)
        hw_snap = rm.snapshot().to_dict()
        eff_run_id = run_id or f"evorun-{uuid.uuid4().hex[:10]}"
        started_utc = dt.datetime.now(dt.UTC).isoformat()

        total_steps = (
            1
            + (1 if cfg.include_tensor_evolution else 0)
            + (1 if cfg.include_behavioral_evolution else 0)
            + (1 if cfg.include_representation_evolution else 0)
            + 1
        )
        provenance: dict[str, Any] = {
            "run_id": eff_run_id,
            "project_id": project_id,
            "content_cache_key": cache_key,
            "checkpoint_fingerprints": {
                c.checkpoint_id: {
                    "version_label": c.version_label,
                    "sha256": c.sha256_fingerprint,
                    "tensor_metadata_hash": c.tensor_metadata_hash,
                }
                for c in ckpts
            },
            "evaluation_dataset_fingerprint": suite_fp,
            "random_seed": cfg.seed,
            "software_versions": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "neural_diff": __version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "safetensors": safetensors.__version__,
            },
            "hardware": hw_snap,
            "selected_device": rm.resolve_device(),
            "device_reason": rm.device_reason,
            "started_at_utc": started_utc,
            "completed_at_utc": None,
        }

        self.store.create_run(
            run_id=eff_run_id,
            project_id=project_id,
            cache_key=cache_key,
            config=cfg.to_dict(),
            provenance=provenance,
            total_steps=total_steps,
        )

        def _notify(msg: str, step: int) -> None:
            self.store.update_run_progress(eff_run_id, msg, step, total_steps)
            if on_progress is not None:
                on_progress(msg, step, total_steps)

        warnings: list[str] = []
        done = 0

        with rm.managed_execution(cancellation_token):
            # Step 1: Verify checkpoint health on disk before running engines
            _notify("Validating checkpoint health & SHA-256 fingerprints", done)
            healthy_ids: set[str] = set()
            for ckpt in ckpts:
                if cancellation_token is not None:
                    cancellation_token.check()
                st, err = revalidate_checkpoint_health(ckpt, policy=self.policy)
                if st == CheckpointValidationStatus.CORRUPTED_OR_MISSING:
                    warnings.append(
                        f"Checkpoint '{ckpt.version_label}' ({ckpt.checkpoint_id}) failed health check: {err}"
                    )
                    self.store.update_checkpoint(
                        project_id,
                        ckpt.checkpoint_id,
                        validation_status=CheckpointValidationStatus.CORRUPTED_OR_MISSING,
                    )
                else:
                    healthy_ids.add(ckpt.checkpoint_id)
            done += 1

            # Step 2: Multi-Generation Tensor & Cumulative Distance Engine
            tensor_transitions: list[dict[str, Any]] = []
            raw_records_map: dict[tuple[str, str, str], list[Any]] = {}
            if cfg.include_tensor_evolution:
                _notify("Computing multi-generation & direct base-to-current tensor diffs", done)
                t_engine = MultiGenTensorEvolutionEngine(
                    policy=self.policy,
                    tolerances=cfg.tolerances,
                    chunk_elements=cfg.chunk_elements,
                    top_k=cfg.top_k_tensors,
                    local_files_only=cfg.local_files_only,
                )
                t_objs, raw_records_map, t_warns = t_engine.run_all_transitions(
                    dag,
                    base_checkpoint_id=proj.base_checkpoint_id,
                    cancellation_token=cancellation_token,
                )
                tensor_transitions = [t.to_dict() for t in t_objs]
                warnings.extend(t_warns)
                done += 1

            # Step 3: Behavioral Evolution Engine
            beh_payload: dict[str, Any] | None = None
            if cfg.include_behavioral_evolution and active_suite is not None:
                _notify("Running deterministic multi-generation behavioral suite", done)
                b_engine = BehavioralEvolutionEngine(
                    store=self.store,
                    resource_manager=rm,
                    policy=self.policy,
                    max_new_tokens=cfg.max_new_tokens,
                    seed=cfg.seed,
                    prompt_mode=cfg.prompt_mode,
                    local_files_only=cfg.local_files_only,
                    use_cache=cfg.allow_cache and not cfg.force_rerun,
                )
                beh_payload, b_warns = b_engine.run(
                    dag,
                    base_checkpoint_id=proj.base_checkpoint_id,
                    suite=active_suite,
                    custom_generator=custom_generator,
                    cancellation_token=cancellation_token,
                )
                warnings.extend(b_warns)
                done += 1

            # Step 4: Representation Evolution Engine (Linear CKA, Cosine, Logit Lens, Patching)
            rep_payload: dict[str, Any] | None = None
            if cfg.include_representation_evolution:
                _notify("Capturing forward-hook representations & Linear CKA matrices", done)
                r_engine = RepresentationEvolutionEngine(
                    resource_manager=rm,
                    seed=cfg.seed,
                    enable_causal_patching=cfg.enable_causal_patching,
                )
                rep_payload, r_warns = r_engine.run(
                    dag,
                    base_checkpoint_id=proj.base_checkpoint_id,
                    cancellation_token=cancellation_token,
                )
                warnings.extend(r_warns)
                done += 1

            # Step 5: Regression Origin, Layer Evolution & Evidence-Backed Insights
            _notify("Synthesizing layer evolution, regression origins & insights", done)
            reg_engine = RegressionOriginEngine(
                overall_pass_rate_drop_threshold=cfg.overall_pass_rate_drop_threshold,
                category_score_drop_threshold=cfg.category_score_drop_threshold,
            )
            regression_events = reg_engine.detect(
                dag=dag,
                behavioral_evolution=beh_payload,
                tensor_transitions=tensor_transitions,
                representation_evolution=rep_payload,
            )

            ckpt_labels = {c.checkpoint_id: c.version_label for c in ckpts}
            layer_payload = (
                LayerEvolutionEngine().analyze(
                    raw_records_map=raw_records_map,
                    checkpoint_labels=ckpt_labels,
                    representation_evolution=rep_payload,
                    regression_events=regression_events,
                )
                if cfg.include_layer_evolution
                else {"epistemic_status": "OBSERVED", "layer_indices": [], "transition_profiles": []}
            )

            insights = generate_evolution_insights(
                tensor_transitions=tensor_transitions,
                layer_evolution=layer_payload,
                behavioral_evolution=beh_payload,
                representation_evolution=rep_payload,
                regression_events=regression_events,
            )
            done += 1

        completed_utc = dt.datetime.now(dt.UTC).isoformat()
        provenance["completed_at_utc"] = completed_utc

        branches = dag.enumerate_branches()
        overview = {
            "total_checkpoints": len(ckpts),
            "healthy_checkpoints": len(healthy_ids),
            "total_transformations": len(dag.relationships),
            "branch_count": len(branches),
            "detected_regressions": len(regression_events),
            "most_modified_layers": layer_payload.get("most_modified_layers", []),
            "least_modified_layers": layer_payload.get("least_modified_layers", []),
            "peak_cumulative_l2": max(
                (
                    float(t["l2_distance"])
                    for t in tensor_transitions
                    if t.get("scope") == "base_to_current" and t.get("l2_distance") is not None
                ),
                default=0.0,
            ),
            "min_base_cosine_similarity": min(
                (
                    float(t["global_cosine_similarity"])
                    for t in tensor_transitions
                    if t.get("scope") == "base_to_current"
                    and t.get("global_cosine_similarity") is not None
                ),
                default=1.0,
            ),
        }

        self.store.save_completed_run(
            run_id=eff_run_id,
            project_id=project_id,
            provenance=provenance,
            overview=overview,
            tensor_transitions=tensor_transitions,
            layer_evolution=layer_payload,
            behavioral_evolution=beh_payload,
            representation_evolution=rep_payload,
            regression_events=regression_events,
            insights=insights,
            warnings=warnings,
        )
        return {**self.store.get_run(eff_run_id), "cache_hit": False}
