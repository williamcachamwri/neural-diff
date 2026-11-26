"""Unified Comparison Orchestrator coordinating Modules A through H."""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.activation_analysis.comparator import ActivationDiffEngine
from core.behavioral_analysis.runner import BehavioralDiffEngine
from core.loaders.model_loader import ModelLoader
from core.regression_engine.engine import (
    RegressionBaseline,
    RegressionEngine,
    RegressionThresholds,
)
from core.report_engine.exporter import (
    ComparisonBundle,
    default_reproducibility_metadata,
)
from core.resource_manager import CancellationToken, ResourceManager
from core.security import SecurityPolicy
from core.structural_analysis.diff import StructuralDiffEngine
from core.structural_analysis.neuron_attention import NeuronAttentionAnalyzer
from core.tensor_engine.comparator import TensorDiffEngine
from core.tensor_engine.metrics import DiffTolerances

VALID_MODULES: tuple[str, ...] = (
    "structural",
    "tensor",
    "neuron",
    "activation",
    "behavioral",
    "regression",
)


@dataclass
class ComparisonRequestConfig:
    """Configuration for an end-to-end NEURAL DIFF comparison run."""

    model_a: str
    model_b: str
    modules: list[str] = field(
        default_factory=lambda: ["structural", "tensor", "neuron", "regression"]
    )
    experiment_type: str = "SYNTHETIC FIXTURE EXPERIMENT"
    tolerances: DiffTolerances = field(default_factory=DiffTolerances)
    thresholds: RegressionThresholds = field(default_factory=RegressionThresholds)
    preferred_device: str = "auto"
    chunk_elements: int = 1_048_576
    activation_prompts: list[str] | None = None
    behavioral_categories: list[str] | None = None
    behavioral_max_cases: int | None = None
    max_new_tokens: int = 128
    seed: int = 42
    prompt_mode: str = "chat_if_available"
    local_files_only: bool = False
    baseline_path: str | None = None


ProgressCallback = Callable[[str, int, int], None]


class ComparisonOrchestrator:
    """Executes genuine multi-module model comparisons with real step-boundary telemetry."""

    def __init__(
        self,
        *,
        policy: SecurityPolicy | None = None,
        resource_manager: ResourceManager | None = None,
    ) -> None:
        self.policy = policy or SecurityPolicy()
        self.resource_manager = resource_manager

    def run(
        self,
        config: ComparisonRequestConfig,
        *,
        comparison_id: str | None = None,
        cancellation_token: CancellationToken | None = None,
        on_progress: ProgressCallback | None = None,
        historical_runs: list[dict[str, Any]] | None = None,
    ) -> ComparisonBundle:
        selected = [m for m in VALID_MODULES if m in set(config.modules)]
        if not selected:
            selected = ["structural"]

        rm = self.resource_manager or ResourceManager(
            preferred_device=config.preferred_device,
            chunk_elements=config.chunk_elements,
        )
        loader = ModelLoader(policy=self.policy)
        comp_id = comparison_id or f"ndiff-{uuid.uuid4().hex[:10]}"
        created_utc = dt.datetime.now(dt.timezone.utc).isoformat()

        total_steps = len(selected) + 1
        completed_steps = 0

        def _notify(stage_msg: str) -> None:
            if on_progress is not None:
                on_progress(stage_msg, completed_steps, total_steps)

        with rm.managed_execution(cancellation_token) as telemetry:
            _notify("Loading and validating checkpoint metadata")
            with (
                loader.load(config.model_a, local_files_only=config.local_files_only) as handle_a,
                loader.load(config.model_b, local_files_only=config.local_files_only) as handle_b,
            ):
                completed_steps += 1
                meta_a = handle_a.metadata
                meta_b = handle_b.metadata

                structural_res = None
                tensor_res = None
                neuron_res = None

                if "structural" in selected:
                    if cancellation_token is not None:
                        cancellation_token.check()
                    _notify("Computing structural & compatibility diff")
                    structural_res = StructuralDiffEngine(
                        chunk_elements=config.chunk_elements
                    ).compare(handle_a, handle_b)
                    completed_steps += 1

                if "tensor" in selected:
                    if cancellation_token is not None:
                        cancellation_token.check()
                    _notify("Computing chunked float64 tensor diff")
                    tensor_res = TensorDiffEngine(
                        tolerances=config.tolerances,
                        chunk_elements=config.chunk_elements,
                    ).compare(handle_a, handle_b, cancellation_token=cancellation_token)
                    completed_steps += 1

                if "neuron" in selected:
                    if cancellation_token is not None:
                        cancellation_token.check()
                    _notify("Analyzing attention heads, MLP coordinates, embeddings, and norms")
                    neuron_res = NeuronAttentionAnalyzer().analyze(
                        handle_a, handle_b, cancellation_token=cancellation_token
                    )
                    completed_steps += 1

            rm.cleanup()

            activation_res = None
            if "activation" in selected:
                if cancellation_token is not None:
                    cancellation_token.check()
                _notify("Running controlled forward-hook activation comparison")
                act_engine = ActivationDiffEngine(
                    resource_manager=rm,
                    policy=self.policy,
                    seed=config.seed,
                    dtype="bfloat16" if rm.resolve_device().startswith("cuda") else "float32",
                )
                activation_res = act_engine.compare_pretrained(
                    config.model_a,
                    config.model_b,
                    prompts=config.activation_prompts,
                    local_files_only=config.local_files_only,
                    cancellation_token=cancellation_token,
                )
                completed_steps += 1

            behavioral_res = None
            if "behavioral" in selected:
                if cancellation_token is not None:
                    cancellation_token.check()
                _notify("Running deterministic paired behavioral evaluation suite")
                beh_engine = BehavioralDiffEngine(
                    resource_manager=rm,
                    policy=self.policy,
                    max_new_tokens=config.max_new_tokens,
                    seed=config.seed,
                    prompt_mode=config.prompt_mode,
                )
                behavioral_res = beh_engine.compare_pretrained(
                    config.model_a,
                    config.model_b,
                    categories=config.behavioral_categories,
                    max_cases=config.behavioral_max_cases,
                    local_files_only=config.local_files_only,
                    cancellation_token=cancellation_token,
                )
                completed_steps += 1

            regression_res = None
            if "regression" in selected:
                if cancellation_token is not None:
                    cancellation_token.check()
                _notify("Evaluating regression policy thresholds")
                loaded_baseline = (
                    RegressionBaseline.load(config.baseline_path, policy=self.policy)
                    if config.baseline_path
                    else None
                )
                regression_res = RegressionEngine(thresholds=config.thresholds).evaluate(
                    model_a_source=config.model_a,
                    model_b_source=config.model_b,
                    structural=structural_res,
                    tensor_diff=tensor_res,
                    activation=activation_res,
                    behavioral=behavioral_res,
                    baseline=loaded_baseline,
                    historical_runs=historical_runs,
                )
                completed_steps += 1

            _notify("Completed comparison")

        repro = default_reproducibility_metadata()
        repro.update(
            {
                "seed": config.seed,
                "device_selected": rm.resolve_device(),
                "device_reason": rm.device_reason,
                "tolerances": config.tolerances.to_dict(),
                "thresholds": config.thresholds.to_dict(),
            }
        )

        return ComparisonBundle(
            comparison_id=comp_id,
            created_at_utc=created_utc,
            experiment_type=config.experiment_type,
            model_a=meta_a,
            model_b=meta_b,
            selected_modules=selected,
            structural=structural_res,
            tensor_diff=tensor_res,
            neuron_attention=neuron_res,
            activation=activation_res,
            behavioral=behavioral_res,
            regression=regression_res,
            execution_telemetry=telemetry,
            reproducibility=repro,
        )
