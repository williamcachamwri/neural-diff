"""Registry mapping behavioral evaluation categories to deterministic evaluators."""

from __future__ import annotations

from benchmarks.evaluators.base import CaseEvalResult, DeterministicEvaluator, EvalCase
from benchmarks.evaluators.code_preservation import CodePreservationEvaluator
from benchmarks.evaluators.conciseness import ConcisenessEvaluator
from benchmarks.evaluators.factual import FactualAccuracyEvaluator
from benchmarks.evaluators.instruction_following import InstructionFollowingEvaluator
from benchmarks.evaluators.json_reliability import JsonReliabilityEvaluator
from benchmarks.evaluators.math_reasoning import MathReasoningEvaluator
from benchmarks.evaluators.refusal import RefusalBehaviorEvaluator
from benchmarks.evaluators.vietnamese import VietnameseQualityEvaluator

ALL_CATEGORIES: tuple[str, ...] = (
    "vietnamese",
    "code_preservation",
    "json_reliability",
    "math_reasoning",
    "instruction_following",
    "conciseness",
    "refusal_behavior",
    "factual_accuracy",
)


def build_evaluator_registry() -> dict[str, DeterministicEvaluator]:
    evaluators: list[DeterministicEvaluator] = [
        VietnameseQualityEvaluator(),
        CodePreservationEvaluator(),
        JsonReliabilityEvaluator(),
        MathReasoningEvaluator(),
        InstructionFollowingEvaluator(),
        ConcisenessEvaluator(),
        RefusalBehaviorEvaluator(),
        FactualAccuracyEvaluator(),
    ]
    return {ev.category: ev for ev in evaluators}


def evaluate_case(case: EvalCase, output_text: str) -> CaseEvalResult:
    registry = build_evaluator_registry()
    evaluator = registry.get(case.category)
    if evaluator is None:
        raise ValueError(
            f"Unsupported behavioral category '{case.category}'. Supported: {list(registry.keys())}"
        )
    return evaluator.evaluate(case, output_text)
