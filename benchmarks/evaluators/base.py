"""Base abstractions and schemas for deterministic behavioral evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from core.types import EpistemicStatus


@dataclass(frozen=True)
class EvalCase:
    """Single deterministic behavioral benchmark prompt and verification contract."""

    case_id: str
    category: str
    prompt: str
    description: str
    expected: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CaseEvalResult:
    """Deterministic evaluation result for one model output on one `EvalCase`."""

    case_id: str
    category: str
    passed: bool
    score: float
    evaluator_type: str  # Strictly "deterministic" unless an explicit model judge is invoked
    epistemic_status: EpistemicStatus
    submetrics: dict[str, Any]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["epistemic_status"] = self.epistemic_status.value
        return data


class DeterministicEvaluator(ABC):
    """Base class for deterministic, zero-LLM-judge behavioral evaluators."""

    category: str

    @abstractmethod
    def evaluate(self, case: EvalCase, output_text: str) -> CaseEvalResult:
        """Score `output_text` deterministically against `case.expected`."""
