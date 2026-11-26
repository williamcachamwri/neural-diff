"""Factual accuracy deterministic ground-truth QA evaluator."""

from __future__ import annotations

from benchmarks.evaluators.base import CaseEvalResult, DeterministicEvaluator, EvalCase
from core.types import EpistemicStatus


class FactualAccuracyEvaluator(DeterministicEvaluator):
    """Checks ground-truth canonical facts while penalizing contradictory claims."""

    category = "factual_accuracy"

    def evaluate(self, case: EvalCase, output_text: str) -> CaseEvalResult:
        lower = output_text.strip().lower()
        accepted_aliases: list[str] = [str(a).lower() for a in case.expected.get("accepted_answers", [])]
        required_all: list[str] = [str(r).lower() for r in case.expected.get("required_all", [])]
        contradictions: list[str] = [str(c).lower() for c in case.expected.get("contradictions", [])]

        alias_hit = True if not accepted_aliases else any(a in lower for a in accepted_aliases)
        all_hits = sum(1 for r in required_all if r in lower)
        all_ratio = (all_hits / len(required_all)) if required_all else 1.0
        contradiction_found = any(c in lower for c in contradictions)

        if not lower:
            score = 0.0
            passed = False
        else:
            base = (0.6 * (1.0 if alias_hit else 0.0)) + (0.4 * all_ratio)
            if not required_all:
                base = 1.0 if alias_hit else 0.0
            if contradiction_found:
                base *= 0.1
            score = round(base, 6)
            passed = alias_hit and (all_ratio >= 1.0) and (not contradiction_found)

        return CaseEvalResult(
            case_id=case.case_id,
            category=self.category,
            passed=passed,
            score=score,
            evaluator_type="deterministic",
            epistemic_status=EpistemicStatus.OBSERVED,
            submetrics={
                "accepted_answer_matched": alias_hit,
                "required_all_ratio": round(all_ratio, 4),
                "contradiction_detected": contradiction_found,
            },
            rationale=(
                f"Factual check: answer_matched={alias_hit}, required_all={all_ratio:.0%}, "
                f"contradiction={contradiction_found}."
            ),
        )
