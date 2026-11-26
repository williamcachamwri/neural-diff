"""Conciseness and verbosity efficiency deterministic evaluator."""

from __future__ import annotations

import re

from benchmarks.evaluators.base import CaseEvalResult, DeterministicEvaluator, EvalCase
from core.types import EpistemicStatus

_FILLER_PHRASES = (
    "sure!",
    "certainly!",
    "of course!",
    "here is the answer",
    "here's the answer",
    "as an ai language model",
    "as a large language model",
    "i would be happy to",
    "let me explain",
)


class ConcisenessEvaluator(DeterministicEvaluator):
    """Scores factual directness and token/word economy without rewarding empty or wrong responses."""

    category = "conciseness"

    def evaluate(self, case: EvalCase, output_text: str) -> CaseEvalResult:
        text = output_text.strip()
        words = re.findall(r"\S+", text)
        word_count = len(words)

        accepted_facts: list[str] = [str(f).lower() for f in case.expected.get("accepted_facts", [])]
        target_max_words = max(1, int(case.expected.get("target_max_words", 15)))
        hard_max_words = max(target_max_words, int(case.expected.get("hard_max_words", target_max_words * 2)))

        lower = text.lower()
        fact_present = True if not accepted_facts else any(f in lower for f in accepted_facts)
        if not text or not fact_present:
            return CaseEvalResult(
                case_id=case.case_id,
                category=self.category,
                passed=False,
                score=0.0,
                evaluator_type="deterministic",
                epistemic_status=EpistemicStatus.OBSERVED,
                submetrics={
                    "fact_present": False,
                    "word_count": word_count,
                    "target_max_words": target_max_words,
                    "filler_free": True,
                },
                rationale="Required core answer fact missing; conciseness cannot be credited on incorrect output.",
            )

        filler_hits = [p for p in _FILLER_PHRASES if p in lower]
        filler_free = len(filler_hits) == 0

        if word_count <= target_max_words:
            brevity_score = 1.0
        elif word_count <= hard_max_words:
            brevity_score = max(
                0.35,
                1.0 - 0.65 * ((word_count - target_max_words) / max(1, hard_max_words - target_max_words)),
            )
        else:
            brevity_score = max(0.10, target_max_words / word_count)

        filler_penalty = 0.0 if filler_free else 0.20
        score = round(max(0.0, min(1.0, brevity_score - filler_penalty)), 6)
        passed = fact_present and word_count <= target_max_words and filler_free

        return CaseEvalResult(
            case_id=case.case_id,
            category=self.category,
            passed=passed,
            score=score,
            evaluator_type="deterministic",
            epistemic_status=EpistemicStatus.OBSERVED,
            submetrics={
                "fact_present": True,
                "word_count": word_count,
                "target_max_words": target_max_words,
                "filler_free": filler_free,
                "filler_phrases_detected": filler_hits,
            },
            rationale=(
                f"Fact present in {word_count} words (target <= {target_max_words}, filler_free={filler_free})."
            ),
        )
