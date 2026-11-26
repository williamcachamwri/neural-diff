"""Instruction following deterministic constraint evaluator."""

from __future__ import annotations

import re

from benchmarks.evaluators.base import CaseEvalResult, DeterministicEvaluator, EvalCase
from core.types import EpistemicStatus

_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+\S+", flags=re.MULTILINE)


class InstructionFollowingEvaluator(DeterministicEvaluator):
    """Verifies hard structural, lexical, and length constraints (IFEval-style deterministic checks)."""

    category = "instruction_following"

    def evaluate(self, case: EvalCase, output_text: str) -> CaseEvalResult:
        text = output_text.strip()
        words = text.split()
        word_count = len(words)
        exp = case.expected

        checks: dict[str, bool] = {}

        if "max_words" in exp:
            checks["max_words"] = 0 < word_count <= int(exp["max_words"])
        if "min_words" in exp:
            checks["min_words"] = word_count >= int(exp["min_words"])
        if "exact_bullet_count" in exp:
            bullets = _BULLET_LINE_RE.findall(text)
            checks["exact_bullet_count"] = len(bullets) == int(exp["exact_bullet_count"])
        if "must_include" in exp:
            checks["must_include"] = all(
                str(sub).lower() in text.lower() for sub in exp["must_include"]
            )
        if "must_exclude" in exp:
            checks["must_exclude"] = all(
                str(sub).lower() not in text.lower() for sub in exp["must_exclude"]
            )
        if "starts_with" in exp:
            prefix = str(exp["starts_with"])
            checks["starts_with"] = text.lower().startswith(prefix.lower())
        if "ends_with" in exp:
            suffix = str(exp["ends_with"])
            checks["ends_with"] = text.lower().endswith(suffix.lower())
        if exp.get("no_commas"):
            checks["no_commas"] = "," not in text and text != ""
        if exp.get("all_uppercase"):
            alpha_chars = [c for c in text if c.isalpha()]
            checks["all_uppercase"] = bool(alpha_chars) and all(c.isupper() for c in alpha_chars)

        if not checks:
            checks["non_empty"] = bool(text)

        satisfied = sum(1 for ok in checks.values() if ok)
        total = len(checks)
        score = round(satisfied / total, 6)
        passed = satisfied == total

        return CaseEvalResult(
            case_id=case.case_id,
            category=self.category,
            passed=passed,
            score=score,
            evaluator_type="deterministic",
            epistemic_status=EpistemicStatus.OBSERVED,
            submetrics={
                "word_count": word_count,
                "constraints_satisfied": satisfied,
                "constraints_total": total,
                "constraint_details": checks,
            },
            rationale=f"Satisfied {satisfied}/{total} instruction constraints: {checks}.",
        )
