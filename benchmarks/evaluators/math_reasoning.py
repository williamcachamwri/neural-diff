"""Mathematical reasoning deterministic numerical answer evaluator."""

from __future__ import annotations

import math
import re

from benchmarks.evaluators.base import CaseEvalResult, DeterministicEvaluator, EvalCase
from core.types import EpistemicStatus

_BOXED_RE = re.compile(r"\\boxed\{\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?)\s*\}")
_HASH_ANS_RE = re.compile(r"####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?)")
_EXPLICIT_ANS_RE = re.compile(
    r"(?:answer|result|kết quả|giá trị)\s*(?:is|là|=|:)\s*\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?)",
    flags=re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")


def _parse_numeric_token(token: str) -> float | None:
    cleaned = token.replace(",", "").strip()
    if "/" in cleaned:
        parts = cleaned.split("/", 1)
        try:
            denom = float(parts[1])
            if denom == 0.0:
                return None
            return float(parts[0]) / denom
        except ValueError:
            return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_final_number(text: str) -> tuple[float | None, str]:
    """Extract predicted numerical answer prioritizing explicit markers (`\\boxed`, `####`, `answer is`)."""
    for label, pattern in (
        ("boxed", _BOXED_RE),
        ("gsm8k_hash", _HASH_ANS_RE),
        ("explicit_phrase", _EXPLICIT_ANS_RE),
    ):
        matches = pattern.findall(text)
        if matches:
            val = _parse_numeric_token(matches[-1])
            if val is not None:
                return val, label

    all_nums = _NUMBER_RE.findall(text)
    if all_nums:
        val = _parse_numeric_token(all_nums[-1])
        if val is not None:
            return val, "last_number"
    return None, "none"


class MathReasoningEvaluator(DeterministicEvaluator):
    """Evaluates mathematical accuracy via deterministic answer extraction and tolerance comparison."""

    category = "math_reasoning"

    def evaluate(self, case: EvalCase, output_text: str) -> CaseEvalResult:
        target = float(case.expected["answer"])
        tol = float(case.expected.get("tolerance", 1e-5))

        pred, source = extract_final_number(output_text)
        if pred is None or math.isnan(pred) or math.isinf(pred):
            return CaseEvalResult(
                case_id=case.case_id,
                category=self.category,
                passed=False,
                score=0.0,
                evaluator_type="deterministic",
                epistemic_status=EpistemicStatus.OBSERVED,
                submetrics={
                    "extracted_answer": None,
                    "expected_answer": target,
                    "extraction_method": source,
                    "abs_error": None,
                },
                rationale="No valid numeric answer could be extracted from model output.",
            )

        abs_err = abs(pred - target)
        exact_match = abs_err <= tol
        # Partial credit only if explicit formatting was used and relative error < 5% (still `passed=False`)
        rel_err = abs_err / max(abs(target), 1.0)
        score = 1.0 if exact_match else (0.15 if rel_err <= 0.05 else 0.0)

        return CaseEvalResult(
            case_id=case.case_id,
            category=self.category,
            passed=exact_match,
            score=score,
            evaluator_type="deterministic",
            epistemic_status=EpistemicStatus.OBSERVED,
            submetrics={
                "extracted_answer": pred,
                "expected_answer": target,
                "extraction_method": source,
                "abs_error": round(abs_err, 8),
            },
            rationale=(
                f"Predicted {pred} via `{source}` vs expected {target} (abs_error={abs_err:.6g})."
            ),
        )
