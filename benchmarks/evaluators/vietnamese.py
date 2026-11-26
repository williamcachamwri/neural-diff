"""Vietnamese language quality deterministic evaluator."""

from __future__ import annotations

import re
import unicodedata

from benchmarks.evaluators.base import CaseEvalResult, DeterministicEvaluator, EvalCase
from core.types import EpistemicStatus

# Precomposed NFC Vietnamese characters with tonal or vowel-specific diacritics
_VIETNAMESE_DIACRITIC_CHARS = set(
    "àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ"
)

_COMMON_VIETNAMESE_FUNCTION_WORDS = {
    "là",
    "của",
    "và",
    "trong",
    "một",
    "các",
    "được",
    "cho",
    "với",
    "không",
    "có",
    "những",
    "này",
    "khi",
    "để",
}


class VietnameseQualityEvaluator(DeterministicEvaluator):
    """Evaluates Vietnamese Unicode NFC integrity, diacritic richness, and required task terms."""

    category = "vietnamese"

    def evaluate(self, case: EvalCase, output_text: str) -> CaseEvalResult:
        normalized = unicodedata.normalize("NFC", output_text.strip())
        if not normalized:
            return CaseEvalResult(
                case_id=case.case_id,
                category=self.category,
                passed=False,
                score=0.0,
                evaluator_type="deterministic",
                epistemic_status=EpistemicStatus.OBSERVED,
                submetrics={
                    "non_empty": False,
                    "mojibake_free": True,
                    "diacritic_word_ratio": 0.0,
                    "keyword_hit_ratio": 0.0,
                },
                rationale="Empty model response.",
            )

        mojibake_free = "\ufffd" not in normalized
        words = re.findall(r"[^\W\d_]+", normalized, flags=re.UNICODE)
        word_count = len(words)

        diacritic_words = sum(
            1 for w in words if any(ch in _VIETNAMESE_DIACRITIC_CHARS for ch in w)
        )
        diacritic_ratio = (diacritic_words / word_count) if word_count > 0 else 0.0

        lower_words = {w.lower() for w in words}
        vi_func_hits = len(lower_words & _COMMON_VIETNAMESE_FUNCTION_WORDS)

        required_keywords: list[str] = [
            unicodedata.normalize("NFC", str(k)).lower()
            for k in case.expected.get("required_keywords", [])
        ]
        any_of_keywords: list[str] = [
            unicodedata.normalize("NFC", str(k)).lower()
            for k in case.expected.get("any_of_keywords", [])
        ]
        min_words = int(case.expected.get("min_words", 4))

        norm_lower = normalized.lower()
        req_hits = sum(1 for kw in required_keywords if kw in norm_lower)
        req_ratio = (req_hits / len(required_keywords)) if required_keywords else 1.0
        any_hit = True if not any_of_keywords else any(kw in norm_lower for kw in any_of_keywords)
        length_ok = word_count >= min_words

        # Vietnamese orthographic fluency: authentic Vietnamese prose typically carries >= 25% diacritic words
        diacritic_score = min(1.0, diacritic_ratio / 0.25)
        func_bonus = 1.0 if (vi_func_hits >= 1 or word_count < 6) else 0.5

        raw_score = (
            0.35 * diacritic_score
            + 0.40 * req_ratio
            + 0.15 * (1.0 if any_hit else 0.0)
            + 0.10 * (1.0 if (mojibake_free and length_ok) else 0.0)
        ) * func_bonus
        if not mojibake_free:
            raw_score *= 0.2

        score = round(max(0.0, min(1.0, raw_score)), 6)
        passed = (
            mojibake_free
            and length_ok
            and diacritic_ratio >= 0.20
            and req_ratio >= 1.0
            and any_hit
        )

        return CaseEvalResult(
            case_id=case.case_id,
            category=self.category,
            passed=passed,
            score=score,
            evaluator_type="deterministic",
            epistemic_status=EpistemicStatus.OBSERVED,
            submetrics={
                "non_empty": True,
                "mojibake_free": mojibake_free,
                "word_count": word_count,
                "diacritic_words": diacritic_words,
                "diacritic_word_ratio": round(diacritic_ratio, 4),
                "vietnamese_function_words": vi_func_hits,
                "keyword_hit_ratio": round(req_ratio, 4),
                "any_of_satisfied": any_hit,
            },
            rationale=(
                f"Vietnamese NFC check: {diacritic_words}/{word_count} diacritic words "
                f"({diacritic_ratio:.1%}), required_keywords={req_ratio:.0%}, any_of={any_hit}."
            ),
        )
