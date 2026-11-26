"""JSON reliability and schema conformance deterministic evaluator."""

from __future__ import annotations

import json
import re
from typing import Any

from benchmarks.evaluators.base import CaseEvalResult, DeterministicEvaluator, EvalCase
from core.types import EpistemicStatus

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", flags=re.DOTALL | re.IGNORECASE)

_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "str": (str,),
    "string": (str,),
    "int": (int,),
    "integer": (int,),
    "float": (int, float),
    "number": (int, float),
    "bool": (bool,),
    "boolean": (bool,),
    "list": (list,),
    "array": (list,),
    "dict": (dict,),
    "object": (dict,),
}


def _extract_json_object(raw: str) -> Any:
    m = _JSON_BLOCK_RE.search(raw)
    if m:
        return json.loads(m.group(1))
    # Scan first balanced `{...}` or `[...]`
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw):
        if ch in ("{", "["):
            try:
                obj, _end = decoder.raw_decode(raw[idx:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("No valid JSON object or array found.")


def _check_type(value: Any, expected_type_name: str) -> bool:
    tname = expected_type_name.lower()
    if tname in ("int", "integer", "float", "number") and isinstance(value, bool):
        return False
    allowed = _TYPE_MAP.get(tname)
    if allowed is None:
        return True
    return isinstance(value, allowed)


class JsonReliabilityEvaluator(DeterministicEvaluator):
    """Evaluates raw/extracted JSON validity, required keys, field types, and exact field values."""

    category = "json_reliability"

    def evaluate(self, case: EvalCase, output_text: str) -> CaseEvalResult:
        stripped = output_text.strip()
        strict_raw_valid = False
        extracted_valid = False
        parsed: Any = None

        if stripped:
            try:
                parsed = json.loads(stripped)
                strict_raw_valid = True
                extracted_valid = True
            except json.JSONDecodeError:
                try:
                    parsed = _extract_json_object(stripped)
                    extracted_valid = True
                except Exception:
                    extracted_valid = False

        require_strict = bool(case.expected.get("require_strict_raw_json", False))
        req_keys: list[str] = [str(k) for k in case.expected.get("required_keys", [])]
        field_types: dict[str, str] = dict(case.expected.get("field_types", {}))
        exact_fields: dict[str, Any] = dict(case.expected.get("exact_values", {}))

        if not extracted_valid or not isinstance(parsed, dict):
            is_array_ok = extracted_valid and case.expected.get("root_type") == "array" and isinstance(parsed, list)
            if not is_array_ok:
                return CaseEvalResult(
                    case_id=case.case_id,
                    category=self.category,
                    passed=False,
                    score=0.1 if extracted_valid else 0.0,
                    evaluator_type="deterministic",
                    epistemic_status=EpistemicStatus.OBSERVED,
                    submetrics={
                        "strict_raw_valid": strict_raw_valid,
                        "extracted_valid": extracted_valid,
                        "key_coverage": 0.0,
                        "type_conformance": 0.0,
                        "exact_value_match": 0.0,
                    },
                    rationale="Response did not contain a valid JSON object matching expected root structure.",
                )

        if isinstance(parsed, dict):
            key_hits = sum(1 for k in req_keys if k in parsed)
            key_ratio = (key_hits / len(req_keys)) if req_keys else 1.0

            type_hits = sum(
                1 for k, tname in field_types.items() if k in parsed and _check_type(parsed[k], tname)
            )
            type_ratio = (type_hits / len(field_types)) if field_types else 1.0

            val_hits = 0
            for k, expected_v in exact_fields.items():
                if k in parsed:
                    actual_v = parsed[k]
                    if isinstance(expected_v, str) and isinstance(actual_v, str):
                        if actual_v.strip().lower() == expected_v.strip().lower():
                            val_hits += 1
                    elif actual_v == expected_v:
                        val_hits += 1
            val_ratio = (val_hits / len(exact_fields)) if exact_fields else 1.0
        else:
            min_len = int(case.expected.get("min_items", 1))
            key_ratio = 1.0 if len(parsed) >= min_len else 0.0
            type_ratio = 1.0
            val_ratio = 1.0

        strict_bonus = 1.0 if strict_raw_valid else (0.5 if not require_strict else 0.0)
        score = round(
            0.30 * 1.0
            + 0.15 * strict_bonus
            + 0.25 * key_ratio
            + 0.15 * type_ratio
            + 0.15 * val_ratio,
            6,
        )
        passed = (
            extracted_valid
            and (strict_raw_valid or not require_strict)
            and key_ratio >= 1.0
            and type_ratio >= 1.0
            and val_ratio >= 1.0
        )

        return CaseEvalResult(
            case_id=case.case_id,
            category=self.category,
            passed=passed,
            score=score,
            evaluator_type="deterministic",
            epistemic_status=EpistemicStatus.OBSERVED,
            submetrics={
                "strict_raw_valid": strict_raw_valid,
                "extracted_valid": extracted_valid,
                "key_coverage": round(key_ratio, 4),
                "type_conformance": round(type_ratio, 4),
                "exact_value_match": round(val_ratio, 4),
            },
            rationale=(
                f"JSON valid (strict={strict_raw_valid}, extracted={extracted_valid}), "
                f"keys={key_ratio:.0%}, types={type_ratio:.0%}, values={val_ratio:.0%}."
            ),
        )
