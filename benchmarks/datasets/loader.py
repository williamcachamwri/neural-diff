"""Versioned benchmark suite loader and validator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.evaluators.base import EvalCase
from benchmarks.evaluators.registry import ALL_CATEGORIES
from core.exceptions import MalformedCheckpointError
from core.security import SecurityPolicy, read_validated_json, validate_safe_path

DEFAULT_SUITE_PATH = Path(__file__).resolve().parent / "default_suite_v1.json"


@dataclass(frozen=True)
class BenchmarkSuite:
    """Loaded and validated deterministic behavioral benchmark suite."""

    suite_id: str
    version: str
    description: str
    cases: list[EvalCase]

    def filter_categories(self, categories: list[str] | None) -> BenchmarkSuite:
        if not categories:
            return self
        wanted = set(categories)
        filtered = [c for c in self.cases if c.category in wanted]
        return BenchmarkSuite(
            suite_id=self.suite_id,
            version=self.version,
            description=self.description,
            cases=filtered,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "version": self.version,
            "description": self.description,
            "case_count": len(self.cases),
            "cases": [c.to_dict() for c in self.cases],
        }


def load_benchmark_suite(
    suite_path: str | Path | None = None,
    *,
    categories: list[str] | None = None,
    policy: SecurityPolicy | None = None,
) -> BenchmarkSuite:
    """Load and validate a deterministic JSON benchmark suite."""
    path = DEFAULT_SUITE_PATH if suite_path is None else validate_safe_path(suite_path, policy=policy)
    raw = read_validated_json(path)

    suite_id = str(raw.get("suite_id", path.stem))
    version = str(raw.get("version", "0.0.0"))
    description = str(raw.get("description", ""))
    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise MalformedCheckpointError(f"Benchmark suite '{path.name}' contains no valid `cases` list.")

    seen_ids: set[str] = set()
    cases: list[EvalCase] = []
    for idx, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise MalformedCheckpointError(f"Invalid case entry at index {idx} in '{path.name}'.")
        cid = str(item.get("case_id", "")).strip()
        cat = str(item.get("category", "")).strip()
        prompt = str(item.get("prompt", "")).strip()
        if not cid or not prompt or cat not in ALL_CATEGORIES:
            raise MalformedCheckpointError(
                f"Invalid case fields (case_id='{cid}', category='{cat}') at index {idx} in '{path.name}'."
            )
        if cid in seen_ids:
            raise MalformedCheckpointError(f"Duplicate case_id '{cid}' in '{path.name}'.")
        seen_ids.add(cid)
        expected = item.get("expected", {})
        if not isinstance(expected, dict):
            raise MalformedCheckpointError(f"`expected` must be a JSON object for case '{cid}'.")
        cases.append(
            EvalCase(
                case_id=cid,
                category=cat,
                prompt=prompt,
                description=str(item.get("description", "")),
                expected=expected,
            )
        )

    suite = BenchmarkSuite(
        suite_id=suite_id,
        version=version,
        description=description,
        cases=cases,
    )
    return suite.filter_categories(categories)
