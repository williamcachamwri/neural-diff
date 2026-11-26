"""Code preservation and Python AST syntactic validity evaluator."""

from __future__ import annotations

import ast
import difflib
import re

from benchmarks.evaluators.base import CaseEvalResult, DeterministicEvaluator, EvalCase
from core.types import EpistemicStatus

_FENCED_CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", flags=re.DOTALL | re.IGNORECASE)


def extract_code_candidate(text: str) -> tuple[str, bool]:
    """Extract fenced code block if present; otherwise return stripped text."""
    m = _FENCED_CODE_RE.search(text)
    if m:
        return m.group(1).strip(), True
    return text.strip(), False


class CodePreservationEvaluator(DeterministicEvaluator):
    """Evaluates syntactic AST parseability, identifier/signature retention, and diff preservation."""

    category = "code_preservation"

    def evaluate(self, case: EvalCase, output_text: str) -> CaseEvalResult:
        code, was_fenced = extract_code_candidate(output_text)
        # Support prompt-continuation style models that emit body lines after a prompt header
        prefix_header = str(case.expected.get("completion_prefix", ""))

        ast_valid = False
        parsed_tree: ast.Module | None = None
        for candidate in ([code, f"{prefix_header}\n{code}"] if prefix_header else [code]):
            if not candidate.strip():
                continue
            try:
                parsed_tree = ast.parse(candidate)
                code = candidate
                ast_valid = True
                break
            except SyntaxError:
                continue

        defined_funcs: dict[str, list[str]] = {}
        defined_classes: set[str] = set()
        if parsed_tree is not None:
            for node in ast.walk(parsed_tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defined_funcs[node.name] = [arg.arg for arg in node.args.args]
                elif isinstance(node, ast.ClassDef):
                    defined_classes.add(node.name)

        req_funcs: list[str] = [str(f) for f in case.expected.get("required_functions", [])]
        req_args: dict[str, list[str]] = dict(case.expected.get("required_signature_args", {}))
        req_tokens: list[str] = [str(t) for t in case.expected.get("required_code_substrings", [])]
        forbidden_tokens: list[str] = [str(t) for t in case.expected.get("forbidden_substrings", [])]
        reference_code: str = str(case.expected.get("reference_code", ""))
        min_sim = float(case.expected.get("min_similarity", 0.0))

        func_hits = sum(1 for fn in req_funcs if fn in defined_funcs)
        func_ratio = (func_hits / len(req_funcs)) if req_funcs else 1.0

        sig_checks: list[bool] = []
        for fn, expected_arg_list in req_args.items():
            sig_checks.append(defined_funcs.get(fn) == list(expected_arg_list))
        sig_ratio = (sum(1 for ok in sig_checks if ok) / len(sig_checks)) if sig_checks else 1.0

        tok_hits = sum(1 for tok in req_tokens if tok in code)
        tok_ratio = (tok_hits / len(req_tokens)) if req_tokens else 1.0
        forbidden_free = all(ft not in code for ft in forbidden_tokens)

        sim_ratio = (
            difflib.SequenceMatcher(None, reference_code.strip(), code.strip()).ratio()
            if reference_code
            else 1.0
        )
        sim_ok = sim_ratio >= min_sim

        raw_score = (
            0.40 * (1.0 if ast_valid else 0.0)
            + 0.20 * func_ratio
            + 0.15 * sig_ratio
            + 0.15 * tok_ratio
            + 0.10 * (1.0 if (forbidden_free and sim_ok) else 0.0)
        )
        if not ast_valid:
            raw_score = min(raw_score, 0.25)

        score = round(max(0.0, min(1.0, raw_score)), 6)
        passed = (
            ast_valid
            and func_ratio >= 1.0
            and sig_ratio >= 1.0
            and tok_ratio >= 1.0
            and forbidden_free
            and sim_ok
        )

        return CaseEvalResult(
            case_id=case.case_id,
            category=self.category,
            passed=passed,
            score=score,
            evaluator_type="deterministic",
            epistemic_status=EpistemicStatus.OBSERVED,
            submetrics={
                "ast_valid": ast_valid,
                "was_fenced": was_fenced,
                "defined_functions": sorted(defined_funcs.keys()),
                "defined_classes": sorted(defined_classes),
                "function_retention_ratio": round(func_ratio, 4),
                "signature_match_ratio": round(sig_ratio, 4),
                "required_construct_ratio": round(tok_ratio, 4),
                "forbidden_free": forbidden_free,
                "reference_similarity": round(sim_ratio, 4),
            },
            rationale=(
                f"AST valid={ast_valid}, functions={func_ratio:.0%}, signatures={sig_ratio:.0%}, "
                f"constructs={tok_ratio:.0%}, similarity={sim_ratio:.2f}."
            ),
        )
