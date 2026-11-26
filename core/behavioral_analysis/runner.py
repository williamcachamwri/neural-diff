"""Behavioral Diff Engine (Module F) — paired inference and deterministic suite evaluation."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from benchmarks.datasets.loader import BenchmarkSuite, load_benchmark_suite
from benchmarks.evaluators.base import CaseEvalResult, EvalCase
from benchmarks.evaluators.registry import evaluate_case
from core.exceptions import ModelCompatibilityError
from core.loaders.model_loader import ModelLoader
from core.resource_manager import CancellationToken, ResourceManager
from core.security import SecurityPolicy
from core.statistics.significance import (
    PairedStatisticalSummary,
    compute_paired_significance,
)
from core.types import EpistemicStatus


@dataclass
class PairedCaseComparison:
    """Actual paired outputs and deterministic evaluation scores for one benchmark case."""

    case: EvalCase
    formatted_input_a: str
    formatted_input_b: str
    output_a: str
    output_b: str
    latency_ms_a: float
    latency_ms_b: float
    eval_a: CaseEvalResult
    eval_b: CaseEvalResult
    score_delta: float
    status_change: str  # "improved" | "regressed" | "both_passed" | "both_failed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "formatted_input_a": self.formatted_input_a,
            "formatted_input_b": self.formatted_input_b,
            "output_a": self.output_a,
            "output_b": self.output_b,
            "latency_ms_a": self.latency_ms_a,
            "latency_ms_b": self.latency_ms_b,
            "eval_a": self.eval_a.to_dict(),
            "eval_b": self.eval_b.to_dict(),
            "score_delta": self.score_delta,
            "status_change": self.status_change,
        }


@dataclass
class CategoryBehavioralSummary:
    """Aggregated deterministic scores for one behavioral evaluation category."""

    category: str
    case_count: int
    pass_rate_a: float
    pass_rate_b: float
    pass_rate_delta: float
    mean_score_a: float
    mean_score_b: float
    mean_score_delta: float
    improved_cases: int
    regressed_cases: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "case_count": self.case_count,
            "pass_rate_a": self.pass_rate_a,
            "pass_rate_b": self.pass_rate_b,
            "pass_rate_delta": self.pass_rate_delta,
            "mean_score_a": self.mean_score_a,
            "mean_score_b": self.mean_score_b,
            "mean_score_delta": self.mean_score_delta,
            "improved_cases": self.improved_cases,
            "regressed_cases": self.regressed_cases,
        }


@dataclass
class BehavioralDiffReport:
    """Complete Module F Behavioral Diff Report with anti-cherry-picking paired statistics."""

    epistemic_status: EpistemicStatus
    evaluator_methodology: str
    suite_id: str
    suite_version: str
    model_a_id: str
    model_b_id: str
    revision_a: str | None
    revision_b: str | None
    inference_parameters: dict[str, Any]
    hardware_telemetry: dict[str, Any]
    overall_statistics: PairedStatisticalSummary
    category_summaries: list[CategoryBehavioralSummary]
    cases: list[PairedCaseComparison]

    def to_dict(self) -> dict[str, Any]:
        return {
            "epistemic_status": self.epistemic_status.value,
            "evaluator_methodology": self.evaluator_methodology,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "model_a_id": self.model_a_id,
            "model_b_id": self.model_b_id,
            "revision_a": self.revision_a,
            "revision_b": self.revision_b,
            "inference_parameters": self.inference_parameters,
            "hardware_telemetry": self.hardware_telemetry,
            "overall_statistics": self.overall_statistics.to_dict(),
            "category_summaries": [c.to_dict() for c in self.category_summaries],
            "cases": [c.to_dict() for c in self.cases],
        }


class BehavioralDiffEngine:
    """Executes deterministic paired behavioral benchmarks across two checkpoints or generators."""

    def __init__(
        self,
        *,
        resource_manager: ResourceManager | None = None,
        policy: SecurityPolicy | None = None,
        max_new_tokens: int = 128,
        seed: int = 42,
        prompt_mode: str = "chat_if_available",
        dtype: str = "bfloat16",
    ) -> None:
        self.resource_manager = resource_manager or ResourceManager()
        self.policy = policy or SecurityPolicy()
        self.max_new_tokens = max(8, int(max_new_tokens))
        self.seed = seed
        self.prompt_mode = prompt_mode
        self.dtype = dtype

    def compare_generators(
        self,
        gen_a: Callable[[EvalCase], tuple[str, str, float]],
        gen_b: Callable[[EvalCase], tuple[str, str, float]],
        *,
        suite: BenchmarkSuite | None = None,
        model_a_id: str = "generator_a",
        model_b_id: str = "generator_b",
        cancellation_token: CancellationToken | None = None,
    ) -> BehavioralDiffReport:
        """Compare two `(formatted_input, output_text, latency_ms)` callables on `suite`."""
        active_suite = suite or load_benchmark_suite(policy=self.policy)
        outputs_a: list[tuple[str, str, float]] = []
        outputs_b: list[tuple[str, str, float]] = []

        with self.resource_manager.managed_execution(cancellation_token) as telemetry:
            for case in active_suite.cases:
                if cancellation_token is not None:
                    cancellation_token.check()
                outputs_a.append(gen_a(case))
            for case in active_suite.cases:
                if cancellation_token is not None:
                    cancellation_token.check()
                outputs_b.append(gen_b(case))

        return self._score_and_aggregate(
            suite=active_suite,
            outputs_a=outputs_a,
            outputs_b=outputs_b,
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            revision_a=None,
            revision_b=None,
            inference_params={"mode": "callable_generator", "seed": self.seed},
            telemetry=telemetry,
        )

    def compare_pretrained(
        self,
        source_a: str | Path,
        source_b: str | Path,
        *,
        suite: BenchmarkSuite | None = None,
        categories: list[str] | None = None,
        max_cases: int | None = None,
        local_files_only: bool = False,
        cancellation_token: CancellationToken | None = None,
    ) -> BehavioralDiffReport:
        """Sequentially run identical `BenchmarkSuite` cases on Model A and Model B (`trust_remote_code=False`)."""
        active_suite = suite or load_benchmark_suite(categories=categories, policy=self.policy)
        if max_cases is not None and max_cases > 0:
            active_suite = BenchmarkSuite(
                suite_id=active_suite.suite_id,
                version=active_suite.version,
                description=active_suite.description,
                cases=active_suite.cases[:max_cases],
            )

        loader = ModelLoader(policy=self.policy)
        device = self.resource_manager.resolve_device()

        with loader.load(source_a, local_files_only=local_files_only) as ha:
            meta_a = ha.metadata
        with loader.load(source_b, local_files_only=local_files_only) as hb:
            meta_b = hb.metadata

        dir_a = Path(meta_a.resolved_path) if Path(meta_a.resolved_path).is_dir() else Path(meta_a.resolved_path).parent
        dir_b = Path(meta_b.resolved_path) if Path(meta_b.resolved_path).is_dir() else Path(meta_b.resolved_path).parent

        if not (dir_a / "config.json").is_file() or not (dir_b / "config.json").is_file():
            raise ModelCompatibilityError(
                "Behavioral generation requires Hugging Face `config.json` and tokenizer files in both models."
            )

        with self.resource_manager.managed_execution(cancellation_token) as telemetry:
            outputs_a = self._generate_suite_outputs(
                dir_a, active_suite.cases, device=device, cancellation_token=cancellation_token
            )
            self.resource_manager.cleanup()
            outputs_b = self._generate_suite_outputs(
                dir_b, active_suite.cases, device=device, cancellation_token=cancellation_token
            )
            self.resource_manager.cleanup()

        return self._score_and_aggregate(
            suite=active_suite,
            outputs_a=outputs_a,
            outputs_b=outputs_b,
            model_a_id=meta_a.model_id,
            model_b_id=meta_b.model_id,
            revision_a=meta_a.revision,
            revision_b=meta_b.revision,
            inference_params={
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": self.max_new_tokens,
                "seed": self.seed,
                "prompt_mode": self.prompt_mode,
                "device": device,
                "dtype": self.dtype,
                "trust_remote_code": False,
            },
            telemetry=telemetry,
        )

    def _generate_suite_outputs(
        self,
        model_dir: Path,
        cases: list[EvalCase],
        *,
        device: str,
        cancellation_token: CancellationToken | None,
    ) -> list[tuple[str, str, float]]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=False,
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        torch_dtype = (
            torch.bfloat16
            if (device.startswith("cuda") and self.dtype == "bfloat16")
            else torch.float32
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            trust_remote_code=False,
            local_files_only=True,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        torch.nn.Module.to(model, device)
        model.eval()

        has_chat_template = bool(getattr(tokenizer, "chat_template", None))
        results: list[tuple[str, str, float]] = []

        try:
            with torch.no_grad():
                for case in cases:
                    if cancellation_token is not None:
                        cancellation_token.check()

                    used_qa_framing = False
                    if self.prompt_mode == "chat_if_available" and has_chat_template:
                        formatted = str(
                            tokenizer.apply_chat_template(
                                [{"role": "user", "content": case.prompt}],
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                        )
                    elif self.prompt_mode == "identical_raw":
                        formatted = case.prompt
                    else:
                        formatted = f"Question: {case.prompt}\nAnswer:"
                        used_qa_framing = True

                    enc = tokenizer(formatted, return_tensors="pt")
                    input_ids = enc["input_ids"].to(device)
                    attention_mask = enc["attention_mask"].to(device)
                    if int(input_ids.shape[1]) == 0:
                        fb_id = int(tokenizer.bos_token_id or tokenizer.eos_token_id or 0)
                        input_ids = torch.tensor([[fb_id]], dtype=torch.long, device=device)
                        attention_mask = torch.ones_like(input_ids, device=device)
                    prompt_len = int(input_ids.shape[1])

                    torch.manual_seed(self.seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(self.seed)

                    t0 = time.perf_counter()
                    gen_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    new_tokens = gen_ids[0, prompt_len:]
                    decoded = str(tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()
                    if used_qa_framing:
                        # Trim trailing multi-turn self-continuation on base pretrained models
                        for stop_tok in ("\n\nQuestion:", "\nQuestion:"):
                            if stop_tok in decoded:
                                decoded = decoded.split(stop_tok, 1)[0].strip()

                    results.append((formatted, decoded, round(elapsed_ms, 3)))
        finally:
            del model
            self.resource_manager.cleanup()

        return results

    @staticmethod
    def _score_and_aggregate(
        *,
        suite: BenchmarkSuite,
        outputs_a: list[tuple[str, str, float]],
        outputs_b: list[tuple[str, str, float]],
        model_a_id: str,
        model_b_id: str,
        revision_a: str | None,
        revision_b: str | None,
        inference_params: dict[str, Any],
        telemetry: dict[str, Any],
    ) -> BehavioralDiffReport:
        paired_cases: list[PairedCaseComparison] = []
        scores_a: list[float] = []
        scores_b: list[float] = []
        passed_a: list[bool] = []
        passed_b: list[bool] = []

        by_cat: dict[str, list[PairedCaseComparison]] = {}

        for case, (fmt_a, out_a, lat_a), (fmt_b, out_b, lat_b) in zip(
            suite.cases, outputs_a, outputs_b, strict=True
        ):
            ea = evaluate_case(case, out_a)
            eb = evaluate_case(case, out_b)
            delta = round(eb.score - ea.score, 6)

            if (not ea.passed) and eb.passed:
                status_change = "improved"
            elif ea.passed and (not eb.passed):
                status_change = "regressed"
            elif ea.passed and eb.passed:
                status_change = "both_passed"
            else:
                status_change = "both_failed"

            rec = PairedCaseComparison(
                case=case,
                formatted_input_a=fmt_a,
                formatted_input_b=fmt_b,
                output_a=out_a,
                output_b=out_b,
                latency_ms_a=lat_a,
                latency_ms_b=lat_b,
                eval_a=ea,
                eval_b=eb,
                score_delta=delta,
                status_change=status_change,
            )
            paired_cases.append(rec)
            by_cat.setdefault(case.category, []).append(rec)

            scores_a.append(ea.score)
            scores_b.append(eb.score)
            passed_a.append(ea.passed)
            passed_b.append(eb.passed)

        overall_stats = compute_paired_significance(scores_a, scores_b, passed_a, passed_b)

        cat_summaries: list[CategoryBehavioralSummary] = []
        for cat_name, crecs in by_cat.items():
            n = len(crecs)
            pra = sum(1 for r in crecs if r.eval_a.passed) / n
            prb = sum(1 for r in crecs if r.eval_b.passed) / n
            msa = sum(r.eval_a.score for r in crecs) / n
            msb = sum(r.eval_b.score for r in crecs) / n
            imp = sum(1 for r in crecs if r.status_change == "improved")
            reg = sum(1 for r in crecs if r.status_change == "regressed")
            cat_summaries.append(
                CategoryBehavioralSummary(
                    category=cat_name,
                    case_count=n,
                    pass_rate_a=round(pra, 6),
                    pass_rate_b=round(prb, 6),
                    pass_rate_delta=round(prb - pra, 6),
                    mean_score_a=round(msa, 6),
                    mean_score_b=round(msb, 6),
                    mean_score_delta=round(msb - msa, 6),
                    improved_cases=imp,
                    regressed_cases=reg,
                )
            )

        return BehavioralDiffReport(
            epistemic_status=EpistemicStatus.OBSERVED,
            evaluator_methodology=(
                "100% deterministic programmatic evaluators (Unicode NFC diacritic analysis, Python AST "
                "syntax/signature parsing, strict JSON schema parsing, numerical answer extraction, "
                "constraint verification, refusal calibration, and contradiction-checked QA). "
                "Zero external LLM-as-a-judge calls."
            ),
            suite_id=suite.suite_id,
            suite_version=suite.version,
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            revision_a=revision_a,
            revision_b=revision_b,
            inference_parameters=inference_params,
            hardware_telemetry=telemetry,
            overall_statistics=overall_stats,
            category_summaries=cat_summaries,
            cases=paired_cases,
        )
