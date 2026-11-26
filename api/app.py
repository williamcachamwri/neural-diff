"""FastAPI application factory and REST endpoints for NEURAL DIFF (Phase 11)."""

from __future__ import annotations

import datetime as dt
import os
import threading
import uuid
from pathlib import Path
from typing import Any

import torch
import transformers
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

import safetensors
from api.routes.diagnostics import build_diagnostics_router
from api.schemas import (
    CreateBaselineRequest,
    InspectModelRequest,
    RunComparisonRequest,
    RuntimeSettingsUpdate,
    ValidateCompatibilityRequest,
)
from api.security_middleware import DEFAULT_MAX_API_BODY_BYTES, SecurityHardeningMiddleware
from api.store import ExperimentStore
from benchmarks.datasets.loader import load_benchmark_suite
from core import __version__
from core.exceptions import ModelNotFoundError, NeuralDiffError, OperationCancelledError
from core.loaders.model_loader import ModelLoader
from core.orchestrator import ComparisonOrchestrator, ComparisonRequestConfig
from core.regression_engine.engine import RegressionBaseline, RegressionThresholds
from core.report_engine.exporter import ReportEngine
from core.resource_manager import CancellationToken, ResourceManager
from core.security import SecurityPolicy
from core.structural_analysis.compatibility import validate_compatibility
from core.tensor_engine.metrics import DiffTolerances

REPO_ROOT = Path(__file__).resolve().parent.parent

VERIFIED_HUB_PRESETS: list[dict[str, Any]] = [
    {
        "id": "Qwen/Qwen2.5-0.5B",
        "label": "Qwen2.5-0.5B (Base)",
        "license": "apache-2.0",
        "gated": False,
        "approx_size_mb": 990,
        "recommended_pair": "Qwen/Qwen2.5-0.5B-Instruct",
    },
    {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "label": "Qwen2.5-0.5B-Instruct (Aligned)",
        "license": "apache-2.0",
        "gated": False,
        "approx_size_mb": 990,
        "recommended_pair": "Qwen/Qwen2.5-0.5B",
    },
    {
        "id": "HuggingFaceTB/SmolLM2-135M",
        "label": "SmolLM2-135M (Base)",
        "license": "apache-2.0",
        "gated": False,
        "approx_size_mb": 270,
        "recommended_pair": "HuggingFaceTB/SmolLM2-135M-Instruct",
    },
    {
        "id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "label": "SmolLM2-135M-Instruct (Aligned)",
        "license": "apache-2.0",
        "gated": False,
        "approx_size_mb": 270,
        "recommended_pair": "HuggingFaceTB/SmolLM2-135M",
    },
    {
        "id": "google/gemma-3-270m",
        "label": "Gemma 3 270M (Gated — Requires HF_TOKEN)",
        "license": "gemma",
        "gated": True,
        "approx_size_mb": 540,
        "recommended_pair": None,
    },
]


def create_app(
    *,
    db_path: str | Path | None = None,
    policy: SecurityPolicy | None = None,
    max_body_bytes: int = DEFAULT_MAX_API_BODY_BYTES,
) -> FastAPI:
    store = ExperimentStore(db_path=db_path)
    base_policy = policy or SecurityPolicy()
    active_tokens: dict[str, CancellationToken] = {}
    tokens_lock = threading.Lock()

    app = FastAPI(
        title="NEURAL DIFF API",
        description="Git for AI Brains — Structural, Neural & Behavioral Intelligence Comparison",
        version=__version__,
    )
    app.state.store = store
    app.state.policy = base_policy

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )
    app.add_middleware(SecurityHardeningMiddleware, max_body_bytes=max_body_bytes)

    @app.exception_handler(NeuralDiffError)
    async def _handle_domain_error(_request, exc: NeuralDiffError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    def _effective_policy() -> SecurityPolicy:
        settings = store.get_settings()
        roots = list(base_policy.allowed_roots)
        db_parent = store.db_path.parent.resolve()
        if db_parent not in roots:
            roots.append(db_parent)
        return SecurityPolicy(
            allowed_roots=roots,
            allow_pytorch_weights_only=bool(settings.get("allow_pytorch_weights_only", True)),
            reject_symlinks_outside_roots=base_policy.reject_symlinks_outside_roots,
            max_header_bytes=base_policy.max_header_bytes,
            max_json_bytes=base_policy.max_json_bytes,
            max_tensor_elements=base_policy.max_tensor_elements,
        )

    @app.get("/api/health")
    def get_health() -> dict[str, Any]:
        settings = store.get_settings()
        rm = ResourceManager(preferred_device=str(settings.get("preferred_device", "auto")))
        snap = rm.snapshot()
        exps = store.list_experiments(limit=200)
        baselines = store.list_baselines()
        return {
            "status": "ok",
            "version": __version__,
            "device": rm.resolve_device(),
            "device_reason": rm.device_reason,
            "hardware": snap.to_dict(),
            "software": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "safetensors": safetensors.__version__,
            },
            "counts": {
                "experiments": len(exps),
                "completed_experiments": sum(1 for e in exps if e["status"] == "completed"),
                "baselines": len(baselines),
                "active_jobs": len(active_tokens),
            },
        }

    @app.get("/api/models/discover")
    def discover_models() -> dict[str, Any]:
        local_models: list[dict[str, Any]] = []
        scan_dirs = [
            REPO_ROOT / "data",
            REPO_ROOT / "examples",
            REPO_ROOT / "tests" / "fixtures",
        ]
        for sdir in scan_dirs:
            if not sdir.is_dir():
                continue
            for st_file in sorted(sdir.rglob("*.safetensors")):
                # Group HF directories with config.json by directory; otherwise list standalone file
                if (st_file.parent / "config.json").is_file() and st_file.name == "model.safetensors":
                    rel = st_file.parent.relative_to(REPO_ROOT).as_posix()
                    local_models.append(
                        {
                            "source": rel,
                            "label": st_file.parent.name,
                            "kind": "local_hf_dir",
                            "synthetic_fixture": True,
                        }
                    )
                elif not (st_file.parent / "config.json").is_file():
                    rel = st_file.relative_to(REPO_ROOT).as_posix()
                    local_models.append(
                        {
                            "source": rel,
                            "label": st_file.name,
                            "kind": "local_safetensors",
                            "synthetic_fixture": True,
                        }
                    )

        # Check cached HF Hub models
        hf_cache_candidates = [
            Path(os.environ["HF_HOME"]) / "hub" if "HF_HOME" in os.environ else None,
            REPO_ROOT / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "huggingface" / "hub",
        ]
        cached_ids: set[str] = set()
        for cdir in hf_cache_candidates:
            if cdir and cdir.is_dir():
                for entry in cdir.glob("models--*"):
                    parts = entry.name.split("--")
                    if len(parts) >= 3:
                        cached_ids.add(f"{parts[1]}/{'--'.join(parts[2:])}")

        hub_presets = [
            {**preset, "cached_locally": preset["id"] in cached_ids}
            for preset in VERIFIED_HUB_PRESETS
        ]
        return {"local_models": local_models, "hub_presets": hub_presets}

    @app.post("/api/models/inspect")
    def inspect_model(req: InspectModelRequest) -> dict[str, Any]:
        loader = ModelLoader(policy=_effective_policy())
        with loader.load(req.source, local_files_only=req.local_files_only) as handle:
            return {
                "metadata": handle.metadata.to_dict(),
                "sample_tensors": [d.to_dict() for d in list(handle.tensors.values())[:25]],
            }

    @app.post("/api/models/validate")
    def validate_models(req: ValidateCompatibilityRequest) -> dict[str, Any]:
        loader = ModelLoader(policy=_effective_policy())
        with (
            loader.load(req.model_a, local_files_only=req.local_files_only) as ha,
            loader.load(req.model_b, local_files_only=req.local_files_only) as hb,
        ):
            comp = validate_compatibility(ha, hb)
            return {
                "model_a": ha.metadata.to_dict(),
                "model_b": hb.metadata.to_dict(),
                "compatibility": comp.to_dict(),
            }

    @app.get("/api/benchmarks/suites")
    def get_benchmark_suites() -> dict[str, Any]:
        suite = load_benchmark_suite(policy=_effective_policy())
        by_cat: dict[str, int] = {}
        for c in suite.cases:
            by_cat[c.category] = by_cat.get(c.category, 0) + 1
        return {
            "suite_id": suite.suite_id,
            "version": suite.version,
            "description": suite.description,
            "total_cases": len(suite.cases),
            "categories": by_cat,
        }

    def _execute_comparison_job(
        comp_id: str,
        req: RunComparisonRequest,
        token: CancellationToken,
    ) -> dict[str, Any]:
        pol = _effective_policy()
        orch = ComparisonOrchestrator(policy=pol)

        # Resolve optional persisted baseline into temporary or direct baseline check
        baseline_file: Path | None = None
        if req.baseline_id:
            b_dict = store.get_baseline(req.baseline_id)
            if b_dict is None:
                raise ModelNotFoundError(f"Baseline '{req.baseline_id}' not found.")
            tmp_dir = store.db_path.parent / "baselines"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            baseline_file = tmp_dir / f"{req.baseline_id}.json"
            RegressionBaseline(**b_dict).save(baseline_file)

        past_runs = [
            {
                "comparison_id": e["comparison_id"],
                "created_at_utc": e["created_at_utc"],
                "verdict": e["verdict"],
                "summary": e["summary"],
            }
            for e in store.list_experiments(limit=10)
            if e["status"] == "completed" and e["comparison_id"] != comp_id
        ]

        inferred_synth = any(
            marker in req.model_a.lower() or marker in req.model_b.lower()
            for marker in ("synth", "fixture", "demo_fixtures")
        )
        effective_exp_type = (
            "SYNTHETIC FIXTURE EXPERIMENT" if inferred_synth else req.experiment_type
        )

        cfg = ComparisonRequestConfig(
            model_a=req.model_a,
            model_b=req.model_b,
            modules=req.modules,
            experiment_type=effective_exp_type,
            tolerances=DiffTolerances(
                atol=req.tolerances.atol,
                rtol=req.tolerances.rtol,
                zero_threshold=req.tolerances.zero_threshold,
            ),
            thresholds=RegressionThresholds(**req.thresholds.model_dump()),
            preferred_device=req.preferred_device,
            chunk_elements=req.chunk_elements,
            activation_prompts=req.activation_prompts,
            behavioral_categories=req.behavioral_categories,
            behavioral_max_cases=req.behavioral_max_cases,
            max_new_tokens=req.max_new_tokens,
            seed=req.seed,
            prompt_mode=req.prompt_mode,
            local_files_only=req.local_files_only,
            baseline_path=str(baseline_file) if baseline_file else None,
        )

        def _progress(msg: str, done: int, total: int) -> None:
            store.update_progress(comp_id, msg, done, total)

        bundle = orch.run(
            cfg,
            comparison_id=comp_id,
            cancellation_token=token,
            on_progress=_progress,
            historical_runs=past_runs,
        )

        exporter = ReportEngine()
        bundle_json = exporter.to_json(bundle, include_full_tensor_records=True)
        md_report = exporter.to_markdown(bundle)
        html_report = exporter.to_html(bundle)

        summary_card = {
            "global_changed_pct": bundle.tensor_diff.global_changed_pct if bundle.tensor_diff else None,
            "global_mean_cosine": bundle.tensor_diff.global_mean_cosine if bundle.tensor_diff else None,
            "behavioral_pass_rate_a": (
                bundle.behavioral.overall_statistics.pass_rate_a if bundle.behavioral else None
            ),
            "behavioral_pass_rate_b": (
                bundle.behavioral.overall_statistics.pass_rate_b if bundle.behavioral else None
            ),
            "behavioral_score_delta": (
                bundle.behavioral.overall_statistics.mean_score_delta if bundle.behavioral else None
            ),
            "pooled_final_cka": (
                bundle.activation.cross_prompt_pooled_cka_final_layer if bundle.activation else None
            ),
            "duration_seconds": bundle.execution_telemetry.get("duration_seconds"),
        }
        verdict = bundle.regression.verdict if bundle.regression else None

        store.save_completed(
            comparison_id=comp_id,
            model_a_id=bundle.model_a.model_id,
            model_b_id=bundle.model_b.model_id,
            verdict=verdict,
            summary_dict=summary_card,
            bundle_json=bundle_json,
            markdown_report=md_report,
            html_report=html_report,
        )
        result = store.get_experiment(comp_id)
        assert result is not None
        return result

    @app.post("/api/comparisons/run")
    def run_comparison(req: RunComparisonRequest) -> dict[str, Any]:
        comp_id = f"ndiff-{uuid.uuid4().hex[:10]}"
        now_utc = dt.datetime.now(dt.UTC).isoformat()
        token = CancellationToken()

        inferred_synth = any(
            marker in req.model_a.lower() or marker in req.model_b.lower()
            for marker in ("synth", "fixture", "demo_fixtures")
        )
        effective_exp_type = (
            "SYNTHETIC FIXTURE EXPERIMENT" if inferred_synth else req.experiment_type
        )

        store.create_job(
            comparison_id=comp_id,
            created_at_utc=now_utc,
            experiment_type=effective_exp_type,
            model_a_id=req.model_a,
            model_b_id=req.model_b,
            selected_modules=req.modules,
            total_steps=len(req.modules) + 1,
        )

        if not req.async_job:
            with tokens_lock:
                active_tokens[comp_id] = token
            try:
                return _execute_comparison_job(comp_id, req, token)
            except OperationCancelledError as exc:
                store.save_failed(comp_id, exc.to_dict(), cancelled=True)
                raise
            except NeuralDiffError as exc:
                store.save_failed(comp_id, exc.to_dict(), cancelled=False)
                raise
            finally:
                with tokens_lock:
                    active_tokens.pop(comp_id, None)

        def _bg_worker() -> None:
            try:
                _execute_comparison_job(comp_id, req, token)
            except OperationCancelledError as exc:
                store.save_failed(comp_id, exc.to_dict(), cancelled=True)
            except NeuralDiffError as exc:
                store.save_failed(comp_id, exc.to_dict(), cancelled=False)
            except Exception as exc:
                store.save_failed(
                    comp_id,
                    {"error": "INTERNAL_EXECUTION_ERROR", "message": str(exc), "http_status": 500},
                )
            finally:
                with tokens_lock:
                    active_tokens.pop(comp_id, None)

        with tokens_lock:
            active_tokens[comp_id] = token
        worker = threading.Thread(target=_bg_worker, name=f"job-{comp_id}", daemon=True)
        worker.start()
        job = store.get_experiment(comp_id)
        assert job is not None
        return job

    @app.post("/api/comparisons/{comparison_id}/cancel")
    def cancel_comparison(comparison_id: str) -> dict[str, Any]:
        with tokens_lock:
            tok = active_tokens.get(comparison_id)
        if tok is None:
            exp = store.get_experiment(comparison_id)
            if exp is None:
                raise ModelNotFoundError(f"Comparison '{comparison_id}' not found.")
            return {"comparison_id": comparison_id, "cancelled": False, "status": exp["status"]}
        tok.cancel("Cancelled via REST API.")
        return {"comparison_id": comparison_id, "cancelled": True, "status": "cancelling"}

    @app.get("/api/comparisons")
    def list_comparisons(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
        return {"experiments": store.list_experiments(limit=limit)}

    @app.get("/api/comparisons/{comparison_id}")
    def get_comparison(comparison_id: str) -> dict[str, Any]:
        exp = store.get_experiment(comparison_id)
        if exp is None:
            raise ModelNotFoundError(f"Comparison '{comparison_id}' not found.")
        return exp

    @app.get("/api/comparisons/{comparison_id}/tensors")
    def query_comparison_tensors(
        comparison_id: str,
        search: str = Query(default="", max_length=120),
        role: str = Query(default="", max_length=40),
        status: str = Query(default="", max_length=40),
        sort_by: str = Query(default="relative_l2", pattern=r"^(relative_l2|max_abs_diff|cosine_similarity|changed_pct|name)$"),
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        exp = store.get_experiment(comparison_id)
        if exp is None or exp.get("bundle") is None:
            raise ModelNotFoundError(f"Completed comparison '{comparison_id}' not found.")
        td = exp["bundle"].get("tensor_diff")
        if td is None:
            return {"total": 0, "records": []}
        recs: list[dict[str, Any]] = list(td.get("records", []))
        if search:
            q = search.lower()
            recs = [r for r in recs if q in r["name"].lower()]
        if role:
            recs = [r for r in recs if r["canonical_role"] == role]
        if status:
            recs = [r for r in recs if r["status"] == status]

        if sort_by == "name":
            recs.sort(key=lambda r: str(r["name"]))
        elif sort_by == "cosine_similarity":
            def _cos_key(rec: dict[str, Any]) -> float:
                m = rec.get("metrics")
                val = m.get("cosine_similarity") if isinstance(m, dict) else None
                return float(val) if isinstance(val, (int, float)) else 2.0

            recs.sort(key=_cos_key)
        else:
            def _metric_key(rec: dict[str, Any]) -> float:
                m = rec.get("metrics")
                val = m.get(sort_by) if isinstance(m, dict) else None
                return float(val) if isinstance(val, (int, float)) else -1.0

            recs.sort(key=_metric_key, reverse=True)

        total = len(recs)
        return {"total": total, "offset": offset, "limit": limit, "records": recs[offset : offset + limit]}

    @app.get("/api/comparisons/{comparison_id}/export/{fmt}")
    def export_comparison_report(comparison_id: str, fmt: str) -> Response:
        if fmt not in ("json", "markdown", "html"):
            raise HTTPException(status_code=400, detail="Export format must be json, markdown, or html.")
        exported = store.get_export(comparison_id, fmt)
        if exported is None:
            raise ModelNotFoundError(f"Export '{fmt}' for comparison '{comparison_id}' not found.")
        content, media_type = exported
        ext = "md" if fmt == "markdown" else fmt
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{comparison_id}.{ext}"'},
        )

    @app.get("/api/baselines")
    def list_baselines() -> dict[str, Any]:
        return {"baselines": store.list_baselines()}

    @app.post("/api/baselines/from-comparison")
    def create_baseline_from_comparison(req: CreateBaselineRequest) -> dict[str, Any]:
        exp = store.get_experiment(req.comparison_id)
        if exp is None or exp.get("bundle") is None:
            raise ModelNotFoundError(f"Completed comparison '{req.comparison_id}' not found.")
        bundle = exp["bundle"]
        beh = bundle.get("behavioral")
        if beh is None:
            raise HTTPException(
                status_code=422,
                detail="Cannot create a behavioral baseline from a run without behavioral results.",
            )
        use_b = req.side == "b"
        stats = beh["overall_statistics"]
        meta = bundle["model_b"] if use_b else bundle["model_a"]
        cat_scores = {
            c["category"]: (c["mean_score_b"] if use_b else c["mean_score_a"])
            for c in beh["category_summaries"]
        }
        cat_pass = {
            c["category"]: (c["pass_rate_b"] if use_b else c["pass_rate_a"])
            for c in beh["category_summaries"]
        }
        baseline = RegressionBaseline(
            baseline_id=req.baseline_id,
            created_at_utc=dt.datetime.now(dt.UTC).isoformat(),
            model_id=meta["model_id"],
            revision=meta.get("revision"),
            suite_id=beh["suite_id"],
            suite_version=beh["suite_version"],
            overall_pass_rate=stats["pass_rate_b"] if use_b else stats["pass_rate_a"],
            overall_mean_score=stats["mean_score_b"] if use_b else stats["mean_score_a"],
            category_scores=cat_scores,
            category_pass_rates=cat_pass,
            tensor_count=meta["tensor_count"],
            total_parameters=meta["total_parameters"],
        )
        store.save_baseline(baseline.to_dict())
        return baseline.to_dict()

    @app.delete("/api/baselines/{baseline_id}")
    def delete_baseline(baseline_id: str) -> dict[str, Any]:
        deleted = store.delete_baseline(baseline_id)
        if not deleted:
            raise ModelNotFoundError(f"Baseline '{baseline_id}' not found.")
        return {"deleted": True, "baseline_id": baseline_id}

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        return store.get_settings()

    @app.put("/api/settings")
    def put_settings(req: RuntimeSettingsUpdate) -> dict[str, Any]:
        return store.update_settings(req.model_dump())

    app.include_router(
        build_diagnostics_router(store=store, policy_provider=_effective_policy)
    )

    return app


app = create_app()
