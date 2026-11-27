"""FastAPI REST router for Neural Evolution Tracking (`/api/evolution`, Section 11)."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import Field

from api.schemas import StrictModel, TolerancesInput
from core.evolution.exporter import EvolutionReportExporter
from core.evolution.orchestrator import EvolutionAnalysisConfig, EvolutionOrchestrator
from core.evolution.registry import inspect_and_build_checkpoint_record
from core.evolution.store import EvolutionStore
from core.evolution.tensor_evolution import MultiGenTensorEvolutionEngine
from core.evolution.types import TransformationType
from core.exceptions import (
    EvolutionEntityNotFoundError,
    NeuralDiffError,
    OperationCancelledError,
)
from core.resource_manager import CancellationToken
from core.security import SecurityPolicy
from core.tensor_engine.metrics import DiffTolerances


class CreateEvolutionProjectRequest(StrictModel):
    name: str = Field(..., min_length=1, max_length=160)
    description: str = Field(default="", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=24)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateEvolutionProjectRequest(StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=2000)
    base_checkpoint_id: str | None = Field(default=None, max_length=80)
    tags: list[str] | None = Field(default=None, max_length=24)


class RegisterCheckpointRequest(StrictModel):
    source: str = Field(..., min_length=1, max_length=512)
    version_label: str | None = Field(default=None, max_length=120)
    model_name: str | None = Field(default=None, max_length=160)
    parent_checkpoint_ids: list[str] = Field(default_factory=list, max_length=16)
    merge_weights: dict[str, float] | None = None
    transformation_type: str | None = Field(default=None, max_length=80)
    transformation_metadata: dict[str, Any] = Field(default_factory=dict)
    training_metadata: dict[str, Any] | None = None
    quantization_metadata: dict[str, Any] | None = None
    experiment_notes: str = Field(default="", max_length=4000)
    local_files_only: bool = False


class UpdateCheckpointRequest(StrictModel):
    version_label: str | None = Field(default=None, min_length=1, max_length=120)
    model_name: str | None = Field(default=None, min_length=1, max_length=160)
    transformation_type: str | None = Field(default=None, max_length=80)
    transformation_metadata: dict[str, Any] | None = None
    experiment_notes: str | None = Field(default=None, max_length=4000)
    parent_checkpoint_ids: list[str] | None = Field(default=None, max_length=16)


class CreateRelationshipRequest(StrictModel):
    parent_checkpoint_id: str = Field(..., min_length=2, max_length=80)
    child_checkpoint_id: str = Field(..., min_length=2, max_length=80)
    transformation_type: str | None = Field(default=None, max_length=80)
    merge_weight: float | None = Field(default=None, ge=0.0, le=10.0)
    transformation_metadata: dict[str, Any] = Field(default_factory=dict)


class CompareEvolutionVersionsRequest(StrictModel):
    checkpoint_a_id: str = Field(..., min_length=2, max_length=80)
    checkpoint_b_id: str = Field(..., min_length=2, max_length=80)
    tolerances: TolerancesInput = Field(default_factory=TolerancesInput)
    top_k_tensors: int = Field(default=12, ge=1, le=100)
    local_files_only: bool = False


class RunEvolutionAnalysisRequest(StrictModel):
    include_tensor_evolution: bool = True
    include_layer_evolution: bool = True
    include_behavioral_evolution: bool = True
    include_representation_evolution: bool = True
    enable_causal_patching: bool = True
    preferred_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    tolerances: TolerancesInput = Field(default_factory=TolerancesInput)
    chunk_elements: int = Field(default=1_048_576, ge=4096, le=16_777_216)
    top_k_tensors: int = Field(default=10, ge=1, le=64)
    behavioral_categories: list[str] | None = Field(default=None, max_length=12)
    behavioral_max_cases: int | None = Field(default=None, ge=1, le=200)
    max_new_tokens: int = Field(default=64, ge=8, le=512)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    prompt_mode: Literal["chat_if_available", "identical_raw", "qa_framed"] = "chat_if_available"
    overall_pass_rate_drop_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    category_score_drop_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    local_files_only: bool = False
    allow_cache: bool = True
    force_rerun: bool = False
    async_job: bool = False


def build_evolution_router(
    *,
    evo_store: EvolutionStore,
    policy_provider: Callable[[], SecurityPolicy],
) -> APIRouter:
    router = APIRouter(prefix="/api/evolution", tags=["evolution"])
    active_tokens: dict[str, CancellationToken] = {}
    tokens_lock = threading.Lock()

    @router.get("/overview")
    def get_evolution_overview() -> dict[str, Any]:
        return {
            "summary": evo_store.global_analytics_summary(),
            "supported_transformations": [t.value for t in TransformationType],
            "schema_migrations": evo_store.list_migrations(),
            "active_jobs": len(active_tokens),
        }

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------
    @router.post("/projects")
    def create_project(req: CreateEvolutionProjectRequest) -> dict[str, Any]:
        proj = evo_store.create_project(
            name=req.name,
            description=req.description,
            tags=req.tags,
            metadata=req.metadata,
        )
        return proj.to_dict()

    @router.get("/projects")
    def list_projects(
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        total, items = evo_store.list_projects(limit=limit, offset=offset)
        return {"total": total, "limit": limit, "offset": offset, "projects": items}

    @router.get("/projects/{project_id}")
    def get_project_detail(project_id: str) -> dict[str, Any]:
        proj = evo_store.get_project(project_id)
        ckpts = evo_store.list_checkpoints(project_id)
        dag = evo_store.get_dag(project_id)
        _, runs = evo_store.list_runs(project_id, limit=10)
        latest_completed = next((r for r in runs if r["status"] == "completed"), None)
        latest_bundle = (
            evo_store.get_run(latest_completed["run_id"]) if latest_completed else None
        )
        return {
            "project": proj.to_dict(),
            "checkpoints": [c.to_dict() for c in ckpts],
            "graph": dag.to_graph_payload(proj.base_checkpoint_id),
            "runs": runs,
            "latest_analysis": latest_bundle,
        }

    @router.put("/projects/{project_id}")
    def update_project(project_id: str, req: UpdateEvolutionProjectRequest) -> dict[str, Any]:
        updated = evo_store.update_project(
            project_id,
            name=req.name,
            description=req.description,
            base_checkpoint_id=req.base_checkpoint_id,
            tags=req.tags,
        )
        return updated.to_dict()

    @router.delete("/projects/{project_id}")
    def delete_project(project_id: str) -> dict[str, Any]:
        deleted = evo_store.delete_project(project_id)
        if not deleted:
            raise EvolutionEntityNotFoundError(f"Evolution project '{project_id}' not found.")
        return {"deleted": True, "project_id": project_id}

    # ------------------------------------------------------------------
    # Checkpoints & Relationships
    # ------------------------------------------------------------------
    @router.post("/projects/{project_id}/checkpoints")
    def register_checkpoint(
        project_id: str,
        req: RegisterCheckpointRequest,
    ) -> dict[str, Any]:
        existing = evo_store.list_checkpoints(project_id)
        ckpt = inspect_and_build_checkpoint_record(
            project_id=project_id,
            source=req.source,
            version_label=req.version_label,
            model_name=req.model_name,
            parent_checkpoint_ids=req.parent_checkpoint_ids,
            transformation_type=req.transformation_type,
            transformation_metadata=req.transformation_metadata,
            training_metadata_override=req.training_metadata,
            quantization_metadata_override=req.quantization_metadata,
            experiment_notes=req.experiment_notes,
            existing_project_checkpoints=existing,
            policy=policy_provider(),
            local_files_only=req.local_files_only,
        )
        saved = evo_store.add_checkpoint(ckpt, merge_weights=req.merge_weights)
        return saved.to_dict()

    @router.get("/projects/{project_id}/checkpoints")
    def list_checkpoints(
        project_id: str,
        transformation_type: str | None = Query(default=None, max_length=80),
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        all_items = evo_store.list_checkpoints(
            project_id, transformation_type=transformation_type
        )
        sliced = all_items[offset : offset + limit]
        return {
            "total": len(all_items),
            "limit": limit,
            "offset": offset,
            "checkpoints": [c.to_dict() for c in sliced],
        }

    @router.get("/projects/{project_id}/checkpoints/{checkpoint_id}")
    def get_checkpoint(project_id: str, checkpoint_id: str) -> dict[str, Any]:
        return evo_store.get_checkpoint(project_id, checkpoint_id).to_dict()

    @router.patch("/projects/{project_id}/checkpoints/{checkpoint_id}")
    def update_checkpoint(
        project_id: str,
        checkpoint_id: str,
        req: UpdateCheckpointRequest,
    ) -> dict[str, Any]:
        updated = evo_store.update_checkpoint(
            project_id,
            checkpoint_id,
            version_label=req.version_label,
            model_name=req.model_name,
            transformation_type=req.transformation_type,
            transformation_metadata=req.transformation_metadata,
            experiment_notes=req.experiment_notes,
            parent_checkpoint_ids=req.parent_checkpoint_ids,
        )
        return updated.to_dict()

    @router.delete("/projects/{project_id}/checkpoints/{checkpoint_id}")
    def delete_checkpoint(project_id: str, checkpoint_id: str) -> dict[str, Any]:
        removed = evo_store.remove_checkpoint(project_id, checkpoint_id)
        return {"deleted": removed, "checkpoint_id": checkpoint_id}

    @router.post("/projects/{project_id}/relationships")
    def create_relationship(
        project_id: str,
        req: CreateRelationshipRequest,
    ) -> dict[str, Any]:
        rel = evo_store.add_relationship(
            project_id,
            parent_checkpoint_id=req.parent_checkpoint_id,
            child_checkpoint_id=req.child_checkpoint_id,
            transformation_type=req.transformation_type,
            merge_weight=req.merge_weight,
            transformation_metadata=req.transformation_metadata,
        )
        return rel.to_dict()

    @router.delete("/projects/{project_id}/relationships/{relationship_id}")
    def delete_relationship(project_id: str, relationship_id: str) -> dict[str, Any]:
        removed = evo_store.remove_relationship(project_id, relationship_id)
        if not removed:
            raise EvolutionEntityNotFoundError(
                f"Relationship '{relationship_id}' not found in project '{project_id}'."
            )
        return {"deleted": True, "relationship_id": relationship_id}

    @router.get("/projects/{project_id}/graph")
    def get_project_graph(project_id: str) -> dict[str, Any]:
        proj = evo_store.get_project(project_id)
        dag = evo_store.get_dag(project_id)
        return dag.to_graph_payload(proj.base_checkpoint_id)

    # ------------------------------------------------------------------
    # Arbitrary / Consecutive / Branch-to-Branch Pair Comparison
    # ------------------------------------------------------------------
    @router.post("/projects/{project_id}/compare")
    def compare_arbitrary_versions(
        project_id: str,
        req: CompareEvolutionVersionsRequest,
    ) -> dict[str, Any]:
        proj = evo_store.get_project(project_id)
        dag = evo_store.get_dag(project_id)
        ca = evo_store.get_checkpoint(project_id, req.checkpoint_a_id)
        cb = evo_store.get_checkpoint(project_id, req.checkpoint_b_id)

        if (ca.checkpoint_id, cb.checkpoint_id) in dag.edge_by_pair:
            scope = "consecutive"
        elif ca.checkpoint_id == proj.base_checkpoint_id:
            scope = "base_to_current"
        else:
            scope = "branch_to_branch"

        lca_id = dag.lowest_common_ancestor(ca.checkpoint_id, cb.checkpoint_id)
        engine = MultiGenTensorEvolutionEngine(
            policy=policy_provider(),
            tolerances=DiffTolerances(
                atol=req.tolerances.atol,
                rtol=req.tolerances.rtol,
                zero_threshold=req.tolerances.zero_threshold,
            ),
            top_k=req.top_k_tensors,
            local_files_only=req.local_files_only,
        )
        tr, _ = engine.compare_pair(ca, cb, scope=scope)
        return {
            "project_id": project_id,
            "lowest_common_ancestor_id": lca_id,
            "lowest_common_ancestor_label": (
                dag.checkpoints_by_id[lca_id].version_label if lca_id else None
            ),
            "directed_path": dag.shortest_directed_path(ca.checkpoint_id, cb.checkpoint_id),
            "comparison": tr.to_dict(),
        }

    # ------------------------------------------------------------------
    # Evolution Analysis Execution, History, Regressions & Export
    # ------------------------------------------------------------------
    def _build_cfg(req: RunEvolutionAnalysisRequest) -> EvolutionAnalysisConfig:
        return EvolutionAnalysisConfig(
            include_tensor_evolution=req.include_tensor_evolution,
            include_layer_evolution=req.include_layer_evolution,
            include_behavioral_evolution=req.include_behavioral_evolution,
            include_representation_evolution=req.include_representation_evolution,
            enable_causal_patching=req.enable_causal_patching,
            preferred_device=req.preferred_device,
            tolerances=DiffTolerances(
                atol=req.tolerances.atol,
                rtol=req.tolerances.rtol,
                zero_threshold=req.tolerances.zero_threshold,
            ),
            chunk_elements=req.chunk_elements,
            top_k_tensors=req.top_k_tensors,
            behavioral_categories=req.behavioral_categories,
            behavioral_max_cases=req.behavioral_max_cases,
            max_new_tokens=req.max_new_tokens,
            seed=req.seed,
            prompt_mode=req.prompt_mode,
            overall_pass_rate_drop_threshold=req.overall_pass_rate_drop_threshold,
            category_score_drop_threshold=req.category_score_drop_threshold,
            local_files_only=req.local_files_only,
            allow_cache=req.allow_cache,
            force_rerun=req.force_rerun,
        )

    @router.post("/projects/{project_id}/analyze")
    def run_evolution_analysis(
        project_id: str,
        req: RunEvolutionAnalysisRequest,
    ) -> dict[str, Any]:
        _ = evo_store.get_project(project_id)
        cfg = _build_cfg(req)
        orch = EvolutionOrchestrator(store=evo_store, policy=policy_provider())
        token = CancellationToken()

        if not req.async_job:
            run_id = f"evorun-{uuid.uuid4().hex[:10]}"
            with tokens_lock:
                active_tokens[run_id] = token
            try:
                return orch.run_analysis(
                    project_id,
                    cfg,
                    run_id=run_id,
                    cancellation_token=token,
                )
            except OperationCancelledError as exc:
                evo_store.save_failed_run(run_id, exc.to_dict(), cancelled=True)
                raise
            except NeuralDiffError as exc:
                try:
                    evo_store.save_failed_run(run_id, exc.to_dict(), cancelled=False)
                except Exception as inner_exc:
                    _ = inner_exc
                raise
            finally:
                with tokens_lock:
                    active_tokens.pop(run_id, None)

        run_id = f"evorun-{uuid.uuid4().hex[:10]}"
        cache_key = orch.compute_content_cache_key(project_id, cfg)
        evo_store.create_run(
            run_id=run_id,
            project_id=project_id,
            cache_key=cache_key,
            config=cfg.to_dict(),
            provenance={"run_id": run_id, "project_id": project_id, "status": "queued"},
            total_steps=5,
        )

        def _bg() -> None:
            try:
                orch.run_analysis(
                    project_id,
                    cfg,
                    run_id=run_id,
                    cancellation_token=token,
                )
            except OperationCancelledError as exc:
                evo_store.save_failed_run(run_id, exc.to_dict(), cancelled=True)
            except NeuralDiffError as exc:
                evo_store.save_failed_run(run_id, exc.to_dict(), cancelled=False)
            except Exception as exc:
                evo_store.save_failed_run(
                    run_id,
                    {"error": "INTERNAL_EXECUTION_ERROR", "message": str(exc), "http_status": 500},
                )
            finally:
                with tokens_lock:
                    active_tokens.pop(run_id, None)

        with tokens_lock:
            active_tokens[run_id] = token
        worker = threading.Thread(target=_bg, name=f"evo-{run_id}", daemon=True)
        worker.start()
        return {
            "run_id": run_id,
            "project_id": project_id,
            "status": "running",
            "stage_message": "Queued background evolution analysis",
        }

    @router.post("/runs/{run_id}/cancel")
    def cancel_evolution_run(run_id: str) -> dict[str, Any]:
        with tokens_lock:
            tok = active_tokens.get(run_id)
        if tok is None:
            existing = evo_store.get_run(run_id)
            return {"run_id": run_id, "cancelled": False, "status": existing["status"]}
        tok.cancel("Cancelled via Evolution REST API.")
        return {"run_id": run_id, "cancelled": True, "status": "cancelling"}

    @router.get("/projects/{project_id}/runs")
    def list_project_runs(
        project_id: str,
        limit: int = Query(default=25, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        total, items = evo_store.list_runs(project_id, limit=limit, offset=offset)
        return {"total": total, "limit": limit, "offset": offset, "runs": items}

    @router.get("/runs/{run_id}")
    def get_evolution_run(run_id: str) -> dict[str, Any]:
        return evo_store.get_run(run_id)

    @router.get("/projects/{project_id}/regressions")
    def get_project_regressions(
        project_id: str,
        category: str | None = Query(default=None, max_length=80),
        transformation_type: str | None = Query(default=None, max_length=80),
    ) -> dict[str, Any]:
        events = evo_store.list_latest_regressions(
            project_id,
            category=category,
            transformation_type=transformation_type,
        )
        return {
            "project_id": project_id,
            "total": len(events),
            "regressions": events,
        }

    @router.get("/runs/{run_id}/export/{fmt}")
    def export_evolution_report(run_id: str, fmt: str) -> Response:
        fmt_l = fmt.lower().strip()
        if fmt_l not in ("json", "csv", "markdown", "md", "html", "pdf"):
            raise HTTPException(
                status_code=400,
                detail="Export format must be json, csv, markdown, html, or pdf.",
            )
        bundle = evo_store.get_run(run_id)
        if bundle.get("status") != "completed":
            raise HTTPException(
                status_code=409,
                detail=f"Run '{run_id}' is in status '{bundle.get('status')}' and cannot be exported yet.",
            )
        exporter = EvolutionReportExporter()
        if fmt_l == "json":
            return Response(
                content=exporter.to_json(bundle),
                media_type="application/json",
                headers={"Content-Disposition": f'attachment; filename="{run_id}.json"'},
            )
        if fmt_l == "csv":
            return Response(
                content=exporter.to_csv(bundle),
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{run_id}.csv"'},
            )
        if fmt_l in ("markdown", "md"):
            return Response(
                content=exporter.to_markdown(bundle),
                media_type="text/markdown; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{run_id}.md"'},
            )
        if fmt_l == "html":
            return Response(
                content=exporter.to_html(bundle),
                media_type="text/html; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{run_id}.html"'},
            )
        pdf_bytes = exporter.to_pdf(bundle)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{run_id}.pdf"'},
        )

    return router
