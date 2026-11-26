"""On-demand mathematical diagnostics router (SVD spectrum, 2:4 sparsity, weight-tying, history trends)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from api.store import ExperimentStore
from core.exceptions import ModelNotFoundError
from core.loaders.model_loader import ModelLoader
from core.regression_engine.diff_of_diffs import RunSnapshot, compare_run_history
from core.security import SecurityPolicy
from core.statistics.histograms import compute_paired_histogram
from core.structural_analysis.sparsity_pruning import analyze_matrix_sparsity
from core.structural_analysis.tie_weights import compare_weight_tying
from core.tensor_engine.task_vector import analyze_task_vector_spectrum
from core.tensor_engine.top_elements import extract_top_k_element_deltas


class TensorProbeRequest(BaseModel):
    model_a: str = Field(..., min_length=1, max_length=512)
    model_b: str = Field(..., min_length=1, max_length=512)
    tensor_name: str = Field(..., min_length=1, max_length=512)
    local_files_only: bool = False
    histogram_bins: int = Field(default=24, ge=4, le=128)
    top_k: int = Field(default=8, ge=1, le=64)


class WeightTyingRequest(BaseModel):
    model_a: str = Field(..., min_length=1, max_length=512)
    model_b: str = Field(..., min_length=1, max_length=512)
    local_files_only: bool = False


def build_diagnostics_router(
    *,
    store: ExperimentStore,
    policy_provider: Callable[[], SecurityPolicy],
) -> APIRouter:
    router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])

    @router.post("/tensor-probe")
    def probe_tensor(req: TensorProbeRequest) -> dict[str, Any]:
        loader = ModelLoader(policy=policy_provider())
        with (
            loader.load(req.model_a, local_files_only=req.local_files_only) as ha,
            loader.load(req.model_b, local_files_only=req.local_files_only) as hb,
        ):
            if req.tensor_name not in ha.tensors or req.tensor_name not in hb.tensors:
                raise ModelNotFoundError(
                    f"Tensor '{req.tensor_name}' was not found in both checkpoints."
                )
            ta = ha.get_tensor(req.tensor_name)
            tb = hb.get_tensor(req.tensor_name)
            if tuple(ta.shape) != tuple(tb.shape):
                raise HTTPException(
                    status_code=422,
                    detail=f"Shape mismatch for '{req.tensor_name}': {list(ta.shape)} vs {list(tb.shape)}.",
                )

            hist = compute_paired_histogram(ta, tb, bins=req.histogram_bins).to_dict()
            top_elems = [
                r.to_dict() for r in extract_top_k_element_deltas(ta, tb, k=req.top_k)
            ]

            svd_data: dict[str, Any] | None = None
            sparsity_b: dict[str, Any] | None = None
            if ta.ndim == 2:
                svd_data = analyze_task_vector_spectrum(ta, tb).to_dict()
                sparsity_b = analyze_matrix_sparsity(tb).to_dict()

            return {
                "tensor_name": req.tensor_name,
                "shape": list(ta.shape),
                "svd_spectrum": svd_data,
                "sparsity_b": sparsity_b,
                "histogram": hist,
                "top_elements": top_elems,
            }

    @router.post("/weight-tying")
    def check_weight_tying(req: WeightTyingRequest) -> dict[str, Any]:
        loader = ModelLoader(policy=policy_provider())
        with (
            loader.load(req.model_a, local_files_only=req.local_files_only) as ha,
            loader.load(req.model_b, local_files_only=req.local_files_only) as hb,
        ):
            return compare_weight_tying(ha, hb)

    @router.get("/history-trends")
    def get_history_trends(
        drop_tolerance: float = Query(default=0.05, ge=0.0, le=1.0),
    ) -> dict[str, Any]:
        summaries = store.list_experiments(limit=50)
        snapshots: list[RunSnapshot] = []
        for item in reversed(summaries):
            if item.get("status") != "completed":
                continue
            full = store.get_experiment(item["comparison_id"])
            bundle = full.get("bundle") if full else None
            beh = bundle.get("behavioral") if isinstance(bundle, dict) else None
            if not isinstance(beh, dict):
                continue
            stats = beh.get("overall_statistics", {})
            cats = {
                c["category"]: float(c["mean_score_b"])
                for c in beh.get("category_summaries", [])
                if "category" in c and "mean_score_b" in c
            }
            if cats:
                snapshots.append(
                    RunSnapshot(
                        run_id=str(item["comparison_id"]),
                        pass_rate=float(stats.get("pass_rate_b", 0.0)),
                        category_scores=cats,
                    )
                )

        if len(snapshots) < 2:
            return {
                "available_behavioral_runs": len(snapshots),
                "comparison": None,
            }
        return {
            "available_behavioral_runs": len(snapshots),
            "comparison": compare_run_history(
                snapshots, drop_tolerance=drop_tolerance
            ).to_dict(),
        }

    return router
