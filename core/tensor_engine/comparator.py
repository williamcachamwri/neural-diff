"""High-level checkpoint Tensor Diff Engine (Module B)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from core.loaders.base import LoadedModel
from core.resource_manager import CancellationToken
from core.statistics.distributions import (
    DistributionShiftSummary,
    TensorDistributionSummary,
    compute_distribution_shift,
    summarize_tensor_chunks,
)
from core.tensor_engine.metrics import (
    DiffTolerances,
    TensorDiffMetrics,
    compute_chunked_tensor_metrics,
)
from core.types import CanonicalRole, EpistemicStatus, TensorDescriptor


class TensorDiffStatus(StrEnum):
    """Comparison outcome classification per tensor key."""

    IDENTICAL = "identical"
    WITHIN_TOLERANCE = "within_tolerance"
    MODIFIED = "modified"
    SHAPE_MISMATCH = "shape_mismatch"
    MISSING_IN_B = "missing_in_b"
    ADDED_IN_B = "added_in_b"


@dataclass
class TensorComparisonRecord:
    """Full comparison record for a single named parameter across Model A and Model B."""

    name: str
    status: TensorDiffStatus
    epistemic_status: EpistemicStatus
    layer_index: int | None
    canonical_role: CanonicalRole
    shape_a: tuple[int, ...] | None
    shape_b: tuple[int, ...] | None
    dtype_a: str | None
    dtype_b: str | None
    dtype_changed: bool
    metrics: TensorDiffMetrics | None
    distribution_a: TensorDistributionSummary | None
    distribution_b: TensorDistributionSummary | None
    distribution_shift: DistributionShiftSummary | None
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "epistemic_status": self.epistemic_status.value,
            "layer_index": self.layer_index,
            "canonical_role": self.canonical_role.value,
            "shape_a": list(self.shape_a) if self.shape_a is not None else None,
            "shape_b": list(self.shape_b) if self.shape_b is not None else None,
            "dtype_a": self.dtype_a,
            "dtype_b": self.dtype_b,
            "dtype_changed": self.dtype_changed,
            "metrics": self.metrics.to_dict() if self.metrics is not None else None,
            "distribution_a": self.distribution_a.to_dict() if self.distribution_a is not None else None,
            "distribution_b": self.distribution_b.to_dict() if self.distribution_b is not None else None,
            "distribution_shift": (
                self.distribution_shift.to_dict() if self.distribution_shift is not None else None
            ),
            "notes": self.notes,
        }


@dataclass
class LayerDiffSummary:
    """Aggregated tensor diff metrics for a single transformer layer index."""

    layer_index: int
    tensor_count: int
    total_parameters: int
    changed_parameters: int
    changed_pct: float
    mean_cosine_similarity: float | None
    min_cosine_similarity: float | None
    mean_relative_l2: float | None
    max_relative_l2: float | None
    max_abs_diff: float | None
    role_breakdown: dict[str, dict[str, float | None]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "tensor_count": self.tensor_count,
            "total_parameters": self.total_parameters,
            "changed_parameters": self.changed_parameters,
            "changed_pct": self.changed_pct,
            "mean_cosine_similarity": self.mean_cosine_similarity,
            "min_cosine_similarity": self.min_cosine_similarity,
            "mean_relative_l2": self.mean_relative_l2,
            "max_relative_l2": self.max_relative_l2,
            "max_abs_diff": self.max_abs_diff,
            "role_breakdown": self.role_breakdown,
        }


@dataclass
class TensorDiffReport:
    """Aggregate report returned by `TensorDiffEngine`."""

    tolerances: DiffTolerances
    total_union_tensors: int
    compared_tensors: int
    identical_tensors: int
    within_tolerance_tensors: int
    modified_tensors: int
    shape_mismatch_tensors: int
    missing_in_b_tensors: int
    added_in_b_tensors: int
    total_compared_parameters: int
    total_changed_parameters: int
    global_changed_pct: float
    global_mean_cosine: float | None
    global_mean_relative_l2: float | None
    global_max_abs_diff: float | None
    has_any_nan_or_inf: bool
    records: list[TensorComparisonRecord]
    layer_summaries: list[LayerDiffSummary]
    top_diverged_names: list[str]

    def to_dict(self, *, include_records: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "epistemic_status": EpistemicStatus.OBSERVED.value,
            "tolerances": self.tolerances.to_dict(),
            "summary": {
                "total_union_tensors": self.total_union_tensors,
                "compared_tensors": self.compared_tensors,
                "identical_tensors": self.identical_tensors,
                "within_tolerance_tensors": self.within_tolerance_tensors,
                "modified_tensors": self.modified_tensors,
                "shape_mismatch_tensors": self.shape_mismatch_tensors,
                "missing_in_b_tensors": self.missing_in_b_tensors,
                "added_in_b_tensors": self.added_in_b_tensors,
                "total_compared_parameters": self.total_compared_parameters,
                "total_changed_parameters": self.total_changed_parameters,
                "global_changed_pct": self.global_changed_pct,
                "global_mean_cosine": self.global_mean_cosine,
                "global_mean_relative_l2": self.global_mean_relative_l2,
                "global_max_abs_diff": self.global_max_abs_diff,
                "has_any_nan_or_inf": self.has_any_nan_or_inf,
            },
            "layer_summaries": [ls.to_dict() for ls in self.layer_summaries],
            "top_diverged_names": self.top_diverged_names,
        }
        if include_records:
            payload["records"] = [r.to_dict() for r in self.records]
        return payload


class TensorDiffEngine:
    """Memory-bounded tensor comparison engine across two `LoadedModel` instances."""

    def __init__(
        self,
        tolerances: DiffTolerances | None = None,
        chunk_elements: int = 1_048_576,
        include_distributions: bool = True,
    ) -> None:
        self.tolerances = tolerances or DiffTolerances()
        self.chunk_elements = max(4096, int(chunk_elements))
        self.include_distributions = include_distributions

    def compare(
        self,
        model_a: LoadedModel,
        model_b: LoadedModel,
        *,
        tensor_filter: set[str] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> TensorDiffReport:
        descs_a = model_a.tensors
        descs_b = model_b.tensors
        ordered_keys = list(dict.fromkeys([*descs_a.keys(), *descs_b.keys()]))
        if tensor_filter is not None:
            ordered_keys = [k for k in ordered_keys if k in tensor_filter]

        records: list[TensorComparisonRecord] = []
        for name in ordered_keys:
            if cancellation_token is not None:
                cancellation_token.check()
            da = descs_a.get(name)
            db = descs_b.get(name)
            rec = self._compare_single_entry(model_a, model_b, name, da, db, cancellation_token)
            records.append(rec)

        return self._aggregate(records)

    def _compare_single_entry(
        self,
        model_a: LoadedModel,
        model_b: LoadedModel,
        name: str,
        da: TensorDescriptor | None,
        db: TensorDescriptor | None,
        token: CancellationToken | None,
    ) -> TensorComparisonRecord:
        notes: list[str] = []
        if da is not None and db is None:
            dist_a = (
                summarize_tensor_chunks(
                    model_a.iter_tensor_chunks(name, self.chunk_elements),
                    zero_threshold=self.tolerances.zero_threshold,
                )
                if self.include_distributions
                else None
            )
            notes.append("Tensor exists in Model A but is absent in Model B.")
            return TensorComparisonRecord(
                name=name,
                status=TensorDiffStatus.MISSING_IN_B,
                epistemic_status=EpistemicStatus.OBSERVED,
                layer_index=da.layer_index,
                canonical_role=da.canonical_role,
                shape_a=da.shape,
                shape_b=None,
                dtype_a=da.dtype,
                dtype_b=None,
                dtype_changed=False,
                metrics=None,
                distribution_a=dist_a,
                distribution_b=None,
                distribution_shift=None,
                notes=notes,
            )

        if da is None and db is not None:
            dist_b = (
                summarize_tensor_chunks(
                    model_b.iter_tensor_chunks(name, self.chunk_elements),
                    zero_threshold=self.tolerances.zero_threshold,
                )
                if self.include_distributions
                else None
            )
            notes.append("Tensor was added in Model B (absent in Model A).")
            return TensorComparisonRecord(
                name=name,
                status=TensorDiffStatus.ADDED_IN_B,
                epistemic_status=EpistemicStatus.OBSERVED,
                layer_index=db.layer_index,
                canonical_role=db.canonical_role,
                shape_a=None,
                shape_b=db.shape,
                dtype_a=None,
                dtype_b=db.dtype,
                dtype_changed=False,
                metrics=None,
                distribution_a=None,
                distribution_b=dist_b,
                distribution_shift=None,
                notes=notes,
            )

        assert da is not None and db is not None
        dtype_changed = da.dtype != db.dtype
        if dtype_changed:
            notes.append(f"Dtype changed from {da.dtype} to {db.dtype}; promoted to float64 for comparison.")

        if da.shape != db.shape:
            notes.append(
                f"Shape mismatch ({list(da.shape)} vs {list(db.shape)}); elementwise metrics skipped, distribution shift computed."
            )
            dist_a = summarize_tensor_chunks(
                model_a.iter_tensor_chunks(name, self.chunk_elements),
                zero_threshold=self.tolerances.zero_threshold,
            )
            dist_b = summarize_tensor_chunks(
                model_b.iter_tensor_chunks(name, self.chunk_elements),
                zero_threshold=self.tolerances.zero_threshold,
            )
            shift = compute_distribution_shift(dist_a, dist_b)
            return TensorComparisonRecord(
                name=name,
                status=TensorDiffStatus.SHAPE_MISMATCH,
                epistemic_status=EpistemicStatus.OBSERVED,
                layer_index=da.layer_index if da.layer_index is not None else db.layer_index,
                canonical_role=da.canonical_role,
                shape_a=da.shape,
                shape_b=db.shape,
                dtype_a=da.dtype,
                dtype_b=db.dtype,
                dtype_changed=dtype_changed,
                metrics=None,
                distribution_a=dist_a,
                distribution_b=dist_b,
                distribution_shift=shift,
                notes=notes,
            )

        metrics = compute_chunked_tensor_metrics(
            model_a.iter_tensor_chunks(name, self.chunk_elements),
            model_b.iter_tensor_chunks(name, self.chunk_elements),
            tolerances=self.tolerances,
            chunk_elements=self.chunk_elements,
            cancellation_token=token,
        )

        dist_a = None
        dist_b = None
        shift = None
        if self.include_distributions:
            dist_a = summarize_tensor_chunks(
                model_a.iter_tensor_chunks(name, self.chunk_elements),
                zero_threshold=self.tolerances.zero_threshold,
            )
            dist_b = summarize_tensor_chunks(
                model_b.iter_tensor_chunks(name, self.chunk_elements),
                zero_threshold=self.tolerances.zero_threshold,
            )
            shift = compute_distribution_shift(dist_a, dist_b)

        if metrics.nan_count_a or metrics.nan_count_b or metrics.inf_count_a or metrics.inf_count_b:
            notes.append(
                f"Non-finite values detected (NaN A={metrics.nan_count_a}, B={metrics.nan_count_b}; "
                f"Inf A={metrics.inf_count_a}, B={metrics.inf_count_b})."
            )

        if metrics.exact_equal and not dtype_changed:
            status = TensorDiffStatus.IDENTICAL
        elif metrics.within_tolerance:
            status = TensorDiffStatus.WITHIN_TOLERANCE
        else:
            status = TensorDiffStatus.MODIFIED

        return TensorComparisonRecord(
            name=name,
            status=status,
            epistemic_status=EpistemicStatus.OBSERVED,
            layer_index=da.layer_index,
            canonical_role=da.canonical_role,
            shape_a=da.shape,
            shape_b=db.shape,
            dtype_a=da.dtype,
            dtype_b=db.dtype,
            dtype_changed=dtype_changed,
            metrics=metrics,
            distribution_a=dist_a,
            distribution_b=dist_b,
            distribution_shift=shift,
            notes=notes,
        )

    def _aggregate(self, records: list[TensorComparisonRecord]) -> TensorDiffReport:
        identical = sum(1 for r in records if r.status == TensorDiffStatus.IDENTICAL)
        within_tol = sum(1 for r in records if r.status == TensorDiffStatus.WITHIN_TOLERANCE)
        modified = sum(1 for r in records if r.status == TensorDiffStatus.MODIFIED)
        shape_mm = sum(1 for r in records if r.status == TensorDiffStatus.SHAPE_MISMATCH)
        missing_b = sum(1 for r in records if r.status == TensorDiffStatus.MISSING_IN_B)
        added_b = sum(1 for r in records if r.status == TensorDiffStatus.ADDED_IN_B)

        compared_recs = [r for r in records if r.metrics is not None]
        total_params = sum(r.metrics.num_elements for r in compared_recs if r.metrics)
        changed_params = sum(r.metrics.changed_elements for r in compared_recs if r.metrics)
        global_changed_pct = round((changed_params / total_params) * 100.0, 6) if total_params > 0 else 0.0

        cos_vals = [
            r.metrics.cosine_similarity
            for r in compared_recs
            if r.metrics and r.metrics.cosine_similarity is not None
        ]
        rel_l2_vals = [
            r.metrics.relative_l2
            for r in compared_recs
            if r.metrics and r.metrics.relative_l2 is not None
        ]
        max_abs_vals = [
            r.metrics.max_abs_diff
            for r in compared_recs
            if r.metrics and r.metrics.max_abs_diff is not None
        ]
        has_non_finite = any(
            (r.metrics.non_finite_pair_elements > 0) for r in compared_recs if r.metrics
        )

        # Group by layer_index
        by_layer: dict[int, list[TensorComparisonRecord]] = {}
        for r in compared_recs:
            if r.layer_index is not None:
                by_layer.setdefault(r.layer_index, []).append(r)

        layer_summaries: list[LayerDiffSummary] = []
        for layer_idx in sorted(by_layer):
            lrecs = by_layer[layer_idx]
            lp = sum(r.metrics.num_elements for r in lrecs if r.metrics)
            lc = sum(r.metrics.changed_elements for r in lrecs if r.metrics)
            lcos = [
                r.metrics.cosine_similarity
                for r in lrecs
                if r.metrics and r.metrics.cosine_similarity is not None
            ]
            lrel = [
                r.metrics.relative_l2
                for r in lrecs
                if r.metrics and r.metrics.relative_l2 is not None
            ]
            lmax = [
                r.metrics.max_abs_diff
                for r in lrecs
                if r.metrics and r.metrics.max_abs_diff is not None
            ]
            role_map: dict[str, list[TensorComparisonRecord]] = {}
            for r in lrecs:
                role_map.setdefault(r.canonical_role.value, []).append(r)

            role_breakdown: dict[str, dict[str, float | None]] = {}
            for role_name, rlist in sorted(role_map.items()):
                rcos = [
                    x.metrics.cosine_similarity
                    for x in rlist
                    if x.metrics and x.metrics.cosine_similarity is not None
                ]
                rrel = [
                    x.metrics.relative_l2
                    for x in rlist
                    if x.metrics and x.metrics.relative_l2 is not None
                ]
                rmax = [
                    x.metrics.max_abs_diff
                    for x in rlist
                    if x.metrics and x.metrics.max_abs_diff is not None
                ]
                role_breakdown[role_name] = {
                    "mean_cosine": round(sum(rcos) / len(rcos), 8) if rcos else None,
                    "mean_relative_l2": round(sum(rrel) / len(rrel), 8) if rrel else None,
                    "max_abs_diff": max(rmax) if rmax else None,
                }

            layer_summaries.append(
                LayerDiffSummary(
                    layer_index=layer_idx,
                    tensor_count=len(lrecs),
                    total_parameters=lp,
                    changed_parameters=lc,
                    changed_pct=round((lc / lp) * 100.0, 6) if lp > 0 else 0.0,
                    mean_cosine_similarity=round(sum(lcos) / len(lcos), 8) if lcos else None,
                    min_cosine_similarity=min(lcos) if lcos else None,
                    mean_relative_l2=round(sum(lrel) / len(lrel), 8) if lrel else None,
                    max_relative_l2=max(lrel) if lrel else None,
                    max_abs_diff=max(lmax) if lmax else None,
                    role_breakdown=role_breakdown,
                )
            )

        ranked = sorted(
            compared_recs,
            key=lambda r: (
                1 if (r.metrics and r.metrics.relative_l2_infinite) else 0,
                (r.metrics.relative_l2 or 0.0) if r.metrics else 0.0,
                (r.metrics.max_abs_diff or 0.0) if r.metrics else 0.0,
            ),
            reverse=True,
        )
        top_diverged = [
            r.name
            for r in ranked[:15]
            if r.status not in (TensorDiffStatus.IDENTICAL,)
        ]

        return TensorDiffReport(
            tolerances=self.tolerances,
            total_union_tensors=len(records),
            compared_tensors=len(compared_recs),
            identical_tensors=identical,
            within_tolerance_tensors=within_tol,
            modified_tensors=modified,
            shape_mismatch_tensors=shape_mm,
            missing_in_b_tensors=missing_b,
            added_in_b_tensors=added_b,
            total_compared_parameters=total_params,
            total_changed_parameters=changed_params,
            global_changed_pct=global_changed_pct,
            global_mean_cosine=round(sum(cos_vals) / len(cos_vals), 8) if cos_vals else None,
            global_mean_relative_l2=round(sum(rel_l2_vals) / len(rel_l2_vals), 8) if rel_l2_vals else None,
            global_max_abs_diff=max(max_abs_vals) if max_abs_vals else None,
            has_any_nan_or_inf=has_non_finite,
            records=records,
            layer_summaries=layer_summaries,
            top_diverged_names=top_diverged,
        )
