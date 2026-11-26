"""Structural Diff Engine (Module C) — added, removed, renamed, shape, dtype, and layer diffs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.loaders.base import LoadedModel
from core.structural_analysis.compatibility import CompatibilityReport, validate_compatibility
from core.tensor_engine.metrics import DiffTolerances, compute_chunked_tensor_metrics
from core.types import EpistemicStatus, TensorDescriptor


@dataclass(frozen=True)
class RenamedTensorCandidate:
    """Inferred rename pairing between an unmatched Model A tensor and unmatched Model B tensor."""

    old_name: str
    new_name: str
    shape: tuple[int, ...]
    canonical_role: str
    layer_index_a: int | None
    layer_index_b: int | None
    cosine_similarity: float
    relative_l2: float | None
    exact_equal: bool
    epistemic_status: EpistemicStatus = EpistemicStatus.INFERRED

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        data["epistemic_status"] = self.epistemic_status.value
        return data


@dataclass(frozen=True)
class ShapeChangeRecord:
    """Observed shape change for a shared parameter key."""

    name: str
    shape_a: tuple[int, ...]
    shape_b: tuple[int, ...]
    elements_a: int
    elements_b: int
    delta_elements: int
    canonical_role: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape_a": list(self.shape_a),
            "shape_b": list(self.shape_b),
            "elements_a": self.elements_a,
            "elements_b": self.elements_b,
            "delta_elements": self.delta_elements,
            "canonical_role": self.canonical_role,
        }


@dataclass(frozen=True)
class DtypeChangeRecord:
    """Observed dtype transition for a shared parameter key."""

    name: str
    dtype_a: str
    dtype_b: str
    shape_a: tuple[int, ...]
    shape_b: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype_a": self.dtype_a,
            "dtype_b": self.dtype_b,
            "shape_a": list(self.shape_a),
            "shape_b": list(self.shape_b),
        }


@dataclass(frozen=True)
class LayerStructuralDelta:
    """Structural comparison of a single transformer block index."""

    layer_index: int
    present_in_a: bool
    present_in_b: bool
    tensor_count_a: int
    tensor_count_b: int
    parameters_a: int
    parameters_b: int
    delta_parameters: int
    roles_a: list[str]
    roles_b: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StructuralDiffReport:
    """Complete structural diff report between Model A and Model B."""

    model_a_parameters: int
    model_b_parameters: int
    delta_parameters: int
    model_a_bytes: int
    model_b_bytes: int
    delta_bytes: int
    shared_tensors: list[str]
    added_tensors: list[dict[str, Any]]
    removed_tensors: list[dict[str, Any]]
    inferred_renames: list[RenamedTensorCandidate]
    shape_changes: list[ShapeChangeRecord]
    dtype_changes: list[DtypeChangeRecord]
    layer_deltas: list[LayerStructuralDelta]
    compatibility: CompatibilityReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "epistemic_status": EpistemicStatus.OBSERVED.value,
            "model_a_parameters": self.model_a_parameters,
            "model_b_parameters": self.model_b_parameters,
            "delta_parameters": self.delta_parameters,
            "model_a_bytes": self.model_a_bytes,
            "model_b_bytes": self.model_b_bytes,
            "delta_bytes": self.delta_bytes,
            "counts": {
                "shared": len(self.shared_tensors),
                "added": len(self.added_tensors),
                "removed": len(self.removed_tensors),
                "inferred_renames": len(self.inferred_renames),
                "shape_changes": len(self.shape_changes),
                "dtype_changes": len(self.dtype_changes),
                "layers_compared": len(self.layer_deltas),
            },
            "shared_tensors": self.shared_tensors,
            "added_tensors": self.added_tensors,
            "removed_tensors": self.removed_tensors,
            "inferred_renames": [r.to_dict() for r in self.inferred_renames],
            "shape_changes": [s.to_dict() for s in self.shape_changes],
            "dtype_changes": [d.to_dict() for d in self.dtype_changes],
            "layer_deltas": [ld.to_dict() for ld in self.layer_deltas],
            "compatibility": self.compatibility.to_dict(),
        }


class StructuralDiffEngine:
    """Detects structural topology, key, shape, dtype, layer, and rename differences."""

    def __init__(
        self,
        *,
        enable_rename_inference: bool = True,
        rename_min_cosine: float = 0.985,
        max_rename_candidates: int = 128,
        chunk_elements: int = 1_048_576,
    ) -> None:
        self.enable_rename_inference = enable_rename_inference
        self.rename_min_cosine = rename_min_cosine
        self.max_rename_candidates = max_rename_candidates
        self.chunk_elements = chunk_elements

    def compare(self, model_a: LoadedModel, model_b: LoadedModel) -> StructuralDiffReport:
        descs_a = model_a.tensors
        descs_b = model_b.tensors
        keys_a = set(descs_a.keys())
        keys_b = set(descs_b.keys())

        shared = [k for k in descs_a if k in keys_b]
        raw_removed = [k for k in descs_a if k not in keys_b]
        raw_added = [k for k in descs_b if k not in keys_a]

        shape_changes: list[ShapeChangeRecord] = []
        dtype_changes: list[DtypeChangeRecord] = []
        for k in shared:
            da = descs_a[k]
            db = descs_b[k]
            if da.shape != db.shape:
                shape_changes.append(
                    ShapeChangeRecord(
                        name=k,
                        shape_a=da.shape,
                        shape_b=db.shape,
                        elements_a=da.num_elements,
                        elements_b=db.num_elements,
                        delta_elements=db.num_elements - da.num_elements,
                        canonical_role=da.canonical_role.value,
                    )
                )
            if da.dtype != db.dtype:
                dtype_changes.append(
                    DtypeChangeRecord(
                        name=k,
                        dtype_a=da.dtype,
                        dtype_b=db.dtype,
                        shape_a=da.shape,
                        shape_b=db.shape,
                    )
                )

        compatibility = validate_compatibility(model_a, model_b)

        # Conservative rename inference: never pair across distinct model_types unless shapes,
        # canonical roles, layer alignment, and high cosine similarity agree.
        inferred_renames: list[RenamedTensorCandidate] = []
        arch_a_type = model_a.metadata.architecture.model_type
        arch_b_type = model_b.metadata.architecture.model_type
        cross_arch_distinct = bool(arch_a_type and arch_b_type and arch_a_type != arch_b_type)

        if (
            self.enable_rename_inference
            and raw_removed
            and raw_added
            and not cross_arch_distinct
            and len(raw_removed) <= self.max_rename_candidates
            and len(raw_added) <= self.max_rename_candidates
        ):
            inferred_renames = self._infer_renames(model_a, model_b, raw_removed, raw_added)

        matched_old = {r.old_name for r in inferred_renames}
        matched_new = {r.new_name for r in inferred_renames}

        removed_tensors = [
            descs_a[k].to_dict() for k in raw_removed if k not in matched_old
        ]
        added_tensors = [
            descs_b[k].to_dict() for k in raw_added if k not in matched_new
        ]

        layer_deltas = self._compute_layer_deltas(descs_a, descs_b)

        return StructuralDiffReport(
            model_a_parameters=model_a.metadata.total_parameters,
            model_b_parameters=model_b.metadata.total_parameters,
            delta_parameters=model_b.metadata.total_parameters - model_a.metadata.total_parameters,
            model_a_bytes=model_a.metadata.total_bytes,
            model_b_bytes=model_b.metadata.total_bytes,
            delta_bytes=model_b.metadata.total_bytes - model_a.metadata.total_bytes,
            shared_tensors=shared,
            added_tensors=added_tensors,
            removed_tensors=removed_tensors,
            inferred_renames=inferred_renames,
            shape_changes=shape_changes,
            dtype_changes=dtype_changes,
            layer_deltas=layer_deltas,
            compatibility=compatibility,
        )

    def _infer_renames(
        self,
        model_a: LoadedModel,
        model_b: LoadedModel,
        removed_keys: list[str],
        added_keys: list[str],
    ) -> list[RenamedTensorCandidate]:
        descs_a = model_a.tensors
        descs_b = model_b.tensors

        # Bucket added keys by (shape, canonical_role, layer_index) first, then (shape, canonical_role)
        candidates_scored: list[tuple[float, RenamedTensorCandidate]] = []
        for old_k in removed_keys:
            da = descs_a[old_k]
            if da.num_elements == 0:
                continue
            for new_k in added_keys:
                db = descs_b[new_k]
                if da.shape != db.shape:
                    continue
                if da.canonical_role != db.canonical_role:
                    continue
                if (
                    da.layer_index is not None
                    and db.layer_index is not None
                    and da.layer_index != db.layer_index
                ):
                    continue
                metrics = compute_chunked_tensor_metrics(
                    model_a.iter_tensor_chunks(old_k, self.chunk_elements),
                    model_b.iter_tensor_chunks(new_k, self.chunk_elements),
                    tolerances=DiffTolerances(),
                    chunk_elements=self.chunk_elements,
                )
                cos = metrics.cosine_similarity
                if cos is not None and cos >= self.rename_min_cosine:
                    cand = RenamedTensorCandidate(
                        old_name=old_k,
                        new_name=new_k,
                        shape=da.shape,
                        canonical_role=da.canonical_role.value,
                        layer_index_a=da.layer_index,
                        layer_index_b=db.layer_index,
                        cosine_similarity=round(cos, 8),
                        relative_l2=metrics.relative_l2,
                        exact_equal=metrics.exact_equal,
                    )
                    score = cos + (0.01 if metrics.exact_equal else 0.0)
                    candidates_scored.append((score, cand))

        # Greedy bipartite matching from highest similarity downward
        candidates_scored.sort(key=lambda item: item[0], reverse=True)
        used_old: set[str] = set()
        used_new: set[str] = set()
        accepted: list[RenamedTensorCandidate] = []
        for _score, cand in candidates_scored:
            if cand.old_name in used_old or cand.new_name in used_new:
                continue
            used_old.add(cand.old_name)
            used_new.add(cand.new_name)
            accepted.append(cand)
        return accepted

    @staticmethod
    def _compute_layer_deltas(
        descs_a: dict[str, TensorDescriptor],
        descs_b: dict[str, TensorDescriptor],
    ) -> list[LayerStructuralDelta]:
        layers_a: dict[int, list[TensorDescriptor]] = {}
        layers_b: dict[int, list[TensorDescriptor]] = {}
        for d in descs_a.values():
            if d.layer_index is not None:
                layers_a.setdefault(d.layer_index, []).append(d)
        for d in descs_b.values():
            if d.layer_index is not None:
                layers_b.setdefault(d.layer_index, []).append(d)

        all_layers = sorted(set(layers_a.keys()) | set(layers_b.keys()))
        deltas: list[LayerStructuralDelta] = []
        for idx in all_layers:
            la = layers_a.get(idx, [])
            lb = layers_b.get(idx, [])
            pa = sum(d.num_elements for d in la)
            pb = sum(d.num_elements for d in lb)
            deltas.append(
                LayerStructuralDelta(
                    layer_index=idx,
                    present_in_a=bool(la),
                    present_in_b=bool(lb),
                    tensor_count_a=len(la),
                    tensor_count_b=len(lb),
                    parameters_a=pa,
                    parameters_b=pb,
                    delta_parameters=pb - pa,
                    roles_a=sorted({d.canonical_role.value for d in la}),
                    roles_b=sorted({d.canonical_role.value for d in lb}),
                )
            )
        return deltas
