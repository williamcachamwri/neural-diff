"""Safetensors single-file, multi-file, and sharded checkpoint readers."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from adapters.architectures.registry import classify_tensor_role
from core.exceptions import MalformedCheckpointError, ModelNotFoundError
from core.loaders.base import CheckpointReader
from core.security import (
    SecurityPolicy,
    read_validated_json,
    validate_safe_shard_filename,
    validate_safetensors_header,
)
from core.types import TensorDescriptor
from safetensors import safe_open

_SAFETENSORS_TO_TORCH_DTYPE_NAME: dict[str, str] = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "U16": "uint16",
    "U32": "uint32",
    "U64": "uint64",
    "BOOL": "bool",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}


def normalize_safetensors_dtype(raw_dtype: str) -> str:
    return _SAFETENSORS_TO_TORCH_DTYPE_NAME.get(raw_dtype, raw_dtype.lower())


class SingleSafetensorsReader(CheckpointReader):
    """Lazy, memory-mapped reader for a single `.safetensors` file."""

    def __init__(self, file_path: Path, policy: SecurityPolicy | None = None) -> None:
        self.file_path = file_path
        self.policy = policy or SecurityPolicy()
        header = validate_safetensors_header(file_path, policy=self.policy)
        self._custom_meta: dict[str, str] = dict(header.get("__metadata__", {}))
        self._descriptors: dict[str, TensorDescriptor] = {}

        for name, info in header.items():
            if name == "__metadata__":
                continue
            shape = tuple(int(d) for d in info["shape"])
            raw_dtype = str(info["dtype"])
            offsets = info["data_offsets"]
            byte_size = int(offsets[1]) - int(offsets[0])
            num_elements = math.prod(shape) if shape else 1
            layer_idx, role = classify_tensor_role(name)
            self._descriptors[name] = TensorDescriptor(
                name=name,
                shape=shape,
                dtype=normalize_safetensors_dtype(raw_dtype),
                num_elements=num_elements,
                byte_size=byte_size,
                shard_path=file_path.name,
                layer_index=layer_idx,
                canonical_role=role,
            )

    def descriptors(self) -> dict[str, TensorDescriptor]:
        return dict(self._descriptors)

    def custom_metadata(self) -> dict[str, str]:
        return dict(self._custom_meta)

    def get_tensor(self, name: str) -> torch.Tensor:
        if name not in self._descriptors:
            raise ModelNotFoundError(f"Tensor '{name}' not found in {self.file_path.name}.")
        try:
            with safe_open(str(self.file_path), framework="pt", device="cpu") as handle:
                return handle.get_tensor(name)
        except Exception as exc:
            raise MalformedCheckpointError(
                f"Failed reading tensor '{name}' from {self.file_path.name}: {exc}"
            ) from exc

    def iter_tensor_chunks(self, name: str, chunk_elements: int = 1_048_576) -> Iterator[torch.Tensor]:
        if name not in self._descriptors:
            raise ModelNotFoundError(f"Tensor '{name}' not found in {self.file_path.name}.")
        desc = self._descriptors[name]
        step_elems = max(1, int(chunk_elements))

        # Scalar or 1D/small tensor: direct slice or load
        if len(desc.shape) == 0 or desc.num_elements <= step_elems:
            yield self.get_tensor(name).reshape(-1)
            return

        leading_dim = desc.shape[0]
        row_elems = math.prod(desc.shape[1:]) if len(desc.shape) > 1 else 1
        if leading_dim == 0 or row_elems == 0:
            yield self.get_tensor(name).reshape(-1)
            return

        if row_elems > step_elems:
            # Single leading row exceeds chunk target: load tensor and yield 1D slices
            flat = self.get_tensor(name).reshape(-1)
            for start in range(0, flat.numel(), step_elems):
                yield flat[start : start + step_elems]
            return

        rows_per_chunk = max(1, step_elems // row_elems)
        try:
            with safe_open(str(self.file_path), framework="pt", device="cpu") as handle:
                tensor_slice = handle.get_slice(name)
                for row_start in range(0, leading_dim, rows_per_chunk):
                    row_end = min(leading_dim, row_start + rows_per_chunk)
                    sub = tensor_slice[row_start:row_end].reshape(-1)
                    # Sub-slice cleanly if still slightly larger than step_elems
                    for offset in range(0, sub.numel(), step_elems):
                        yield sub[offset : offset + step_elems]
        except Exception as exc:
            raise MalformedCheckpointError(
                f"Failed streaming slice for '{name}' from {self.file_path.name}: {exc}"
            ) from exc


class MultiSafetensorsReader(CheckpointReader):
    """Unified lazy reader over sharded (`model.safetensors.index.json`) or multi-file Safetensors."""

    def __init__(
        self,
        shard_readers: dict[str, SingleSafetensorsReader],
        weight_map: dict[str, str],
        index_metadata: dict[str, str] | None = None,
    ) -> None:
        self._shard_readers = shard_readers
        self._weight_map = weight_map
        self._custom_meta: dict[str, str] = dict(index_metadata or {})
        self._descriptors: dict[str, TensorDescriptor] = {}

        for shard_name, reader in shard_readers.items():
            for k, v in reader.custom_metadata().items():
                self._custom_meta.setdefault(f"{shard_name}:{k}", v)

        for tensor_name, shard_name in weight_map.items():
            shard_reader = shard_readers.get(shard_name)
            if shard_reader is None:
                raise MalformedCheckpointError(
                    f"Shard '{shard_name}' referenced by tensor '{tensor_name}' is missing."
                )
            shard_descs = shard_reader.descriptors()
            if tensor_name not in shard_descs:
                raise MalformedCheckpointError(
                    f"Shard index maps '{tensor_name}' to '{shard_name}', but tensor is absent in shard."
                )
            self._descriptors[tensor_name] = shard_descs[tensor_name]

    @classmethod
    def from_index_file(
        cls,
        index_path: Path,
        policy: SecurityPolicy | None = None,
    ) -> MultiSafetensorsReader:
        active_policy = policy or SecurityPolicy()
        index_json = read_validated_json(index_path, max_bytes=active_policy.max_json_bytes)
        weight_map = index_json.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise MalformedCheckpointError(
                f"Invalid or empty `weight_map` in {index_path.name}.",
                details={"path": str(index_path)},
            )
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in weight_map.items()):
            raise MalformedCheckpointError(
                f"`weight_map` keys and shard filenames must be strings in {index_path.name}."
            )

        base_dir = index_path.parent
        unique_shards = sorted(set(weight_map.values()))
        shard_readers: dict[str, SingleSafetensorsReader] = {}
        for shard_name in unique_shards:
            shard_path = validate_safe_shard_filename(shard_name, base_dir, policy=active_policy)
            shard_readers[shard_name] = SingleSafetensorsReader(shard_path, policy=active_policy)

        raw_meta: dict[str, Any] = index_json.get("metadata", {}) if isinstance(index_json.get("metadata"), dict) else {}
        str_meta = {str(k): str(v) for k, v in raw_meta.items()}
        return cls(shard_readers=shard_readers, weight_map=weight_map, index_metadata=str_meta)

    @classmethod
    def from_shard_files(
        cls,
        files: list[Path],
        policy: SecurityPolicy | None = None,
    ) -> MultiSafetensorsReader:
        active_policy = policy or SecurityPolicy()
        if not files:
            raise MalformedCheckpointError("No Safetensors files provided.")
        shard_readers: dict[str, SingleSafetensorsReader] = {}
        weight_map: dict[str, str] = {}
        for fp in sorted(files, key=lambda p: p.name):
            reader = SingleSafetensorsReader(fp, policy=active_policy)
            shard_readers[fp.name] = reader
            for tensor_name in reader.descriptors():
                if tensor_name in weight_map:
                    raise MalformedCheckpointError(
                        f"Duplicate tensor '{tensor_name}' found across '{weight_map[tensor_name]}' and '{fp.name}'."
                    )
                weight_map[tensor_name] = fp.name
        return cls(shard_readers=shard_readers, weight_map=weight_map)

    @property
    def shard_count(self) -> int:
        return len(self._shard_readers)

    def descriptors(self) -> dict[str, TensorDescriptor]:
        return dict(self._descriptors)

    def custom_metadata(self) -> dict[str, str]:
        return dict(self._custom_meta)

    def get_tensor(self, name: str) -> torch.Tensor:
        shard_name = self._weight_map.get(name)
        if shard_name is None:
            raise ModelNotFoundError(f"Tensor '{name}' not found in sharded Safetensors checkpoint.")
        return self._shard_readers[shard_name].get_tensor(name)

    def iter_tensor_chunks(self, name: str, chunk_elements: int = 1_048_576) -> Iterator[torch.Tensor]:
        shard_name = self._weight_map.get(name)
        if shard_name is None:
            raise ModelNotFoundError(f"Tensor '{name}' not found in sharded Safetensors checkpoint.")
        return self._shard_readers[shard_name].iter_tensor_chunks(name, chunk_elements=chunk_elements)

    def close(self) -> None:
        for reader in self._shard_readers.values():
            reader.close()
