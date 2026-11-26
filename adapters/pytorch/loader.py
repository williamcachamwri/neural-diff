"""Safe PyTorch (`weights_only=True`) single and sharded checkpoint readers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from adapters.architectures.registry import classify_tensor_role
from core.exceptions import MalformedCheckpointError, ModelNotFoundError, SecurityValidationError
from core.loaders.base import CheckpointReader
from core.security import (
    SecurityPolicy,
    inspect_pytorch_archive_safety,
    read_validated_json,
    validate_safe_shard_filename,
)
from core.types import TensorDescriptor


def _normalize_torch_dtype(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _extract_state_dict(payload: Any, source_name: str) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise MalformedCheckpointError(
            f"Expected dictionary state_dict in {source_name}, got {type(payload).__name__}."
        )

    candidate: dict[str, Any] = payload
    for wrapper_key in ("state_dict", "model_state_dict", "model"):
        inner = payload.get(wrapper_key)
        if isinstance(inner, dict) and inner and all(isinstance(v, torch.Tensor) for v in inner.values()):
            candidate = inner
            break

    tensors: dict[str, torch.Tensor] = {}
    for k, v in candidate.items():
        if isinstance(k, str) and isinstance(v, torch.Tensor):
            tensors[k] = v.detach().cpu()

    if not tensors:
        raise MalformedCheckpointError(
            f"PyTorch archive {source_name} contained no valid `torch.Tensor` entries."
        )
    return tensors


class SafePyTorchReader(CheckpointReader):
    """Loads legacy PyTorch `.pt`/`.pth`/`.bin` checkpoints strictly via static opcode audit + `weights_only=True`."""

    def __init__(
        self,
        shard_files: list[Path],
        weight_map: dict[str, str] | None = None,
        policy: SecurityPolicy | None = None,
    ) -> None:
        self.policy = policy or SecurityPolicy()
        if not self.policy.allow_pytorch_weights_only:
            raise SecurityValidationError(
                "PyTorch pickle-based checkpoints are disabled by active SecurityPolicy; use Safetensors."
            )
        if not shard_files:
            raise MalformedCheckpointError("No PyTorch checkpoint files specified.")

        self._shard_paths: dict[str, Path] = {fp.name: fp for fp in shard_files}
        self._shard_cache: dict[str, dict[str, torch.Tensor]] = {}
        self._descriptors: dict[str, TensorDescriptor] = {}
        self._weight_map: dict[str, str] = {}
        self._security_notes: list[str] = []

        for fp in shard_files:
            opcode_notes = inspect_pytorch_archive_safety(fp)
            self._security_notes.append(
                f"Scanned {fp.name}: {len(opcode_notes)} safe tensor rebuild globals; loaded with weights_only=True."
            )
            try:
                raw = torch.load(str(fp), map_location="cpu", weights_only=True)
            except Exception as exc:
                raise MalformedCheckpointError(
                    f"Safe `torch.load(..., weights_only=True)` rejected {fp.name}: {exc}"
                ) from exc
            state_dict = _extract_state_dict(raw, fp.name)
            self._shard_cache[fp.name] = state_dict

            for tname, tensor in state_dict.items():
                if weight_map is not None:
                    expected_shard = weight_map.get(tname)
                    if expected_shard != fp.name:
                        continue
                if tname in self._descriptors:
                    raise MalformedCheckpointError(
                        f"Duplicate tensor '{tname}' across PyTorch shards."
                    )
                layer_idx, role = classify_tensor_role(tname)
                shape = tuple(int(d) for d in tensor.shape)
                numel = int(tensor.numel())
                byte_size = numel * int(tensor.element_size())
                self._descriptors[tname] = TensorDescriptor(
                    name=tname,
                    shape=shape,
                    dtype=_normalize_torch_dtype(tensor.dtype),
                    num_elements=numel,
                    byte_size=byte_size,
                    shard_path=fp.name,
                    layer_index=layer_idx,
                    canonical_role=role,
                )
                self._weight_map[tname] = fp.name

        if weight_map is not None:
            missing = [k for k in weight_map if k not in self._descriptors]
            if missing:
                raise MalformedCheckpointError(
                    f"Sharded PyTorch index references {len(missing)} missing tensor(s), e.g. '{missing[0]}'."
                )

    @classmethod
    def from_single_file(cls, file_path: Path, policy: SecurityPolicy | None = None) -> SafePyTorchReader:
        return cls(shard_files=[file_path], policy=policy)

    @classmethod
    def from_index_file(cls, index_path: Path, policy: SecurityPolicy | None = None) -> SafePyTorchReader:
        active_policy = policy or SecurityPolicy()
        index_json = read_validated_json(index_path, max_bytes=active_policy.max_json_bytes)
        weight_map = index_json.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise MalformedCheckpointError(f"Invalid `weight_map` in {index_path.name}.")
        base_dir = index_path.parent
        unique_shards = sorted(set(str(v) for v in weight_map.values()))
        shard_paths = [
            validate_safe_shard_filename(sname, base_dir, policy=active_policy)
            for sname in unique_shards
        ]
        str_map = {str(k): str(v) for k, v in weight_map.items()}
        return cls(shard_files=shard_paths, weight_map=str_map, policy=active_policy)

    @property
    def security_notes(self) -> list[str]:
        return list(self._security_notes)

    @property
    def shard_count(self) -> int:
        return len(self._shard_paths)

    def descriptors(self) -> dict[str, TensorDescriptor]:
        return dict(self._descriptors)

    def custom_metadata(self) -> dict[str, str]:
        return {"loader": "torch.load(weights_only=True)"}

    def get_tensor(self, name: str) -> torch.Tensor:
        shard_name = self._weight_map.get(name)
        if shard_name is None:
            raise ModelNotFoundError(f"Tensor '{name}' not found in PyTorch checkpoint.")
        return self._shard_cache[shard_name][name]

    def close(self) -> None:
        self._shard_cache.clear()
