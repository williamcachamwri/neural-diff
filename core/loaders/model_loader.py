"""Unified Model Loader (Module A) orchestrating Safetensors, Safe PyTorch, and Hugging Face."""

from __future__ import annotations

from pathlib import Path

from adapters.architectures.registry import extract_architecture_metadata
from adapters.huggingface.loader import (
    detect_snapshot_revision,
    open_directory_checkpoint,
    resolve_hf_hub_to_local_dir,
)
from adapters.pytorch.loader import SafePyTorchReader
from adapters.safetensors.loader import MultiSafetensorsReader, SingleSafetensorsReader
from core.exceptions import ModelNotFoundError, UnsupportedFormatError
from core.loaders.base import CheckpointReader, LoadedModel
from core.security import SecurityPolicy, is_valid_hf_repo_id, validate_safe_path
from core.types import ModelFormat, ModelMetadata


class ModelLoader:
    """Security-hardened unified loader for local checkpoints, directories, and HF Hub IDs."""

    def __init__(
        self,
        policy: SecurityPolicy | None = None,
        allow_hub_download: bool = True,
        hub_cache_dir: Path | None = None,
    ) -> None:
        self.policy = policy or SecurityPolicy()
        self.allow_hub_download = allow_hub_download
        self.hub_cache_dir = hub_cache_dir

    def load(
        self,
        source: str | Path,
        *,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> LoadedModel:
        """Load metadata and attach a lazy `CheckpointReader` for `source`."""
        raw_str = str(source).strip()
        if not raw_str:
            raise ModelNotFoundError("Model source identifier cannot be empty.")

        # Determine whether `source` is an existing local path (or looks like a local path) vs HF Hub ID
        looks_like_local = (
            isinstance(source, Path)
            or raw_str.startswith((".", "/", "\\"))
            or ":" in raw_str
            or "\\" in raw_str
            or raw_str.endswith((".safetensors", ".pt", ".pth", ".bin", ".json"))
            or (Path(raw_str).exists() and self.policy.is_within_allowed_roots(Path(raw_str).resolve(strict=False)))
        )

        if not looks_like_local and is_valid_hf_repo_id(raw_str):
            if not self.allow_hub_download and not local_files_only:
                local_files_only = True
            resolved_dir, detected_rev = resolve_hf_hub_to_local_dir(
                raw_str,
                revision=revision,
                cache_dir=self.hub_cache_dir,
                local_files_only=local_files_only,
            )
            # Ensure resolved cache directory is permitted under active policy
            if not self.policy.is_within_allowed_roots(resolved_dir):
                self.policy.allowed_roots.append(resolved_dir.parent.parent)
            reader, _fmt, shard_count, notes = open_directory_checkpoint(resolved_dir, policy=self.policy)
            arch = extract_architecture_metadata(resolved_dir)
            return self._assemble_loaded_model(
                model_id=raw_str,
                resolved_path=resolved_dir,
                fmt=ModelFormat.HF_HUB_CACHED,
                revision=detected_rev,
                reader=reader,
                shard_count=shard_count,
                arch=arch,
                security_notes=notes,
            )

        safe_path = validate_safe_path(source, policy=self.policy, must_exist=True)

        if safe_path.is_dir():
            reader, fmt, shard_count, notes = open_directory_checkpoint(safe_path, policy=self.policy)
            arch = extract_architecture_metadata(safe_path)
            detected_rev = revision or detect_snapshot_revision(safe_path)
            return self._assemble_loaded_model(
                model_id=safe_path.name,
                resolved_path=safe_path,
                fmt=fmt,
                revision=detected_rev,
                reader=reader,
                shard_count=shard_count,
                arch=arch,
                security_notes=notes,
            )

        # Single file or shard index file
        name_lower = safe_path.name.lower()
        parent_arch = extract_architecture_metadata(safe_path.parent)
        reader_obj: CheckpointReader

        if name_lower.endswith(".safetensors.index.json"):
            m_reader = MultiSafetensorsReader.from_index_file(safe_path, policy=self.policy)
            return self._assemble_loaded_model(
                model_id=safe_path.parent.name or safe_path.name,
                resolved_path=safe_path,
                fmt=ModelFormat.SAFETENSORS_SHARDED,
                revision=revision,
                reader=m_reader,
                shard_count=m_reader.shard_count,
                arch=parent_arch,
                security_notes=["Verified sharded Safetensors index and shard headers."],
            )

        if name_lower.endswith(".safetensors"):
            s_reader = SingleSafetensorsReader(safe_path, policy=self.policy)
            return self._assemble_loaded_model(
                model_id=safe_path.stem,
                resolved_path=safe_path,
                fmt=ModelFormat.SAFETENSORS_SINGLE,
                revision=revision,
                reader=s_reader,
                shard_count=1,
                arch=parent_arch,
                security_notes=["Verified Safetensors binary header; zero code execution."],
            )

        if name_lower.endswith(".bin.index.json"):
            p_reader = SafePyTorchReader.from_index_file(safe_path, policy=self.policy)
            return self._assemble_loaded_model(
                model_id=safe_path.parent.name or safe_path.name,
                resolved_path=safe_path,
                fmt=ModelFormat.PYTORCH_SAFE,
                revision=revision,
                reader=p_reader,
                shard_count=p_reader.shard_count,
                arch=parent_arch,
                security_notes=p_reader.security_notes,
            )

        if name_lower.endswith((".pt", ".pth", ".bin")):
            p_reader = SafePyTorchReader.from_single_file(safe_path, policy=self.policy)
            reader_obj = p_reader
            return self._assemble_loaded_model(
                model_id=safe_path.stem,
                resolved_path=safe_path,
                fmt=ModelFormat.PYTORCH_SAFE,
                revision=revision,
                reader=reader_obj,
                shard_count=1,
                arch=parent_arch,
                security_notes=p_reader.security_notes,
            )

        raise UnsupportedFormatError(
            f"Unsupported checkpoint file extension for '{safe_path.name}'. Supported: `.safetensors`, `.safetensors.index.json`, `.pt`, `.pth`, `.bin`, or Hugging Face directory.",
            details={"path": str(safe_path)},
        )

    @staticmethod
    def _assemble_loaded_model(
        *,
        model_id: str,
        resolved_path: Path,
        fmt: ModelFormat,
        revision: str | None,
        reader: CheckpointReader,
        shard_count: int,
        arch,
        security_notes: list[str],
    ) -> LoadedModel:
        descs = reader.descriptors()
        total_params = sum(d.num_elements for d in descs.values())
        total_bytes = sum(d.byte_size for d in descs.values())
        dtypes = sorted({d.dtype for d in descs.values()})

        metadata = ModelMetadata(
            model_id=model_id,
            resolved_path=str(resolved_path),
            format=fmt,
            revision=revision,
            tensor_count=len(descs),
            total_parameters=total_params,
            total_bytes=total_bytes,
            shard_count=shard_count,
            dtypes_present=dtypes,
            architecture=arch,
            custom_metadata=reader.custom_metadata(),
            security_notes=security_notes,
        )
        return LoadedModel(metadata=metadata, reader=reader)


def load_model(
    source: str | Path,
    *,
    policy: SecurityPolicy | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> LoadedModel:
    """Convenience entrypoint to load a model checkpoint or directory safely."""
    loader = ModelLoader(policy=policy)
    return loader.load(source, revision=revision, local_files_only=local_files_only)
