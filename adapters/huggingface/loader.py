"""Hugging Face local directory and Hub snapshot resolver (zero remote code execution)."""

from __future__ import annotations

import os
import re
from pathlib import Path

from core.exceptions import (
    MalformedCheckpointError,
    ModelAccessBlockedError,
    ModelNotFoundError,
    UnsupportedFormatError,
)
from core.loaders.base import CheckpointReader
from core.security import SecurityPolicy, is_valid_hf_repo_id
from core.types import ModelFormat

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def detect_snapshot_revision(resolved_dir: Path) -> str | None:
    """Extract 40-char git revision hash if path resides inside an HF Hub `snapshots/<sha>` folder."""
    if _SHA_RE.match(resolved_dir.name) and resolved_dir.parent.name == "snapshots":
        return resolved_dir.name
    return None


def resolve_hf_hub_to_local_dir(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
) -> tuple[Path, str | None]:
    """Download or resolve a Hugging Face Hub repository strictly excluding executable code."""
    if not is_valid_hf_repo_id(repo_id):
        raise ModelNotFoundError(
            f"Invalid Hugging Face repository identifier: '{repo_id}'.",
            details={"repo_id": repo_id},
        )

    effective_cache = cache_dir
    if effective_cache is None:
        env_hf = os.environ.get("HF_HOME")
        if env_hf:
            effective_cache = Path(env_hf) / "hub"
        else:
            repo_root = Path(__file__).resolve().parent.parent.parent
            effective_cache = repo_root / ".cache" / "huggingface" / "hub"
    effective_cache.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import (
            GatedRepoError,
            LocalEntryNotFoundError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError as exc:
        raise UnsupportedFormatError(f"huggingface_hub is not available: {exc}") from exc

    try:
        local_dir_str = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(effective_cache),
            local_files_only=local_files_only,
            allow_patterns=[
                "*.safetensors",
                "*.json",
                "tokenizer.model",
                "vocab.txt",
                "merges.txt",
                "*.bin",
            ],
            ignore_patterns=[
                "*.py",
                "*.pyc",
                "*.so",
                "*.dll",
                "*.dylib",
                "*.msgpack",
                "*.h5",
                "*.ot",
                "*.onnx",
            ],
        )
    except GatedRepoError as exc:
        raise ModelAccessBlockedError(
            f"Model '{repo_id}' is gated on Hugging Face Hub and requires an accepted license + HF_TOKEN.",
            details={"repo_id": repo_id, "upstream_error": str(exc)},
        ) from exc
    except RepositoryNotFoundError as exc:
        msg = str(exc)
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            raise ModelAccessBlockedError(
                f"Access blocked (HTTP 401/403) for Hugging Face model '{repo_id}'.",
                details={"repo_id": repo_id, "upstream_error": msg},
            ) from exc
        raise ModelNotFoundError(
            f"Hugging Face repository '{repo_id}' was not found.",
            details={"repo_id": repo_id, "upstream_error": msg},
        ) from exc
    except (RevisionNotFoundError, LocalEntryNotFoundError) as exc:
        raise ModelNotFoundError(
            f"Could not resolve '{repo_id}' (revision={revision}, local_files_only={local_files_only}): {exc}",
            details={"repo_id": repo_id, "revision": revision},
        ) from exc
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            raise ModelAccessBlockedError(
                f"Access blocked for Hugging Face model '{repo_id}': {msg}",
                details={"repo_id": repo_id},
            ) from exc
        raise MalformedCheckpointError(
            f"Failed resolving Hugging Face model '{repo_id}': {exc}",
            details={"repo_id": repo_id},
        ) from exc

    resolved_dir = Path(local_dir_str).resolve()
    detected_rev = detect_snapshot_revision(resolved_dir) or revision
    return resolved_dir, detected_rev


def open_directory_checkpoint(
    directory: Path,
    policy: SecurityPolicy | None = None,
) -> tuple[CheckpointReader, ModelFormat, int, list[str]]:
    """Open checkpoints inside a validated directory, strictly preferring Safetensors over pickle."""
    from adapters.pytorch.loader import SafePyTorchReader
    from adapters.safetensors.loader import MultiSafetensorsReader, SingleSafetensorsReader

    active_policy = policy or SecurityPolicy()
    notes: list[str] = ["Automatic remote/custom Python code execution disabled (trust_remote_code=False)."]

    py_files = list(directory.glob("*.py"))
    if py_files:
        notes.append(
            f"Ignored {len(py_files)} custom Python file(s) in checkpoint directory without executing them."
        )

    st_index = directory / "model.safetensors.index.json"
    if st_index.is_file():
        reader = MultiSafetensorsReader.from_index_file(st_index, policy=active_policy)
        notes.append("Loaded sharded Safetensors via model.safetensors.index.json.")
        return reader, ModelFormat.SAFETENSORS_SHARDED, reader.shard_count, notes

    st_single = directory / "model.safetensors"
    if st_single.is_file():
        s_reader = SingleSafetensorsReader(st_single, policy=active_policy)
        notes.append("Loaded Safetensors via model.safetensors.")
        return s_reader, ModelFormat.HF_LOCAL_DIR, 1, notes

    st_files = sorted(directory.glob("*.safetensors"))
    if len(st_files) == 1:
        s_reader = SingleSafetensorsReader(st_files[0], policy=active_policy)
        notes.append(f"Loaded Safetensors file {st_files[0].name}.")
        return s_reader, ModelFormat.HF_LOCAL_DIR, 1, notes
    if len(st_files) > 1:
        m_reader = MultiSafetensorsReader.from_shard_files(st_files, policy=active_policy)
        notes.append(f"Loaded {len(st_files)} Safetensors shard files.")
        return m_reader, ModelFormat.SAFETENSORS_SHARDED, m_reader.shard_count, notes

    pt_index = directory / "pytorch_model.bin.index.json"
    if pt_index.is_file():
        p_reader = SafePyTorchReader.from_index_file(pt_index, policy=active_policy)
        notes.extend(p_reader.security_notes)
        return p_reader, ModelFormat.PYTORCH_SAFE, p_reader.shard_count, notes

    pt_single = directory / "pytorch_model.bin"
    if pt_single.is_file():
        p_reader = SafePyTorchReader.from_single_file(pt_single, policy=active_policy)
        notes.extend(p_reader.security_notes)
        return p_reader, ModelFormat.PYTORCH_SAFE, 1, notes

    pt_candidates = sorted(
        [*directory.glob("*.pt"), *directory.glob("*.pth"), *directory.glob("*.bin")]
    )
    if len(pt_candidates) == 1:
        p_reader = SafePyTorchReader.from_single_file(pt_candidates[0], policy=active_policy)
        notes.extend(p_reader.security_notes)
        return p_reader, ModelFormat.PYTORCH_SAFE, 1, notes
    if len(pt_candidates) > 1:
        p_reader = SafePyTorchReader(shard_files=pt_candidates, policy=active_policy)
        notes.extend(p_reader.security_notes)
        return p_reader, ModelFormat.PYTORCH_SAFE, p_reader.shard_count, notes

    raise MalformedCheckpointError(
        f"No supported checkpoint files (`.safetensors`, `.pt`, `.pth`, `.bin`) found in directory '{directory}'.",
        details={"directory": str(directory)},
    )
