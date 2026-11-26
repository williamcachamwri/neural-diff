"""Security controls for path validation, Safetensors binary headers, and pickle inspection."""

from __future__ import annotations

import io
import json
import math
import os
import pickletools
import re
import struct
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.exceptions import MalformedCheckpointError, SecurityValidationError

DEFAULT_MAX_HEADER_BYTES = 100 * 1024 * 1024  # 100 MiB Safetensors header ceiling
DEFAULT_MAX_JSON_CONFIG_BYTES = 16 * 1024 * 1024  # 16 MiB config/index ceiling
DEFAULT_MAX_TENSOR_ELEMENTS = 4_000_000_000  # 4B elements per single tensor ceiling

_HF_REPO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")

_SAFETENSORS_DTYPE_BYTES: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

# Safe globals permitted inside PyTorch `weights_only=True` tensor archives
_ALLOWED_PICKLE_GLOBALS: set[tuple[str, str]] = {
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_tensor"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch", "FloatStorage"),
    ("torch", "DoubleStorage"),
    ("torch", "HalfStorage"),
    ("torch", "BFloat16Storage"),
    ("torch", "LongStorage"),
    ("torch", "IntStorage"),
    ("torch", "ShortStorage"),
    ("torch", "CharStorage"),
    ("torch", "ByteStorage"),
    ("torch", "BoolStorage"),
    ("torch.storage", "_load_from_bytes"),
    ("collections", "OrderedDict"),
    ("_codecs", "encode"),
}


def default_allowed_roots() -> list[Path]:
    """Build default safe filesystem roots (cwd, package root, HF cache, system temp, env roots)."""
    repo_root = Path(__file__).resolve().parent.parent
    roots = [
        Path.cwd().resolve(),
        repo_root.resolve(),
        Path(tempfile.gettempdir()).resolve(),
    ]
    extra_env = os.environ.get("NEURAL_DIFF_ALLOWED_ROOTS", "")
    if extra_env:
        for raw_entry in extra_env.split(os.pathsep):
            cleaned = raw_entry.strip()
            if cleaned:
                roots.append(Path(cleaned).resolve())
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home).resolve())
    default_hf_cache = Path.home() / ".cache" / "huggingface"
    roots.append(default_hf_cache.resolve())
    local_cache = repo_root / ".cache"
    roots.append(local_cache.resolve())
    return roots


@dataclass
class SecurityPolicy:
    """Configurable security guardrails for checkpoint loading and API requests."""

    allowed_roots: list[Path] = field(default_factory=default_allowed_roots)
    allow_pytorch_weights_only: bool = True
    reject_symlinks_outside_roots: bool = True
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES
    max_json_bytes: int = DEFAULT_MAX_JSON_CONFIG_BYTES
    max_tensor_elements: int = DEFAULT_MAX_TENSOR_ELEMENTS

    def is_within_allowed_roots(self, candidate: Path) -> bool:
        resolved = candidate.resolve(strict=False)
        for root in self.allowed_roots:
            root_resolved = root.resolve(strict=False)
            try:
                resolved.relative_to(root_resolved)
                return True
            except ValueError:
                continue
        return False


def is_valid_hf_repo_id(identifier: str) -> bool:
    """Return True if identifier strictly matches `namespace/model_name` without traversal."""
    if not identifier or ".." in identifier or "\\" in identifier or "\x00" in identifier:
        return False
    if identifier.startswith(("/", ".", "~")) or ":" in identifier:
        return False
    return bool(_HF_REPO_ID_PATTERN.match(identifier))


def validate_safe_path(
    raw_path: str | Path,
    policy: SecurityPolicy | None = None,
    *,
    must_exist: bool = True,
) -> Path:
    """Validate and canonicalize a local filesystem path against traversal and device attacks."""
    active_policy = policy or SecurityPolicy()
    text = str(raw_path)

    if not text or not text.strip():
        raise SecurityValidationError("Empty path is not permitted.")

    if "\x00" in text:
        raise SecurityValidationError(
            "Null byte detected in path.",
            details={"raw_path": repr(text)},
        )

    lowered = text.strip().lower()
    if lowered.startswith(("http://", "https://", "file://", "ftp://")):
        raise SecurityValidationError(
            "Remote or URI scheme paths are rejected in local path validation.",
            details={"raw_path": text},
        )

    if text.startswith(("\\\\?\\", "\\\\.\\", "//?/", "//./")) or (
        text.startswith("\\\\") and not text.startswith("\\\\localhost\\")
    ):
        raise SecurityValidationError(
            "Windows UNC or raw device namespace paths are rejected.",
            details={"raw_path": text},
        )

    # Normalize separators to detect explicit parent-directory traversal attempts
    normalized_parts = re.split(r"[\\/]+", text)
    if ".." in normalized_parts:
        candidate_raw = Path(text)
        resolved = (
            (Path.cwd() / candidate_raw).resolve(strict=False)
            if not candidate_raw.is_absolute()
            else candidate_raw.resolve(strict=False)
        )
        if not active_policy.is_within_allowed_roots(resolved):
            raise SecurityValidationError(
                "Path traversal outside allowed root directories is forbidden.",
                details={"raw_path": text, "resolved_path": str(resolved)},
            )
        # Even within roots, reject explicit traversal tokens on raw strings when escaping base
        if not candidate_raw.is_absolute():
            raise SecurityValidationError(
                "Relative parent traversal ('..') is forbidden.",
                details={"raw_path": text},
            )

    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate

    resolved = candidate.resolve(strict=False)
    if not active_policy.is_within_allowed_roots(resolved):
        raise SecurityValidationError(
            "Target path resides outside configured allowed directories.",
            details={
                "resolved_path": str(resolved),
                "allowed_roots": [str(r) for r in active_policy.allowed_roots],
            },
        )

    if candidate.exists() and candidate.is_symlink() and active_policy.reject_symlinks_outside_roots:
        link_target = candidate.resolve(strict=True)
        if not active_policy.is_within_allowed_roots(link_target):
            raise SecurityValidationError(
                "Symlink target escapes allowed directories.",
                details={"path": str(candidate), "target": str(link_target)},
            )

    if must_exist and not resolved.exists():
        from core.exceptions import ModelNotFoundError

        raise ModelNotFoundError(
            f"Path does not exist: {resolved}",
            details={"resolved_path": str(resolved)},
        )

    return resolved


def validate_safe_shard_filename(shard_name: str, base_dir: Path, policy: SecurityPolicy | None = None) -> Path:
    """Ensure a shard filename from `model.safetensors.index.json` stays strictly inside `base_dir`."""
    if not shard_name or "\x00" in shard_name:
        raise SecurityValidationError("Invalid empty or null-containing shard filename.")
    if Path(shard_name).is_absolute() or "\\" in shard_name or ".." in shard_name.split("/"):
        raise SecurityValidationError(
            "Shard index references a path outside the checkpoint directory.",
            details={"shard_name": shard_name},
        )
    resolved_base = base_dir.resolve(strict=True)
    shard_path = (resolved_base / shard_name).resolve(strict=False)
    try:
        shard_path.relative_to(resolved_base)
    except ValueError as exc:
        raise SecurityValidationError(
            "Shard path escaped checkpoint directory.",
            details={"shard_name": shard_name, "base_dir": str(resolved_base)},
        ) from exc
    return validate_safe_path(shard_path, policy=policy, must_exist=True)


def read_validated_json(file_path: Path, max_bytes: int = DEFAULT_MAX_JSON_CONFIG_BYTES) -> dict[str, Any]:
    """Safely parse a JSON configuration or shard index file with size enforcement."""
    try:
        stat = file_path.stat()
    except OSError as exc:
        raise MalformedCheckpointError(f"Unable to stat JSON file {file_path}: {exc}") from exc

    if stat.st_size > max_bytes:
        raise SecurityValidationError(
            f"JSON file exceeds maximum allowed size ({stat.st_size} > {max_bytes} bytes).",
            details={"path": str(file_path), "size": stat.st_size, "max_bytes": max_bytes},
        )

    try:
        raw_text = file_path.read_text(encoding="utf-8")
        parsed = json.loads(raw_text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MalformedCheckpointError(
            f"Malformed JSON in {file_path.name}: {exc}",
            details={"path": str(file_path)},
        ) from exc

    if not isinstance(parsed, dict):
        raise MalformedCheckpointError(
            f"Expected top-level JSON object in {file_path.name}, got {type(parsed).__name__}.",
            details={"path": str(file_path)},
        )
    return parsed


def validate_safetensors_header(
    file_path: Path,
    policy: SecurityPolicy | None = None,
) -> dict[str, Any]:
    """Validate a `.safetensors` binary header before passing to C/Rust memory mappers."""
    active_policy = policy or SecurityPolicy()
    try:
        file_size = file_path.stat().st_size
    except OSError as exc:
        raise MalformedCheckpointError(f"Cannot inspect {file_path}: {exc}") from exc

    if file_size < 8:
        raise MalformedCheckpointError(
            f"Truncated Safetensors file ({file_size} bytes < 8-byte header prefix).",
            details={"path": str(file_path), "file_size": file_size},
        )

    with file_path.open("rb") as fh:
        prefix = fh.read(8)
        (header_len,) = struct.unpack("<Q", prefix)

        if header_len == 0:
            raise MalformedCheckpointError(
                "Safetensors header length cannot be 0.",
                details={"path": str(file_path)},
            )
        if header_len > active_policy.max_header_bytes:
            raise SecurityValidationError(
                f"Safetensors header length ({header_len} bytes) exceeds safety limit ({active_policy.max_header_bytes} bytes).",
                details={"path": str(file_path), "header_len": header_len},
            )
        if 8 + header_len > file_size:
            raise MalformedCheckpointError(
                f"Safetensors header length ({header_len}) exceeds file size ({file_size}).",
                details={"path": str(file_path), "header_len": header_len, "file_size": file_size},
            )

        raw_header = fh.read(header_len)

    try:
        header_obj = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MalformedCheckpointError(
            f"Corrupted Safetensors JSON header in {file_path.name}: {exc}",
            details={"path": str(file_path)},
        ) from exc

    if not isinstance(header_obj, dict):
        raise MalformedCheckpointError(
            "Safetensors header must be a JSON object.",
            details={"path": str(file_path)},
        )

    data_region_bytes = file_size - (8 + header_len)
    spans: list[tuple[int, int, str]] = []

    for key, info in header_obj.items():
        if key == "__metadata__":
            if not isinstance(info, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in info.items()
            ):
                raise MalformedCheckpointError(
                    "Safetensors `__metadata__` must be a string-to-string dictionary.",
                    details={"path": str(file_path)},
                )
            continue

        if not isinstance(info, dict):
            raise MalformedCheckpointError(
                f"Malformed descriptor for tensor '{key}'.",
                details={"path": str(file_path), "tensor": key},
            )

        dtype = info.get("dtype")
        shape = info.get("shape")
        offsets = info.get("data_offsets")

        if dtype not in _SAFETENSORS_DTYPE_BYTES:
            raise MalformedCheckpointError(
                f"Unsupported or invalid Safetensors dtype '{dtype}' for tensor '{key}'.",
                details={"path": str(file_path), "tensor": key, "dtype": dtype},
            )
        if not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0 for d in shape):
            raise MalformedCheckpointError(
                f"Invalid tensor shape {shape} for tensor '{key}'.",
                details={"path": str(file_path), "tensor": key, "shape": shape},
            )

        num_elements = math.prod(shape) if shape else 1
        if num_elements > active_policy.max_tensor_elements:
            raise SecurityValidationError(
                f"Tensor '{key}' element count ({num_elements}) exceeds safety ceiling ({active_policy.max_tensor_elements}).",
                details={"tensor": key, "num_elements": num_elements},
            )

        expected_bytes = num_elements * _SAFETENSORS_DTYPE_BYTES[dtype]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(x, int) and x >= 0 for x in offsets)
        ):
            raise MalformedCheckpointError(
                f"Invalid `data_offsets` for tensor '{key}'.",
                details={"tensor": key, "offsets": offsets},
            )

        begin, end = offsets
        if begin > end or end > data_region_bytes or (end - begin) != expected_bytes:
            raise MalformedCheckpointError(
                f"Out-of-bounds or mismatched byte span [{begin}, {end}] (expected {expected_bytes} bytes within {data_region_bytes}) for tensor '{key}'.",
                details={
                    "tensor": key,
                    "offsets": [begin, end],
                    "expected_bytes": expected_bytes,
                    "data_region_bytes": data_region_bytes,
                },
            )
        spans.append((begin, end, key))

    # Verify non-overlapping spans
    spans.sort(key=lambda item: (item[0], item[1]))
    for idx in range(1, len(spans)):
        prev_begin, prev_end, prev_key = spans[idx - 1]
        cur_begin, cur_end, cur_key = spans[idx]
        if cur_begin < prev_end and prev_end > prev_begin and cur_end > cur_begin:
            raise MalformedCheckpointError(
                f"Overlapping data buffers detected between '{prev_key}' and '{cur_key}'.",
                details={"prev": [prev_begin, prev_end], "current": [cur_begin, cur_end]},
            )

    return header_obj


def inspect_pytorch_archive_safety(file_path: Path) -> list[str]:
    """Statically scan pickle streams inside a PyTorch archive for disallowed globals before loading."""
    findings: list[str] = []
    pickle_blobs: list[tuple[str, bytes]] = []

    if zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path, "r") as zf:
            for member in zf.infolist():
                if member.filename.endswith(".pkl"):
                    if member.file_size > DEFAULT_MAX_JSON_CONFIG_BYTES:
                        raise SecurityValidationError(
                            f"Oversized pickle stream '{member.filename}' inside PyTorch archive.",
                            details={"member": member.filename, "size": member.file_size},
                        )
                    pickle_blobs.append((member.filename, zf.read(member.filename)))
    else:
        size = file_path.stat().st_size
        if size > DEFAULT_MAX_HEADER_BYTES:
            # Do not read massive raw non-zip pickle files into memory; reject directly
            raise SecurityValidationError(
                "Legacy non-zip pickle file exceeds safe static-inspection size; convert to Safetensors.",
                details={"path": str(file_path), "size": size},
            )
        pickle_blobs.append((file_path.name, file_path.read_bytes()))

    if not pickle_blobs:
        raise MalformedCheckpointError(
            f"No pickle metadata stream found in PyTorch checkpoint {file_path.name}.",
            details={"path": str(file_path)},
        )

    for blob_name, raw_bytes in pickle_blobs:
        stream = io.BytesIO(raw_bytes)
        stack_strings: list[str] = []
        try:
            for opcode, arg, _pos in pickletools.genops(stream):
                opname = opcode.name
                if opname in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "STRING") and isinstance(arg, str):
                    stack_strings.append(arg)
                    if len(stack_strings) > 8:
                        stack_strings.pop(0)
                elif opname == "GLOBAL" and isinstance(arg, str):
                    parts = arg.split(" ", 1)
                    pair = (parts[0], parts[1]) if len(parts) == 2 else (arg, "")
                    if pair not in _ALLOWED_PICKLE_GLOBALS:
                        raise SecurityValidationError(
                            f"Unsafe pickle GLOBAL '{pair[0]}.{pair[1]}' rejected in {blob_name}.",
                            details={"global": f"{pair[0]}.{pair[1]}", "archive": str(file_path)},
                        )
                    findings.append(f"Verified safe global {pair[0]}.{pair[1]}")
                elif opname == "STACK_GLOBAL":
                    if len(stack_strings) >= 2:
                        mod_name, attr_name = stack_strings[-2], stack_strings[-1]
                        if (mod_name, attr_name) not in _ALLOWED_PICKLE_GLOBALS:
                            raise SecurityValidationError(
                                f"Unsafe pickle STACK_GLOBAL '{mod_name}.{attr_name}' rejected in {blob_name}.",
                                details={"global": f"{mod_name}.{attr_name}", "archive": str(file_path)},
                            )
                        findings.append(f"Verified safe stack_global {mod_name}.{attr_name}")
                    else:
                        raise SecurityValidationError(
                            f"Opaque STACK_GLOBAL opcode rejected in {blob_name}.",
                            details={"archive": str(file_path)},
                        )
                elif opname in ("INST", "OBJ", "EXT1", "EXT2", "EXT4"):
                    raise SecurityValidationError(
                        f"Disallowed pickle opcode '{opname}' detected in {blob_name}.",
                        details={"opcode": opname, "archive": str(file_path)},
                    )
        except SecurityValidationError:
            raise
        except Exception as exc:
            raise MalformedCheckpointError(
                f"Malformed pickle stream in {file_path.name}: {exc}",
                details={"path": str(file_path)},
            ) from exc

    return findings
