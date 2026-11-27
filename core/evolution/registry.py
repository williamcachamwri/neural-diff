"""Checkpoint Registry inspector — SHA-256 fingerprinting, LoRA/quant/training metadata & duplicate detection."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import torch
import transformers

import safetensors
from core.evolution.types import (
    CheckpointValidationStatus,
    RegisteredCheckpoint,
    TransformationType,
    normalize_transformation_type,
)
from core.loaders.model_loader import ModelLoader
from core.security import SecurityPolicy, read_validated_json

HASH_CHUNK_BYTES = 4 * 1024 * 1024  # 4 MiB streaming chunks


def compute_checkpoint_sha256_and_size(resolved_path: Path) -> tuple[str, int]:
    """Compute streaming SHA-256 fingerprint and total disk byte size without loading into RAM."""
    hasher = hashlib.sha256()
    total_bytes = 0

    if resolved_path.is_file():
        files = [resolved_path]
    else:
        weight_exts = (".safetensors", ".pt", ".pth", ".bin")
        weight_files = sorted(
            p for p in resolved_path.iterdir() if p.is_file() and p.name.lower().endswith(weight_exts)
        )
        cfg_files = sorted(
            p
            for p in resolved_path.iterdir()
            if p.is_file()
            and p.name in ("config.json", "adapter_config.json", "model.safetensors.index.json")
        )
        files = cfg_files + weight_files

    for fp in files:
        # Include filename boundary so multi-file layouts cannot collide by concatenation
        hasher.update(fp.name.encode("utf-8") + b"\x00")
        with fp.open("rb") as fh:
            while True:
                chunk = fh.read(HASH_CHUNK_BYTES)
                if not chunk:
                    break
                total_bytes += len(chunk)
                hasher.update(chunk)

    return hasher.hexdigest(), total_bytes


def compute_tensor_metadata_hash(descriptors: dict[str, Any]) -> str:
    """Compute canonical SHA-256 hash over sorted `(tensor_name, shape, dtype)` descriptors."""
    canonical_items = [
        {
            "name": name,
            "shape": list(desc.shape),
            "dtype": desc.dtype,
            "num_elements": desc.num_elements,
        }
        for name, desc in sorted(descriptors.items(), key=lambda kv: kv[0])
    ]
    payload = json.dumps(canonical_items, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _extract_optional_directory_metadata(
    directory: Path | None,
    raw_config: dict[str, Any],
    custom_meta: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Extract real tokenizer, quantization, training, and LoRA adapter metadata without inventing values."""
    tokenizer_meta: dict[str, Any] = {}
    quantization_meta: dict[str, Any] = {}
    training_meta: dict[str, Any] = {}
    lora_meta: dict[str, Any] = {}

    if isinstance(raw_config.get("quantization_config"), dict):
        quantization_meta.update(raw_config["quantization_config"])

    if custom_meta:
        for k in ("format", "pt_version", "quantization", "training_step", "global_step"):
            if k in custom_meta:
                training_meta[f"header_{k}"] = custom_meta[k]

    if directory is None or not directory.is_dir():
        return tokenizer_meta, quantization_meta, training_meta, lora_meta

    # Full config.json nested quantization_config extraction
    cfg_json_path = directory / "config.json"
    if cfg_json_path.is_file():
        try:
            full_cfg = read_validated_json(cfg_json_path)
            if isinstance(full_cfg.get("quantization_config"), dict):
                quantization_meta.update(full_cfg["quantization_config"])
        except Exception as exc:
            _ = exc

    # Tokenizer metadata
    tok_cfg_path = directory / "tokenizer_config.json"
    if tok_cfg_path.is_file():
        try:
            tok_cfg = read_validated_json(tok_cfg_path)
            for key in ("tokenizer_class", "model_max_length", "padding_side", "clean_up_tokenization_spaces"):
                if key in tok_cfg and isinstance(tok_cfg[key], (str, int, float, bool)):
                    tokenizer_meta[key] = tok_cfg[key]
            tokenizer_meta["has_chat_template"] = bool(tok_cfg.get("chat_template"))
        except Exception as exc:
            _ = exc

    # Standalone quantize_config.json (GPTQ / AWQ convention)
    q_cfg_path = directory / "quantize_config.json"
    if q_cfg_path.is_file():
        try:
            q_cfg = read_validated_json(q_cfg_path)
            for key in ("bits", "group_size", "desc_act", "sym", "quant_method", "zero_point"):
                if key in q_cfg and isinstance(q_cfg[key], (str, int, float, bool)):
                    quantization_meta[key] = q_cfg[key]
        except Exception as exc:
            _ = exc

    # PEFT LoRA / QLoRA adapter_config.json
    adapter_cfg_path = directory / "adapter_config.json"
    if adapter_cfg_path.is_file():
        try:
            a_cfg = read_validated_json(adapter_cfg_path)
            for key in (
                "peft_type",
                "base_model_name_or_path",
                "r",
                "lora_alpha",
                "lora_dropout",
                "bias",
                "task_type",
                "use_rslora",
                "use_dora",
            ):
                if key in a_cfg and isinstance(a_cfg[key], (str, int, float, bool, type(None))):
                    lora_meta[key] = a_cfg[key]
            if isinstance(a_cfg.get("target_modules"), list):
                lora_meta["target_modules"] = [str(m) for m in a_cfg["target_modules"][:32]]
        except Exception as exc:
            _ = exc

    # Trainer state (Hugging Face Trainer convention)
    trainer_state_path = directory / "trainer_state.json"
    if trainer_state_path.is_file():
        try:
            t_state = read_validated_json(trainer_state_path)
            for key in ("epoch", "global_step", "max_steps", "best_metric", "total_flos"):
                if key in t_state and isinstance(t_state[key], (int, float, str)):
                    training_meta[key] = t_state[key]
        except Exception as exc:
            _ = exc

    return tokenizer_meta, quantization_meta, training_meta, lora_meta


def inspect_and_build_checkpoint_record(
    *,
    project_id: str,
    source: str | Path,
    version_label: str | None = None,
    model_name: str | None = None,
    parent_checkpoint_ids: list[str] | None = None,
    transformation_type: str | TransformationType | None = None,
    transformation_metadata: dict[str, Any] | None = None,
    training_metadata_override: dict[str, Any] | None = None,
    quantization_metadata_override: dict[str, Any] | None = None,
    experiment_notes: str = "",
    existing_project_checkpoints: list[RegisteredCheckpoint] | None = None,
    policy: SecurityPolicy | None = None,
    local_files_only: bool = False,
    checkpoint_id: str | None = None,
) -> RegisteredCheckpoint:
    """Inspect a real checkpoint using `ModelLoader`, compute SHA-256, and detect duplicates."""
    active_policy = policy or SecurityPolicy()
    loader = ModelLoader(policy=active_policy)

    with loader.load(source, local_files_only=local_files_only) as handle:
        meta = handle.metadata
        resolved = Path(meta.resolved_path)
        ckpt_dir = resolved if resolved.is_dir() else resolved.parent

        sha256_hex, disk_bytes = compute_checkpoint_sha256_and_size(resolved)
        tensor_meta_hash = compute_tensor_metadata_hash(handle.tensors)

        tok_meta, quant_meta, train_meta, lora_meta = _extract_optional_directory_metadata(
            ckpt_dir,
            meta.architecture.raw_config,
            meta.custom_metadata,
        )
        if meta.architecture.tokenizer_class and "tokenizer_class" not in tok_meta:
            tok_meta["tokenizer_class"] = meta.architecture.tokenizer_class
        if meta.architecture.tokenizer_vocab_size is not None:
            tok_meta["vocab_size"] = meta.architecture.tokenizer_vocab_size

        if quantization_metadata_override:
            quant_meta.update(quantization_metadata_override)
        if training_metadata_override:
            train_meta.update(training_metadata_override)

        # Infer transformation type only when unspecified and unambiguous from disk artifacts
        parents = list(parent_checkpoint_ids or [])
        if transformation_type is not None:
            norm_transform = normalize_transformation_type(transformation_type)
        elif lora_meta:
            norm_transform = TransformationType.QLORA if quant_meta else TransformationType.LORA
        elif len(parents) > 1:
            norm_transform = TransformationType.MODEL_MERGING
        elif not parents:
            norm_transform = TransformationType.BASE_MODEL
        elif quant_meta:
            norm_transform = TransformationType.QUANTIZATION
        else:
            norm_transform = TransformationType.UNKNOWN_TRANSFORMATION

        # Duplicate detection by exact content SHA-256 or identical tensor metadata + weights
        duplicate_of: str | None = None
        for existing in existing_project_checkpoints or []:
            if existing.sha256_fingerprint == sha256_hex:
                duplicate_of = existing.checkpoint_id
                break

        is_lora_only = bool(lora_meta) and all(
            "lora_" in tname.lower() or "modules_to_save" in tname.lower()
            for tname in handle.tensors
        )
        if duplicate_of is not None:
            val_status = CheckpointValidationStatus.DUPLICATE_CONTENT
        elif is_lora_only:
            val_status = CheckpointValidationStatus.ADAPTER_ONLY
        else:
            val_status = CheckpointValidationStatus.VERIFIED

        inferred_layers = meta.architecture.num_hidden_layers
        if inferred_layers is None:
            layer_indices = [
                d.layer_index for d in handle.tensors.values() if d.layer_index is not None
            ]
            if layer_indices:
                inferred_layers = max(layer_indices) + 1

        cid = checkpoint_id or f"ckpt-{uuid.uuid4().hex[:10]}"
        eff_name = (model_name or meta.model_id).strip()
        eff_label = (version_label or eff_name).strip()

        framework_info = {
            "format": meta.format.value,
            "torch_observer": torch.__version__,
            "safetensors_observer": safetensors.__version__,
            "transformers_observer": transformers.__version__,
        }
        if meta.revision:
            framework_info["source_revision"] = meta.revision

        return RegisteredCheckpoint(
            checkpoint_id=cid,
            project_id=project_id,
            version_label=eff_label,
            model_name=eff_name,
            source_path=str(source),
            resolved_path=str(resolved),
            architecture=meta.architecture.architecture_class,
            model_family=meta.architecture.model_type,
            parameter_count=meta.total_parameters,
            layer_count=inferred_layers,
            hidden_size=meta.architecture.hidden_size,
            intermediate_size=meta.architecture.intermediate_size,
            num_attention_heads=meta.architecture.num_attention_heads,
            num_key_value_heads=meta.architecture.num_key_value_heads,
            vocab_size=meta.architecture.vocab_size,
            tensor_count=meta.tensor_count,
            checkpoint_format=meta.format.value,
            dtypes_present=list(meta.dtypes_present),
            file_size_bytes=disk_bytes or meta.total_bytes,
            sha256_fingerprint=sha256_hex,
            tensor_metadata_hash=tensor_meta_hash,
            parent_checkpoint_ids=parents,
            transformation_type=norm_transform,
            transformation_metadata=dict(transformation_metadata or {}),
            created_at_utc=dt.datetime.now(dt.UTC).isoformat(),
            framework_info=framework_info,
            tokenizer_metadata=tok_meta,
            quantization_metadata=quant_meta,
            training_metadata=train_meta,
            lora_metadata=lora_meta,
            experiment_notes=experiment_notes,
            validation_status=val_status,
            duplicate_of_checkpoint_id=duplicate_of,
            security_notes=list(meta.security_notes),
        )


def revalidate_checkpoint_health(
    ckpt: RegisteredCheckpoint,
    *,
    policy: SecurityPolicy | None = None,
) -> tuple[CheckpointValidationStatus, str | None]:
    """Verify a previously registered checkpoint still exists, passes header validation, and matches SHA-256."""
    active_policy = policy or SecurityPolicy()
    loader = ModelLoader(policy=active_policy)
    try:
        with loader.load(ckpt.source_path, local_files_only=True) as handle:
            resolved = Path(handle.metadata.resolved_path)
            sha_now, _ = compute_checkpoint_sha256_and_size(resolved)
            if sha_now != ckpt.sha256_fingerprint:
                return (
                    CheckpointValidationStatus.CORRUPTED_OR_MISSING,
                    f"SHA-256 mismatch: expected {ckpt.sha256_fingerprint[:12]}, got {sha_now[:12]}.",
                )
            return ckpt.validation_status, None
    except Exception as exc:
        return CheckpointValidationStatus.CORRUPTED_OR_MISSING, str(exc)
