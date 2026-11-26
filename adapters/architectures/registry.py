"""Architecture metadata extraction and canonical tensor-role mapping."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core.security import read_validated_json
from core.types import ArchitectureMetadata, CanonicalRole

_LAYER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:^|\.)layers\.(\d+)\."),
    re.compile(r"(?:^|\.)h\.(\d+)\."),
    re.compile(r"(?:^|\.)layer\.(\d+)\."),
    re.compile(r"(?:^|\.)blocks\.(\d+)\."),
    re.compile(r"(?:^|\.)block\.(\d+)\."),
    re.compile(r"(?:^|\.)decoder\.layers\.(\d+)\."),
    re.compile(r"(?:^|\.)encoder\.layers\.(\d+)\."),
)


def classify_tensor_role(tensor_name: str) -> tuple[int | None, CanonicalRole]:
    """Infer `(layer_index, CanonicalRole)` from a parameter name without forcing false matches."""
    lower = tensor_name.lower()
    layer_idx: int | None = None
    for pat in _LAYER_PATTERNS:
        m = pat.search(lower)
        if m:
            layer_idx = int(m.group(1))
            break

    # Embeddings & Language Model Head
    if any(
        tok in lower
        for tok in (
            "embed_tokens",
            "wte",
            "wpe",
            "word_embeddings",
            "position_embeddings",
            "tok_embeddings",
            "embed_positions",
        )
    ):
        return (layer_idx, CanonicalRole.EMBEDDING)

    if any(tok in lower for tok in ("lm_head", "embed_out", "output.weight")):
        if layer_idx is None:
            return (None, CanonicalRole.LM_HEAD)

    # Normalization layers
    if any(
        tok in lower
        for tok in (
            "layernorm",
            "layer_norm",
            "rmsnorm",
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            ".ln_1",
            ".ln_2",
            ".ln_f",
            "final_layernorm",
            "model.norm.",
            "transformer.norm.",
            "q_norm",
            "k_norm",
        )
    ) or lower in ("norm.weight", "norm.bias"):
        return (layer_idx, CanonicalRole.NORM)

    # Attention projections
    if any(tok in lower for tok in ("qkv_proj", "query_key_value", "c_attn", "in_proj_weight", "wqkv")):
        return (layer_idx, CanonicalRole.ATTN_QKV_FUSED)
    if any(tok in lower for tok in ("q_proj", ".wq.", "self.query.", "q_lin")):
        return (layer_idx, CanonicalRole.ATTN_Q)
    if any(tok in lower for tok in ("k_proj", ".wk.", "self.key.", "k_lin")):
        return (layer_idx, CanonicalRole.ATTN_K)
    if any(tok in lower for tok in ("v_proj", ".wv.", "self.value.", "v_lin")):
        return (layer_idx, CanonicalRole.ATTN_V)
    if any(
        tok in lower
        for tok in ("o_proj", "out_proj", ".wo.", "attn.c_proj", "attention.output.dense", "dense_4h_to_h")
    ) and "mlp" not in lower:
        return (layer_idx, CanonicalRole.ATTN_O)

    # MLP / Feed-Forward projections
    if any(tok in lower for tok in ("gate_proj", ".w1.", "wi_0")):
        return (layer_idx, CanonicalRole.MLP_GATE)
    if any(tok in lower for tok in ("up_proj", ".w3.", "wi_1", "fc1", "c_fc", "intermediate.dense")):
        return (layer_idx, CanonicalRole.MLP_UP)
    if any(tok in lower for tok in ("down_proj", ".w2.", "wo.", "fc2", "mlp.c_proj")):
        return (layer_idx, CanonicalRole.MLP_DOWN)
    if "mlp" in lower or "ffn" in lower:
        return (layer_idx, CanonicalRole.MLP_FC)

    return (layer_idx, CanonicalRole.OTHER)


def extract_architecture_metadata(directory: Path | None) -> ArchitectureMetadata:
    """Extract architecture and tokenizer metadata from a local checkpoint directory if present."""
    if directory is None or not directory.is_dir():
        return ArchitectureMetadata()

    config_data: dict[str, Any] = {}
    config_path = directory / "config.json"
    if config_path.is_file():
        config_data = read_validated_json(config_path)

    # Nested text_config support (e.g. Gemma 3 / multimodal wrappers)
    effective_cfg: dict[str, Any] = dict(config_data)
    if isinstance(config_data.get("text_config"), dict):
        merged = dict(config_data["text_config"])
        for k, v in config_data.items():
            if k not in merged and k != "text_config":
                merged[k] = v
        effective_cfg = merged

    archs = config_data.get("architectures")
    arch_class = archs[0] if isinstance(archs, list) and archs and isinstance(archs[0], str) else None

    hidden_size = _first_int(effective_cfg, ("hidden_size", "n_embd", "d_model"))
    intermediate_size = _first_int(effective_cfg, ("intermediate_size", "n_inner", "d_ff", "ffn_dim"))
    num_layers = _first_int(effective_cfg, ("num_hidden_layers", "n_layer", "num_layers"))
    num_heads = _first_int(effective_cfg, ("num_attention_heads", "n_head", "num_heads"))
    num_kv_heads = _first_int(effective_cfg, ("num_key_value_heads", "num_kv_heads")) or num_heads
    vocab_size = _first_int(effective_cfg, ("vocab_size",))
    max_pos = _first_int(effective_cfg, ("max_position_embeddings", "n_positions", "max_seq_len"))

    rms_eps_raw = effective_cfg.get("rms_norm_eps", effective_cfg.get("layer_norm_epsilon"))
    rms_eps = float(rms_eps_raw) if isinstance(rms_eps_raw, (int, float)) else None

    tie_emb = effective_cfg.get("tie_word_embeddings")
    tie_word_embeddings = bool(tie_emb) if isinstance(tie_emb, bool) else None

    dtype_val = effective_cfg.get("torch_dtype", effective_cfg.get("dtype"))
    torch_dtype = str(dtype_val) if dtype_val is not None else None

    tokenizer_class: str | None = None
    tokenizer_vocab_size: int | None = None
    tok_cfg_path = directory / "tokenizer_config.json"
    if tok_cfg_path.is_file():
        try:
            tok_cfg = read_validated_json(tok_cfg_path)
            tc = tok_cfg.get("tokenizer_class")
            if isinstance(tc, str):
                tokenizer_class = tc
        except Exception:
            pass

    tok_json_path = directory / "tokenizer.json"
    if tok_json_path.is_file():
        try:
            tok_json = read_validated_json(tok_json_path, max_bytes=64 * 1024 * 1024)
            model_sec = tok_json.get("model")
            if isinstance(model_sec, dict) and isinstance(model_sec.get("vocab"), dict):
                base_vocab_len = len(model_sec["vocab"])
                added_len = len(tok_json.get("added_tokens", [])) if isinstance(tok_json.get("added_tokens"), list) else 0
                tokenizer_vocab_size = base_vocab_len + added_len
        except Exception:
            pass

    sanitized_raw = {
        k: v
        for k, v in config_data.items()
        if isinstance(v, (str, int, float, bool, type(None))) or (isinstance(v, list) and len(v) <= 16)
    }

    return ArchitectureMetadata(
        architecture_class=arch_class,
        model_type=str(effective_cfg["model_type"]) if "model_type" in effective_cfg else None,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        vocab_size=vocab_size,
        max_position_embeddings=max_pos,
        rms_norm_eps=rms_eps,
        tie_word_embeddings=tie_word_embeddings,
        torch_dtype=torch_dtype,
        bos_token_id=effective_cfg.get("bos_token_id"),
        eos_token_id=effective_cfg.get("eos_token_id"),
        tokenizer_class=tokenizer_class,
        tokenizer_vocab_size=tokenizer_vocab_size,
        raw_config=sanitized_raw,
    )


def _first_int(cfg: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        val = cfg.get(key)
        if isinstance(val, int) and not isinstance(val, bool):
            return val
    return None
