"""Deterministic synthetic checkpoint builders for unit, integration, and regression tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn


class TinyTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int = 16, intermediate_size: int = 32) -> None:
        super().__init__()
        self.self_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(intermediate_size, hidden_size, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.self_attn(x)
        return h + self.mlp(h)


class TinyCausalModel(nn.Module):
    """Minimal deterministic causal model with `.layers.{i}.self_attn` and `.layers.{i}.mlp`."""

    def __init__(self, vocab_size: int = 64, hidden_size: int = 16, num_layers: int = 2) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [TinyTransformerBlock(hidden_size=hidden_size, intermediate_size=hidden_size * 2) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for block in self.layers:
            x = block(x)
        return self.lm_head(x)


def make_synthetic_hf_checkpoint_pair(
    base_dir: Path,
    *,
    vocab_size: int = 128,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    perturbation_scale: float = 0.05,
) -> tuple[Path, Path]:
    """Create two deterministic Qwen2/Llama-shaped Safetensors directories on disk."""
    dir_a = base_dir / "synth_model_a"
    dir_b = base_dir / "synth_model_b"
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    head_dim = hidden_size // num_heads
    kv_dim = num_kv_heads * head_dim

    cfg = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "rms_norm_eps": 1e-6,
        "torch_dtype": "float32",
    }
    (dir_a / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (dir_b / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    g = torch.Generator().manual_seed(1234)
    tensors_a: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn((vocab_size, hidden_size), generator=g),
        "model.norm.weight": torch.ones((hidden_size,)),
        "lm_head.weight": torch.randn((vocab_size, hidden_size), generator=g),
    }
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}"
        tensors_a[f"{prefix}.input_layernorm.weight"] = torch.ones((hidden_size,))
        tensors_a[f"{prefix}.self_attn.q_proj.weight"] = torch.randn((hidden_size, hidden_size), generator=g)
        tensors_a[f"{prefix}.self_attn.k_proj.weight"] = torch.randn((kv_dim, hidden_size), generator=g)
        tensors_a[f"{prefix}.self_attn.v_proj.weight"] = torch.randn((kv_dim, hidden_size), generator=g)
        tensors_a[f"{prefix}.self_attn.o_proj.weight"] = torch.randn((hidden_size, hidden_size), generator=g)
        tensors_a[f"{prefix}.post_attention_layernorm.weight"] = torch.ones((hidden_size,))
        tensors_a[f"{prefix}.mlp.gate_proj.weight"] = torch.randn((intermediate_size, hidden_size), generator=g)
        tensors_a[f"{prefix}.mlp.up_proj.weight"] = torch.randn((intermediate_size, hidden_size), generator=g)
        tensors_a[f"{prefix}.mlp.down_proj.weight"] = torch.randn((hidden_size, intermediate_size), generator=g)

    g_delta = torch.Generator().manual_seed(5678)
    tensors_b: dict[str, torch.Tensor] = {}
    for name, t in tensors_a.items():
        if "layers.0.input_layernorm" in name:
            # Leave one tensor strictly identical
            tensors_b[name] = t.clone()
        else:
            noise = torch.randn(t.shape, generator=g_delta) * perturbation_scale * (1.5 if "layers.1" in name else 0.5)
            tensors_b[name] = t + noise

    save_file(tensors_a, str(dir_a / "model.safetensors"), metadata={"source": "synthetic_fixture_a"})
    save_file(tensors_b, str(dir_b / "model.safetensors"), metadata={"source": "synthetic_fixture_b"})
    return dir_a, dir_b


def make_sharded_safetensors_checkpoint(
    target_dir: Path,
    tensors: dict[str, torch.Tensor],
) -> Path:
    """Split `tensors` across two `.safetensors` shards with a valid `model.safetensors.index.json`."""
    target_dir.mkdir(parents=True, exist_ok=True)
    items = list(tensors.items())
    mid = max(1, len(items) // 2)
    shard1_tensors = dict(items[:mid])
    shard2_tensors = dict(items[mid:])

    s1_name = "model-00001-of-00002.safetensors"
    s2_name = "model-00002-of-00002.safetensors"
    save_file(shard1_tensors, str(target_dir / s1_name))
    if shard2_tensors:
        save_file(shard2_tensors, str(target_dir / s2_name))

    weight_map: dict[str, str] = {}
    for k in shard1_tensors:
        weight_map[k] = s1_name
    for k in shard2_tensors:
        weight_map[k] = s2_name

    index_path = target_dir / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"metadata": {"total_shards": "2"}, "weight_map": weight_map}, indent=2),
        encoding="utf-8",
    )
    return index_path
