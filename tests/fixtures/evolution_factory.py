"""Deterministic multi-generation checkpoint builders with analytically known deltas for Evolution tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def _write_qwen_dir(
    target_dir: Path,
    tensors: dict[str, torch.Tensor],
    *,
    vocab_size: int = 64,
    hidden_size: int = 16,
    intermediate_size: int = 32,
    num_layers: int = 3,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    extra_config: dict | None = None,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
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
        "tie_word_embeddings": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "torch_dtype": "float32",
    }
    if extra_config:
        cfg.update(extra_config)
    (target_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    save_file(tensors, str(target_dir / "model.safetensors"))
    return target_dir


def build_base_qwen_tensors(
    *,
    seed: int = 101,
    vocab_size: int = 64,
    hidden_size: int = 16,
    intermediate_size: int = 32,
    num_layers: int = 3,
    num_heads: int = 4,
    num_kv_heads: int = 2,
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    kv_dim = (hidden_size // num_heads) * num_kv_heads
    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn((vocab_size, hidden_size), generator=g) * 0.2,
        "model.norm.weight": torch.ones((hidden_size,), dtype=torch.float32),
        "lm_head.weight": torch.randn((vocab_size, hidden_size), generator=g) * 0.2,
    }
    for layer in range(num_layers):
        p = f"model.layers.{layer}"
        tensors[f"{p}.input_layernorm.weight"] = torch.ones((hidden_size,), dtype=torch.float32)
        tensors[f"{p}.post_attention_layernorm.weight"] = torch.ones((hidden_size,), dtype=torch.float32)
        tensors[f"{p}.self_attn.q_proj.weight"] = torch.randn((hidden_size, hidden_size), generator=g) * 0.2
        tensors[f"{p}.self_attn.k_proj.weight"] = torch.randn((kv_dim, hidden_size), generator=g) * 0.2
        tensors[f"{p}.self_attn.v_proj.weight"] = torch.randn((kv_dim, hidden_size), generator=g) * 0.2
        tensors[f"{p}.self_attn.o_proj.weight"] = torch.randn((hidden_size, hidden_size), generator=g) * 0.2
        tensors[f"{p}.mlp.gate_proj.weight"] = torch.randn((intermediate_size, hidden_size), generator=g) * 0.2
        tensors[f"{p}.mlp.up_proj.weight"] = torch.randn((intermediate_size, hidden_size), generator=g) * 0.2
        tensors[f"{p}.mlp.down_proj.weight"] = torch.randn((hidden_size, intermediate_size), generator=g) * 0.2
    return tensors


def create_known_evolution_checkpoints(root: Path) -> dict[str, Path]:
    """Build controlled multi-generation & branched checkpoints with exact analytical properties:

    - `ckpt_a` (Base): original weights.
    - `ckpt_b` (SFT A): ONLY `model.layers.0.self_attn.q_proj.weight` shifted by exact `+0.25` everywhere
      (`16x16 = 256` elements -> exact `max_abs_diff = 0.25`, `l2_distance = sqrt(256 * 0.25^2) = 4.0`).
    - `ckpt_c` (DPO A): partially reverts `q_proj` (`+0.125` from Base, i.e. `-0.125` from B) AND modifies
      `model.layers.2.mlp.down_proj.weight` (`16x32 = 512` elements) by `+0.5` (`l2 = sqrt(512 * 0.25) = sqrt(128)`).
      Proves direct `Base -> C` L2 (`sqrt(256 * 0.125^2 + 512 * 0.5^2) = sqrt(4 + 128) = sqrt(132)`) != `L2(A->B) + L2(B->C)`.
    - `ckpt_d` (Quantized): simulated coarse 4-bit uniform quantization of `ckpt_c` + `quantization_config`.
    - `ckpt_e` (Pruned): magnitude pruning zeroing small weights in `ckpt_d` (increases exact sparsity).
    - `ckpt_sft_b` (SFT B branch): modifies Layer 1 `mlp.up_proj.weight` by `+0.30`.
    - `ckpt_merged` (Model Merging `[ckpt_c, ckpt_sft_b]`): 0.5 * C + 0.5 * SFT_B.
    - `ckpt_incompat` (Incompatible architecture): hidden_size=24, num_layers=2.
    - `ckpt_lora` (PEFT LoRA adapter directory): `adapter_config.json` + `adapter_model.safetensors`.
    """
    base_t = build_base_qwen_tensors()
    dir_a = _write_qwen_dir(root / "ckpt_a_base", base_t)

    # B: exact +0.25 on 16x16 q_proj of layer 0 -> L2(A, B) == 4.0
    t_b = {k: v.clone() for k, v in base_t.items()}
    t_b["model.layers.0.self_attn.q_proj.weight"] = (
        base_t["model.layers.0.self_attn.q_proj.weight"] + 0.25
    )
    dir_b = _write_qwen_dir(root / "ckpt_b_sft_a", t_b)

    # C: +0.125 from Base on layer 0 q_proj AND +0.50 from Base on layer 2 down_proj (16x32=512 elems)
    t_c = {k: v.clone() for k, v in base_t.items()}
    t_c["model.layers.0.self_attn.q_proj.weight"] = (
        base_t["model.layers.0.self_attn.q_proj.weight"] + 0.125
    )
    t_c["model.layers.2.mlp.down_proj.weight"] = (
        base_t["model.layers.2.mlp.down_proj.weight"] + 0.50
    )
    dir_c = _write_qwen_dir(root / "ckpt_c_dpo_a", t_c)

    # D: coarse uniform step quantization (step = 0.10) + quantization_config metadata
    t_d = {k: torch.round(v / 0.10) * 0.10 for k, v in t_c.items()}
    dir_d = _write_qwen_dir(
        root / "ckpt_d_quant",
        t_d,
        extra_config={"quantization_config": {"quant_method": "simulated_int4", "bits": 4}},
    )

    # E: magnitude pruning zeroing |w| < 0.15 on Layer 1 & Layer 2
    t_e = {k: v.clone() for k, v in t_d.items()}
    for k in list(t_e.keys()):
        if "layers.1." in k or "layers.2." in k:
            mask = t_e[k].abs() >= 0.15
            t_e[k] = t_e[k] * mask.to(t_e[k].dtype)
    dir_e = _write_qwen_dir(root / "ckpt_e_pruned", t_e)

    # SFT B (second branch from Base)
    t_sft_b = {k: v.clone() for k, v in base_t.items()}
    t_sft_b["model.layers.1.mlp.up_proj.weight"] = (
        base_t["model.layers.1.mlp.up_proj.weight"] + 0.30
    )
    dir_sft_b = _write_qwen_dir(root / "ckpt_sft_b", t_sft_b)

    # Merged ([DPO A, SFT B])
    t_merged = {k: 0.5 * t_c[k] + 0.5 * t_sft_b[k] for k in base_t}
    dir_merged = _write_qwen_dir(root / "ckpt_merged", t_merged)

    # Incompatible architecture (hidden_size=24, num_layers=2)
    t_incompat = build_base_qwen_tensors(
        seed=202, vocab_size=80, hidden_size=24, intermediate_size=48, num_layers=2, num_heads=4, num_kv_heads=2
    )
    dir_incompat = _write_qwen_dir(
        root / "ckpt_incompat",
        t_incompat,
        vocab_size=80,
        hidden_size=24,
        intermediate_size=48,
        num_layers=2,
    )

    # PEFT LoRA adapter-only directory
    dir_lora = root / "ckpt_lora_adapter"
    dir_lora.mkdir(parents=True, exist_ok=True)
    (dir_lora / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "ckpt_a_base",
                "r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones((4, 16)) * 0.02,
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones((16, 4)) * 0.02,
        },
        str(dir_lora / "adapter_model.safetensors"),
    )

    return {
        "a": dir_a,
        "b": dir_b,
        "c": dir_c,
        "d": dir_d,
        "e": dir_e,
        "sft_b": dir_sft_b,
        "merged": dir_merged,
        "incompat": dir_incompat,
        "lora": dir_lora,
    }
