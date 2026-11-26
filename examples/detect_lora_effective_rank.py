#!/usr/bin/env python3
"""Example: Distinguish a low-rank LoRA update from a full-rank fine-tune via Shannon effective rank."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.tensor_engine.task_vector import analyze_task_vector_spectrum  # noqa: E402


def main() -> None:
    g = torch.Generator().manual_seed(42)
    base_w = torch.randn((256, 256), generator=g)

    # Simulated rank-4 LoRA update: W_B = W_A + B @ A
    lora_b = torch.randn((256, 4), generator=g) * 0.05
    lora_a = torch.randn((4, 256), generator=g) * 0.05
    lora_merged = base_w + (lora_b @ lora_a)

    # Simulated full-rank SFT update
    full_sft = base_w + torch.randn((256, 256), generator=g) * 0.02

    spec_lora = analyze_task_vector_spectrum(base_w, lora_merged)
    spec_full = analyze_task_vector_spectrum(base_w, full_sft)

    print(
        json.dumps(
            {
                "lora_rank4_update": spec_lora.to_dict(),
                "full_rank_sft_update": spec_full.to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
