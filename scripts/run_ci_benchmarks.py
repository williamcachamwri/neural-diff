#!/usr/bin/env python3
"""Deterministic CI numerical & mechanistic benchmark gate for NEURAL DIFF."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from transformers import GPT2Config, GPT2LMHeadModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.activation_analysis.metrics import compute_linear_cka  # noqa: E402
from core.activation_analysis.patching import run_residual_patching  # noqa: E402
from core.orchestrator import ComparisonOrchestrator, ComparisonRequestConfig  # noqa: E402
from core.structural_analysis.sparsity_pruning import analyze_matrix_sparsity  # noqa: E402
from core.tensor_engine.task_vector import analyze_task_vector_spectrum  # noqa: E402
from tests.fixtures.checkpoint_factory import make_synthetic_hf_checkpoint_pair  # noqa: E402


def run_ci_benchmarks() -> dict[str, Any]:
    results: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="neural_diff_ci_bench_") as tmp:
        root = Path(tmp)
        dir_a, dir_b = make_synthetic_hf_checkpoint_pair(root)

        # 1. Identity invariant check (Model A vs Model A)
        t0 = time.perf_counter()
        id_bundle = ComparisonOrchestrator().run(
            ComparisonRequestConfig(
                model_a=str(dir_a),
                model_b=str(dir_a),
                modules=["structural", "tensor"],
                local_files_only=True,
            )
        )
        id_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert id_bundle.tensor_diff is not None
        assert id_bundle.tensor_diff.global_max_abs_diff == 0.0
        assert id_bundle.tensor_diff.global_mean_cosine is not None
        assert abs(id_bundle.tensor_diff.global_mean_cosine - 1.0) < 1e-7
        assert not id_bundle.tensor_diff.has_any_nan_or_inf
        results["identity_gate"] = {
            "elapsed_ms": round(id_elapsed_ms, 2),
            "global_max_abs_diff": id_bundle.tensor_diff.global_max_abs_diff,
            "global_mean_cosine": id_bundle.tensor_diff.global_mean_cosine,
            "status": "PASS",
        }

        # 2. Full structural + tensor + neuron + regression pipeline benchmark
        t1 = time.perf_counter()
        diff_bundle = ComparisonOrchestrator().run(
            ComparisonRequestConfig(
                model_a=str(dir_a),
                model_b=str(dir_b),
                modules=["structural", "tensor", "neuron", "regression"],
                local_files_only=True,
            )
        )
        diff_elapsed_ms = (time.perf_counter() - t1) * 1000.0
        assert diff_bundle.tensor_diff is not None
        assert diff_bundle.tensor_diff.global_max_abs_diff is not None
        assert diff_bundle.tensor_diff.global_max_abs_diff > 0.0
        assert diff_bundle.tensor_diff.global_mean_relative_l2 is not None
        assert math.isfinite(diff_bundle.tensor_diff.global_mean_relative_l2)
        results["checkpoint_diff_pipeline"] = {
            "elapsed_ms": round(diff_elapsed_ms, 2),
            "compared_tensors": diff_bundle.tensor_diff.compared_tensors,
            "modified_tensors": diff_bundle.tensor_diff.modified_tensors,
            "global_mean_relative_l2": round(diff_bundle.tensor_diff.global_mean_relative_l2, 6),
            "status": "PASS",
        }

        # 3. SVD spectrum, Roy-Vetterli Shannon effective rank & 2:4 sparsity gate
        t2 = time.perf_counter()
        g = torch.Generator().manual_seed(42)
        base_w = torch.randn((256, 256), generator=g)
        lora_w = base_w + (
            torch.randn((256, 4), generator=g) * 0.05
        ) @ (torch.randn((4, 256), generator=g) * 0.05)
        dense_w = base_w + torch.randn((256, 256), generator=g) * 0.02
        spec_lora = analyze_task_vector_spectrum(base_w, lora_w)
        spec_dense = analyze_task_vector_spectrum(base_w, dense_w)

        sparse_24 = base_w.clone().reshape(256, 64, 4)
        sparse_24[:, :, :2] = 0.0
        sparsity_prof = analyze_matrix_sparsity(sparse_24.reshape(256, 256))
        svd_elapsed_ms = (time.perf_counter() - t2) * 1000.0
        assert 1.0 <= spec_lora.shannon_effective_rank <= 4.05
        assert spec_dense.shannon_effective_rank > 100.0
        assert sparsity_prof.is_2_to_4_semi_structured is True
        results["svd_and_sparsity_gate"] = {
            "elapsed_ms": round(svd_elapsed_ms, 2),
            "lora_rank4_effective_rank": round(spec_lora.shannon_effective_rank, 4),
            "dense_sft_effective_rank": round(spec_dense.shannon_effective_rank, 4),
            "lora_regime": spec_lora.inferred_update_regime,
            "dense_regime": spec_dense.inferred_update_regime,
            "semi_structured_2_4_detected": sparsity_prof.is_2_to_4_semi_structured,
            "status": "PASS",
        }

        # 4. Mechanistic Linear CKA & Causal Residual Activation Patching gate
        t3 = time.perf_counter()
        torch.manual_seed(42)
        cfg = GPT2Config(
            vocab_size=128,
            n_positions=32,
            n_ctx=32,
            n_embd=32,
            n_layer=2,
            n_head=4,
            bos_token_id=0,
            eos_token_id=1,
        )
        model = GPT2LMHeadModel(cfg).eval()
        clean_ids = torch.randint(0, 128, (1, 12))
        corrupted_ids = torch.randint(0, 128, (1, 12))

        with torch.no_grad():
            hs_clean = model(input_ids=clean_ids, output_hidden_states=True).hidden_states[-1]
        cka_identity = compute_linear_cka(hs_clean.reshape(-1, 32), hs_clean.reshape(-1, 32))
        assert cka_identity is not None and abs(cka_identity - 1.0) < 1e-6

        patch_report = run_residual_patching(
            model,
            model.transformer.h,
            clean_ids,
            corrupted_ids,
            correct_token_id=7,
            wrong_token_id=19,
        )
        mech_elapsed_ms = (time.perf_counter() - t3) * 1000.0
        assert len(patch_report.sites) == 2
        assert all(math.isfinite(s.recovery_fraction) for s in patch_report.sites)
        results["mechanistic_cka_and_patching"] = {
            "elapsed_ms": round(mech_elapsed_ms, 2),
            "identity_linear_cka": round(cka_identity, 6),
            "patched_sites": len(patch_report.sites),
            "epistemic_status": patch_report.epistemic_status,
            "status": "PASS",
        }

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic CI numerical benchmarks.")
    parser.add_argument("--output", default=None, help="Optional JSON output file path.")
    args = parser.parse_args()

    summary = run_ci_benchmarks()
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
