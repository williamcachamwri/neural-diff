# NEURAL DIFF Report — `ndiff-real-cross-arch`

- **Experiment Classification:** `REAL MODEL EXPERIMENT`
- **Generated (UTC):** `2026-11-26T19:35:27.181216+00:00`
- **Model A (Baseline):** `Qwen/Qwen2.5-0.5B-Instruct` (`494,032,768` params, format=`huggingface_hub_cached`, rev=`7ae557604adf67be50417f59c2c2f167def9a775`)
- **Model B (Candidate):** `HuggingFaceTB/SmolLM2-135M-Instruct` (`134,515,008` params, format=`huggingface_hub_cached`, rev=`12fd25f77366fa6b3b4b768ec3050bf629380bac`)
- **Executed Modules:** `structural, activation, regression`

## 1. Epistemic Classification Ledger

| Status | Finding | Source |
| :--- | :--- | :--- |
| `OBSERVED` | Model A (494,032,768 params) vs Model B (134,515,008 params) share 218 tensor keys. | Module C — StructuralDiffEngine |
| `NOT VERIFIED` | Causal mediation of individual neurons or circuits via counterfactual activation patching was not executed in this observational diff run. | Epistemic boundary declaration (Causal Patching Gate) |

## 2. Structural & Compatibility Diff (`OBSERVED` / `INFERRED`)

- **Compatibility Summary:** Incompatible parameter spaces (0 shape-matched shared tensors); structural and behavioral comparisons remain supported.
- **Parameter Delta:** `-359,517,760` (`494,032,768` -> `134,515,008`)
- **Shared Keys:** `218` | **Added:** `54` | **Removed:** `72` | **Inferred Renames (`INFERRED`):** `0` | **Shape Changes:** `218` | **Dtype Changes:** `0`

## 4. Activation Analysis (`OBSERVED`)

- **Controlled Prompts Executed:** `4`
- **Mean Final-Token Logit JS Divergence:** `None`
- **Cross-Prompt Final-Layer Linear CKA:** `0.96675814`

## 6. Regression Gate Verdict

- **Verdict:** **`FAILED_REGRESSION`** (Highest Severity: `HIGH`)
- **Policy:** `production-default-v1` | **Findings:** `1`
  - `[HIGH]` (`OBSERVED`) **Removed tensors detected in candidate model**: 72 tensor(s) were removed.

## 7. Execution & Reproducibility Metadata

```json
{
  "reproducibility": {
    "captured_at_utc": "2026-11-26T19:35:32.069740+00:00",
    "python_version": "3.12.16",
    "platform": "Windows-11-10.0.26320-SP0",
    "torch_version": "2.10.0+cu128",
    "transformers_version": "5.24.1",
    "safetensors_version": "0.8.0",
    "neural_diff_version": "0.1.0",
    "seed": 42,
    "device_selected": "cuda:0",
    "device_reason": "Verified CUDA kernel execution on NVIDIA GeForce RTX 5050 Laptop GPU (7014 MB free VRAM).",
    "tolerances": {
      "atol": 1e-06,
      "rtol": 1e-05,
      "zero_threshold": 0.0,
      "eps": 1e-12
    },
    "thresholds": {
      "policy_name": "production-default-v1",
      "forbid_nan_or_inf": true,
      "forbid_missing_tensors": true,
      "forbid_shape_mismatches": true,
      "max_overall_pass_rate_drop": 0.05,
      "critical_overall_pass_rate_drop": 0.15,
      "max_overall_score_drop": 0.05,
      "max_category_score_drop": 0.1,
      "critical_category_score_drop": 0.25,
      "zero_tolerance_refusal_regression": true,
      "min_final_hidden_cka": 0.6
    }
  },
  "telemetry": {
    "before": {
      "timestamp_monotonic": 5350.3629891,
      "process_rss_mb": 1958.38,
      "system_available_ram_mb": 2735.4,
      "system_total_ram_mb": 16088.54,
      "cuda_available": true,
      "cuda_allocated_mb": 9.12,
      "cuda_reserved_mb": 22.0,
      "cuda_total_vram_mb": 8150.56,
      "selected_device": "cuda:0"
    },
    "after_pre_cleanup": {
      "timestamp_monotonic": 5355.1561256,
      "process_rss_mb": 2545.77,
      "system_available_ram_mb": 2105.57,
      "system_total_ram_mb": 16088.54,
      "cuda_available": true,
      "cuda_allocated_mb": 9.12,
      "cuda_reserved_mb": 22.0,
      "cuda_total_vram_mb": 8150.56,
      "selected_device": "cuda:0"
    },
    "after_post_cleanup": {
      "timestamp_monotonic": 5355.2514679,
      "process_rss_mb": 2545.77,
      "system_available_ram_mb": 2105.53,
      "system_total_ram_mb": 16088.54,
      "cuda_available": true,
      "cuda_allocated_mb": 9.12,
      "cuda_reserved_mb": 22.0,
      "cuda_total_vram_mb": 8150.56,
      "selected_device": "cuda:0"
    },
    "duration_seconds": 4.792252,
    "rss_delta_mb": 587.39,
    "cuda_allocated_peak_delta_mb": 0.0
  }
}
```
