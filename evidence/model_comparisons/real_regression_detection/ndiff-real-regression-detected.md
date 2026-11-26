# NEURAL DIFF Report — `ndiff-real-regression-detected`

- **Experiment Classification:** `REAL MODEL EXPERIMENT`
- **Generated (UTC):** `2026-11-26T19:35:32.152073+00:00`
- **Model A (Baseline):** `Qwen/Qwen2.5-0.5B-Instruct` (`494,032,768` params, format=`huggingface_hub_cached`, rev=`7ae557604adf67be50417f59c2c2f167def9a775`)
- **Model B (Candidate):** `Qwen/Qwen2.5-0.5B` (`494,032,768` params, format=`huggingface_hub_cached`, rev=`060db6499f32faf8b98477b0a26969ef7d8b9987`)
- **Executed Modules:** `structural, tensor, behavioral, regression`

## 1. Epistemic Classification Ledger

| Status | Finding | Source |
| :--- | :--- | :--- |
| `OBSERVED` | Model A (494,032,768 params) vs Model B (494,032,768 params) share 290 tensor keys. | Module C — StructuralDiffEngine |
| `OBSERVED` | 97.9318% of aligned parameters exceeded numerical tolerance (mean cosine=0.99849455, max abs diff=0.5). | Module B — TensorDiffEngine |
| `NOT VERIFIED` | Causal mediation of individual neurons or circuits via counterfactual activation patching was not executed in this observational diff run. | Epistemic boundary declaration (Causal Patching Gate) |

## 2. Structural & Compatibility Diff (`OBSERVED` / `INFERRED`)

- **Compatibility Summary:** Fully aligned checkpoints (290/290 tensors match in name and shape).
- **Parameter Delta:** `+0` (`494,032,768` -> `494,032,768`)
- **Shared Keys:** `290` | **Added:** `0` | **Removed:** `0` | **Inferred Renames (`INFERRED`):** `0` | **Shape Changes:** `0` | **Dtype Changes:** `0`

## 3. Tensor Diff Summary (`OBSERVED`)

- **Compared Tensors:** `290` (Identical: `0`, Within Tolerance: `0`, Modified: `290`)
- **Global Changed Parameters:** `483,814,955 / 494,032,768` (`97.9318%`)
- **Global Mean Cosine Similarity:** `0.99849455`
- **Global Mean Relative L2:** `0.04499653`
- **Global Max Absolute Diff:** `0.5`

### Top Diverged Tensors

| Tensor | Role | Shape | Cosine | Relative L2 | Max Abs Diff | Changed % |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| `model.embed_tokens.weight` | `embedding` | `[151936, 896]` | `0.9939060338269035` | `0.11421452624733086` | `0.01904296875` | `98.33%` |
| `model.layers.22.self_attn.q_proj.weight` | `attn_q` | `[896, 896]` | `0.9945007652288852` | `0.10656235531883512` | `0.01812744140625` | `98.44%` |
| `model.layers.0.self_attn.o_proj.weight` | `attn_o` | `[896, 896]` | `0.994686347338939` | `0.10633417194911025` | `0.015380859375` | `98.37%` |
| `model.layers.23.self_attn.q_proj.weight` | `attn_q` | `[896, 896]` | `0.9947515864773084` | `0.10409851369852419` | `0.02197265625` | `98.40%` |
| `model.layers.17.self_attn.q_proj.weight` | `attn_q` | `[896, 896]` | `0.9951505634048962` | `0.10000356375931678` | `0.017333984375` | `98.24%` |
| `model.layers.20.self_attn.q_proj.weight` | `attn_q` | `[896, 896]` | `0.9954553806659053` | `0.0968709981571832` | `0.01729583740234375` | `98.29%` |
| `model.layers.18.self_attn.q_proj.weight` | `attn_q` | `[896, 896]` | `0.9955723579361279` | `0.09584888521650524` | `0.018310546875` | `98.21%` |
| `model.layers.19.self_attn.q_proj.weight` | `attn_q` | `[896, 896]` | `0.9956311983894981` | `0.0957113833606916` | `0.01446533203125` | `98.17%` |
| `model.layers.11.self_attn.q_proj.weight` | `attn_q` | `[896, 896]` | `0.9956729439766868` | `0.09519487143190504` | `0.02734375` | `98.21%` |
| `model.layers.21.self_attn.q_proj.weight` | `attn_q` | `[896, 896]` | `0.9956430912685806` | `0.09506786597176253` | `0.01513671875` | `98.35%` |

## 5. Deterministic Behavioral Comparison (`OBSERVED`)

- **Suite:** `neural-diff-core-v1@1.0.0` (`16` paired cases)
- **Pass Rate:** Model A `62.5%` -> Model B `37.5%` (Delta `-25.0%`)
- **Mean Score:** Model A `0.6964` -> Model B `0.5495` (Delta `-0.1469`, 95% Bootstrap CI `[-0.4000, +0.1396]`, McNemar p=`0.2891`)

| Category | Cases | Pass A | Pass B | Mean Score A | Mean Score B | Delta |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `vietnamese` | 3 | 67% | 0% | 0.867 | 0.500 | -0.367 |
| `code_preservation` | 3 | 67% | 33% | 0.700 | 0.417 | -0.283 |
| `json_reliability` | 3 | 100% | 100% | 0.925 | 0.925 | +0.000 |
| `math_reasoning` | 3 | 0% | 67% | 0.000 | 0.667 | +0.667 |
| `instruction_following` | 3 | 67% | 0% | 0.889 | 0.389 | -0.500 |
| `conciseness` | 1 | 100% | 0% | 1.000 | 0.100 | -0.900 |

## 6. Regression Gate Verdict

- **Verdict:** **`FAILED_REGRESSION`** (Highest Severity: `CRITICAL`)
- **Policy:** `production-default-v1` | **Findings:** `5`
  - `[CRITICAL]` (`OBSERVED`) **Overall behavioral pass-rate drop**: Pass rate decreased from 62.5% to 37.5% (drop=25.0%).
  - `[CRITICAL]` (`OBSERVED`) **Behavioral regression in 'vietnamese'**: Category 'vietnamese' mean score changed 0.867 -> 0.500 (2 regressed case(s)).
  - `[CRITICAL]` (`OBSERVED`) **Behavioral regression in 'code_preservation'**: Category 'code_preservation' mean score changed 0.700 -> 0.417 (1 regressed case(s)).
  - `[CRITICAL]` (`OBSERVED`) **Behavioral regression in 'instruction_following'**: Category 'instruction_following' mean score changed 0.889 -> 0.389 (2 regressed case(s)).
  - `[CRITICAL]` (`OBSERVED`) **Behavioral regression in 'conciseness'**: Category 'conciseness' mean score changed 1.000 -> 0.100 (1 regressed case(s)).

## 7. Execution & Reproducibility Metadata

```json
{
  "reproducibility": {
    "captured_at_utc": "2026-11-26T19:37:29.335798+00:00",
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
      "timestamp_monotonic": 5355.3340426,
      "process_rss_mb": 2548.05,
      "system_available_ram_mb": 2105.09,
      "system_total_ram_mb": 16088.54,
      "cuda_available": true,
      "cuda_allocated_mb": 9.12,
      "cuda_reserved_mb": 22.0,
      "cuda_total_vram_mb": 8150.56,
      "selected_device": "cuda:0"
    },
    "after_pre_cleanup": {
      "timestamp_monotonic": 5472.4205666,
      "process_rss_mb": 2532.07,
      "system_available_ram_mb": 2027.98,
      "system_total_ram_mb": 16088.54,
      "cuda_available": true,
      "cuda_allocated_mb": 9.12,
      "cuda_reserved_mb": 22.0,
      "cuda_total_vram_mb": 8150.56,
      "selected_device": "cuda:0"
    },
    "after_post_cleanup": {
      "timestamp_monotonic": 5472.5175051,
      "process_rss_mb": 2532.07,
      "system_available_ram_mb": 2027.96,
      "system_total_ram_mb": 16088.54,
      "cuda_available": true,
      "cuda_allocated_mb": 9.12,
      "cuda_reserved_mb": 22.0,
      "cuda_total_vram_mb": 8150.56,
      "selected_device": "cuda:0"
    },
    "duration_seconds": 117.085701,
    "rss_delta_mb": -15.98,
    "cuda_allocated_peak_delta_mb": 0.0
  }
}
```
