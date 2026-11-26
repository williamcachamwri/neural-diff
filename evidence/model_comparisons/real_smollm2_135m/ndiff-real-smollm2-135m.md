# NEURAL DIFF Report — `ndiff-real-smollm2-135m`

- **Experiment Classification:** `REAL MODEL EXPERIMENT`
- **Generated (UTC):** `2026-11-26T19:28:16.894908+00:00`
- **Model A (Baseline):** `HuggingFaceTB/SmolLM2-135M` (`134,515,008` params, format=`huggingface_hub_cached`, rev=`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`)
- **Model B (Candidate):** `HuggingFaceTB/SmolLM2-135M-Instruct` (`134,515,008` params, format=`huggingface_hub_cached`, rev=`12fd25f77366fa6b3b4b768ec3050bf629380bac`)
- **Executed Modules:** `structural, tensor, neuron, activation, behavioral, regression`

## 1. Epistemic Classification Ledger

| Status | Finding | Source |
| :--- | :--- | :--- |
| `OBSERVED` | Model A (134,515,008 params) vs Model B (134,515,008 params) share 272 tensor keys. | Module C — StructuralDiffEngine |
| `OBSERVED` | 97.6728% of aligned parameters exceeded numerical tolerance (mean cosine=0.99695139, max abs diff=0.154296875). | Module B — TensorDiffEngine |
| `CORRELATED` | Internal hidden-state and MLP activation shifts co-occurred alongside observed behavioral score deltas; without targeted activation patching, internal-to-behavioral linkage is correlational. | Module E + Module F joint comparison |
| `NOT VERIFIED` | Causal mediation of individual neurons or circuits via counterfactual activation patching was not executed in this observational diff run. | Epistemic boundary declaration (Causal Patching Gate) |

## 2. Structural & Compatibility Diff (`OBSERVED` / `INFERRED`)

- **Compatibility Summary:** Fully aligned checkpoints (272/272 tensors match in name and shape).
- **Parameter Delta:** `+0` (`134,515,008` -> `134,515,008`)
- **Shared Keys:** `272` | **Added:** `0` | **Removed:** `0` | **Inferred Renames (`INFERRED`):** `0` | **Shape Changes:** `0` | **Dtype Changes:** `0`

## 3. Tensor Diff Summary (`OBSERVED`)

- **Compared Tensors:** `272` (Identical: `0`, Within Tolerance: `0`, Modified: `272`)
- **Global Changed Parameters:** `131,384,568 / 134,515,008` (`97.6728%`)
- **Global Mean Cosine Similarity:** `0.99695139`
- **Global Mean Relative L2:** `0.06869105`
- **Global Max Absolute Diff:** `0.154296875`

### Top Diverged Tensors

| Tensor | Role | Shape | Cosine | Relative L2 | Max Abs Diff | Changed % |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| `model.layers.0.self_attn.v_proj.weight` | `attn_v` | `[192, 576]` | `0.9497899668829543` | `0.3277855319552071` | `0.06982421875` | `99.56%` |
| `model.layers.0.input_layernorm.weight` | `norm` | `[576]` | `0.9477459966318127` | `0.32347254791950886` | `0.04742431640625` | `99.65%` |
| `model.layers.0.self_attn.o_proj.weight` | `attn_o` | `[576, 576]` | `0.9737072694428626` | `0.23309864359154678` | `0.0714111328125` | `99.30%` |
| `model.layers.2.self_attn.v_proj.weight` | `attn_v` | `[192, 576]` | `0.9908209788333087` | `0.13623524344458093` | `0.074462890625` | `98.76%` |
| `model.layers.28.self_attn.q_proj.weight` | `attn_q` | `[576, 576]` | `0.9923374563716212` | `0.12409708652162905` | `0.1083984375` | `98.43%` |
| `model.layers.26.self_attn.k_proj.weight` | `attn_k` | `[192, 576]` | `0.9927073167839536` | `0.12112873554220427` | `0.091796875` | `98.27%` |
| `model.layers.1.self_attn.v_proj.weight` | `attn_v` | `[192, 576]` | `0.9927851071266209` | `0.12097428984547345` | `0.06536865234375` | `98.60%` |
| `model.layers.26.self_attn.q_proj.weight` | `attn_q` | `[576, 576]` | `0.9928293970926857` | `0.12012753266618584` | `0.141876220703125` | `98.42%` |
| `model.layers.25.self_attn.q_proj.weight` | `attn_q` | `[576, 576]` | `0.9928173552797419` | `0.12010338992004753` | `0.1083984375` | `98.41%` |
| `model.layers.24.self_attn.k_proj.weight` | `attn_k` | `[192, 576]` | `0.9929393685739755` | `0.11945886007031832` | `0.11376953125` | `98.15%` |

## 4. Activation Analysis (`OBSERVED`)

- **Controlled Prompts Executed:** `4`
- **Mean Final-Token Logit JS Divergence:** `0.20172417`
- **Cross-Prompt Final-Layer Linear CKA:** `0.9517821`

## 5. Deterministic Behavioral Comparison (`OBSERVED`)

- **Suite:** `neural-diff-core-v1@1.0.0` (`24` paired cases)
- **Pass Rate:** Model A `20.8%` -> Model B `33.3%` (Delta `+12.5%`)
- **Mean Score:** Model A `0.4322` -> Model B `0.6141` (Delta `+0.1819`, 95% Bootstrap CI `[+0.0433, +0.3488]`, McNemar p=`0.3750`)

| Category | Cases | Pass A | Pass B | Mean Score A | Mean Score B | Delta |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `vietnamese` | 3 | 0% | 0% | 0.500 | 0.383 | -0.117 |
| `code_preservation` | 3 | 0% | 67% | 0.200 | 0.750 | +0.550 |
| `json_reliability` | 3 | 0% | 67% | 0.350 | 0.925 | +0.575 |
| `math_reasoning` | 3 | 0% | 0% | 0.000 | 0.000 | +0.000 |
| `instruction_following` | 3 | 0% | 0% | 0.333 | 0.611 | +0.278 |
| `conciseness` | 3 | 33% | 0% | 0.608 | 0.777 | +0.169 |
| `refusal_behavior` | 3 | 67% | 67% | 0.800 | 0.800 | +0.000 |
| `factual_accuracy` | 3 | 67% | 67% | 0.667 | 0.667 | +0.000 |

## 6. Regression Gate Verdict

- **Verdict:** **`FAILED_REGRESSION`** (Highest Severity: `HIGH`)
- **Policy:** `production-default-v1` | **Findings:** `1`
  - `[HIGH]` (`OBSERVED`) **Behavioral regression in 'vietnamese'**: Category 'vietnamese' mean score changed 0.500 -> 0.383 (0 regressed case(s)).

## 7. Execution & Reproducibility Metadata

```json
{
  "reproducibility": {
    "captured_at_utc": "2026-11-26T19:31:28.117654+00:00",
    "python_version": "3.12.16",
    "platform": "Windows-11-10.0.26320-SP0",
    "torch_version": "2.10.0+cu128",
    "transformers_version": "5.24.1",
    "safetensors_version": "0.8.0",
    "neural_diff_version": "0.1.0",
    "seed": 42,
    "device_selected": "cuda:0",
    "device_reason": "Verified CUDA kernel execution on NVIDIA GeForce RTX 5050 Laptop GPU (7048 MB free VRAM).",
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
      "timestamp_monotonic": 4920.0769556,
      "process_rss_mb": 1078.36,
      "system_available_ram_mb": 3142.67,
      "system_total_ram_mb": 16088.54,
      "cuda_available": true,
      "cuda_allocated_mb": 0.0,
      "cuda_reserved_mb": 0.0,
      "cuda_total_vram_mb": 8150.56,
      "selected_device": "cuda:0"
    },
    "after_pre_cleanup": {
      "timestamp_monotonic": 5111.2067861,
      "process_rss_mb": 2064.07,
      "system_available_ram_mb": 2473.12,
      "system_total_ram_mb": 16088.54,
      "cuda_available": true,
      "cuda_allocated_mb": 8.12,
      "cuda_reserved_mb": 20.0,
      "cuda_total_vram_mb": 8150.56,
      "selected_device": "cuda:0"
    },
    "after_post_cleanup": {
      "timestamp_monotonic": 5111.2995091,
      "process_rss_mb": 2064.07,
      "system_available_ram_mb": 2473.15,
      "system_total_ram_mb": 16088.54,
      "cuda_available": true,
      "cuda_allocated_mb": 8.12,
      "cuda_reserved_mb": 20.0,
      "cuda_total_vram_mb": 8150.56,
      "selected_device": "cuda:0"
    },
    "duration_seconds": 191.128353,
    "rss_delta_mb": 985.71,
    "cuda_allocated_peak_delta_mb": 8.12
  }
}
```
