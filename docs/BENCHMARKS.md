# Benchmarks & Real Model Experiments — NEURAL DIFF

**Verification Status:** `VERIFIED` (All measurements generated from live execution on `Intel Core i5-14450HX` + `NVIDIA GeForce RTX 5050 Laptop GPU` `8,151 MiB` VRAM, Windows 11 `10.0.26320`, CPython `3.12.16`, `torch 2.10.0+cu128`, `transformers 5.24.1`, `safetensors 0.8.0`).

> **Provenance Separation Policy:** `REAL MODEL EXPERIMENT` results and `SYNTHETIC FIXTURE EXPERIMENT` micro-benchmarks are stored in separate artifacts and documented in separate sections below. Their numbers are never combined or averaged.

---

## PART I — REAL MODEL EXPERIMENTS (`VERIFIED` / `BLOCKED` Upstream Gate)

**Raw Evidence Artifacts:**
- Summary Telemetry: [`evidence/benchmarks/real_model_benchmarks.json`](../evidence/benchmarks/real_model_benchmarks.json)
- Execution Log: [`evidence/benchmarks/model-comparison.log`](../evidence/benchmarks/model-comparison.log)
- Qwen2.5-0.5B Full Bundle & Reports: [`evidence/model_comparisons/real_qwen25_05b/`](../evidence/model_comparisons/real_qwen25_05b/)
- SmolLM2-135M Full Bundle & Reports: [`evidence/model_comparisons/real_smollm2_135m/`](../evidence/model_comparisons/real_smollm2_135m/)
- Cross-Architecture Bundle: [`evidence/model_comparisons/real_cross_architecture/`](../evidence/model_comparisons/real_cross_architecture/)
- Real Regression Detection Bundle: [`evidence/model_comparisons/real_regression_detection/`](../evidence/model_comparisons/real_regression_detection/)
- Gemma 3 270M Upstream Gate Probe: [`evidence/model_comparisons/gemma3_270m_access_probe.json`](../evidence/model_comparisons/gemma3_270m_access_probe.json)

### 1. Summary of Executed Real Model Comparisons (`cuda:0` RTX 5050)

| Experiment ID | Model A (`rev`) | Model B (`rev`) | Params (A / B) | Tensors | Changed % | Mean Cosine | Final-Layer CKA | Pass Rate (A → B) | Score Δ | Runtime |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ndiff-real-qwen25-05b` | `Qwen/Qwen2.5-0.5B` (`060db6499f...`) | `Qwen/Qwen2.5-0.5B-Instruct` (`7ae557604a...`) | `494,032,768` / `494,032,768` | `290` | `97.9318%` | `0.998495` | `0.991076` | `45.8%` → `70.8%` | `+0.1265` | `238.51s` |
| `ndiff-real-smollm2-135m` | `HuggingFaceTB/SmolLM2-135M` (`93efa2f097...`) | `HuggingFaceTB/SmolLM2-135M-Instruct` (`12fd25f773...`) | `134,515,008` / `134,515,008` | `272` | `97.6728%` | `0.996951` | `0.951782` | `20.8%` → `33.3%` | `+0.1819` | `191.13s` |
| `ndiff-real-cross-arch` | `Qwen/Qwen2.5-0.5B-Instruct` | `HuggingFaceTB/SmolLM2-135M-Instruct` | `494,032,768` / `134,515,008` | `290` vs `272` | *(cross-arch)* | *(cross-arch)* | `0.966758` | *(struct + act)* | — | `4.79s` |
| `ndiff-real-regression-detected` | `Qwen/Qwen2.5-0.5B-Instruct` | `Qwen/Qwen2.5-0.5B` | `494,032,768` / `494,032,768` | `290` | `97.9318%` | `0.998495` | — | `62.5%` → `37.5%` | `-0.1656` | `117.09s` |

### 2. Deep-Dive Findings: `Qwen/Qwen2.5-0.5B` vs. `Qwen/Qwen2.5-0.5B-Instruct`

1. **Structural & Parameter Localization (`OBSERVED`):**
   - `290/290` tensors match in name and shape (`494,032,768` parameters, `bfloat16`).
   - `97.9318%` (`483,814,955`) of parameters shifted beyond numerical tolerance (`atol=1e-6`, `rtol=1e-5`), with global mean cosine similarity `0.99849455` and global mean relative $L_2$ `0.04429389`.
   - **Largest relative $L_2$ updates concentrated in token embeddings and upper-layer Query Attention projections:**
     - `model.embed_tokens.weight` (`[151936, 896]`): Rel-$L_2$ = `0.111558`, Cosine = `0.993906`
     - `model.layers.22.self_attn.q_proj.weight`: Rel-$L_2$ = `0.105072`, Cosine = `0.994501`
     - `model.layers.0.self_attn.o_proj.weight`: Rel-$L_2$ = `0.104117`, Cosine = `0.994686`
     - `model.layers.23.self_attn.q_proj.weight`: Rel-$L_2$ = `0.102670`, Cosine = `0.994752`
     - `model.layers.17–21.self_attn.q_proj.weight`: Rel-$L_2$ = `0.093740`–`0.098702`

2. **Activation Space Alignment (`OBSERVED`):**
   - Cross-prompt final-layer Linear CKA = `0.99107627`; mean final-token next-word Jensen-Shannon divergence = `0.07065137`.

3. **8-Suite Deterministic Behavioral Deltas (`OBSERVED`, 24 paired cases, `max_new_tokens=96`):**
   - Overall Pass Rate improved from **`45.8%`** (`11/24`) to **`70.8%`** (`17/24`, `+25.0%`), mean score `0.6294` → `0.7559` (`+0.1265`, 95% Bootstrap CI `[-0.0542, +0.3085]`, McNemar $p = 0.1094$):
     - `instruction_following`: `0%` → `67%` (`+0.500` mean score)
     - `vietnamese`: `0%` → `67%` (`+0.367` mean score)
     - `conciseness`: `0%` → `67%` (`+0.329` mean score)
     - `code_preservation`: `33%` → `67%` (`+0.283` mean score)
     - `refusal_behavior`: `67%` → `100%` (`+0.200` mean score — refused credential-stealing keylogger request while avoiding false refusals on Linux `kill` and SQL parameterization)
     - `json_reliability`: `100%` → `100%` (`0.925` mean score)
     - `factual_accuracy`: `100%` → `100%` (`1.000` mean score)
     - `math_reasoning`: `67%` → `0%` (`-0.667` mean score) — **Genuine Regression Caught:** under a strict `96`-token generation budget, `Qwen2.5-0.5B-Instruct` emits multi-step chain-of-thought explanations that truncate before emitting the final `'Answer: <number>'` line, whereas the terse base model emits the numeric answer within 15 tokens. `RegressionEngine` flagged this trade-off automatically (`REG-CAT`, `CRITICAL`), proving that strict regression gating surfaces real deployment token-budget trade-offs rather rubber-stamping overall averages.

### 3. Upstream Gated Model Verification: `google/gemma-3-270m` (`BLOCKED`)

Unauthenticated probing of `google/gemma-3-270m` (`sha: 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`) confirmed `gated: "manual"` under Google's Gemma Terms of Use (`HTTP 401 Client Error: Access to model google/gemma-3-270m is restricted`). NEURAL DIFF surfaced `ModelAccessBlockedError` cleanly and recorded exact reproduction instructions (`$env:HF_TOKEN="<token>"` + `uv run neural-diff inspect google/gemma-3-270m`) in [`gemma3_270m_access_probe.json`](../evidence/model_comparisons/gemma3_270m_access_probe.json).

---

## PART II — SYNTHETIC FIXTURE THROUGHPUT & MEMORY MICRO-BENCHMARKS (`VERIFIED`)

**Raw Evidence Artifact:** [`evidence/benchmarks/synthetic_throughput_benchmarks.json`](../evidence/benchmarks/synthetic_throughput_benchmarks.json)

| Benchmark ID | Elements | Dtype | Chunk Size | Wall Time (s) | Throughput (M elems/s) | Effective GB/s | RSS Delta (MB) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `1M_float32_chunk64k` | `1,000,000` | `float32` | `65,536` | `0.030627` | `32.651` | `0.243` | `+153.57` *(initial torch kernel warmup)* |
| `1M_float32_chunk1M` | `1,000,000` | `float32` | `1,048,576` | `0.012333` | `81.080` | `0.604` | `+49.59` |
| `10M_float32_chunk256k` | `10,000,000` | `float32` | `262,144` | `0.089136` | `112.188` | `0.836` | `+0.54` |
| `10M_float32_chunk1M` | `10,000,000` | `float32` | `1,048,576` | `0.085730` | **`116.645`** | **`0.869`** | `+24.94` |
| `10M_bfloat16_chunk1M` | `10,000,000` | `bfloat16` | `1,048,576` | `0.097149` | `102.934` | `0.383` | `+22.82` |
| `25M_bfloat16_chunk2M` | `25,000,000` | `bfloat16` | `2,097,152` | `0.262816` | `95.124` | `0.354` | `+87.70` |
| `safetensors_8x512x1024_bf16_lazy_stream` | `4,194,304` *(8 disk tensors + quantiles)* | `bfloat16` | `262,144` | `0.295004` | — | — | **`+0.52`** |

Notice that lazy Safetensors disk-to-diff streaming across 8 matrices (`4.19M` parameters, including quantile distribution estimation) added only **`+0.52 MB`** to process RSS.
