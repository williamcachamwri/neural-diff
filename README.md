# NEURAL DIFF

**Git for AI Brains — Structural, Neural & Behavioral Intelligence Comparison**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](pyproject.toml)
[![TypeScript](https://img.shields.io/badge/Frontend-React_%2B_TypeScript_%2B_Vite-3178C6.svg)](frontend/package.json)
[![Coverage: 90%](https://img.shields.io/badge/Coverage-90%25_(4219_stmts)-brightgreen.svg)](evidence/tests/coverage.log)

When two neural checkpoints differ—after supervised fine-tuning, RLHF/DPO alignment, model merging, pruning, or quantization—file-level hashes only tell you *that* a binary changed, never *where* or *with what behavioral consequence*.

**NEURAL DIFF** is a full-stack open-source engineering platform that diffs checkpoints across five complementary layers:
1. **Structural & Compatibility Diff (`Module C`):** Added, removed, shape-changed, dtype-changed, and conservatively inferred renamed tensors (`INFERRED`), plus multi-dimensional compatibility validation.
2. **Out-of-Core Float64 Tensor Diff (`Module B`):** Chunked streaming reductions (`max_abs_diff`, `mean_abs_diff`, `RMSE`, `relative_l2`, `cosine_similarity`, tolerance-bounded `changed_pct`, zero/nonzero transitions, `NaN`/`Inf` counters, and Wasserstein-1 quantile shifts).
3. **Attention Head, MLP Basis & Embedding Localization (`Module D`):** Grouped-Query Attention (`q`/`k`/`v`/`o`) per-head slicing, MLP coordinate norm shifts (with explicit superposition caveats), batched vocabulary row drift, and `RMSNorm`/`LayerNorm` gain shifts.
4. **Controlled Forward-Hook Activation Diff (`Module E`):** Sequential memory-bounded forward-hook capture of hidden states, attention outputs, and MLP intermediates, reporting per-layer cosine, **Linear CKA**, and next-token KL / Jensen-Shannon logit divergence.
5. **8-Suite Deterministic Behavioral Diff & Regression Gate (`Modules F & G`):** Zero-LLM-judge deterministic evaluators across Vietnamese quality, Python AST code preservation, strict JSON reliability, mathematical reasoning, instruction constraints, conciseness, refusal calibration, and factual QA—paired with 95% bootstrap confidence intervals, exact McNemar $p$-values, versioned baselines, and CLI failure reproduction commands.

---

## Live Application Previews (Captured via Playwright Chromium)

| Dashboard (`CUDA:0` RTX 5050) | 24-Layer Relative $L_2$ Heatmap & Head Slicing |
| :---: | :---: |
| ![Dashboard](evidence/frontend/01-dashboard.png) | ![Layer Analysis](evidence/frontend/06-layer-analysis.png) |

| Tensor Diff Explorer (`Qwen2.5-0.5B`) | 8-Suite Deterministic Behavioral Diff |
| :---: | :---: |
| ![Tensor Diff](evidence/frontend/04-tensor-diff.png) | ![Behavioral Diff](evidence/frontend/08-behavioral-diff.png) |

| Activation Alignment & Linear CKA | Automated Regression Gate & CLI Reproduction |
| :---: | :---: |
| ![Activation Diff](evidence/frontend/07-activation-comparison.png) | ![Regression Gate](evidence/frontend/09-regression-report.png) |

Full metadata and provenance for all 15 captured views are cataloged in [`evidence/SCREENSHOT_INDEX.md`](evidence/SCREENSHOT_INDEX.md).

---

## Verified Real-Model Benchmark Highlights (`NVIDIA RTX 5050`, `torch 2.10.0+cu128`)

All real-model experiments are strictly separated from synthetic micro-benchmarks (see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) and [`evidence/benchmarks/real_model_benchmarks.json`](evidence/benchmarks/real_model_benchmarks.json)):

- **`Qwen/Qwen2.5-0.5B` (`060db6499f`) → `Qwen/Qwen2.5-0.5B-Instruct` (`7ae557604a`)** (`494,032,768` parameters, `290` tensors):
  - **Weight Drift:** `97.9318%` of parameters exceeded tolerance (`mean cosine = 0.998495`, `mean rel-L2 = 0.044294`). Largest relative $L_2$ updates localized in `model.embed_tokens.weight` (`0.1116`) and upper Query projections `layers.17–23.self_attn.q_proj.weight` (`0.0937`–`0.1051`).
  - **Activation Alignment:** Cross-prompt final-layer Linear CKA = `0.991076`; final-token JS divergence = `0.070651`.
  - **Deterministic Behavioral Suite (24 paired cases):** Overall pass rate rose from **`45.8%`** to **`70.8%`** (`+25.0%`), driven by `instruction_following` (`0% → 67%`), `vietnamese` (`0% → 67%`), `conciseness` (`0% → 67%`), `code_preservation` (`33% → 67%`), and `refusal_behavior` (`67% → 100%`), while the regression gate automatically caught a `math_reasoning` verbose chain-of-thought truncation under a `96`-token budget.
- **`HuggingFaceTB/SmolLM2-135M` (`93efa2f097`) → `SmolLM2-135M-Instruct` (`12fd25f773`)** (`134,515,008` parameters, `272` tensors):
  - **Weight & Activation:** `97.6728%` changed (`mean cosine = 0.996951`, final-layer CKA = `0.951782`); behavioral pass rate rose `20.8% → 33.3%` (`+0.1819` mean score delta).
- **Chunked Tensor Engine Throughput:** Up to **`116.65M` elements/sec** (`float64`-accumulated) with **`+0.52 MB`** RSS delta during lazy Safetensors streaming.

---

## Epistemic Classification Discipline

Every report and UI view explicitly labels claim strength:
- `OBSERVED`: Direct deterministic calculation over checkpoint tensors, shapes, activations, or outputs.
- `INFERRED`: Rule-based structural rename matching across unmatched same-role tensors (`cos >= 0.985`).
- `CORRELATED`: Co-occurring internal activation drift and behavioral changes without counterfactual intervention.
- `CAUSALLY VERIFIED`: Reserved strictly for counterfactual activation-patching verification.
- `NOT VERIFIED`: Unexecuted claim boundary explicitly disclosed.

---

## Reproducible Quickstart

```powershell
# 1. Provision locked Python 3.12 + PyTorch cu128 environment
python -m uv sync --locked --python 3.12

# 2. Provision frontend dependencies and build production assets
npm --prefix frontend ci
npm --prefix frontend run build

# 3. Run CLI comparison (exports .json, .md, and standalone SVG-heatmap .html)
.\.venv\Scripts\python.exe -m core.cli compare `
  --model-a Qwen/Qwen2.5-0.5B `
  --model-b Qwen/Qwen2.5-0.5B-Instruct `
  --modules structural,tensor,neuron,activation,behavioral,regression `
  --real-experiment `
  --output-dir ./reports

# 4. Launch Full-Stack Web Workbench
.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000
# In a second terminal:
npm --prefix frontend run dev
```

---

## Documentation Index

- [Installation & Setup](docs/INSTALLATION.md)
- [System Architecture](docs/ARCHITECTURE.md)
- [Environment Audit](docs/ENVIRONMENT_AUDIT.md)
- [Technical Research](docs/RESEARCH.md)
- [Competitive Analysis](docs/COMPETITIVE_ANALYSIS.md)
- [Tensor & Structural Math](docs/TENSOR_COMPARISON.md)
- [Activation Analysis & CKA](docs/ACTIVATION_ANALYSIS.md)
- [Behavioral Evaluation & Regression](docs/BEHAVIORAL_EVALUATION.md)
- [Benchmarks & Real Model Experiments](docs/BENCHMARKS.md)
- [Security Controls & Audit](docs/SECURITY.md)
- [Known Limitations](docs/LIMITATIONS.md)
- [Roadmap](docs/ROADMAP.md)
- [Contributing Guide](CONTRIBUTING.md) • [Changelog](CHANGELOG.md)
- [Final Engineering Report](evidence/FINAL_ENGINEERING_REPORT.md) • [Screenshot Index](evidence/SCREENSHOT_INDEX.md)

---

## License

Released under the [MIT License](LICENSE).
