# Changelog

All notable changes to **NEURAL DIFF** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-11-27

### Added
- **Module I — Neural Evolution Tracking (`core/evolution/`):**
  - Validated Directed Acyclic Graph (`EvolutionDAG`) supporting linear chains, experimental branches, multi-parent Model Merging, Lowest Common Ancestor (LCA), and 3-color DFS circular dependency rejection (`CYCLIC_DEPENDENCY_ERROR`).
  - Streaming 4 MiB chunked SHA-256 checkpoint registry (`registry.py`) with duplicate content detection, PEFT LoRA `adapter_config.json` & quantization metadata extraction, and versioned SQLite schema migrations (`EvolutionStore`).
  - Multi-Generation & Direct Cumulative Tensor Evolution (`tensor_evolution.py`) strictly separating `consecutive`, direct `base_to_current` (`W_current - W_base`), and `branch_to_branch` scopes with path-cancellation ratios.
  - Dedicated Layer Evolution Engine (`layer_evolution.py`), Multi-Generation Behavioral Evolution Engine (`behavioral_evolution.py`) with Wilson 95% CIs and surviving-regression tracking, Representation Evolution Engine (`representation_evolution.py`) with Centered Linear CKA matrices & cross-checkpoint causal patching, and Automatic Regression Origin Pinpointing (`regression_origin.py`) strictly distinguishing `CORRELATED` from `CAUSALLY VERIFIED`.
  - Pure-Python `%PDF-1.4` multi-page vector report generator (`pdf_writer.py`) alongside `JSON`, `CSV`, and `HTML` (`exporter.py`), 20 REST endpoints under `/api/evolution/*`, `neural-diff evolution` CLI subcommands, and the interactive ReactFlow `EvolutionStudioView` UI.
- **CI/CD Hardening (`.github/workflows/`):** 8 specialized GitHub Actions pipelines (`backend-ci.yml`, `frontend-ci.yml`, `e2e-playwright.yml`, `security-audit.yml`, `codeql.yml`, `numerical-benchmarks.yml`, `docs-and-evidence.yml`, and `package-and-release.yml`).

## [0.1.0] - 2026-11-27

### Added
- **Module A — Unified Model Loader (`core/loaders`, `adapters/`):** Single, multi-file, and sharded Safetensors lazy slice readers; opcode-inspected `weights_only=True` PyTorch checkpoint loader; Hugging Face local directory and Hub snapshot resolver (`trust_remote_code=False`, executable files excluded).
- **Module B — Chunked Tensor Diff Engine (`core/tensor_engine`, `core/statistics/distributions.py`):** Out-of-core `float64`-accumulated max/mean abs diff, RMSE, relative $L_2$, cosine similarity, tolerance-bounded changed percentage, zero/nonzero transitions, NaN/Inf counters, and Wasserstein-1 quantile distribution shifts.
- **Module C — Structural & Compatibility Analysis (`core/structural_analysis`):** Multi-dimensional compatibility validator, added/removed/shape/dtype diffing, layer topology comparison, and conservative cross-architecture-guarded rename inference (`EpistemicStatus.INFERRED`).
- **Module D — Neuron & Attention Analysis (`core/structural_analysis/neuron_attention.py`):** GQA-aware per-head `q`/`k`/`v`/`o` slicing, MLP coordinate shift analysis with explicit superposition warnings, batched token-embedding drift, and normalization gain analysis.
- **Module E — Activation Diff Engine (`core/activation_analysis`):** Sequential memory-bounded forward-hook capture for hidden states, attention outputs, and MLP intermediates; Linear CKA; per-token drift; and next-token KL/Jensen-Shannon logit divergence.
- **Module F — Deterministic Behavioral Diff Engine (`core/behavioral_analysis`, `benchmarks/`):** 24-case `neural-diff-core-v1` benchmark suite and 8 zero-judge deterministic evaluators (`vietnamese`, `code_preservation`, `json_reliability`, `math_reasoning`, `instruction_following`, `conciseness`, `refusal_behavior`, `factual_accuracy`) with paired bootstrap 95% CIs and exact McNemar $p$-values.
- **Module G — Regression Engine (`core/regression_engine`):** Configurable severity thresholds, versioned baseline snapshots, and deterministic CLI failure reproduction commands.
- **Module H — Report Engine (`core/report_engine`):** Strict RFC-8259 JSON, GitHub-flavored Markdown, and self-contained SVG-heatmap HTML report exporters with epistemic ledgers.
- **Interfaces (`core/cli.py`, `api/`, `frontend/`):** `neural-diff` CLI, hardened FastAPI REST backend with SQLite persistence and cooperative cancellation, and 12-view React + TypeScript + Vite web application.
- **Verification & Evidence (`tests/`, `scripts/`, `evidence/`):** Unit, Hypothesis property, integration, golden regression, security, Vitest, and Playwright E2E test suites; real-model comparisons (`SmolLM2-135M`, `Qwen2.5-0.5B`); SHA-256 evidence manifest generator and verifier.

### Security
- Remediated transitive `tinypool`/`@vitest/mocker` npm advisories (`vitest@^5.1.2`, `vite@^8.4.0` -> `0` npm vulnerabilities) and Python `setuptools` advisory (`setuptools>=84.1.0` -> `0` `pip-audit` vulnerabilities).
