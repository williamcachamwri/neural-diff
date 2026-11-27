# Changelog

All notable changes to **NEURAL DIFF** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and changes are organized by subsystem and engineering impact.

---

## [Unreleased]

### Added
- **4-Tier Epistemic Rigor, Separated Regression/Causal Flags & Statistical Safeguards ([`core/evolution/regression_origin.py`](core/evolution/regression_origin.py), [`core/evolution/behavioral_evolution.py`](core/evolution/behavioral_evolution.py)):**
  - Explicitly separated `behavioral_regression_observed` (binary pass-rate drop), `score_regression_observed` (continuous rubric score drop), and `causal_intervention_verified` (case-complete token-logit counterfactual provenance) across backend events, benchmark JSON, 5-format exports, and Studio UI cards; `score_only` regressions (`behavioral_regression_observed = false`) are strictly capped at `CORRELATED` / `OBSERVED` so single-token logit patching is never conflated with continuous rubric score causation.
  - Introduced `EpistemicStatus.CAUSAL_SUPPORT` (`"CAUSAL SUPPORT"`) and `CausalInterventionProvenance` (`validate_causal_verification_provenance`) requiring `case_targeted`, `target_vs_foil_verified`, `negative_control_verified`, and case+site `verification_scope` to be explicitly `true` for every failed case in an event (with `prompt_fingerprint`, `failed_case_id`, `layer_index`, `intervention_type`, `target_token`, `foil_token`, `negative_control_result`, and `recovery_measurement`; missing evidence never defaults to `true`).
  - Added per-category paired McNemar raw $p$-values, step-down **Holm-Bonferroni** family-wise error adjustment (`holm_bonferroni_p_value`) with deterministic tie-breaking, explicit `regression_kind` classification (`pass_rate_and_score`, `pass_rate_only`, `score_only`), sample-size floors (`min_category_sample_size = 8`, `is_exploratory`, `statistically_significant`), binomial discrete power disclosures, and custom `benchmark_suite_path` support.
- **Exact $O(V+E)$ DAG Branch Counting, Maximal Common Ancestors, Lineage-State DP & 7-Way Scope Taxonomy ([`core/evolution/dag.py`](core/evolution/dag.py), [`core/evolution/regression_origin.py`](core/evolution/regression_origin.py), [`api/routes/evolution.py`](api/routes/evolution.py)):**
  - Added `count_total_branches()` (reverse-topological DP), lazy `iter_branches()`, `maximal_common_ancestors()` alongside deterministic tie-broken `lowest_common_ancestor()`, explicit truncation metadata (`branches_truncated`, `analyzed_branch_count`, `max_branches_limit`), $O(|V|\cdot|E|)$ complete-DAG lineage-state DP across multi-root, multi-merge, and recovery transitions (`leaf_outcomes`, `surviving_leaf_ids`, `recovered_leaf_ids`), and `classify_pair_scope()` (`identical`, `consecutive`, `base_to_current`, `ancestor_to_descendant`, `descendant_to_ancestor`, `branch_to_branch`, `unrelated`).
- **Orthogonal 5-Domain Checkpoint SHA-256 Fingerprinting ([`core/evolution/registry.py`](core/evolution/registry.py), [`core/evolution/store.py`](core/evolution/store.py)):**
  - Added migration `0002_checkpoint_component_fingerprints` and `compute_checkpoint_component_fingerprints()` separating `weights_sha256` (raw tensor slices), `architecture_config_sha256`, `tokenizer_sha256` (`tokenizer.json`, `tokenizer.model`, `spiece.model`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `special_tokens_map.json`, `added_tokens.json`, `chat_template.jinja`, `chat_template.json`), `adapter_sha256`, and `metadata_sha256` (`.safetensors` descriptors & `__metadata__` across both primary weights and coexisting PEFT adapter files), invalidating behavioral caches whenever any component changes.
- **Precision Rounded Studio UI, Optical Signal Cables & Brand Mark ([`frontend/src/`](frontend/src/)):**
  - Redesigned Studio shell with floating `16px`-rounded glass inspector sheets, 3-layer optical Bezier/PCB signal cables, `⌘K` Quick Command Palette, non-overlapping `EvoOpticalEdge` DAG labels, and transparent squircle `ND` brand mark.
- **Minimalist 5-Format Visual Report Primitives ([`core/report_engine/visuals.py`](core/report_engine/visuals.py)):**
  - Data-driven SVG generators following Architectural Monochrome (`#09090b` void, `#121216` hairline surfaces, `#f4f4f5` titanium signal, `#d6c5a9` champagne luminance): `svg_evolution_dag` (branched & merged Bezier DAG), `svg_dual_line_chart` (oscilloscope area + dual trajectory), `svg_matrix_heatmap` (luminance pixel grid with hover tooltips), `svg_radar_polygon` (multi-series polar capability polygon), `svg_lollipop_chart`, and `svg_segmented_ratio_bar`.
  - Markdown visual helpers (`md_bar` proportional Unicode `█░` gauges and `human_label` enum normalization) plus dynamic `mermaid` `flowchart LR` generation across both Pairwise (`ReportEngine.to_markdown`) and Evolution (`EvolutionReportExporter.to_markdown`) reports.
  - Pure-Python `%PDF-1.4` vector trajectory polyline chart (`MinimalPDFBuilder.add_polyline_chart` in [`core/evolution/pdf_writer.py`](core/evolution/pdf_writer.py)) alongside Swiss-grid hairline tables and horizontal vector bars.
  - Evolution Markdown export endpoint (`GET /api/evolution/runs/{run_id}/export/markdown`) and CLI `--output-dir` 5-format export (`json`, `csv`, `markdown`, `html`, `pdf`).
- **Studio Interactive SVG Chart Library ([`frontend/src/components/charts/StudioVisuals.tsx`](frontend/src/components/charts/StudioVisuals.tsx)):**
  - Added `DualTrajectoryChart` (with optional 95% Wilson CI band), `PolarRadarChart`, `PixelMatrixHeatmap`, and `DivergingDeltaBars` across `ActivationComparisonView`, `BehavioralDiffView`, `EvolutionStudioView` (3-chart overview triptych), and `EvolutionAnalyticsTabs`.
  - Upgraded `ReportViewerView` and `ExperimentExportSection` to render live sandboxed Visual HTML reports (`<iframe sandbox="">`) by default alongside instant `.html`, `.md`, `.pdf`, `.csv`, and `.json` downloads.
- **10 Multi-Node GitHub Actions CI/CD Workflows ([`.github/workflows/`](.github/workflows)):**
  - Added [`evolution-pipeline.yml`](.github/workflows/evolution-pipeline.yml) (4-node DAG verifying SQLite migrations, 3-color DFS cycle guard, direct cumulative $L_2$ math, live branched/merged lifecycle execution, and strict 5-format XML/SVG/PDF/CSV/JSON validation).
  - Added [`api-contract-and-db.yml`](.github/workflows/api-contract-and-db.yml) (3-node DAG auditing all 37 FastAPI OpenAPI routes, multi-threaded SQLite WAL concurrency + SHA-256 cache invalidation, and CLI JSON stdout contracts).
  - Expanded existing workflows into realistic multi-node job graphs with `Node.js 20 / 22 LTS` matrices, `Python 3.11 / 3.12` matrices, `ubuntu-latest` + `windows-latest` portability verification, and parallel Playwright spec sharding (`e2e.spec.ts` $\parallel$ `evolution.e2e.spec.ts`).
- **Documentation & Live Evidence (`01`..`27`):**
  - Updated primary English [`README.md`](README.md) and all 4 translations ([Vietnamese](docs/i18n/README.vi.md), [Simplified Chinese](docs/i18n/README.zh-CN.md), [Japanese](docs/i18n/README.ja.md), [French](docs/i18n/README.fr.md)) with the 8-Layer System Architecture and all **27 live Playwright Chromium screenshots**.
  - Added [`scripts/refresh_persisted_reports.py`](scripts/refresh_persisted_reports.py) to regenerate persisted SQLite and `evidence/` reports from canonical `ComparisonBundle` JSON records.

### Changed
- **Zero-Slop Report Redesign ([`core/report_engine/exporter.py`](core/report_engine/exporter.py), [`core/evolution/exporter.py`](core/evolution/exporter.py)):** Replaced legacy rainbow `.badge` pills, ALL-CAPS headings, `[OBSERVED]` bullet walls, unformatted float dumps, and raw `<pre>` JSON blocks with restrained `6px` status dots, tabular numerals, hairline Swiss-grid tables, and 5 embedded SVG visualizations per HTML report.

### Fixed & Performance
- **Local Snapshot Resolution in Arbitrary Pair Comparisons ([`core/evolution/tensor_evolution.py`](core/evolution/tensor_evolution.py)):** `MultiGenTensorEvolutionEngine.compare_pair` now loads verified on-disk `resolved_path` snapshots directly before falling back to Hub IDs, eliminating redundant external Hub ETag round-trips and cutting `135M`-parameter pair comparisons to `< 0.5s`.

---

## [0.2.0] - 2026-11-27

### Added
- **Module I — Neural Evolution Tracking ([`core/evolution/`](core/evolution/)):**
  - Validated Directed Acyclic Graph ([`dag.py`](core/evolution/dag.py)) supporting linear chains, experimental branches, multi-parent Model Merging with merge weights, Kahn topological sorting, Lowest Common Ancestor (LCA), and 3-color DFS circular dependency rejection (`CYCLIC_DEPENDENCY_ERROR`, `HTTP 422`).
  - Streaming 4 MiB chunked SHA-256 checkpoint registry ([`registry.py`](core/evolution/registry.py)) supporting all 14 `TransformationType` stages, duplicate content detection (`DUPLICATE_CONTENT`), PEFT LoRA `adapter_config.json` & quantization metadata extraction, and versioned SQLite schema migrations (`0001_evolution_core_tables` in [`store.py`](core/evolution/store.py)).
  - Multi-Generation & Direct Cumulative Tensor Evolution ([`tensor_evolution.py`](core/evolution/tensor_evolution.py)) strictly separating `consecutive`, direct `base_to_current` (`||W_current - W_base||_2`, never summing step norms), and `branch_to_branch` scopes with `path_cancellation_ratio`.
  - Dedicated Layer Evolution Engine ([`layer_evolution.py`](core/evolution/layer_evolution.py)), Multi-Generation Behavioral Evolution Engine ([`behavioral_evolution.py`](core/evolution/behavioral_evolution.py)) with Wilson 95% CIs and surviving-regression tracking, Representation Evolution Engine ([`representation_evolution.py`](core/evolution/representation_evolution.py)) with Centered Linear CKA matrices & cross-checkpoint residual patching, and Automatic Regression Origin Pinpointing ([`regression_origin.py`](core/evolution/regression_origin.py)) strictly distinguishing `CORRELATED` from `CAUSALLY VERIFIED`.
  - 20 REST endpoints under `/api/evolution/*` ([`api/routes/evolution.py`](api/routes/evolution.py)), `neural-diff evolution` CLI subcommands, runnable [`examples/track_model_evolution.py`](examples/track_model_evolution.py), and the real 6-checkpoint `SmolLM2-135M` branched & merged experiment (`evorun-real-smollm2`).

### Fixed
- **Windows Safetensors Memory-Mapped Handle Lifetime ([`adapters/safetensors/loader.py`](adapters/safetensors/loader.py)):** Cloned full-tensor slices returned by `SingleSafetensorsReader.get_tensor` before closing `safe_open(...)`, preventing Windows `0xC0000005` access violations when holding hundreds of tensors across derived checkpoints.
- **Clean-Room Wheel Path Confinement ([`core/security.py`](core/security.py)):** Included `Path.cwd().resolve()` and `NEURAL_DIFF_ALLOWED_ROOTS` in `default_allowed_roots()` so installed wheels invoked outside the source checkout validate user-specified working directories safely.
- **Store Import Isolation ([`core/evolution/store.py`](core/evolution/store.py)):** Decoupled `DEFAULT_EVOLUTION_DB_PATH` from `api.store` to avoid circular package initialization when importing `core.evolution` standalone.

---

## [0.1.0] - 2026-11-27

### Added
- **Module A — Unified Safe Model Loader (`core/loaders/`, `adapters/`):** Single, multi-file, and sharded Safetensors lazy slice readers; opcode-inspected `weights_only=True` PyTorch checkpoint loader; Hugging Face local directory and Hub snapshot resolver (`trust_remote_code=False`, executable files excluded).
- **Module B — Chunked Float64 Tensor & SVD Engine (`core/tensor_engine/`, `core/statistics/`):** Out-of-core `float64`-accumulated max/mean abs diff, RMSE, relative $L_2$, cosine similarity, tolerance-bounded changed percentage, zero/nonzero transitions, NaN/Inf counters, Wasserstein-1 quantile shifts, aligned histograms, top-$k$ coordinate deltas, and SVD Shannon effective rank.
- **Module C — Structural, Weight-Tying & 2:4 Sparsity Analysis (`core/structural_analysis/`):** Multi-dimensional compatibility validator, added/removed/shape/dtype diffing, layer topology comparison, conservative rename inference (`INFERRED`), weight-tying verification, and structured / NVIDIA `2:4` semi-structured sparsity compliance.
- **Module D — Neuron & GQA Attention Analysis (`core/structural_analysis/neuron_attention.py`):** GQA-aware per-head `q`/`k`/`v`/`o` slicing, MLP coordinate shift analysis with superposition disclosures, batched token-embedding drift, and normalization gain analysis.
- **Module E — Activation Diff, Logit Lens & Residual Patching (`core/activation_analysis/`):** Sequential memory-bounded forward-hook capture, Centered Linear CKA, next-token KL/JS logit divergence, observational Logit Lens, and counterfactual residual-stream activation patching (`CAUSALLY VERIFIED`).
- **Module F — Deterministic Behavioral Harness (`core/behavioral_analysis/`, `benchmarks/`):** 24-case `neural-diff-core-v1` suite and 8 zero-LLM-judge deterministic evaluators (`vietnamese`, `code_preservation`, `json_reliability`, `math_reasoning`, `instruction_following`, `conciseness`, `refusal_behavior`, `factual_accuracy`) with paired bootstrap 95% CIs and exact McNemar $p$-values.
- **Module G & Module H — Regression Gate & Report Engine (`core/regression_engine/`, `core/report_engine/`):** Configurable severity thresholds, versioned SQLite/JSON baselines, multi-run `diff-of-diffs` trend tracking, 1-click CLI reproducers, and RFC-8259 JSON / Markdown / HTML exports.
- **Interfaces & Verification (`core/cli.py`, `api/`, `frontend/`, `tests/`):** `neural-diff` CLI, hardened FastAPI backend with WAL SQLite persistence and cooperative cancellation, ComfyUI Nodes 2.0 React 19 + XYFlow Studio, real-model benchmarks (`Qwen2.5-0.5B`, `SmolLM2-135M`), and SHA-256 `EVIDENCE_MANIFEST.json` verification.

### Security
- Remediated transitive `tinypool`/`@vitest/mocker` npm advisories (`vitest@^5.1.2`, `vite@^8.4.0` $\to$ `0` npm vulnerabilities) and Python `setuptools` advisory (`setuptools>=84.1.0` $\to$ `0` `pip-audit` vulnerabilities).
