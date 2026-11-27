# NEURAL DIFF — Final Engineering Report

All measurements, exit codes, file paths, and status classifications in this report come directly from executed commands and artifacts recorded under `evidence/`.

Status labels used strictly per Section 1.1: `VERIFIED`, `PARTIALLY VERIFIED`, `NOT VERIFIED`, `BLOCKED`, `FAILED`, `NOT IMPLEMENTED`.

---

## A. Environment (`VERIFIED`)

Captured via [`scripts/capture_environment.py`](../scripts/capture_environment.py) in [`evidence/environment/environment_post_install.json`](environment/environment_post_install.json) and documented in [`docs/ENVIRONMENT_AUDIT.md`](../docs/ENVIRONMENT_AUDIT.md):

* **Operating System:** Windows 11 (`10.0.26320`, `AMD64`)
* **CPU & Memory:** Intel64 Family 6 Model 183 Stepping 1 (`16` logical CPUs), `15.71 GB` total RAM
* **Storage:** Workspace drive `D:\LLM` (`200.05 GB` total, `88.68 GB` free at baseline)
* **GPU & CUDA:** `NVIDIA GeForce RTX 5050 Laptop GPU` (`sm_120` Blackwell, `8,150.56 MB` VRAM) verified executing live CUDA tensor kernels under `torch==2.10.0+cu128`
* **Apple Silicon / Metal:** Not present (`apple_silicon_hardware: false`, `metal_mps_supported: false`)
* **Toolchains & Lockfiles:** CPython `3.12.16` managed by `uv 0.13.10` ([`pyproject.toml`](../pyproject.toml), [`uv.lock`](../uv.lock)); Node.js + Vite `8.4.0` + TypeScript ([`frontend/package.json`](../frontend/package.json), [`frontend/package-lock.json`](../frontend/package-lock.json))

---

## B. Implementation Status (`VERIFIED`)

| Subsystem | Verification Status | Implementation & Evidence |
| :--- | :--- | :--- |
| **Module A — Safe Model Loader** | `VERIFIED` | Single & sharded Safetensors, Hugging Face local snapshots & Hub cache (`trust_remote_code=False`), opcode-scanned `weights_only=True` PyTorch archives (`core/loaders/`, `adapters/`) |
| **Module B — Float64 Tensor & SVD Engine** | `VERIFIED` | Lazy chunked `float64` accumulation (`max_abs_diff`, `mean_abs_diff`, `rmse`, `relative_l2`, `cosine`, quantiles, Wasserstein-1, aligned histograms, top-$k$ coordinate deltas, SVD task-vector Shannon effective rank) (`core/tensor_engine/`, `core/statistics/`) |
| **Module C — Structural & Compatibility Diff** | `VERIFIED` | Naming-agnostic canonical roles, added/removed keys, cosine rename inference ($\cos \ge 0.985$, `INFERRED`), shape/dtype shifts, weight-tying verification, structured & NVIDIA `2:4` sparsity (`core/structural_analysis/`) |
| **Module D — Neuron & GQA Attention Slicing** | `VERIFIED` | Per-head Q/K/V/O slicing (including GQA `14 Q / 2 KV` heads), MLP basis coordinate analysis, vocabulary embedding drift (`core/structural_analysis/neuron_attention.py`) |
| **Module E — Activation Diff, Logit Lens & Patching** | `VERIFIED` | Identical-prompt forward hooks, orthogonal-invariant Linear CKA, next-token JS divergence (`OBSERVED`), Logit Lens (`OBSERVED`), and counterfactual residual-stream activation patching (`CAUSALLY VERIFIED` on paired prompts) (`core/activation_analysis/`) |
| **Module F — Deterministic Behavioral Harness** | `VERIFIED` | 8 zero-LLM-judge evaluation suites (`24` paired cases), paired bootstrap 95% CI, exact two-sided McNemar test (`core/behavioral_analysis/`, `benchmarks/`) |
| **Module G — Regression Engine & Diff-of-Diffs** | `VERIFIED` | Threshold policy evaluation, severity classification, SQLite/JSON baselines, deterministic CLI reproducers, multi-run historical trend comparator (`core/regression_engine/`) |
| **Module H — Multi-Format Report Engine** | `VERIFIED` | Minimalist Architectural Monochrome HTML (5 inline SVGs), Mermaid + Unicode sparkbar Markdown, and strict JSON exports (`core/report_engine/exporter.py`, `core/report_engine/visuals.py`) |
| **Module I — Neural Evolution Tracking** | `VERIFIED` | Branched & merged DAG (`core/evolution/dag.py`), SHA-256 registry (`registry.py`), direct base-to-current cumulative math (`tensor_evolution.py`), layer engine, behavioral Wilson/McNemar timeline, representation CKA + cross-checkpoint patching, regression origin pinpointing, SQLite migrations (`store.py`), 20 REST routes, and JSON/CSV/MD/HTML/PDF exporters |
| **FastAPI Backend, Diagnostics & CLI** | `VERIFIED` | REST API (`api/app.py`, `api/routes/diagnostics.py`), WAL SQLite store (`data/neural_diff.db`), security hardening middleware, cooperative job cancellation, and `neural-diff` CLI (`core/cli.py`) |
| **ComfyUI Nodes 2.0 Monochrome Frontend Studio** | `VERIFIED` | Unified single-header React 19 + TypeScript + `@xyflow/react` + `framer-motion` studio with 9-node interactive DAG, Bezier/Orthogonal PCB cable routing, proportional head bars, master-detail behavioral workbench, and live SVD probe (`frontend/`) |
| **Multilingual Documentation (5 Languages)** | `VERIFIED` | Primary English [`README.md`](../README.md) + full translations in Vietnamese, Simplified Chinese, Japanese, and French under [`docs/i18n/`](../docs/i18n/) + 12 technical guides under [`docs/`](../docs/) |

---

## C. Automated Tests & Static Analysis (`VERIFIED`)

Executed via `.venv\Scripts\python.exe scripts/run_all_tests.py` and `.venv\Scripts\python.exe scripts/capture_screenshots.py` (summary recorded in [`evidence/tests/test_execution_summary.json`](tests/test_execution_summary.json)):

| Suite | Exact Command | Exit Code | Result | Evidence Log |
| :--- | :--- | ---: | :--- | :--- |
| Ruff Linting | `python -m ruff check core adapters benchmarks api tests scripts` | `0` | **PASS** | [`evidence/tests/ruff-lint.log`](tests/ruff-lint.log) |
| MyPy Typecheck | `python -m mypy core adapters benchmarks api` (`86` files) | `0` | **PASS** | [`evidence/tests/mypy-typecheck.log`](tests/mypy-typecheck.log) |
| Unit Tests | `python -m pytest -m unit -v` | `0` | **47 passed** | [`evidence/tests/unit-tests.log`](tests/unit-tests.log) |
| Property Tests | `python -m pytest -m property -v` (Hypothesis) | `0` | **7 passed** | [`evidence/tests/property-tests.log`](tests/property-tests.log) |
| Integration Tests | `python -m pytest -m integration -v` | `0` | **6 passed** | [`evidence/tests/integration-tests.log`](tests/integration-tests.log) |
| Regression Tests | `python -m pytest -m regression -v` | `0` | **1 passed** | [`evidence/tests/regression-tests.log`](tests/regression-tests.log) |
| Security Tests | `python -m pytest -m security -v` | `0` | **8 passed** | [`evidence/tests/security-tests.log`](tests/security-tests.log) |
| Full Coverage | `python -m pytest --cov=core --cov=adapters --cov=benchmarks --cov=api` | `0` | **69 passed · `89%` (`8,149` stmts)** | [`evidence/tests/coverage.log`](tests/coverage.log) |
| Frontend Typecheck | `npm --prefix frontend run typecheck` | `0` | **PASS** | [`evidence/tests/frontend-typecheck.log`](tests/frontend-typecheck.log) |
| Frontend Unit | `npm --prefix frontend run test` (Vitest) | `0` | **4 passed (3 files)** | [`evidence/tests/frontend-vitest.log`](tests/frontend-vitest.log) |
| Production Build | `npm --prefix frontend run build` (Vite 8) | `0` | **PASS** | [`evidence/tests/frontend-build.log`](tests/frontend-build.log) |
| Live Full-Stack E2E | `npx playwright test` (Chromium) | `0` | **2 passed** | [`evidence/tests/playwright-e2e.log`](tests/playwright-e2e.log) |

**Preserved Historical Failure Logs & Corrections (Section 2.4):**
* [`evidence/tests/ruff-lint-initial-failure.log`](tests/ruff-lint-initial-failure.log) & [`evidence/tests/mypy-typecheck-initial-failure.log`](tests/mypy-typecheck-initial-failure.log) — Initial import sorting and variable-shadowing lints resolved cleanly.
* [`evidence/tests/integration-tests-initial-failure.log`](tests/integration-tests-initial-failure.log) — Custom temporary `db_path` baseline root confinement fixed by registering `store.db_path.parent` in `_effective_policy()`.
* [`evidence/security/pip_audit_initial_failure.log`](security/pip_audit_initial_failure.log) — Transitive `setuptools 81.0.0` advisory remediated by upgrading to `setuptools 84.2.1` (`0` vulnerabilities in [`evidence/security/pip_audit.log`](security/pip_audit.log)).

---

## D. Live Application Screenshots (`VERIFIED`)

All 27 live screenshots (`01`..`15` Pairwise Studio + `16`..`27` Neural Evolution Studio & Exported Report) are indexed with timestamps, browser engine (`Chromium`), and provenance in [`evidence/SCREENSHOT_INDEX.md`](SCREENSHOT_INDEX.md):

1. [`evidence/frontend/01-dashboard.png`](frontend/01-dashboard.png) — 9-node ComfyUI workflow graph canvas (Bezier spline mode).
2. [`evidence/frontend/02-model-selection.png`](frontend/02-model-selection.png) — Model A & Model B comparison configuration.
3. [`evidence/frontend/03-compatibility-validation.png`](frontend/03-compatibility-validation.png) — Live compatibility validation on `Qwen/Qwen2.5-0.5B` vs `Instruct`.
4. [`evidence/frontend/04-tensor-diff.png`](frontend/04-tensor-diff.png) — Float64 Tensor Explorer with live SVD & Sparsity probe control.
5. [`evidence/frontend/05-structural-diff.png`](frontend/05-structural-diff.png) — Cross-architecture structural diff (`Qwen2.5-0.5B` vs `SmolLM2-135M`).
6. [`evidence/frontend/06-layer-analysis.png`](frontend/06-layer-analysis.png) — 24-layer submodule heatmap and proportional GQA head divergence bars.
7. [`evidence/frontend/07-activation-comparison.png`](frontend/07-activation-comparison.png) — Forward-hook Linear CKA alignment and Top-5 next-token probability bars.
8. [`evidence/frontend/08-behavioral-diff.png`](frontend/08-behavioral-diff.png) — 8-suite behavioral breakdown and master-detail paired response comparator.
9. [`evidence/frontend/09-regression-report.png`](frontend/09-regression-report.png) — Regression gate findings and 1-click copyable CLI reproduction commands.
10. [`evidence/frontend/10-benchmark-results.png`](frontend/10-benchmark-results.png) — Experiment ledger separating real-model runs from synthetic fixtures.
11. [`evidence/frontend/11-report-export.png`](frontend/11-report-export.png) — Standalone HTML report preview and `.html`/`.md`/`.json` exports.
12. [`evidence/frontend/12-test-execution-summary.png`](frontend/12-test-execution-summary.png) — Automated test & `90%` statement coverage ledger.
13. [`evidence/frontend/13-error-handling.png`](frontend/13-error-handling.png) — Live `HTTP 403` path-traversal rejection banner.
14. [`evidence/frontend/14-settings.png`](frontend/14-settings.png) — SQLite runtime and safe serialization policy settings.
15. [`evidence/frontend/15-final-application-state.png`](frontend/15-final-application-state.png) — Final studio state in Orthogonal PCB wire-routing mode.
16. [`evidence/frontend/16-evolution-dashboard.png`](frontend/16-evolution-dashboard.png) — Evolution Analytics Dashboard with 3-panel SVG trajectory triptych and empirical findings.
17. [`evidence/frontend/17-evolution-timeline.png`](frontend/17-evolution-timeline.png) — Evolution lineage timeline and live arbitrary pair comparison (`classify_pair_scope` + LCA).
18. [`evidence/frontend/18-evolution-branched-graph.png`](frontend/18-evolution-branched-graph.png) — Interactive branched & merged optical DAG (`v0 → v1a/v1b → v2 → v3 → v4`).
19. [`evidence/frontend/19-evolution-checkpoint-registry.png`](frontend/19-evolution-checkpoint-registry.png) — Checkpoint Registry displaying all 6 SmolLM2-135M checkpoints and verification states.
20. [`evidence/frontend/20-evolution-checkpoint-details.png`](frontend/20-evolution-checkpoint-details.png) — Checkpoint Details inspector showing 5-domain SHA-256 component hashes (`Weights`, `Config`, `Tokenizer`, `Adapter`, `Metadata`).
21. [`evidence/frontend/21-evolution-tensor-comparison.png`](frontend/21-evolution-tensor-comparison.png) — Multi-generation direct cumulative weight distance (`||W_current − W_base||_2` vs step sum), sparsity deltas, and `≠0 → 0` transitions.
22. [`evidence/frontend/22-evolution-layer-heatmap.png`](frontend/22-evolution-layer-heatmap.png) — 30-layer weight evolution pixel heatmap and canonical submodule breakdown.
23. [`evidence/frontend/23-evolution-behavioral-chart.png`](frontend/23-evolution-behavioral-chart.png) — Multi-generation behavioral pass-rate curve (95% Wilson CI) and 8-domain capability radar polygon.
24. [`evidence/frontend/24-evolution-representation-matrix.png`](frontend/24-evolution-representation-matrix.png) — Layer-wise Centered Linear CKA matrix, representation drift, and diagnostic `CAUSAL SUPPORT` patching.
25. [`evidence/frontend/25-evolution-regression-origin.png`](frontend/25-evolution-regression-origin.png) — Automatic Regression Origin pinpointing with 4-tier epistemic labels, `regression_kind` separation, and Holm-Bonferroni small-sample safeguards.
26. [`evidence/frontend/26-evolution-experiment-details.png`](frontend/26-evolution-experiment-details.png) — Experiment reproducibility provenance and 5-format report export workbench.
27. [`evidence/frontend/27-evolution-exported-report.png`](frontend/27-evolution-exported-report.png) — Standalone rendered HTML Evolution Report with 5 embedded data-driven SVGs.

---

## E. Real Model Experiments (`VERIFIED` & `BLOCKED` Disclosed)

Recorded in [`evidence/benchmarks/real_model_benchmarks.json`](benchmarks/real_model_benchmarks.json) (`REAL MODEL EXPERIMENT` strictly separated from synthetic fixtures):

* **`ndiff-real-qwen25-05b` (`VERIFIED`):** `Qwen/Qwen2.5-0.5B` (`060db6499f...`) $\to$ `Qwen/Qwen2.5-0.5B-Instruct` (`7ae557604a...`), `494,032,768` parameters, `290` tensors, `97.9318%` changed, mean cosine `0.998495`, final-layer pooled CKA `0.991076`, behavioral pass rate `45.8% → 70.8%` (`+0.1265` mean score delta, McNemar $p = 0.1094$), `238.51s` on `cuda:0`.
* **`ndiff-real-smollm2-135m` (`VERIFIED`):** `HuggingFaceTB/SmolLM2-135M` (`93efa2f097...`) $\to$ `HuggingFaceTB/SmolLM2-135M-Instruct` (`12fd25f773...`), `134,515,008` parameters, `272` tensors, `97.6728%` changed, mean cosine `0.996951`, final-layer CKA `0.951782`, pass rate `20.8% → 33.3%`, `191.13s`.
* **`ndiff-real-cross-arch` (`VERIFIED`):** `Qwen/Qwen2.5-0.5B-Instruct` vs `HuggingFaceTB/SmolLM2-135M-Instruct`, cross-architecture final-layer pooled CKA `0.966758`, `4.79s`.
* **`ndiff-real-regression-detected` (`VERIFIED`):** `Qwen/Qwen2.5-0.5B-Instruct` $\to$ `Qwen/Qwen2.5-0.5B` downgrade validation, `FAILED_REGRESSION` verdict (`62.5% → 37.5%`), `5` critical findings, `6` deterministic CLI reproducers, `117.09s`.
* **`google/gemma-3-270m` (`BLOCKED`):** Gated Hugging Face repository returned `HTTP 401 Unauthorized` without an authenticated `HF_TOKEN`; preserved in [`evidence/model_comparisons/gemma3_270m_access_probe.json`](model_comparisons/gemma3_270m_access_probe.json).

---

## F. Performance & Memory Usage (`VERIFIED`)

Recorded in [`evidence/benchmarks/synthetic_throughput_benchmarks.json`](benchmarks/synthetic_throughput_benchmarks.json):

* **Peak Streaming Throughput:** `116.645 million elements/sec` (`10M float32` elements, `1M` chunk size, `float64` accumulation, `0.0857s`).
* **`bfloat16` Streaming Throughput:** `102.934 million elements/sec` on `10M` elements (`0.0971s`) and `95.124 million elements/sec` on `25M` elements (`0.2628s`).
* **Lazy Safetensors Memory Overhead:** `+0.52 MB` process RSS delta across `8 × 512 × 1024` (`4.19M` parameters) memory-mapped `bfloat16` slices.

---

## G. Security Audit (`VERIFIED`)

Recorded under [`evidence/security/`](security/) and documented in [`docs/SECURITY.md`](../docs/SECURITY.md):

* **Dependency Vulnerability Scans:** `pip-audit` (`exit 0`, `0` vulnerabilities) and `npm audit` (`exit 0`, `0` vulnerabilities).
* **Repository Secret Scan:** `0` secret findings across tracked files ([`evidence/security/secret_scan.json`](security/secret_scan.json)).
* **Runtime Defenses:** `trust_remote_code=False` hardcoded on Hugging Face loaders, `pickletools` opcode whitelist inspection before `torch.load(..., weights_only=True)`, strict root-confinement path validation (`HTTP 403` verified in unit, security, and Playwright E2E tests), request body size enforcement, and CSP-sandboxed report previews. No system is completely immune to unknown vulnerabilities in upstream native libraries (`libtorch`, OS mmap); residual risks are documented in `docs/SECURITY.md`.

---

## H. Git History (`VERIFIED`)

Verified via [`scripts/capture_git_history.py`](../scripts/capture_git_history.py) in [`evidence/git/git-history.log`](git/git-history.log) and [`evidence/git/git_verification.json`](git/git_verification.json):

* **Commit Count:** `105+` genuine, atomic Conventional Commits on `main` (`feat`, `fix`, `perf`, `refactor`, `style`, `test`, `docs`, `build`, `ci`, `chore`), zero empty commits, zero history rewrites, clean working tree.

---

## I. Evidence Integrity (`VERIFIED`)

* **Manifest:** [`evidence/EVIDENCE_MANIFEST.json`](EVIDENCE_MANIFEST.json) generated from live files on disk via [`scripts/generate_evidence_manifest.py`](../scripts/generate_evidence_manifest.py).
* **Verification:** Executed via `.venv\Scripts\python.exe scripts/verify_evidence.py`, confirming `100%` file existence, byte-size match, and SHA-256 hash match across every recorded artifact. (SHA-256 digests guarantee post-capture tamper detection, not independent external attestation of event execution.)

---

## J. Reproducible Execution Commands

```powershell
# 1. Install pinned Python & Frontend dependencies
python -m uv sync
npm --prefix frontend ci

# 2. Launch API server & Studio UI
.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000
npm --prefix frontend run dev

# 3. Run full test & security verification suites
.\.venv\Scripts\python.exe scripts/run_all_tests.py
.\.venv\Scripts\python.exe scripts/run_security_audit.py

# 4. Run benchmarks, recapture Playwright screenshots, and verify SHA-256 manifest
.\.venv\Scripts\python.exe scripts/run_benchmarks.py
.\.venv\Scripts\python.exe scripts/capture_screenshots.py
.\.venv\Scripts\python.exe scripts/capture_evolution_evidence.py
.\.venv\Scripts\python.exe scripts/verify_evidence.py
```

---

## J.2 Neural Evolution Tracking Extension (`VERIFIED`)

Implemented in `core/evolution/`, `api/routes/evolution.py`, and `frontend/src/views/EvolutionStudioView.tsx` (documented in [`docs/NEURAL_EVOLUTION_TRACKING.md`](../docs/NEURAL_EVOLUTION_TRACKING.md)):

* **Real 6-Checkpoint Branched & Merged Experiment (`evorun-real-smollm2`):** Executed across `SmolLM2-135M` Base -> (`Instruct-SFT` || `SFT-Conservative`) -> `Merged-AB` -> `Quantized-LowBit` -> `Pruned-MLP` ([`evidence/evolution/evolution_benchmark.json`](evolution/evolution_benchmark.json)).
* **Verified Multi-Format Exports:** JSON, CSV, Mermaid-Markdown, 5-SVG HTML, and pure-Python multi-page `%PDF-1.4` reports archived under [`evidence/evolution/exports/`](evolution/exports/).
* **Live Playwright Screenshots (`16`..`27`):** All 12 required Evolution views captured in `evidence/frontend/16-evolution-dashboard.png` through `27-evolution-exported-report.png` and indexed in [`evidence/SCREENSHOT_INDEX.md`](SCREENSHOT_INDEX.md).

---

## K. Residual Limitations (`PARTIALLY VERIFIED` / Disclosed)

1. **Remote CI Runner Hardware Scope:** [`.github/workflows/`](../.github/workflows) executes **10 specialized multi-node GitHub Actions workflows** (`backend-ci.yml`, `frontend-ci.yml`, `evolution-pipeline.yml`, `api-contract-and-db.yml`, `e2e-playwright.yml`, `security-audit.yml`, `codeql.yml`, `numerical-benchmarks.yml`, `docs-and-evidence.yml`, `package-and-release.yml` — `31` job blocks / `37` runner nodes across `ubuntu-latest` + `windows-latest`, `Python 3.11/3.12`, and `Node 20.x/22.x/24.x`); multi-hundred-million-parameter CUDA model benchmarks are executed on local RTX 5050 hardware and archived under `evidence/`.
2. **Sample Size & Exploratory Gating of Default Behavioral Suite:** `default_suite_v1` contains 24 deterministic prompts (3 per category) for tractable local execution; `RegressionOriginEngine` enforces `min_category_sample_size = 8` (`is_exploratory = true`, `statistically_significant = false` when $n < 8$) alongside step-down Holm-Bonferroni adjustment, and supports larger custom suites via `benchmark_suite_path`.
3. **4-Tier Epistemic & Intervention Scope:** Parameter basis shifts (`OBSERVED`) do not identify monosemantic features; generic diagnostic-prompt patching provides transition-level `CAUSAL SUPPORT` only, while `CAUSALLY VERIFIED` is restricted exclusively to case-targeted counterfactual interventions with verified target/foil margins and negative-control checks on the specific failed prompt(s).
