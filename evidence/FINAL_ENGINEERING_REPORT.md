# NEURAL DIFF — Final Engineering Report

Every figure below was taken from a log or JSON file already in `evidence/`. Status labels: VERIFIED / PARTIALLY VERIFIED / NOT VERIFIED / BLOCKED / NOT IMPLEMENTED.

## A. Environment (VERIFIED — `evidence/environment/environment_post_install.json`, `docs/ENVIRONMENT_AUDIT.md`)
- Windows 11 (10.0.26320), AMD64, 16 logical CPUs, 15.71 GB RAM, `D:` 88.7 GB free.
- GPU: NVIDIA GeForce RTX 5050 Laptop (sm_120, 8 GB). It runs CUDA kernels with `torch 2.10.0+cu128`.
- Python 3.12.16 via uv, pinned in `uv.lock`. Node and npm lockfile in `frontend/package-lock.json`.
- Apple Silicon/Metal: not present. Docker: not used.

## B. Implementation
| Component | Status |
|---|---|
| Loader (Safetensors, sharded, HF dir, safe PyTorch) | VERIFIED |
| Float64 chunked tensor diff, tolerances, NaN/Inf | VERIFIED |
| Structural and compatibility diff, rename inference | VERIFIED |
| Neuron/GQA head analysis | VERIFIED (observational only; the code refuses to claim semantic neurons) |
| Activation diff (hooks, linear CKA, logit JS) | VERIFIED |
| 8-suite deterministic behavioral diff (24 cases) | VERIFIED |
| Regression engine, baselines, repro commands | VERIFIED |
| HTML/MD/JSON reports | VERIFIED |
| FastAPI + SQLite + security middleware + cancellation, CLI | VERIFIED |
| React/TS/Vite frontend: 12 views plus ComfyUI-style node canvas | VERIFIED (Playwright E2E exit 0) |
| SVD task-vector and 2:4 sparsity analyzers | VERIFIED at module level (unit tests); not wired into API or UI |
| Logit lens, causal activation patching, diff-of-diffs, diagnostics API router | NOT IMPLEMENTED |
| Causal verification (`CAUSALLY VERIFIED`) | NOT IMPLEMENTED; no code path currently produces it |

## C. Tests (VERIFIED — `evidence/tests/test_execution_summary.json`, all `all_passed: true`)
Run with `.venv\Scripts\python.exe scripts/run_all_tests.py`:
- ruff, mypy, frontend typecheck and production build: exit 0.
- unit 30 passed, property (Hypothesis) 6, integration 3, regression 1, security 5. The full coverage run reports 45 passed, TOTAL coverage 90% over 4430 statements (`coverage.log`).
- Vitest 1 passed. Playwright E2E 1 passed (`playwright-e2e.log`).
- The suite is small: 45 Python tests and one frontend unit test. Coverage was not manipulated, but the test count is modest.
- Preserved failures: `integration-tests-initial-failure.log`, `ruff-lint-initial-failure.log`, `mypy-typecheck-initial-failure.log`, `security/pip_audit_initial_failure.log`. Each was diagnosed and fixed, and the passing rerun logs are kept alongside.

## D. Screenshots (VERIFIED — `evidence/SCREENSHOT_INDEX.md`, `screenshots_metadata.json`)
15 PNGs in `evidence/frontend/` (01-dashboard … 15-final-application-state), captured with headless Chromium via Playwright against the live API and Vite preview. They cover all 14 required views. 12 is rendered from the real test-summary JSON and coverage log. Real-model screenshots use the persisted `ndiff-real-*` runs. The synthetic fixture is badged separately.

## E. Model experiments (VERIFIED — `evidence/benchmarks/real_model_benchmarks.json`, `model-comparison.log`)
REAL MODEL EXPERIMENTS:
- `Qwen/Qwen2.5-0.5B` → `Qwen2.5-0.5B-Instruct` (rev `060db6499f…` → `7ae557604a…`): 494M params, 290 tensors, 97.93% changed, mean cosine 0.998495, final-layer CKA 0.991076, pass rate 45.8% → 70.8%.
- `SmolLM2-135M` → `SmolLM2-135M-Instruct`: 134.5M params, pass rate 20.8% → 33.3%.
- Cross-architecture Qwen-Instruct vs SmolLM2-Instruct: pooled CKA 0.966758.
- Regression run (Instruct → Base): `FAILED_REGRESSION`, 5 findings, 6 CLI reproductions.
- Qwen math_reasoning drops to 0% under the 96-token budget. This is OBSERVED and probably a truncation effect, not an INFERRED capability loss.
- Pass-rate difference: the McNemar p-value is 0.1094 and the bootstrap CI includes 0, so it is not statistically significant on 24 cases.
- `google/gemma-3-270m`: BLOCKED (HTTP 401, gated repo); `evidence/model_comparisons/gemma3_270m_access_probe.json`.

SYNTHETIC FIXTURE EXPERIMENTS (`data/demo_fixtures/synth_model_*`) are kept apart and badged SYNTHETIC. `synthetic_throughput_benchmarks.json` reports peak 116.6M elements/s.

## F. Performance
The Qwen full run took 238.5 s, SmolLM2 191.1 s and the cross-architecture run 4.8 s. Lazy Safetensors streaming added +0.52 MB RSS in the synthetic benchmark. Raw data is in `evidence/benchmarks/`.

## G. Security
- `pip-audit` and `npm audit` report 0 vulnerabilities; the secret scan reports 0 findings (`evidence/security/`).
- 5 security tests cover path traversal, oversized bodies and unsafe serialization. The E2E run also verified a 403 on path traversal.
- No remote code execution (no `trust_remote_code`), and pickle checkpoints are gated.
- Not a claim of complete security. See `docs/SECURITY.md`.

## H. Git history
The count at the time of writing is in `evidence/git/git-history.log`, produced by `scripts/capture_git_history.py`. It is below the requested 100 commits (86 before the final report commits). No filler commits were made to reach it. The queued modules that would have added legitimate commits are listed under limitations.

## I. Evidence integrity
`evidence/EVIDENCE_MANIFEST.json`, checked with `scripts/verify_evidence.py` (result recorded in `evidence/final_report/`). SHA-256 shows files are unchanged after capture. It does not prove that the underlying events happened.

## J. Reproduction
```
python -m uv sync
npm --prefix frontend ci
.venv\Scripts\python.exe -m uvicorn api.app:app --port 8000
npm --prefix frontend run preview
.venv\Scripts\python.exe scripts/run_all_tests.py
.venv\Scripts\python.exe scripts/run_benchmarks.py
.venv\Scripts\python.exe scripts/capture_screenshots.py
```

## K. Limitations
- Fewer than 100 commits.
- GitHub Actions is configured but has not been run remotely; local runs are not equivalent.
- Modules listed as NOT IMPLEMENTED in section B.
- The behavioral suite has only 24 cases, so deltas are low-powered.
- Frontend unit coverage is minimal. UI correctness rests on Playwright E2E and screenshots.
- Gemma 3 270M is not compared.
