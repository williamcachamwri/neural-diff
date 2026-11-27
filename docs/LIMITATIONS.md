# Known Limitations & Epistemic Boundaries — NEURAL DIFF

NEURAL DIFF is engineered with strict honesty about what static parameter comparison, forward-hook activation capture, and deterministic behavioral testing can and cannot prove.

---

## 1. Scientific & Interpretability Limitations

1. **Superposition & Polysemantic Basis Coordinates:**
   MLP row/column norm deltas and embedding row shifts (`Module D`) measure parameter changes along native checkpoint basis coordinates (`OBSERVED`). Under neural superposition, individual coordinate indices frequently encode multiple unrelated concepts simultaneously. NEURAL DIFF does **not** train Sparse Autoencoders (SAEs) or cross-model Crosscoders internally, and never claims that coordinate index $k$ corresponds to a single semantic feature.
2. **Observational Activation Diffing vs. Causal Patching:**
   `ActivationDiffEngine` (`Module E`) measures observational forward-pass divergence (cosine similarity, per-token $L_2$, Linear CKA, and next-token KL/JS divergence). Co-occurring internal activation shifts and downstream behavioral changes are labeled `CORRELATED`, not `CAUSALLY VERIFIED`, because counterfactual activation patching across intermediate residual subspaces is not executed automatically.
3. **Cross-Architecture Parameter & Tokenizer Boundaries:**
   - When two models belong to distinct architecture families (e.g., `Qwen2ForCausalLM` vs `LlamaForCausalLM` with differing hidden widths or layer depths), elementwise tensor subtraction is undefined on mismatched shapes; NEURAL DIFF reports structural topology diffs and value-distribution shifts instead.
   - When tokenizers segment the same prompt string into different token lengths ($T_A \ne T_B$), token-aligned per-step cosine/CKA is skipped with an explicit limitation note, falling back to cross-prompt last-token pooled Linear CKA (`cross_prompt_pooled_cka_final_layer`).

---

## 2. Behavioral Evaluation & Statistical Power Limitations

1. **Finite Deterministic Benchmark Coverage:**
   The default deterministic suite (`neural-diff-core-v1`, 24 cases across 8 domains, $n = 3$ per category) eliminates external LLM-as-a-judge variance and API dependencies, but finite programmatic assertions cannot measure open-ended creative writing nuance, long-horizon multi-turn agentic planning, or subtle pragmatic tone shifts.
2. **Minimum Sample Floor (`min_category_sample_size = 8`) & Binomial Discreteness:**
   - Categories with $n < 8$ are strictly marked `is_exploratory = true` and `statistically_significant = false` regardless of raw score drops.
   - Under exact two-sided binomial McNemar testing, the discrete minimum attainable raw $p$-value for $n$ discordant pairs is $p_{\min}(n) = 2^{1-n}$. At the minimum non-exploratory threshold $n = 8$, $p_{\min}(8) = 2^{-7} = 0.0078125$. Under step-down **Holm-Bonferroni** family-wise error correction across $m \ge 7$ categories (e.g., $m = 8$), even an $8/8$ unidirectional collapse yields $p_{\text{adj}} = 8 \times 0.0078125 = 0.0625 > 0.05$. Non-zero rejection power at $\alpha = 0.05$ for $m = 8$ requires $n \ge 9$, and $\ge 80\%$ power for moderate ($25\text{--}30\%$) pass-rate drops requires $n \ge 30$ per category (supported via `benchmark_suite_path`).
   - Exact McNemar evaluates binary pass/fail flips only; continuous `score_only` regressions where binary pass/fail outcomes do not flip yield `mcnemar_p_value = 1.0` and must be interpreted via paired bootstrap 95% score CIs.
3. **Strict Causal Verification Scoping:**
   In Neural Evolution Tracking, diagnostic-prompt cross-checkpoint patching achieves at most `CAUSAL SUPPORT`. `CAUSALLY VERIFIED` requires all four provenance checks (`case_targeted`, `target_vs_foil_verified`, `negative_control_verified`, and case+site `verification_scope`) to be explicitly verified for every failed case in the event, and applies strictly to the tested prompt fingerprint and layer intervention site.
4. **Prompt Formatting Sensitivity (Base vs. Instruct Checkpoints):**
   Pretrained base models lack chat templates (`tokenizer.chat_template`), whereas instruction-tuned models expect structured role delimiters. `BehavioralDiffEngine` records exact `formatted_input_a` and `formatted_input_b` strings under `chat_if_available` (default) and supports `identical_raw` mode so formatting choices remain auditable.

---

## 3. Infrastructure, Hardware & Upstream Limitations

1. **Gated Upstream Models (`google/gemma-3-270m`):**
   Hugging Face repositories configured with `gated: "manual"` return `HTTP 401 Unauthorized` unless the user manually accepts the upstream license on Hugging Face and supplies `HF_TOKEN`. NEURAL DIFF surfaces this cleanly as `MODEL_ACCESS_BLOCKED` (`HTTP 403`) rather than attempting any authentication bypass.
2. **Apple Silicon Metal (`MPS`) & Multi-Node Sharding:**
   `ResourceManager` includes runtime `torch.backends.mps.is_available()` detection, which is `NOT VERIFIED` on this Windows 11 `AMD64` + NVIDIA RTX 5050 host. Very large multi-GPU models (>14B parameters exceeding local `8 GB` VRAM + `16 GB` RAM) can be compared structurally and via chunked out-of-core Safetensors tensor diffing on disk, but full forward-pass activation/behavioral inference is bounded by local RAM/VRAM capacity.
3. **GitHub Actions Runner Hardware Scope:**
   `.github/workflows/` runs 10 multi-node automated CI workflows (`backend-ci.yml`, `frontend-ci.yml`, `evolution-pipeline.yml`, `api-contract-and-db.yml`, `e2e-playwright.yml`, `security-audit.yml`, `codeql.yml`, `numerical-benchmarks.yml`, `docs-and-evidence.yml`, and `package-and-release.yml`) across CPU-hosted `ubuntu-latest` and `windows-latest` runners; real GPU-accelerated Hugging Face model benchmarks (`Qwen2.5-0.5B`, `SmolLM2-135M`) are verified on local CUDA hardware and archived in `evidence/`.
4. **Localhost Single-User API Scope:**
   The FastAPI server is designed for local workstation execution (`127.0.0.1`) without built-in multi-user authentication or rate-limiting across untrusted public networks.
