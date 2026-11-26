# Competitive Analysis — Model Diffing, Interpretability & Regression Ecosystems

**Verification Status:** `VERIFIED` (Repository metadata and licenses verified in [hub_and_ecosystem_verification.json](../evidence/research/hub_and_ecosystem_verification.json)).

> **Explicit Non-Novelty Statement:** NEURAL DIFF is **not** the first project to inspect neural network weight changes, compare activations, or run behavioral regression suites. Open-source tools already exist within each individual vertical. NEURAL DIFF addresses the integration gap between static weight inspection, internal activation drift, and deterministic behavioral regression gating.

---

## 1. Comparative Matrix

| Project / Product | License | Primary Layer | Structural & Tensor Diff | Controlled Activation Diff | Deterministic Behavioral Suite | Regression Threshold Engine | Interactive Diff UI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`safetensors`** | Apache-2.0 | Safe weight storage | Metadata & slice APIs only | No | No | No | No |
| **`diffai`** | Open Source | CLI tensor diffing | Yes (summary stats) | No | No | Basic exit codes | No |
| **`arcee-ai/mergekit`** | LGPL-3.0 | Weight-space merging | Task-vector math internally | No | No | No | No |
| **`TransformerLens`** | MIT | Single-model mech interp | No (folds weights by default) | Manual scripting | No | No | Notebook widgets |
| **`ndif-team/nnsight`** | MIT | Graph tracing / remote hooks | No | Via custom Python traces | No | No | No |
| **`SAELens` / Diffing Toolkit** | MIT | Sparse feature / crosscoder research | Research scripts | Crosscoder / KL lenses | No | No | Research notebooks |
| **`lm-evaluation-harness`** | MIT | Academic benchmark evaluation | No | No | Broad benchmark tasks | No built-in diff UI | No |
| **`promptfoo`** | MIT | Prompt & app regression testing | No (black-box API focus) | No | Strong assertion engine | Yes | Web viewer |
| **`DeepEval`** | Apache-2.0 | LLM-as-a-judge unit testing | No | No | Mostly judge-driven | Pytest thresholds | Cloud/CLI |
| **NEURAL DIFF** | MIT | Full-stack checkpoint intelligence diff | **Yes** (chunked, sharded, head/MLP aware) | **Yes** (hooks, cosine, L2, CKA) | **Yes** (8 deterministic suites) | **Yes** (versioned baselines & severity) | **Yes** (React + TypeScript workbench) |

---

## 2. Detailed Project Evaluations

### 2.1 `safetensors/safetensors` & File-Level VCS (`git-lfs`, `DVC`)
- **Existing Capabilities:** Zero-copy memory-mapped tensor deserialization, strict JSON header layout, elimination of Python `pickle` RCE vectors; DVC and `git-lfs` track binary checkpoint hashes across commits.
- **Limitations:** Binary hash tracking treats a 1 GB checkpoint as an opaque blob—changing a single bias element changes the SHA-256 hash without revealing *which* layer, attention head, or behavioral capability changed.
- **Relevant Features:** `safe_open` lazy slice extraction and sharded `model.safetensors.index.json` conventions.
- **Architectural Differences:** Storage/transport layer rather than an analytical comparison engine.
- **What NEURAL DIFF Contributes:** Builds directly atop `safetensors` lazy slicing to turn opaque binary diffs into structured tensor, layer, head/MLP, activation, and behavioral diffs.

### 2.2 Weight Diffing Utilities (`diffai`, Ad-hoc PyTorch Scripts)
- **Existing Capabilities:** Command-line diffing of `.safetensors`, `.pt`, and `.npy` files reporting parameter counts, shape changes, and basic L1/L2 norms.
- **Limitations:** Typically stop at flat key-value tensor tables. They do not map flat parameter names across diverse transformer architectures (`q_proj`/`k_proj` vs fused `c_attn`), slice GQA attention heads, correlate weight deltas with forward-pass activation shifts, or verify whether weight drift actually degraded downstream JSON/code/math behavior.
- **Architectural Differences:** Stateless CLI utilities vs. NEURAL DIFF's unified core library + persistent SQLite experiment store + FastAPI backend + React analytical UI.
- **What NEURAL DIFF Contributes:** Architecture-aware structural mapping, rename heuristics without false cross-architecture assumptions, chunked float64-accumulated statistical distributions, and linkage to live inference behavior.

### 2.3 `arcee-ai/mergekit`
- **Existing Capabilities:** Industry-standard weight merging (Linear, SLERP, Task Arithmetic, TIES, DARE, Frankenmerging) with out-of-core lazy tensor execution on consumer hardware.
- **Limitations:** MergeKit *produces* merged checkpoints from YAML recipes, but leaves post-merge diagnostics—verifying which layers absorbed the largest parameter updates, whether activations destabilized, or whether formatting/refusal regressed—to external tools.
- **Relevant Features:** Out-of-core lazy shard reading and task-vector arithmetic ($\theta_{\text{ft}} - \theta_{\text{base}}$).
- **What NEURAL DIFF Contributes:** Acts as the diagnostic counterpart to MergeKit and fine-tuning pipelines—inspecting exact pre/post-merge structural, activation, and behavioral diffs before release.

### 2.4 `TransformerLens` & `NNsight`
- **Existing Capabilities:** Rich hook points (`TransformerLens`) and universal deferred execution graphs (`NNsight`) for circuit analysis, logit lens, and causal activation patching.
- **Limitations:** Designed as low-level Python research frameworks rather than turnkey model-comparison applications. Critically, `TransformerLens` folds LayerNorm scales into linear matrices by default, altering raw checkpoint tensor norms unless explicitly disabled, and covers a curated subset of architectures.
- **Architectural Differences:** Stateful research SDKs focused on single-model intervention vs. NEURAL DIFF's dual-model comparative pipeline over unmodified `nn.Module` checkpoints.
- **What NEURAL DIFF Contributes:** Non-invasive native PyTorch hook capture on unmodified Hugging Face models, automated token-aligned layer/MLP/attention divergence metrics (cosine, L2, Linear CKA), and explicit epistemic tagging (`OBSERVED` vs `CORRELATED` vs `CAUSALLY_VERIFIED`).

### 2.5 `SAELens` & Mechanistic Model-Diffing Research (`science-of-finetuning/diffing-toolkit`)
- **Existing Capabilities:** Decomposes polysemantic activations into sparse dictionary features and trains cross-model *crosscoders* to isolate fine-tuning-specific latent directions.
- **Limitations:** Training crosscoders or SAEs for every checkpoint pair requires substantial GPU hours and large token corpora—impractical for rapid CI/CD checkpoint comparison on a laptop GPU.
- **What NEURAL DIFF Contributes:** Fast, zero-training structural, neuron-norm, and activation-space diagnostics suitable for immediate local verification, paired with explicit documentation warning users not to conflate basis coordinates with monosemantic SAE features.

### 2.6 `lm-evaluation-harness`, `promptfoo`, & `DeepEval`
- **Existing Capabilities:**
  - `lm-evaluation-harness`: Broad academic benchmark coverage (MMLU, GSM8K, ARC, HellaSwag).
  - `promptfoo`: Declarative YAML prompt evaluation, deterministic assertions, CI gating, and web diff tables.
  - `DeepEval`: Pytest-integrated LLM-as-a-judge evaluation metrics.
- **Limitations:** Operate purely at the black-box text/logit boundary. When a fine-tuned or quantized checkpoint regresses on JSON formatting or Vietnamese diacritics, black-box harnesses cannot reveal whether embeddings were resized, `lm_head` drifted, or upper-layer MLPs saturated. Conversely, `DeepEval` relies heavily on external LLM judges that introduce cost and non-determinism.
- **What NEURAL DIFF Contributes:** Couples deterministic, zero-judge behavioral test suites (Vietnamese quality, AST-verified code preservation, strict JSON reliability, math reasoning, instruction constraints, conciseness, refusal boundaries, factual QA) with white-box tensor and activation evidence in one reproducible artifact.
