# Engineering Roadmap — NEURAL DIFF

This roadmap distinguishes capabilities already `VERIFIED` in `v0.1.0` from future extensions (`NOT IMPLEMENTED`).

---

## Delivered in `v0.1.0` (`VERIFIED`)

- Security-hardened Safetensors (single, multi-file, sharded), opcode-audited `weights_only=True` PyTorch, and Hugging Face local/Hub loaders (`trust_remote_code=False`).
- Out-of-core `float64`-accumulated chunked Tensor Diff Engine with configurable tolerances, distribution shifts, and NaN/Inf accounting.
- Architecture-aware Structural Diff Engine with conservative rename inference (`INFERRED`) and GQA-aware Attention Head / MLP basis / Embedding / Norm analyzers.
- Sequential memory-bounded Forward-Hook Activation Diff Engine with Linear CKA and next-token KL/JS divergence.
- 8-domain, 24-case deterministic Behavioral Diff Engine with paired bootstrap 95% CIs and exact McNemar $p$-values.
- Versioned Regression Engine with CLI failure reproduction packs and SQLite experiment/baseline persistence.
- Standalone SVG-equipped HTML, Markdown, and strict RFC-8259 JSON report exporters.
- 12-view React + TypeScript + Vite workbench verified with Vitest and Playwright E2E automation.

---

## Planned Future Milestones (`NOT IMPLEMENTED`)

### Milestone 1 — Counterfactual Activation Patching (`v0.2.0`)
- Optional causal mediation runner patching Layer $L$ / Head $(L, h)$ activations from Model A into Model B on compatible architectures to upgrade correlational findings (`CORRELATED`) to `CAUSALLY VERIFIED` circuit attributions.

### Milestone 2 — Quantized Format Adapters (`v0.3.0`)
- Read-only dequantization adapters for `GGUF`, `GPTQ`, and `AWQ` checkpoints to measure quantization-induced weight, activation, and behavioral degradation directly against `bfloat16` masters.

### Milestone 3 — Sparse Autoencoder / Crosscoder Hooks (`v0.4.0`)
- Plug-in adapter for pre-trained `SAELens` dictionaries on supported model families to project polysemantic residual deltas onto sparse dictionary latents.

### Milestone 4 — Authenticated Team Server Mode (`v0.5.0`)
- Optional token/OIDC authentication middleware, PostgreSQL backend adapter, and PR comment webhook integration for shared MLOps CI pipelines.
