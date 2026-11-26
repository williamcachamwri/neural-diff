# Technical Research — Neural Checkpoint, Activation & Behavioral Diffing

**Verification Status:** `VERIFIED` (Grounded in official specifications, peer-reviewed literature, and live API verification captured in [hub_and_ecosystem_verification.json](../evidence/research/hub_and_ecosystem_verification.json)).

---

## 1. Serialization Formats & Safe Model Ingestion

### 1.1 Safetensors (`safetensors/safetensors`)
Safetensors (Apache-2.0, Rust core with Python bindings) stores neural weights using a deterministic layout:
1. **Bytes `0..8`**: Unsigned 64-bit little-endian integer (`N`) specifying header byte length.
2. **Bytes `8..8+N`**: UTF-8 JSON object mapping tensor names to `{"dtype": ..., "shape": [...], "data_offsets": [begin, end]}` plus an optional `"__metadata__"` string dictionary.
3. **Bytes `8+N..EOF`**: Raw contiguous byte buffers indexed by `data_offsets`.

**Engineering implications for NEURAL DIFF:**
- **Security:** Deserialization parses bounded JSON and slices raw byte ranges—eliminating arbitrary code execution inherent in Python `pickle` opcodes (`GLOBAL`, `REDUCE`). Header validation must still enforce maximum header size (e.g., 100 MB cap), monotonic non-overlapping `data_offsets`, and exact file-length bounds to reject malformed or adversarial files.
- **Memory-Mapped Streaming:** `safetensors.safe_open(path, framework="pt", device="cpu")` allows reading metadata (`keys()`, `get_slice(name)`) and slicing 1D/ND chunks without loading an entire checkpoint into physical RAM.

### 1.2 PyTorch Checkpoints (`torch.load`)
Legacy `.pt` / `.pth` / `pytorch_model.bin` files wrap ZIP archives around Python `pickle` streams plus raw storage files.
- Executing `torch.load(path, weights_only=False)` invokes `pickle.Unpickler`, enabling arbitrary code execution on untrusted checkpoints.
- **NEURAL DIFF Policy:** Only `torch.load(path, map_location="cpu", weights_only=True)` is permitted when loading PyTorch binary checkpoints, and Safetensors is strictly preferred whenever present.

### 1.3 Hugging Face Transformers (`huggingface/transformers`)
Hugging Face model directories combine:
- `config.json`: Architecture identifier (`architectures`, `model_type`), hidden size, head counts (`num_attention_heads`, `num_key_value_heads` for GQA/MQA), MLP intermediate size, normalization epsilon, and vocabulary size.
- Single checkpoint (`model.safetensors`) or sharded index (`model.safetensors.index.json` mapping each parameter name in `weight_map` to shard filenames `model-00001-of-0000N.safetensors`).
- Tokenizer artifacts (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`).
- **Security Boundary:** `trust_remote_code=False` must be enforced unconditionally so custom Python modules inside model repositories are never executed automatically.

---

## 2. Internal Model Inspection & Mechanistic Interpretability

### 2.1 TransformerLens (`TransformerLensOrg/TransformerLens`)
TransformerLens standardizes GPT-style architectures into a unified `HookedTransformer` interface (`blocks.{l}.attn.W_Q`, `hook_resid_post`, etc.).
- **Critical Caveat for Checkpoint Diffing:** By default, TransformerLens folds LayerNorm/RMSNorm scale parameters into adjacent linear weights and centers unembedding matrices to simplify circuit algebra. While valuable for single-model circuit discovery, weight folding mutates the raw parameter values stored in checkpoints. Consequently, **NEURAL DIFF operates directly on unmodified checkpoint tensors for weight diffing** and uses non-invasive PyTorch forward hooks (`register_forward_hook`) on unmodified `nn.Module` layers for activation diffing.

### 2.2 NNsight (`ndif-team/nnsight`)
NNsight wraps arbitrary PyTorch `nn.Module` graphs in deferred execution proxies (`with model.trace(prompt): ...`), supporting both local execution and remote NDIF cluster execution without rewriting model weights.
- **Design Takeaway:** Architecture-agnostic module traversal via native PyTorch hooks avoids brittle weight conversion pipelines while supporting modern GQA architectures (Qwen2.5, Llama-3, SmolLM2, Mistral, Gemma).

### 2.3 Superposition, Polysemanticity & Sparse Autoencoders (`decoderesearch/SAELens`)
Mechanistic interpretability research (Elhage et al., *Toy Models of Superposition*, 2022; Bricken et al., *Towards Monosemanticity*, 2023) demonstrates that neural networks represent more features than basis dimensions via **superposition**:
- Individual MLP coordinate indices (`neuron i`) or residual dimensions are routinely **polysemantic**—responding to unrelated concepts simultaneously.
- **Sparse Autoencoders (SAEs)** and **Model-Diffing Crosscoders** (Lindsey et al., 2024) learn overcomplete sparse dictionaries to disentangle features across checkpoints, at the cost of training auxiliary dictionary models on millions of tokens.
- **NEURAL DIFF Epistemic Rule:** Structural and neuron-level weight comparisons measure *basis-aligned parameter and norm shifts* (`OBSERVED`), never claiming that coordinate index `k` represents a single semantic concept unless independently decomposed and causally intervened upon.

### 2.4 Activation Comparison vs. Activation Patching
- **Observational Activation Diffing:** Passing identical token sequences through Model A and Model B under deterministic inference (`torch.no_grad()`, greedy decoding / fixed seeds) directly measures hidden-state cosine similarity, L2 distance, distribution shift, and **Linear Centered Kernel Alignment (CKA)** (Kornblith et al., 2019), which is invariant to orthogonal transformations and isotropic scaling—critical when comparing layers whose coordinate bases may have rotated.
- **Causal Activation Patching:** Replacing intermediate activations from Run A into Run B (Vig et al., 2020; Meng et al., 2022) tests whether a specific activation subspace *causally mediates* an output logit difference, which requires identical hidden dimensions and compatible residual topologies.
- **Epistemic Labeling:** Observational divergence between Layer $L$ activations and downstream behavioral changes is `CORRELATED` (or `OBSERVED` at the tensor level), NOT `CAUSALLY_VERIFIED`, unless causal intervention confirms mediation.

---

## 3. Weight-Space Arithmetic, Model Merging & Checkpoint Diffing

Research on **Task Arithmetic** (Ilharco et al., 2023), **TIES-Merging** (Yadav et al., 2023), and **DARE** (Yu et al., 2024) formalized the parameter delta ("task vector") between a fine-tuned model $\theta_{\text{B}}$ and base model $\theta_{\text{A}}$:

$$\Delta \theta = \theta_{\text{B}} - \theta_{\text{A}}$$

Key numerical properties required for rigorous tensor diffing:
1. **Precision Promotion:** `bfloat16` has only 7 mantissa bits (machine epsilon $\approx 7.8 \times 10^{-3}$). Accumulating reductions (`sum`, `L2`, `cosine`) directly in `bfloat16` or `float16` causes catastrophic rounding overflow/underflow. All reductions in NEURAL DIFF promote chunks to `float64` (for scalar accumulators) / `float32` (for vector operations).
2. **Relative L2 & Cosine Complementarity:**
   - **Relative L2** $\frac{\|\theta_B - \theta_A\|_2}{\max(\|\theta_A\|_2, \epsilon)}$ captures magnitude change relative to baseline energy.
   - **Cosine Similarity** $\frac{\langle \theta_A, \theta_B \rangle}{\|\theta_A\|_2 \|\theta_B\|_2}$ captures angular rotation independent of uniform rescaling.
3. **Sparsity & Zero Transitions:** Pruning, quantization, and DARE merging introduce exact zeros or revive zeroed parameters; tracking `zero_to_nonzero` and `nonzero_to_zero` counts alongside tolerance-bounded `changed_ratio` exposes structural sparsification.

---

## 4. Deterministic Behavioral Evaluation & Regression Testing

Behavioral evaluation frameworks (`EleutherAI/lm-evaluation-harness`, `promptfoo/promptfoo`, `DeepEval`, UK AISI `Inspect`) highlight two complementary paradigms:
- **Deterministic Programmatic Evaluators:** Exact JSON AST/schema validation, Python `ast.parse` syntax and signature verification, exact mathematical answer extraction, regex/length/constraint checks, Vietnamese Unicode NFC diacritic and lexical validation, and deterministic refusal-boundary classification. These are 100% reproducible, zero-cost, and immune to judge-model drift.
- **Model-Based Judges:** Useful for open-ended prose grading, but introduce non-determinism, self-preference bias, and external API dependencies.
- **NEURAL DIFF Design:** Prioritizes deterministic, auditable evaluators across all 8 required domains (Vietnamese quality, code preservation, JSON reliability, math reasoning, instruction following, conciseness, refusal behavior, factual accuracy), pairing identical prompts across Model A and Model B with paired score deltas and explicit regression severity thresholds.

---

## 5. Live Model Hub Verification for Real Experiments

Live unauthenticated HTTP queries against the Hugging Face Hub API (`evidence/research/hub_and_ecosystem_verification.json`, captured `2026-11-26T18:20:11Z`) verified model licensing and accessibility:

| Model Identifier | Commit SHA (`main`) | License | Gated | Unauth `config.json` HTTP | Suitability for Local RTX 5050 (`8 GB` VRAM) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen2.5-0.5B` | `060db6499f32faf8b98477b0a26969ef7d8b9987` | `apache-2.0` | `false` | `200 OK` | **Primary Base Model** (~1.0 GB in `bfloat16`) |
| `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | `apache-2.0` | `false` | `200 OK` | **Primary Aligned Variant** (identical shape to Base; exposes real SFT/RLHF deltas) |
| `HuggingFaceTB/SmolLM2-135M` | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | `apache-2.0` | `false` | `200 OK` | **Secondary Fast Real Model** (~270 MB; also enables cross-architecture comparison vs Qwen2.5) |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | `12fd25f77366fa6b3b4b768ec3050bf629380bac` | `apache-2.0` | `false` | `200 OK` | **Secondary Aligned Variant** |
| `google/gemma-3-270m` | `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1` | `gemma` | `"manual"` | `401 Unauthorized` | **Gated Upstream** — requires manual user license click-through + `HF_TOKEN`; documented honestly as an upstream access gate when `HF_TOKEN` is absent |
