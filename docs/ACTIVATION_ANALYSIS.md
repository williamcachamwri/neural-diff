# Activation Analysis & Representation Alignment — NEURAL DIFF

**Implementation Modules:**
- `core/activation_analysis/metrics.py` (`compute_linear_cka`, `compare_activation_tensors`)
- `core/activation_analysis/comparator.py` (`capture_module_traces`, `ActivationDiffEngine`)

---

## 1. Controlled Sequential Forward-Hook Capture

Comparing internal activations across two language models on consumer hardware requires avoiding holding two full `nn.Module` weight graphs in GPU VRAM simultaneously. `ActivationDiffEngine.compare_pretrained` executes sequentially:

1. Validate `config.json` and tokenizer presence; enforce `trust_remote_code=False`.
2. Load Model A (`bfloat16` on verified CUDA devices, `float32` on CPU) in `eval()` mode under `torch.no_grad()` and deterministic `torch.manual_seed(seed)`.
3. Attach non-invasive PyTorch forward hooks (`register_forward_hook`) onto:
   - Transformer decoder/encoder blocks (`hidden_state`)
   - Self-attention modules (`attention_output`)
   - MLP activation/projection blocks (`mlp_intermediate`)
4. Detach captured `[1, T, D]` tensors to CPU `float32`, remove all hooks in a `finally` block, delete Model A, and reclaim GPU/host memory via `ResourceManager.cleanup()`.
5. Repeat the identical prompt suite on Model B and compute layer-by-layer alignment metrics.

---

## 2. Representation Similarity Metrics

### 2.1 Direct Cosine & Per-Token $L_2$ Drift
When Model A and Model B share identical sequence tokenization ($T_A = T_B$) and hidden dimension ($D_A = D_B$), NEURAL DIFF computes:
- Flattened layer-activation cosine similarity and relative $L_2$ difference.
- Per-token cosine similarity $c_t = \cos(h_{A,t}, h_{B,t})$ and $L_2$ distance $\|h_{B,t} - h_{A,t}\|_2$ for $t \in \{0, \dots, T-1\}$, pinpointing `most_diverged_token_index`.

### 2.2 Linear Centered Kernel Alignment (Linear CKA)
When fine-tuning or cross-architecture differences rotate internal coordinate bases—or when hidden widths differ ($D_A \ne D_B$)—elementwise subtraction is undefined or misleading. Given centered representations $\tilde{X} \in \mathbb{R}^{N \times D_A}$ and $\tilde{Y} \in \mathbb{R}^{N \times D_B}$ across $N \ge 2$ aligned tokens (or across $N \ge 2$ prompts for last-token pooled representations):

$$\text{CKA}_{\text{linear}}(X, Y) = \frac{\|\tilde{Y}^\top \tilde{X}\|_F^2}{\|\tilde{X}^\top \tilde{X}\|_F \cdot \|\tilde{Y}^\top \tilde{Y}\|_F} \in [0, 1]$$

Linear CKA is invariant to orthogonal transformations ($\tilde{Y} = \tilde{X} Q$, $Q^\top Q = I$) and positive isotropic scaling ($\tilde{Y} = \alpha \tilde{X}$, $\alpha > 0$), verified via Hypothesis property tests in `tests/property/test_tensor_properties.py`.

### 2.3 Next-Token Logit KL & Jensen-Shannon Divergence
When vocabulary sizes match, final-token next-word probability distributions $P_A = \text{softmax}(z_A)$ and $P_B = \text{softmax}(z_B)$ are compared via:
- Kullback-Leibler divergence $D_{\text{KL}}(P_A \| P_B)$
- Symmetric bounded Jensen-Shannon divergence:
  $$D_{\text{JS}}(P_A, P_B) = \frac{1}{2} D_{\text{KL}}\left(P_A \,\middle\|\, \frac{P_A + P_B}{2}\right) + \frac{1}{2} D_{\text{KL}}\left(P_B \,\middle\|\, \frac{P_A + P_B}{2}\right)$$
alongside Top-5 predicted tokens and probabilities for each model.

---

## 3. Cross-Architecture & Epistemic Limitations

1. **Differing Tokenizers ($T_A \ne T_B$):** Token-by-token elementwise subtraction and per-prompt token CKA require equal sequence lengths. When tokenizers segment a prompt into different token counts, NEURAL DIFF records an explicit limitation note and falls back to **cross-prompt last-token pooled Linear CKA** (`cross_prompt_pooled_cka_final_layer`) plus activation value distribution shifts.
2. **Observation vs. Causation:** Co-occurrence of high layer activation drift and behavioral changes is classified as `CORRELATED` (and individual tensor deltas as `OBSERVED`), never `CAUSALLY VERIFIED`, because observational hooks do not perform counterfactual activation patching.
