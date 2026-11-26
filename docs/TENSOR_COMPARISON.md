# Tensor & Structural Comparison Methodology — NEURAL DIFF

**Implementation Modules:**
- `core/tensor_engine/metrics.py` (`compute_chunked_tensor_metrics`, `compare_tensors`)
- `core/tensor_engine/comparator.py` (`TensorDiffEngine`)
- `core/statistics/distributions.py` (`summarize_tensor_chunks`, `compute_distribution_shift`)
- `core/structural_analysis/diff.py` (`StructuralDiffEngine`)
- `core/structural_analysis/neuron_attention.py` (`NeuronAttentionAnalyzer`)

---

## 1. Out-of-Core Float64 Accumulation

Neural network checkpoints commonly store weights in `bfloat16` (7 mantissa bits) or `float16` (10 mantissa bits). Accumulating sums of squares over millions of elements in 16-bit or 32-bit precision causes rounding drift and overflow.

NEURAL DIFF streams flattened chunks of size $C$ (`1,048,576` elements by default) via `iter_aligned_chunk_pairs`—which aligns heterogeneous shard/row chunk boundaries element-for-element—and casts each chunk to `torch.float64` before updating scalar accumulators.

---

## 2. Exact Mathematical Definitions

Given finite-aligned parameter vectors $a, b \in \mathbb{R}^{N_f}$ (where $N_f$ counts indices where both $a_i$ and $b_i$ are finite, out of $N$ total elements):

1. **Maximum Absolute Difference:**
   $$\Delta_{\max} = \max_{1 \le i \le N_f} |b_i - a_i|$$

2. **Mean Absolute Difference:**
   $$\Delta_{\text{mean}} = \frac{1}{N_f} \sum_{i=1}^{N_f} |b_i - a_i|$$

3. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE}(a, b) = \sqrt{\frac{1}{N_f} \sum_{i=1}^{N_f} (b_i - a_i)^2}$$

4. **Relative $L_2$ Difference:**
   $$\text{Rel-}L_2(a, b) = \frac{\|b - a\|_2}{\max(\|a\|_2, \epsilon)}$$
   If $\|b - a\|_2 = 0$, $\text{Rel-}L_2 = 0$. If $\|a\|_2 = 0$ and $\|b\|_2 > 0$, `relative_l2` is serialized as `null` with `relative_l2_infinite: true` to preserve strict RFC-8259 JSON compliance (`allow_nan=False`).

5. **Cosine Similarity:**
   $$\cos(a, b) = \text{clamp}_{[-1, 1]}\left(\frac{\langle a, b \rangle}{\|a\|_2 \|b\|_2}\right)$$
   Zero-vector convention: $\cos(0, 0) = 1.0$; $\cos(0, b) = 0.0$ for $\|b\|_2 > 0$.

6. **Tolerance-Bounded Changed Element Percentage:**
   An element $i$ is counted as changed if either value is non-finite (unless both are identically $+\infty$ or both $-\infty$, noting $\text{NaN} \ne \text{NaN}$) or:
   $$|b_i - a_i| > \text{atol} + \text{rtol} \cdot |a_i|$$
   $$\text{Changed\%} = 100 \times \frac{N_{\text{changed}}}{N}$$

7. **Zero / Nonzero Transitions:**
   Given threshold $\tau_0 \ge 0$ (`zero_threshold`, default `0.0`):
   - `zero_to_nonzero`: $|\{i : |a_i| \le \tau_0 \land |b_i| > \tau_0\}|$
   - `nonzero_to_zero`: $|\{i : |a_i| > \tau_0 \land |b_i| \le \tau_0\}|$

---

## 3. Structural & Submodule Analysis

### 3.1 Conservative Rename Inference (`EpistemicStatus.INFERRED`)
Unmatched removed/added tensor keys are paired as `RenamedTensorCandidate` **only** when:
1. Both models share the same `model_type` (cross-architecture models such as Llama vs GPT-2 never trigger false rename guesses).
2. Shapes match identically and canonical roles (`attn_q`, `attn_o`, `mlp_up`, etc.) match.
3. Layer indices match (when present) and full weight cosine similarity $\ge 0.985$.

### 3.2 Grouped-Query Attention (GQA) Head Slicing (`EpistemicStatus.OBSERVED`)
- For `q_proj`, `k_proj`, and `v_proj` of shape `(H * d_h, D)`, axis `0` is reshaped into `(H, d_h * D)` where $H = H_Q$ for `q_proj` and $H = H_{KV}$ for `k_proj`/`v_proj`.
- For `o_proj` of shape `(D, H_Q * d_h)`, axis `1` is reshaped and transposed into `(H_Q, D * d_h)`.
- Per-head cosine similarity, Frobenius norms, and relative $L_2$ drift isolate exactly which heads changed most during fine-tuning or merging.

### 3.3 Superposition & Polysemanticity Notice
MLP coordinate rows (`gate_proj`, `up_proj`) and columns (`down_proj`) are basis-aligned dimensions. Because neural networks represent features in superposition, `NeuronAttentionAnalyzer` explicitly warns that coordinate index shifts are `OBSERVED` parameter measurements and must not be conflated with monosemantic concepts.
