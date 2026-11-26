# Deterministic Behavioral Evaluation & Regression Gating — NEURAL DIFF

**Implementation Modules:**
- `benchmarks/datasets/default_suite_v1.json` & `benchmarks/datasets/loader.py`
- `benchmarks/evaluators/` (8 deterministic domain evaluators)
- `core/statistics/significance.py` (`compute_paired_significance`)
- `core/behavioral_analysis/runner.py` (`BehavioralDiffEngine`)
- `core/regression_engine/` (`RegressionEngine`, `RegressionBaseline`, `RegressionThresholds`)

---

## 1. Zero-Judge Deterministic Design Philosophy

LLM-as-a-judge pipelines suffer from judge-model drift, self-preference bias, API cost, and non-reproducibility. NEURAL DIFF implements **100% deterministic programmatic evaluators** across 8 behavioral domains (`neural-diff-core-v1`, 24 curated paired test cases):

| Category | Evaluator Class | Deterministic Verification Logic |
| :--- | :--- | :--- |
| `vietnamese` | `VietnameseQualityEvaluator` | Unicode NFC normalization, precomposed Vietnamese vowel/tonal diacritic word ratio, mojibake (`\ufffd`) rejection, Vietnamese function-word check, and required task keywords. |
| `code_preservation` | `CodePreservationEvaluator` | Fenced/raw code extraction, `ast.parse()` Python syntax verification, function/class name retention, exact argument-list signature verification, construct checks, and `SequenceMatcher` similarity. |
| `json_reliability` | `JsonReliabilityEvaluator` | Strict raw `json.loads` vs extracted JSON object parsing, required key coverage, strict Python-vs-JSON boolean/integer type distinction (`isinstance(True, int)` guarded), and exact field matching. |
| `math_reasoning` | `MathReasoningEvaluator` | Priority extraction (`\boxed{...}`, `#### ...`, `Answer: ...`, fraction `a/b`, trailing numeric literal) compared against target answer within `tolerance` (`1e-5`). |
| `instruction_following` | `InstructionFollowingEvaluator` | Hard constraint checks: `max_words`, `min_words`, `exact_bullet_count`, `must_include`, `must_exclude`, `starts_with`, `ends_with`, `no_commas`, and `all_uppercase`. |
| `conciseness` | `ConcisenessEvaluator` | Verifies required core fact presence first (preventing empty/wrong responses from gaming brevity), scores word count against `target_max_words`, and penalizes conversational filler prefixes. |
| `refusal_behavior` | `RefusalBehaviorEvaluator` | Dual-direction calibration: verifies explicit refusal + absence of exploit markers on unsafe requests (`should_refuse: true`), and verifies absence of false refusal + presence of helpful technical guidance on benign dual-use prompts (`should_refuse: false`). |
| `factual_accuracy` | `FactualAccuracyEvaluator` | Canonical answer alias matching paired with explicit `contradictions` penalty so hedged/self-contradicting answers fail. |

---

## 2. Anti-Cherry-Picking Paired Statistical Protocol

Every prompt in the active `BenchmarkSuite` is executed on both Model A and Model B under deterministic greedy decoding (`do_sample=False`, fixed seed), saving verbatim prompts and raw model completions.

`compute_paired_significance` computes over all $N$ paired cases:
- **Mean Score Delta:** $\bar{d} = \frac{1}{N}\sum_{i=1}^N (s_{B,i} - s_{A,i})$
- **95% Paired Bootstrap Confidence Interval:** `2,000` seeded resamples of paired differences $d_i$
- **Win / Tie / Loss Counts**
- **Exact Two-Sided McNemar Binomial $p$-value:** Exact `scipy.stats.binomtest` over discordant binary pass/fail pairs $(N_{01}, N_{10})$.

---

## 3. Regression Threshold Engine & Failure Reproduction

`RegressionEngine` evaluates structural, numerical, activation, and behavioral metrics against `RegressionThresholds`:
- `CRITICAL`: Any `NaN`/`Inf` weight, refusal safety regression (`zero_tolerance_refusal_regression`), overall pass-rate drop $\ge 15\%$, or category score drop $\ge 25\%$.
- `HIGH`: Missing tensors, shared tensor shape mismatches, overall pass-rate drop $\ge 5\%$, category score drop $\ge 10\%$, or drop relative to a saved `RegressionBaseline`.
- `MEDIUM` / `LOW`: Minor score drops, suite version mismatch against baseline, or final-layer Linear CKA below `0.60`.

Every regressed behavioral case emits a `FailureReproduction` record containing the exact prompt, baseline vs candidate outputs, score drop, evaluator rationale, and a deterministic CLI reproduction command (`neural-diff behavior --model-a ... --model-b ... --category ... --seed 42`).
