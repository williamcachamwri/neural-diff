export type ViewId =
  | "dashboard"
  | "compare"
  | "compatibility"
  | "tensors"
  | "structural"
  | "layers"
  | "activations"
  | "behavioral"
  | "regression"
  | "history"
  | "reports"
  | "settings";

export interface HealthResponse {
  status: string;
  version: string;
  device: string;
  device_reason: string;
  hardware: {
    process_rss_mb: number;
    system_available_ram_mb: number;
    system_total_ram_mb: number;
    cuda_available: boolean;
    cuda_allocated_mb: number;
    cuda_reserved_mb: number;
    cuda_total_vram_mb: number;
    selected_device: string;
  };
  software: {
    torch: string;
    transformers: string;
    safetensors: string;
  };
  counts: {
    experiments: number;
    completed_experiments: number;
    baselines: number;
    active_jobs: number;
  };
}

export interface DiscoveredModelsResponse {
  local_models: Array<{
    source: string;
    label: string;
    kind: string;
    synthetic_fixture: boolean;
  }>;
  hub_presets: Array<{
    id: string;
    label: string;
    license: string;
    gated: boolean;
    approx_size_mb: number;
    recommended_pair: string | null;
    cached_locally: boolean;
  }>;
}

export interface ModelMetadata {
  model_id: string;
  resolved_path: string;
  format: string;
  revision: string | null;
  tensor_count: number;
  total_parameters: number;
  total_bytes: number;
  shard_count: number;
  dtypes_present: string[];
  architecture: {
    architecture_class: string | null;
    model_type: string | null;
    hidden_size: number | null;
    intermediate_size: number | null;
    num_hidden_layers: number | null;
    num_attention_heads: number | null;
    num_key_value_heads: number | null;
    vocab_size: number | null;
    torch_dtype: string | null;
  };
  security_notes: string[];
}

export interface CompatibilityIssue {
  code: string;
  severity: string;
  message: string;
  field: string;
  value_a: unknown;
  value_b: unknown;
  epistemic_status: string;
}

export interface CompatibilityReport {
  epistemic_status: string;
  compatible_for_direct_tensor_diff: boolean;
  compatible_for_structural_diff: boolean;
  compatible_for_head_neuron_analysis: boolean;
  compatible_for_activation_diff: boolean;
  compatible_for_behavioral_diff: boolean;
  shared_tensor_count: number;
  shape_matched_tensor_count: number;
  jaccard_key_overlap: number;
  issues: CompatibilityIssue[];
  summary: string;
}

export interface TensorMetrics {
  num_elements: number;
  finite_pair_elements: number;
  non_finite_pair_elements: number;
  nan_count_a: number;
  nan_count_b: number;
  inf_count_a: number;
  inf_count_b: number;
  exact_equal: boolean;
  within_tolerance: boolean;
  max_abs_diff: number | null;
  mean_abs_diff: number | null;
  rmse: number | null;
  l2_diff: number | null;
  l2_norm_a: number | null;
  l2_norm_b: number | null;
  relative_l2: number | null;
  relative_l2_infinite: boolean;
  cosine_similarity: number | null;
  changed_elements: number;
  changed_pct: number;
  zero_to_nonzero: number;
  nonzero_to_zero: number;
}

export interface TensorRecord {
  name: string;
  status: string;
  epistemic_status: string;
  layer_index: number | null;
  canonical_role: string;
  shape_a: number[] | null;
  shape_b: number[] | null;
  dtype_a: string | null;
  dtype_b: string | null;
  dtype_changed: boolean;
  metrics: TensorMetrics | null;
  distribution_a: Record<string, number | boolean | null> | null;
  distribution_b: Record<string, number | boolean | null> | null;
  distribution_shift: Record<string, number | null> | null;
  notes: string[];
}

export interface LayerSummary {
  layer_index: number;
  tensor_count: number;
  total_parameters: number;
  changed_parameters: number;
  changed_pct: number;
  mean_cosine_similarity: number | null;
  min_cosine_similarity: number | null;
  mean_relative_l2: number | null;
  max_relative_l2: number | null;
  max_abs_diff: number | null;
  role_breakdown: Record<
    string,
    {
      mean_cosine: number | null;
      mean_relative_l2: number | null;
      max_abs_diff: number | null;
    }
  >;
}

export interface ComparisonBundle {
  comparison_id: string;
  created_at_utc: string;
  experiment_type: string;
  selected_modules: string[];
  model_a: ModelMetadata;
  model_b: ModelMetadata;
  epistemic_ledger: Array<{
    statement: string;
    epistemic_status: string;
    evidence_source: string;
  }>;
  structural: {
    model_a_parameters: number;
    model_b_parameters: number;
    delta_parameters: number;
    counts: Record<string, number>;
    shared_tensors: string[];
    added_tensors: Array<{ name: string; shape: number[]; dtype: string; canonical_role: string }>;
    removed_tensors: Array<{ name: string; shape: number[]; dtype: string; canonical_role: string }>;
    inferred_renames: Array<{
      old_name: string;
      new_name: string;
      shape: number[];
      cosine_similarity: number;
      epistemic_status: string;
    }>;
    shape_changes: Array<{
      name: string;
      shape_a: number[];
      shape_b: number[];
      delta_elements: number;
    }>;
    dtype_changes: Array<{ name: string; dtype_a: string; dtype_b: string }>;
    layer_deltas: Array<{
      layer_index: number;
      present_in_a: boolean;
      present_in_b: boolean;
      tensor_count_a: number;
      tensor_count_b: number;
      parameters_a: number;
      parameters_b: number;
      delta_parameters: number;
    }>;
    compatibility: CompatibilityReport;
  } | null;
  tensor_diff: {
    tolerances: { atol: number; rtol: number; zero_threshold: number };
    summary: {
      total_union_tensors: number;
      compared_tensors: number;
      identical_tensors: number;
      within_tolerance_tensors: number;
      modified_tensors: number;
      shape_mismatch_tensors: number;
      missing_in_b_tensors: number;
      added_in_b_tensors: number;
      total_compared_parameters: number;
      total_changed_parameters: number;
      global_changed_pct: number;
      global_mean_cosine: number | null;
      global_mean_relative_l2: number | null;
      global_max_abs_diff: number | null;
      has_any_nan_or_inf: boolean;
    };
    layer_summaries: LayerSummary[];
    top_diverged_names: string[];
    records?: TensorRecord[];
  } | null;
  neuron_attention: {
    epistemic_status: string;
    epistemic_caution: string;
    counts: Record<string, number>;
    attention_projections: Array<{
      tensor_name: string;
      layer_index: number | null;
      role: string;
      num_heads: number;
      head_dim: number;
      mean_head_cosine: number;
      min_head_cosine: number;
      max_head_relative_l2: number;
      most_diverged_heads: Array<{
        head_index: number;
        cosine_similarity: number;
        relative_l2: number;
        max_abs_diff: number;
      }>;
    }>;
    mlp_projections: Array<{
      tensor_name: string;
      layer_index: number | null;
      role: string;
      basis_dimension: number;
      mean_basis_cosine: number;
      p95_relative_l2: number;
      max_relative_l2: number;
      top_shifted_indices: Array<{
        basis_index: number;
        relative_l2: number;
        cosine_similarity: number;
      }>;
    }>;
    embeddings_and_heads: Array<{
      tensor_name: string;
      role: string;
      vocab_size: number;
      hidden_size: number;
      mean_token_cosine: number;
      changed_token_pct: number;
    }>;
    normalization_layers: Array<{
      tensor_name: string;
      layer_index: number | null;
      mean_scale_shift: number;
      relative_l2: number;
      cosine_similarity: number;
    }>;
    skipped_notes: string[];
  } | null;
  activation: {
    epistemic_status: string;
    provenance: Record<string, unknown>;
    summary: {
      prompt_count: number;
      mean_hidden_cosine_by_layer: Record<string, number | null>;
      mean_hidden_cka_by_layer: Record<string, number | null>;
      mean_mlp_cosine_by_layer: Record<string, number | null>;
      mean_attn_cosine_by_layer: Record<string, number | null>;
      mean_final_token_js_divergence: number | null;
      cross_prompt_pooled_cka_final_layer: number | null;
    };
    prompt_diffs: Array<{
      prompt: string;
      tokens_a: string[];
      tokens_b: string[];
      tokens_identical: boolean;
      next_token_kl_divergence: number | null;
      next_token_js_divergence: number | null;
      top_tokens_a: Array<{ token_id: number; token: string; prob: number }>;
      top_tokens_b: Array<{ token_id: number; token: string; prob: number }>;
      hidden_state_comparisons: Array<{
        layer_name: string;
        layer_index: number | null;
        cosine_similarity: number | null;
        relative_l2: number | null;
        linear_cka: number | null;
        most_diverged_token_index: number | null;
      }>;
      limitations: string[];
    }>;
    cross_architecture_notes: string[];
  } | null;
  behavioral: {
    epistemic_status: string;
    evaluator_methodology: string;
    suite_id: string;
    suite_version: string;
    overall_statistics: {
      n_cases: number;
      mean_score_a: number;
      mean_score_b: number;
      mean_score_delta: number;
      bootstrap_ci95_low: number;
      bootstrap_ci95_high: number;
      pass_rate_a: number;
      pass_rate_b: number;
      pass_rate_delta: number;
      b_wins: number;
      a_wins: number;
      ties: number;
      mcnemar_p_value: number;
      statistically_significant_95: boolean;
    };
    category_summaries: Array<{
      category: string;
      case_count: number;
      pass_rate_a: number;
      pass_rate_b: number;
      pass_rate_delta: number;
      mean_score_a: number;
      mean_score_b: number;
      mean_score_delta: number;
      improved_cases: number;
      regressed_cases: number;
    }>;
    cases: Array<{
      case: { case_id: string; category: string; prompt: string; description: string };
      output_a: string;
      output_b: string;
      latency_ms_a: number;
      latency_ms_b: number;
      eval_a: { passed: boolean; score: number; rationale: string; submetrics: Record<string, unknown> };
      eval_b: { passed: boolean; score: number; rationale: string; submetrics: Record<string, unknown> };
      score_delta: number;
      status_change: string;
    }>;
  } | null;
  regression: {
    verdict: string;
    highest_severity: string;
    thresholds: Record<string, unknown>;
    finding_counts: Record<string, number>;
    findings: Array<{
      finding_id: string;
      domain: string;
      severity: string;
      epistemic_status: string;
      title: string;
      description: string;
      metric_name: string;
      baseline_value: unknown;
      candidate_value: unknown;
      threshold_value: unknown;
    }>;
    failure_reproductions: Array<{
      case_id: string;
      category: string;
      prompt: string;
      baseline_output: string;
      candidate_output: string;
      baseline_score: number;
      candidate_score: number;
      rationale_candidate: string;
      cli_command: string;
    }>;
    baseline_comparison: Record<string, unknown> | null;
  } | null;
  execution_telemetry: Record<string, unknown>;
  reproducibility: Record<string, unknown>;
}

export interface ExperimentSummaryItem {
  comparison_id: string;
  created_at_utc: string;
  status: "running" | "completed" | "failed" | "cancelled";
  stage_message: string;
  completed_steps: number;
  total_steps: number;
  experiment_type: string;
  model_a_id: string;
  model_b_id: string;
  selected_modules: string[];
  verdict: string | null;
  summary: {
    global_changed_pct: number | null;
    global_mean_cosine: number | null;
    behavioral_pass_rate_a: number | null;
    behavioral_pass_rate_b: number | null;
    behavioral_score_delta: number | null;
    pooled_final_cka: number | null;
    duration_seconds: number | null;
  } | null;
  bundle?: ComparisonBundle | null;
  error: { error: string; message: string; http_status: number; details?: Record<string, unknown> } | null;
}

export interface RuntimeSettings {
  preferred_device: "auto" | "cpu" | "cuda" | "mps";
  atol: number;
  rtol: number;
  zero_threshold: number;
  chunk_elements: number;
  allow_pytorch_weights_only: boolean;
  max_new_tokens: number;
}

export interface BaselineRecord {
  baseline_id: string;
  created_at_utc: string;
  model_id: string;
  revision: string | null;
  suite_id: string;
  suite_version: string;
  overall_pass_rate: number;
  overall_mean_score: number;
}
