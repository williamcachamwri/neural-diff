export interface EvolutionGlobalSummary {
  total_projects: number;
  total_checkpoints: number;
  total_transformations: number;
  total_branches: number;
  total_detected_regressions: number;
  completed_analysis_runs: number;
}

export interface SchemaMigrationRecord {
  migration_id: string;
  description: string;
  checksum_sha256: string;
  applied_at_utc: string;
}

export interface EvolutionOverviewResponse {
  summary: EvolutionGlobalSummary;
  supported_transformations: string[];
  schema_migrations: SchemaMigrationRecord[];
  active_jobs: number;
}

export interface EvolutionProjectSummary {
  project_id: string;
  name: string;
  description: string;
  base_checkpoint_id: string | null;
  created_at_utc: string;
  updated_at_utc: string;
  tags: string[];
  metadata: Record<string, unknown>;
  checkpoint_count: number;
  relationship_count: number;
  branch_count: number;
  regression_count: number;
  latest_run: EvolutionRunSummary | null;
}

export interface EvolutionCheckpoint {
  checkpoint_id: string;
  project_id: string;
  version_label: string;
  model_name: string;
  source_path: string;
  resolved_path: string;
  architecture: string | null;
  model_family: string | null;
  parameter_count: number;
  layer_count: number | null;
  hidden_size: number | null;
  intermediate_size: number | null;
  num_attention_heads: number | null;
  num_key_value_heads: number | null;
  vocab_size: number | null;
  tensor_count: number;
  checkpoint_format: string;
  dtypes_present: string[];
  file_size_bytes: number;
  sha256_fingerprint: string;
  tensor_metadata_hash: string;
  parent_checkpoint_ids: string[];
  transformation_type: string;
  transformation_metadata: Record<string, unknown>;
  created_at_utc: string;
  framework_info: Record<string, string>;
  tokenizer_metadata: Record<string, unknown>;
  quantization_metadata: Record<string, unknown>;
  training_metadata: Record<string, unknown>;
  lora_metadata: Record<string, unknown>;
  experiment_notes: string;
  validation_status: string;
  duplicate_of_checkpoint_id: string | null;
  security_notes: string[];
}

export interface EvolutionGraphNode {
  checkpoint_id: string;
  version_label: string;
  model_name: string;
  transformation_type: string;
  generation_depth: number;
  lane_index: number;
  lane_count: number;
  is_root: boolean;
  is_leaf: boolean;
  is_merge_node: boolean;
  is_branch_point: boolean;
  is_base_model: boolean;
  parent_ids: string[];
  child_ids: string[];
  parameter_count: number;
  layer_count: number | null;
  hidden_size: number | null;
  dtypes_present: string[];
  sha256_fingerprint: string;
  validation_status: string;
  duplicate_of_checkpoint_id: string | null;
  created_at_utc: string;
}

export interface EvolutionGraphEdge {
  relationship_id: string;
  project_id: string;
  parent_checkpoint_id: string;
  child_checkpoint_id: string;
  transformation_type: string;
  merge_weight: number | null;
  transformation_metadata: Record<string, unknown>;
  created_at_utc: string;
}

export interface EvolutionBranchLineage {
  branch_id: string;
  root_checkpoint_id: string;
  leaf_checkpoint_id: string;
  checkpoint_ids: string[];
  version_labels: string[];
  contains_merge: boolean;
}

export interface EvolutionGraphPayload {
  base_checkpoint_id: string | null;
  topological_order: string[];
  root_checkpoint_ids: string[];
  leaf_checkpoint_ids: string[];
  branch_count: number;
  branches: EvolutionBranchLineage[];
  nodes: EvolutionGraphNode[];
  edges: EvolutionGraphEdge[];
}

export interface TransitionTensorRecord {
  scope: "consecutive" | "base_to_current" | "branch_to_branch" | string;
  source_checkpoint_id: string;
  source_label: string;
  target_checkpoint_id: string;
  target_label: string;
  transformation_type: string;
  epistemic_status: string;
  compatible_for_direct_tensor_diff: boolean;
  compatibility_warnings: string[];
  compared_tensors: number;
  modified_tensors: number;
  unchanged_tensors: number;
  modified_tensor_pct: number;
  unchanged_tensor_pct: number;
  added_tensors: string[];
  removed_tensors: string[];
  shape_mismatched_tensors: Array<{
    name: string;
    shape_source: number[] | null;
    shape_target: number[] | null;
  }>;
  total_compared_parameters: number;
  modified_parameters: number;
  modified_parameter_pct: number;
  max_abs_diff: number | null;
  mean_abs_diff: number | null;
  rmse: number | null;
  l2_distance: number | null;
  relative_l2_distance: number | null;
  global_cosine_similarity: number | null;
  mean_tensor_cosine_similarity: number | null;
  zero_to_nonzero_transitions: number;
  nonzero_to_zero_transitions: number;
  sparsity_ratio_source: number | null;
  sparsity_ratio_target: number | null;
  sparsity_delta: number | null;
  distribution_shift_summary: {
    mean_wasserstein_1d: number | null;
    weighted_mean_shift: number | null;
    weighted_std_shift: number | null;
    max_tensor_wasserstein_1d: number | null;
  };
  per_layer_changes: Array<Record<string, unknown>>;
  top_k_modified_tensors: Array<{
    name: string;
    layer_index: number | null;
    canonical_role: string;
    shape: number[];
    status: string;
    relative_l2: number | null;
    l2_diff: number | null;
    max_abs_diff: number | null;
    rmse: number | null;
    cosine_similarity: number | null;
    changed_pct: number;
    zero_to_nonzero: number;
    nonzero_to_zero: number;
  }>;
  sum_of_step_l2_along_path: number | null;
  path_cancellation_ratio: number | null;
}

export interface LayerEvolutionPayload {
  epistemic_status: string;
  layer_indices: number[];
  layer_aggregates: Array<{
    layer_index: number;
    mean_transition_relative_l2: number;
    peak_transition_relative_l2: number;
    transitions_modified: number;
    total_transitions: number;
    consistency_ratio: number;
  }>;
  most_modified_layers: number[];
  least_modified_layers: number[];
  unusually_large_change_layers: Array<{
    layer_index: number;
    peak_relative_l2: number;
    z_score: number;
    epistemic_status: string;
  }>;
  consistently_changing_weight_layers: number[];
  consistently_drifting_representation_layers: number[];
  regression_associated_layers: Array<Record<string, unknown>>;
  heatmap: {
    layer_indices: number[];
    rows: Array<{
      transition_label: string;
      source_checkpoint_id: string;
      target_checkpoint_id: string;
      relative_l2_by_layer: number[];
      cosine_by_layer: number[];
    }>;
  };
  transition_profiles: Array<{
    scope: string;
    source_checkpoint_id: string;
    source_label: string;
    target_checkpoint_id: string;
    target_label: string;
    embedding_changes: Record<string, unknown>;
    output_head_changes: Record<string, unknown>;
    attention_projection_changes: Record<string, unknown>;
    mlp_changes: Record<string, unknown>;
    normalization_changes: Record<string, unknown>;
    layers: Array<Record<string, unknown>>;
  }>;
}

export interface BehavioralTimelineEntry {
  checkpoint_id: string;
  version_label: string;
  transformation_type: string;
  sample_size: number;
  pass_rate: number;
  pass_rate_wilson_ci95: [number, number];
  mean_score: number;
  factual_contradiction_rate: number | null;
  category_scores: Record<string, number>;
  category_pass_rates: Record<string, number>;
}

export interface BehavioralEvolutionPayload {
  epistemic_status: string;
  suite_id: string;
  suite_version: string;
  suite_fingerprint: string;
  sample_size: number;
  statistical_disclosure: string;
  checkpoint_evaluations: Array<{
    checkpoint_id: string;
    version_label: string;
    transformation_type: string;
    sample_size: number;
    passed_count: number;
    pass_rate: number;
    pass_rate_wilson_ci95: [number, number];
    mean_score: number;
    evaluated_at_utc: string;
    random_seed: number;
    categories: Record<
      string,
      {
        category: string;
        sample_size: number;
        passed_count: number;
        pass_rate: number;
        pass_rate_wilson_ci95: [number, number];
        mean_score: number;
      }
    >;
  }>;
  timeline: BehavioralTimelineEntry[];
  consecutive_transitions: Array<{
    scope: string;
    source_checkpoint_id: string;
    source_label: string;
    target_checkpoint_id: string;
    target_label: string;
    transformation_type: string;
    sample_size: number;
    output_consistency_rate: number;
    newly_introduced_failures: string[];
    newly_passed_cases: string[];
    persistent_failures: string[];
    paired_statistics: {
      mean_score_delta: number;
      pass_rate_delta: number;
      mcnemar_p_value: number;
      bootstrap_ci95_low: number;
      bootstrap_ci95_high: number;
      statistically_significant_95: boolean;
    };
    improved_categories: Array<Record<string, unknown>>;
    degraded_categories: Array<Record<string, unknown>>;
    observed_zero_to_positive_categories: Array<Record<string, unknown>>;
  }>;
  base_to_current_transitions: Array<Record<string, unknown>>;
  lineage_persistent_failures: Record<string, string[]>;
  surviving_regressions: Array<{
    branch_id: string;
    introduced_from_label: string;
    introduced_at_label: string;
    transformation_type: string;
    leaf_label: string;
    surviving_failed_case_ids: string[];
    affected_categories: string[];
  }>;
}

export interface RepresentationEvolutionPayload {
  epistemic_status: string;
  prompt_count: number;
  prompts: string[];
  layer_indices: number[];
  evolution_matrix: {
    base_checkpoint_id: string | null;
    layer_indices: number[];
    rows: Array<{
      checkpoint_id: string;
      version_label: string;
      transformation_type: string;
      compatible_with_base: boolean;
      compatibility_warnings: string[];
      layer_cka: Array<number | null>;
      layer_cosine: Array<number | null>;
      mean_cka_to_base: number | null;
      representation_drift: number | null;
      mean_logit_js_to_base: number | null;
    }>;
  };
  consecutive_transitions: Array<{
    source_checkpoint_id: string;
    source_label: string;
    target_checkpoint_id: string;
    target_label: string;
    transformation_type: string;
    compatible: boolean;
    compatibility_warnings: string[];
    mean_cka: number | null;
    mean_drift: number | null;
    mean_hidden_cosine: number | null;
    mean_logit_js_divergence: number | null;
    causal_patching: {
      epistemic_status: string;
      most_causal_layer: number | null;
      peak_recovery_fraction: number;
      scope: string;
    } | null;
  }>;
  consistent_drift_layers: number[];
  logit_lens_by_checkpoint: Record<string, unknown>;
}

export interface RegressionOriginEvent {
  event_id: string;
  branch_id: string;
  first_affected_checkpoint_id: string;
  first_affected_label: string;
  previous_checkpoint_id: string;
  previous_label: string;
  transformation_type: string;
  affected_category: string;
  pass_rate_before: number;
  pass_rate_after: number;
  pass_rate_drop: number;
  score_before: number;
  score_after: number;
  score_drop: number;
  failed_test_case_ids: string[];
  survives_to_leaf: boolean;
  downstream_affected_checkpoints: string[];
  correlated_layers: Array<{
    layer_index: number;
    weight_mean_relative_l2: number | null;
    weight_changed_pct: number | null;
    linear_cka: number | null;
    evidence_relationship: string;
  }>;
  weight_changes: {
    relative_l2_distance?: number | null;
    l2_distance?: number | null;
    max_abs_diff?: number | null;
    modified_parameter_pct?: number | null;
  };
  activation_changes: {
    mean_linear_cka?: number | null;
    mean_representation_drift?: number | null;
    mean_logit_js_divergence?: number | null;
  };
  statistical_confidence: {
    sample_size: number;
    mcnemar_p_value: number;
    confidence_level: string;
  };
  causal_verification: {
    causal_verification_exists: boolean;
    epistemic_status: string;
    most_causal_layer: number | null;
    peak_recovery_fraction: number | null;
    scope_disclaimer: string;
  };
  epistemic_status: string;
  interpretation_note: string;
}

export interface EvolutionInsightItem {
  insight_id: string;
  domain: string;
  statement: string;
  epistemic_status: string;
  checkpoints: string[];
  evaluation_cases: string[];
  supporting_measurements: Record<string, unknown>;
}

export interface EvolutionRunSummary {
  run_id: string;
  project_id: string;
  cache_key: string;
  status: string;
  stage_message: string;
  completed_steps: number;
  total_steps: number;
  started_at_utc: string;
  completed_at_utc: string | null;
  config: Record<string, unknown>;
  overview: {
    total_checkpoints: number;
    healthy_checkpoints: number;
    total_transformations: number;
    branch_count: number;
    detected_regressions: number;
    most_modified_layers: number[];
    least_modified_layers: number[];
    peak_cumulative_l2: number;
    min_base_cosine_similarity: number;
  } | null;
  warnings: string[];
}

export interface EvolutionRunBundle extends EvolutionRunSummary {
  cache_hit?: boolean;
  provenance: {
    run_id: string;
    project_id: string;
    content_cache_key: string;
    checkpoint_fingerprints: Record<
      string,
      { version_label: string; sha256: string; tensor_metadata_hash: string }
    >;
    evaluation_dataset_fingerprint: string | null;
    random_seed: number;
    software_versions: Record<string, string>;
    hardware: Record<string, unknown>;
    selected_device: string;
    device_reason: string;
    started_at_utc: string;
    completed_at_utc: string | null;
  };
  project: {
    project_id: string;
    name: string;
    description: string;
    base_checkpoint_id: string | null;
  };
  checkpoints: EvolutionCheckpoint[];
  graph: EvolutionGraphPayload;
  tensor_transitions: TransitionTensorRecord[];
  layer_evolution: LayerEvolutionPayload | null;
  behavioral_evolution: BehavioralEvolutionPayload | null;
  representation_evolution: RepresentationEvolutionPayload | null;
  regression_events: RegressionOriginEvent[];
  insights: EvolutionInsightItem[];
}

export interface EvolutionProjectDetailResponse {
  project: {
    project_id: string;
    name: string;
    description: string;
    base_checkpoint_id: string | null;
    created_at_utc: string;
    updated_at_utc: string;
    tags: string[];
  };
  checkpoints: EvolutionCheckpoint[];
  graph: EvolutionGraphPayload;
  runs: EvolutionRunSummary[];
  latest_analysis: EvolutionRunBundle | null;
}
