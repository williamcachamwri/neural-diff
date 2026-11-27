import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { EvolutionStudioView } from "../views/EvolutionStudioView";

const mockRunBundle = {
  run_id: "evorun-test01",
  project_id: "evo-proj01",
  cache_key: "abc123def456abc123def456abc123def456abc123def456abc123def456abcd",
  status: "completed",
  stage_message: "Completed",
  completed_steps: 5,
  total_steps: 5,
  started_at_utc: "2026-11-27T05:00:00Z",
  completed_at_utc: "2026-11-27T05:00:04Z",
  config: {},
  warnings: [],
  overview: {
    total_checkpoints: 3,
    healthy_checkpoints: 3,
    total_transformations: 2,
    branch_count: 1,
    detected_regressions: 1,
    most_modified_layers: [2],
    least_modified_layers: [1],
    peak_cumulative_l2: 11.4891,
    min_base_cosine_similarity: 0.9612,
  },
  provenance: {
    run_id: "evorun-test01",
    project_id: "evo-proj01",
    content_cache_key: "abc123def456",
    checkpoint_fingerprints: {},
    evaluation_dataset_fingerprint: "feedface1234",
    random_seed: 42,
    software_versions: { torch: "2.10.0", transformers: "4.50.0", safetensors: "0.5.3" },
    hardware: {},
    selected_device: "cuda:0",
    device_reason: "CUDA available",
    started_at_utc: "2026-11-27T05:00:00Z",
    completed_at_utc: "2026-11-27T05:00:04Z",
  },
  project: {
    project_id: "evo-proj01",
    name: "Qwen Lifecycle Study",
    description: "Base -> SFT -> Quantized",
    base_checkpoint_id: "ckpt-1",
  },
  checkpoints: [
    {
      checkpoint_id: "ckpt-1",
      project_id: "evo-proj01",
      version_label: "Base",
      model_name: "qwen-base",
      source_path: "data/base",
      resolved_path: "/abs/data/base",
      architecture: "Qwen2ForCausalLM",
      model_family: "qwen2",
      parameter_count: 500000,
      layer_count: 3,
      hidden_size: 16,
      intermediate_size: 32,
      num_attention_heads: 4,
      num_key_value_heads: 2,
      vocab_size: 64,
      tensor_count: 30,
      checkpoint_format: "huggingface_local_dir",
      dtypes_present: ["F32"],
      file_size_bytes: 2048000,
      sha256_fingerprint: "11112222333344445555666677778888",
      tensor_metadata_hash: "aaaa1111",
      parent_checkpoint_ids: [],
      transformation_type: "Base Model",
      transformation_metadata: {},
      created_at_utc: "2026-11-27T05:00:00Z",
      framework_info: {},
      tokenizer_metadata: {},
      quantization_metadata: {},
      training_metadata: {},
      lora_metadata: {},
      experiment_notes: "Initial base",
      validation_status: "VERIFIED",
      duplicate_of_checkpoint_id: null,
      security_notes: [],
    },
  ],
  graph: {
    base_checkpoint_id: "ckpt-1",
    topological_order: ["ckpt-1"],
    root_checkpoint_ids: ["ckpt-1"],
    leaf_checkpoint_ids: ["ckpt-1"],
    branch_count: 1,
    branches: [
      {
        branch_id: "branch-01",
        root_checkpoint_id: "ckpt-1",
        leaf_checkpoint_id: "ckpt-1",
        checkpoint_ids: ["ckpt-1"],
        version_labels: ["Base", "SFT", "Quantized"],
        contains_merge: false,
      },
    ],
    nodes: [
      {
        checkpoint_id: "ckpt-1",
        version_label: "Base",
        model_name: "qwen-base",
        transformation_type: "Base Model",
        generation_depth: 0,
        lane_index: 0,
        lane_count: 1,
        is_root: true,
        is_leaf: true,
        is_merge_node: false,
        is_branch_point: false,
        is_base_model: true,
        parent_ids: [],
        child_ids: [],
        parameter_count: 500000,
        layer_count: 3,
        hidden_size: 16,
        dtypes_present: ["F32"],
        sha256_fingerprint: "1111222233334444",
        validation_status: "VERIFIED",
        duplicate_of_checkpoint_id: null,
        created_at_utc: "2026-11-27T05:00:00Z",
      },
    ],
    edges: [],
  },
  tensor_transitions: [
    {
      scope: "base_to_current",
      source_checkpoint_id: "ckpt-1",
      source_label: "Base",
      target_checkpoint_id: "ckpt-1",
      target_label: "Quantized",
      transformation_type: "Quantization",
      epistemic_status: "OBSERVED",
      compatible_for_direct_tensor_diff: true,
      compatibility_warnings: [],
      compared_tensors: 30,
      modified_tensors: 28,
      unchanged_tensors: 2,
      modified_tensor_pct: 93.3,
      unchanged_tensor_pct: 6.7,
      added_tensors: [],
      removed_tensors: [],
      shape_mismatched_tensors: [],
      total_compared_parameters: 500000,
      modified_parameters: 420000,
      modified_parameter_pct: 84.0,
      max_abs_diff: 0.05,
      mean_abs_diff: 0.012,
      rmse: 0.018,
      l2_distance: 4.12,
      relative_l2_distance: 0.085,
      global_cosine_similarity: 0.994,
      mean_tensor_cosine_similarity: 0.995,
      zero_to_nonzero_transitions: 0,
      nonzero_to_zero_transitions: 120,
      sparsity_ratio_source: 0.0,
      sparsity_ratio_target: 0.05,
      sparsity_delta: 0.05,
      distribution_shift_summary: {
        mean_wasserstein_1d: 0.01,
        weighted_mean_shift: 0.0,
        weighted_std_shift: -0.001,
        max_tensor_wasserstein_1d: 0.02,
      },
      per_layer_changes: [],
      top_k_modified_tensors: [],
      sum_of_step_l2_along_path: 5.1,
      path_cancellation_ratio: 0.8078,
    },
  ],
  layer_evolution: null,
  behavioral_evolution: null,
  representation_evolution: null,
  regression_events: [
    {
      event_id: "reg-01",
      branch_id: "branch-01",
      first_affected_checkpoint_id: "ckpt-1",
      first_affected_label: "Quantized",
      previous_checkpoint_id: "ckpt-0",
      previous_label: "DPO",
      transformation_type: "Quantization",
      affected_category: "overall",
      pass_rate_before: 0.91,
      pass_rate_after: 0.78,
      pass_rate_drop: 0.13,
      score_before: 0.91,
      score_after: 0.78,
      score_drop: 0.13,
      failed_test_case_ids: ["math-01", "code-02"],
      survives_to_leaf: true,
      downstream_affected_checkpoints: ["ckpt-1"],
      correlated_layers: [],
      weight_changes: { relative_l2_distance: 0.085 },
      activation_changes: { mean_linear_cka: 0.92 },
      statistical_confidence: {
        sample_size: 24,
        mcnemar_p_value: 0.031,
        confidence_level: "high_statistical_confidence",
      },
      causal_verification: {
        causal_verification_exists: false,
        epistemic_status: "CORRELATED",
        most_causal_layer: null,
        peak_recovery_fraction: null,
        scope_disclaimer: "Correlational only",
      },
      epistemic_status: "CORRELATED",
      interpretation_note: "Observed drop",
    },
  ],
  insights: [
    {
      insight_id: "ins-1",
      domain: "regression_origin",
      statement: "Quantization introduced the largest observed regression (91.0% -> 78.0%).",
      epistemic_status: "CORRELATED",
      checkpoints: ["ckpt-0", "ckpt-1"],
      evaluation_cases: ["math-01"],
      supporting_measurements: {},
    },
  ],
};

describe("EvolutionStudioView", () => {
  it("renders dashboard insights, DAG timeline, regressions, and multi-format export controls", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/api/evolution/overview")) {
          return new Response(
            JSON.stringify({
              summary: {
                total_projects: 1,
                total_checkpoints: 3,
                total_transformations: 2,
                total_branches: 1,
                total_detected_regressions: 1,
                completed_analysis_runs: 1,
              },
              supported_transformations: ["Base Model", "SFT", "DPO", "Quantization"],
              schema_migrations: [],
              active_jobs: 0,
            }),
            { status: 200 }
          );
        }
        if (url.includes("/api/evolution/projects?")) {
          return new Response(
            JSON.stringify({
              total: 1,
              limit: 50,
              offset: 0,
              projects: [
                {
                  project_id: "evo-proj01",
                  name: "Qwen Lifecycle Study",
                  description: "Study",
                  base_checkpoint_id: "ckpt-1",
                  created_at_utc: "2026-11-27T05:00:00Z",
                  updated_at_utc: "2026-11-27T05:00:00Z",
                  tags: [],
                  metadata: {},
                  checkpoint_count: 3,
                  relationship_count: 2,
                  branch_count: 1,
                  regression_count: 1,
                  latest_run: null,
                },
              ],
            }),
            { status: 200 }
          );
        }
        return new Response(
          JSON.stringify({
            project: mockRunBundle.project,
            checkpoints: mockRunBundle.checkpoints,
            graph: mockRunBundle.graph,
            runs: [],
            latest_analysis: mockRunBundle,
          }),
          { status: 200 }
        );
      })
    );

    render(<EvolutionStudioView discovered={null} onError={() => {}} />);

    await waitFor(() => {
      expect(
        screen.getByText(/Quantization introduced the largest observed regression/i)
      ).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId("evo-tab-graph"));
    expect(screen.getByTestId("evo-section-graph")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("evo-tab-regressions"));
    expect(screen.getByTestId("evo-section-regressions")).toBeInTheDocument();
    expect(screen.getByText(/Correlational Only/i)).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("evo-tab-experiments"));
    expect(screen.getByTestId("btn-evo-download-pdf")).toBeInTheDocument();
    expect(screen.getByTestId("btn-evo-download-csv")).toBeInTheDocument();

    vi.unstubAllGlobals();
  });
});
