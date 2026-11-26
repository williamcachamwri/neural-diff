import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";

const mockHealth = {
  status: "ok",
  version: "0.1.0",
  device: "cuda:0",
  device_reason: "Verified CUDA kernel execution on RTX 5050",
  hardware: {
    process_rss_mb: 312.4,
    system_available_ram_mb: 4820.0,
    system_total_ram_mb: 16088.0,
    cuda_available: true,
    cuda_allocated_mb: 0.0,
    cuda_reserved_mb: 0.0,
    cuda_total_vram_mb: 8151.0,
    selected_device: "cuda:0",
  },
  software: {
    torch: "2.10.0+cu128",
    transformers: "5.24.1",
    safetensors: "0.8.0",
  },
  counts: {
    experiments: 1,
    completed_experiments: 1,
    baselines: 1,
    active_jobs: 0,
  },
};

const mockBundle = {
  comparison_id: "ndiff-test001",
  created_at_utc: "2026-11-26T19:00:00Z",
  experiment_type: "SYNTHETIC FIXTURE EXPERIMENT",
  selected_modules: ["structural", "tensor", "neuron", "regression"],
  model_a: {
    model_id: "synth_a",
    resolved_path: "data/synth_a",
    format: "huggingface_local_dir",
    revision: null,
    tensor_count: 21,
    total_parameters: 12480,
    total_bytes: 49920,
    shard_count: 1,
    dtypes_present: ["float32"],
    architecture: {
      architecture_class: "Qwen2ForCausalLM",
      model_type: "qwen2",
      hidden_size: 32,
      intermediate_size: 64,
      num_hidden_layers: 2,
      num_attention_heads: 4,
      num_key_value_heads: 2,
      vocab_size: 128,
      torch_dtype: "float32",
    },
    security_notes: ["Verified Safetensors header"],
  },
  model_b: {
    model_id: "synth_b",
    resolved_path: "data/synth_b",
    format: "huggingface_local_dir",
    revision: null,
    tensor_count: 21,
    total_parameters: 12480,
    total_bytes: 49920,
    shard_count: 1,
    dtypes_present: ["float32"],
    architecture: {
      architecture_class: "Qwen2ForCausalLM",
      model_type: "qwen2",
      hidden_size: 32,
      intermediate_size: 64,
      num_hidden_layers: 2,
      num_attention_heads: 4,
      num_key_value_heads: 2,
      vocab_size: 128,
      torch_dtype: "float32",
    },
    security_notes: ["Verified Safetensors header"],
  },
  epistemic_ledger: [
    {
      statement: "Observed 21 shared tensors.",
      epistemic_status: "OBSERVED",
      evidence_source: "StructuralDiffEngine",
    },
  ],
  structural: {
    model_a_parameters: 12480,
    model_b_parameters: 12480,
    delta_parameters: 0,
    counts: { shared: 21, added: 0, removed: 0 },
    shared_tensors: ["model.norm.weight"],
    added_tensors: [],
    removed_tensors: [],
    inferred_renames: [],
    shape_changes: [],
    dtype_changes: [],
    layer_deltas: [
      {
        layer_index: 0,
        present_in_a: true,
        present_in_b: true,
        tensor_count_a: 9,
        tensor_count_b: 9,
        parameters_a: 4160,
        parameters_b: 4160,
        delta_parameters: 0,
      },
    ],
    compatibility: {
      epistemic_status: "OBSERVED",
      compatible_for_direct_tensor_diff: true,
      compatible_for_structural_diff: true,
      compatible_for_head_neuron_analysis: true,
      compatible_for_activation_diff: true,
      compatible_for_behavioral_diff: true,
      shared_tensor_count: 21,
      shape_matched_tensor_count: 21,
      jaccard_key_overlap: 1.0,
      issues: [],
      summary: "Fully aligned checkpoints (21/21 tensors match in name and shape).",
    },
  },
  tensor_diff: {
    tolerances: { atol: 1e-6, rtol: 1e-5, zero_threshold: 0.0 },
    summary: {
      total_union_tensors: 21,
      compared_tensors: 21,
      identical_tensors: 1,
      within_tolerance_tensors: 0,
      modified_tensors: 20,
      shape_mismatch_tensors: 0,
      missing_in_b_tensors: 0,
      added_in_b_tensors: 0,
      total_compared_parameters: 12480,
      total_changed_parameters: 12448,
      global_changed_pct: 99.74,
      global_mean_cosine: 0.9988,
      global_mean_relative_l2: 0.048,
      global_max_abs_diff: 0.21,
      has_any_nan_or_inf: false,
    },
    layer_summaries: [],
    top_diverged_names: [],
  },
  neuron_attention: null,
  activation: null,
  behavioral: null,
  regression: {
    verdict: "PASSED",
    highest_severity: "INFO",
    thresholds: { policy_name: "production-default-v1" },
    finding_counts: {},
    findings: [],
    failure_reproductions: [],
    baseline_comparison: null,
  },
  execution_telemetry: { duration_seconds: 0.12 },
  reproducibility: {},
};

describe("NEURAL DIFF Frontend Shell & Navigation", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/api/health")) {
        return new Response(JSON.stringify(mockHealth), { status: 200 });
      }
      if (url.endsWith("/api/models/discover")) {
        return new Response(JSON.stringify({ local_models: [], hub_presets: [] }), {
          status: 200,
        });
      }
      if (url.endsWith("/api/comparisons")) {
        return new Response(
          JSON.stringify({
            experiments: [
              {
                comparison_id: "ndiff-test001",
                created_at_utc: "2026-11-26T19:00:00Z",
                status: "completed",
                stage_message: "Completed",
                completed_steps: 5,
                total_steps: 5,
                experiment_type: "SYNTHETIC FIXTURE EXPERIMENT",
                model_a_id: "synth_a",
                model_b_id: "synth_b",
                selected_modules: ["structural", "tensor", "regression"],
                verdict: "PASSED",
                summary: { global_changed_pct: 99.74 },
                error: null,
              },
            ],
          }),
          { status: 200 }
        );
      }
      if (url.endsWith("/api/baselines")) {
        return new Response(JSON.stringify({ baselines: [] }), { status: 200 });
      }
      if (url.includes("/api/comparisons/ndiff-test001/tensors")) {
        return new Response(JSON.stringify({ total: 0, records: [] }), { status: 200 });
      }
      if (url.endsWith("/api/comparisons/ndiff-test001")) {
        return new Response(
          JSON.stringify({
            comparison_id: "ndiff-test001",
            status: "completed",
            bundle: mockBundle,
          }),
          { status: 200 }
        );
      }
      if (url.endsWith("/api/settings")) {
        return new Response(
          JSON.stringify({
            preferred_device: "auto",
            atol: 1e-6,
            rtol: 1e-5,
            zero_threshold: 0,
            chunk_elements: 1048576,
            allow_pytorch_weights_only: true,
            max_new_tokens: 96,
          }),
          { status: 200 }
        );
      }
      return new Response(
        JSON.stringify({ error: "NOT_FOUND", message: "Unknown mock endpoint" }),
        { status: 404 }
      );
    });
  });

  it("renders dashboard hardware telemetry and navigates across core views", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/NEURAL DIFF Engineering Workbench/i)).toBeInTheDocument();
    });
    expect(screen.getByText("CUDA:0")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("nav-compare"));
    expect(screen.getByTestId("view-compare")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("nav-compatibility"));
    expect(screen.getByTestId("view-compatibility")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("nav-structural"));
    await waitFor(() => {
      expect(screen.getByTestId("view-structural")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId("nav-regression"));
    expect(screen.getByTestId("view-regression")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("nav-settings"));
    await waitFor(() => {
      expect(screen.getByTestId("view-settings")).toBeInTheDocument();
    });
  });
});
