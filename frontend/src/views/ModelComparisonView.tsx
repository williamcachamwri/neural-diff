import React, { useState } from "react";
import { apiClient, ApiClientError } from "../api/client";
import type {
  BaselineRecord,
  DiscoveredModelsResponse,
  ExperimentSummaryItem,
} from "../types/api";

interface Props {
  discovered: DiscoveredModelsResponse | null;
  baselines: BaselineRecord[];
  modelA: string;
  modelB: string;
  onChangeModelA: (val: string) => void;
  onChangeModelB: (val: string) => void;
  onComparisonComplete: (exp: ExperimentSummaryItem) => void;
  onError: (err: ApiClientError | Error) => void;
}

const ALL_MODULES = [
  { id: "structural", label: "Structural & Compatibility Diff" },
  { id: "tensor", label: "Chunked Float64 Tensor Diff" },
  { id: "neuron", label: "Attention Heads, MLP & Embeddings" },
  { id: "activation", label: "Forward-Hook Activation Diff & CKA" },
  { id: "behavioral", label: "8-Suite Deterministic Behavioral Diff" },
  { id: "regression", label: "Regression Threshold Gate" },
];

export const ModelComparisonView: React.FC<Props> = ({
  discovered,
  baselines,
  modelA,
  modelB,
  onChangeModelA,
  onChangeModelB,
  onComparisonComplete,
  onError,
}) => {
  const [modules, setModules] = useState<string[]>([
    "structural",
    "tensor",
    "neuron",
    "regression",
  ]);
  const [experimentType, setExperimentType] = useState<
    "REAL MODEL EXPERIMENT" | "SYNTHETIC FIXTURE EXPERIMENT"
  >("REAL MODEL EXPERIMENT");
  const [device, setDevice] = useState<"auto" | "cpu" | "cuda">("auto");
  const [atol, setAtol] = useState<string>("0.000001");
  const [rtol, setRtol] = useState<string>("0.00001");
  const [maxCases, setMaxCases] = useState<string>("24");
  const [baselineId, setBaselineId] = useState<string>("");
  const [runningJob, setRunningJob] = useState<ExperimentSummaryItem | null>(null);

  const toggleModule = (modId: string) => {
    setModules((prev) =>
      prev.includes(modId) ? prev.filter((m) => m !== modId) : [...prev, modId]
    );
  };

  const applyPreset = (a: string, b: string, isReal: boolean) => {
    onChangeModelA(a);
    onChangeModelB(b);
    setExperimentType(isReal ? "REAL MODEL EXPERIMENT" : "SYNTHETIC FIXTURE EXPERIMENT");
  };

  const handleRun = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!modelA.trim() || !modelB.trim()) {
      onError(new Error("Both Model A and Model B must be specified."));
      return;
    }
    if (modules.length === 0) {
      onError(new Error("Select at least one analysis module."));
      return;
    }

    const autoSynth =
      modelA.toLowerCase().includes("synth") ||
      modelA.toLowerCase().includes("fixture") ||
      modelB.toLowerCase().includes("synth") ||
      modelB.toLowerCase().includes("fixture");

    try {
      const job = await apiClient.runComparison({
        model_a: modelA.trim(),
        model_b: modelB.trim(),
        modules,
        experiment_type: autoSynth ? "SYNTHETIC FIXTURE EXPERIMENT" : experimentType,
        preferred_device: device,
        tolerances: {
          atol: Number(atol) || 1e-6,
          rtol: Number(rtol) || 1e-5,
          zero_threshold: 0.0,
        },
        behavioral_max_cases: modules.includes("behavioral") ? Number(maxCases) || 24 : null,
        baseline_id: baselineId || null,
        async_job: true,
      });
      setRunningJob(job);

      // Poll genuine server-side step progress until completion/failure/cancellation
      let current = job;
      while (current.status === "running") {
        await new Promise((r) => setTimeout(r, 450));
        current = await apiClient.getComparison(job.comparison_id);
        setRunningJob(current);
      }

      setRunningJob(null);
      if (current.status === "completed") {
        onComparisonComplete(current);
      } else if (current.error) {
        onError(
          new ApiClientError(
            current.error.error,
            current.error.message,
            current.error.http_status,
            current.error.details || {}
          )
        );
      }
    } catch (err) {
      setRunningJob(null);
      onError(err instanceof Error ? err : new Error(String(err)));
    }
  };

  const handleCancel = async () => {
    if (!runningJob) return;
    try {
      await apiClient.cancelComparison(runningJob.comparison_id);
    } catch (err) {
      onError(err instanceof Error ? err : new Error(String(err)));
    }
  };

  return (
    <div data-testid="view-compare">
      <div className="panel">
        <div className="panel-header">
          <h2 className="panel-title">Configure &amp; Execute Model Comparison</h2>
          <span className="badge badge-info">Real Backend Execution</span>
        </div>

        {discovered && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 12.5, color: "var(--text-muted)", marginBottom: 8 }}>
              Quick-Select Verified Model Pairs:
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              <button
                type="button"
                className="btn"
                onClick={() =>
                  applyPreset("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", true)
                }
              >
                Qwen2.5-0.5B Base vs Instruct (Apache-2.0)
              </button>
              <button
                type="button"
                className="btn"
                onClick={() =>
                  applyPreset(
                    "HuggingFaceTB/SmolLM2-135M",
                    "HuggingFaceTB/SmolLM2-135M-Instruct",
                    true
                  )
                }
              >
                SmolLM2-135M Base vs Instruct (Apache-2.0)
              </button>
              {discovered.local_models.length >= 2 && (
                <button
                  type="button"
                  className="btn"
                  onClick={() =>
                    applyPreset(
                      discovered.local_models[0].source,
                      discovered.local_models[1].source,
                      false
                    )
                  }
                >
                  Local Fixture Pair ({discovered.local_models[0].label} vs{" "}
                  {discovered.local_models[1].label})
                </button>
              )}
            </div>
          </div>
        )}

        <form onSubmit={handleRun}>
          <div className="grid-2">
            <div className="field-group">
              <label htmlFor="input-model-a">Model A (Baseline Path or Hugging Face ID)</label>
              <input
                id="input-model-a"
                data-testid="input-model-a"
                type="text"
                value={modelA}
                onChange={(e) => onChangeModelA(e.target.value)}
                placeholder="e.g. Qwen/Qwen2.5-0.5B or local checkpoint path"
              />
            </div>
            <div className="field-group">
              <label htmlFor="input-model-b">Model B (Candidate Path or Hugging Face ID)</label>
              <input
                id="input-model-b"
                data-testid="input-model-b"
                type="text"
                value={modelB}
                onChange={(e) => onChangeModelB(e.target.value)}
                placeholder="e.g. Qwen/Qwen2.5-0.5B-Instruct or local checkpoint path"
              />
            </div>
          </div>

          <div className="grid-2" style={{ marginTop: 6 }}>
            <div className="field-group">
              <label htmlFor="select-exp-type">Experiment Provenance Classification</label>
              <select
                id="select-exp-type"
                value={experimentType}
                onChange={(e) =>
                  setExperimentType(
                    e.target.value as "REAL MODEL EXPERIMENT" | "SYNTHETIC FIXTURE EXPERIMENT"
                  )
                }
              >
                <option value="REAL MODEL EXPERIMENT">REAL MODEL EXPERIMENT</option>
                <option value="SYNTHETIC FIXTURE EXPERIMENT">SYNTHETIC FIXTURE EXPERIMENT</option>
              </select>
            </div>
            <div className="field-group">
              <label htmlFor="select-device">Preferred Compute Device</label>
              <select
                id="select-device"
                value={device}
                onChange={(e) => setDevice(e.target.value as "auto" | "cpu" | "cuda")}
              >
                <option value="auto">Auto (Verified CUDA GPU with CPU Fallback)</option>
                <option value="cuda">CUDA GPU</option>
                <option value="cpu">CPU Only</option>
              </select>
            </div>
          </div>

          <div className="grid-2">
            <div className="field-group">
              <label htmlFor="input-atol">Absolute Tolerance (atol)</label>
              <input
                id="input-atol"
                type="number"
                step="any"
                value={atol}
                onChange={(e) => setAtol(e.target.value)}
              />
            </div>
            <div className="field-group">
              <label htmlFor="input-rtol">Relative Tolerance (rtol)</label>
              <input
                id="input-rtol"
                type="number"
                step="any"
                value={rtol}
                onChange={(e) => setRtol(e.target.value)}
              />
            </div>
          </div>

          <div className="grid-2">
            <div className="field-group">
              <label htmlFor="input-max-cases">Behavioral Suite Case Limit (max 24)</label>
              <input
                id="input-max-cases"
                type="number"
                min={1}
                max={24}
                value={maxCases}
                onChange={(e) => setMaxCases(e.target.value)}
              />
            </div>
            <div className="field-group">
              <label htmlFor="select-baseline">Optional Versioned Baseline Gate</label>
              <select
                id="select-baseline"
                value={baselineId}
                onChange={(e) => setBaselineId(e.target.value)}
              >
                <option value="">None (Compare Model B directly against Model A)</option>
                {baselines.map((b) => (
                  <option key={b.baseline_id} value={b.baseline_id}>
                    {b.baseline_id} ({b.model_id} — pass {(b.overall_pass_rate * 100).toFixed(0)}%)
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="field-group" style={{ marginTop: 8 }}>
            <label>Analysis Modules</label>
            <div className="checkbox-grid">
              {ALL_MODULES.map((m) => (
                <label key={m.id} className="check-tile">
                  <input
                    type="checkbox"
                    checked={modules.includes(m.id)}
                    onChange={() => toggleModule(m.id)}
                  />
                  <span>{m.label}</span>
                </label>
              ))}
            </div>
          </div>

          <div style={{ marginTop: 16, display: "flex", alignItems: "center", gap: 12 }}>
            <button
              type="submit"
              className="btn btn-primary"
              data-testid="btn-run-comparison"
              disabled={runningJob !== null}
            >
              {runningJob ? "Executing Comparison..." : "Run Comparison"}
            </button>
            {runningJob && (
              <button type="button" className="btn btn-danger" onClick={handleCancel}>
                Cancel Active Job
              </button>
            )}
          </div>
        </form>
      </div>

      {runningJob && (
        <div className="panel" data-testid="live-job-status">
          <h3 className="panel-title">
            Active Backend Job: <span className="mono">{runningJob.comparison_id}</span>
          </h3>
          <div style={{ marginTop: 8, color: "var(--accent)" }}>
            Stage ({runningJob.completed_steps} / {runningJob.total_steps} completed modules):{" "}
            <strong>{runningJob.stage_message}</strong>
          </div>
        </div>
      )}
    </div>
  );
};
