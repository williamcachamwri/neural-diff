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
  { id: "structural", label: "Structural & Topology" },
  { id: "tensor", label: "Float64 Tensor Math" },
  { id: "neuron", label: "Attention Heads & MLP" },
  { id: "activation", label: "Forward-Hook CKA" },
  { id: "behavioral", label: "Behavioral Suites" },
  { id: "regression", label: "Regression Gate" },
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
      onError(new Error("Specify both Model A and Model B."));
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
          <div>
            <h2 className="panel-title">Model Comparison</h2>
            <div className="panel-subtitle" style={{ marginTop: 2 }}>
              Select checkpoints, numerical tolerances, and analysis stages
            </div>
          </div>
        </div>

        {discovered && (
          <div style={{ marginBottom: 18 }}>
            <div style={{ fontSize: 11.5, color: "var(--text-muted)", marginBottom: 8 }}>
              Presets
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              <button
                type="button"
                className="btn"
                onClick={() =>
                  applyPreset("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", true)
                }
              >
                Qwen2.5-0.5B · Base vs Instruct
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
                SmolLM2-135M · Base vs Instruct
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
                  Local Fixture · {discovered.local_models[0].label} vs{" "}
                  {discovered.local_models[1].label}
                </button>
              )}
            </div>
          </div>
        )}

        <form onSubmit={handleRun}>
          <div className="grid-2">
            <div className="field-group">
              <label htmlFor="input-model-a">Model A (Baseline)</label>
              <input
                id="input-model-a"
                data-testid="input-model-a"
                type="text"
                value={modelA}
                onChange={(e) => onChangeModelA(e.target.value)}
                placeholder="Qwen/Qwen2.5-0.5B or local path"
              />
            </div>
            <div className="field-group">
              <label htmlFor="input-model-b">Model B (Candidate)</label>
              <input
                id="input-model-b"
                data-testid="input-model-b"
                type="text"
                value={modelB}
                onChange={(e) => onChangeModelB(e.target.value)}
                placeholder="Qwen/Qwen2.5-0.5B-Instruct or local path"
              />
            </div>
          </div>

          <div className="grid-2">
            <div className="field-group">
              <label htmlFor="select-exp-type">Provenance</label>
              <select
                id="select-exp-type"
                value={experimentType}
                onChange={(e) =>
                  setExperimentType(
                    e.target.value as "REAL MODEL EXPERIMENT" | "SYNTHETIC FIXTURE EXPERIMENT"
                  )
                }
              >
                <option value="REAL MODEL EXPERIMENT">Real model benchmark</option>
                <option value="SYNTHETIC FIXTURE EXPERIMENT">Synthetic fixture</option>
              </select>
            </div>
            <div className="field-group">
              <label htmlFor="select-device">Compute Device</label>
              <select
                id="select-device"
                value={device}
                onChange={(e) => setDevice(e.target.value as "auto" | "cpu" | "cuda")}
              >
                <option value="auto">Auto (CUDA with CPU fallback)</option>
                <option value="cuda">CUDA GPU</option>
                <option value="cpu">CPU</option>
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
              <label htmlFor="input-max-cases">Behavioral Case Limit (1–24)</label>
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
              <label htmlFor="select-baseline">Baseline Gate</label>
              <select
                id="select-baseline"
                value={baselineId}
                onChange={(e) => setBaselineId(e.target.value)}
              >
                <option value="">Direct A → B comparison</option>
                {baselines.map((b) => (
                  <option key={b.baseline_id} value={b.baseline_id}>
                    {b.baseline_id} ({b.model_id} · {(b.overall_pass_rate * 100).toFixed(0)}%)
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="field-group" style={{ marginTop: 6 }}>
            <label>Pipeline Stages</label>
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

          <div style={{ marginTop: 18, display: "flex", alignItems: "center", gap: 10 }}>
            <button
              type="submit"
              className="btn btn-primary"
              data-testid="btn-run-comparison"
              disabled={runningJob !== null}
            >
              {runningJob ? "Running..." : "Run Comparison"}
            </button>
            {runningJob && (
              <button type="button" className="btn btn-danger" onClick={handleCancel}>
                Cancel
              </button>
            )}
          </div>
        </form>
      </div>

      {runningJob && (
        <div className="panel" data-testid="live-job-status">
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
            <span className="mono" style={{ color: "var(--text-main)" }}>
              {runningJob.comparison_id}
            </span>
            <span className="mono" style={{ color: "var(--text-muted)" }}>
              {runningJob.completed_steps} / {runningJob.total_steps}
            </span>
          </div>
          <div
            style={{
              height: 4,
              background: "rgba(255,255,255,0.06)",
              borderRadius: 99,
              overflow: "hidden",
              marginBottom: 8,
            }}
          >
            <div
              style={{
                width: `${Math.round(
                  (runningJob.completed_steps / Math.max(1, runningJob.total_steps)) * 100
                )}%`,
                height: "100%",
                background: "#f4f4f5",
                transition: "width 0.25s ease",
              }}
            />
          </div>
          <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>
            {runningJob.stage_message}
          </div>
        </div>
      )}
    </div>
  );
};
