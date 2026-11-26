import React from "react";
import type {
  ComparisonBundle,
  ExperimentSummaryItem,
  HealthResponse,
  ViewId,
} from "../types/api";

interface Props {
  health: HealthResponse | null;
  experiments: ExperimentSummaryItem[];
  activeBundle: ComparisonBundle | null;
  onNavigate: (view: ViewId) => void;
  onSelectExperiment: (comparisonId: string) => void;
}

export const DashboardView: React.FC<Props> = ({
  health,
  experiments,
  activeBundle,
  onNavigate,
  onSelectExperiment,
}) => {
  const realCount = experiments.filter(
    (e) => e.experiment_type === "REAL MODEL EXPERIMENT" && e.status === "completed"
  ).length;
  const synthCount = experiments.filter(
    (e) => e.experiment_type === "SYNTHETIC FIXTURE EXPERIMENT" && e.status === "completed"
  ).length;

  return (
    <div data-testid="view-dashboard">
      <div className="panel">
        <div className="panel-header">
          <div>
            <h2 className="panel-title">NEURAL DIFF Engineering Workbench</h2>
            <div style={{ color: "var(--text-muted)", fontSize: 13, marginTop: 4 }}>
              Git for AI Brains — Structural, Tensor, Activation &amp; Behavioral Intelligence Comparison
            </div>
          </div>
          <div style={{ display: "flex", gap: 10 }}>
            <button
              className="btn btn-primary"
              data-testid="cta-new-comparison"
              onClick={() => onNavigate("compare")}
            >
              New Model Comparison
            </button>
            <button className="btn" onClick={() => onNavigate("compatibility")}>
              Validate Compatibility
            </button>
          </div>
        </div>

        <div className="kpi-grid">
          <div className="kpi-card">
            <div className="kpi-label">Compute Engine</div>
            <div className="kpi-value mono">{health ? health.device.toUpperCase() : "PROBING..."}</div>
            <div className="kpi-sub">{health?.device_reason ?? "Querying /api/health"}</div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">Host &amp; GPU Memory</div>
            <div className="kpi-value">
              {health
                ? `${(health.hardware.system_available_ram_mb / 1024).toFixed(1)} GB RAM`
                : "—"}
            </div>
            <div className="kpi-sub">
              {health?.hardware.cuda_available
                ? `CUDA Total VRAM: ${(health.hardware.cuda_total_vram_mb / 1024).toFixed(1)} GB`
                : "CPU Mode"}
            </div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">Real Model Experiments</div>
            <div className="kpi-value">{realCount}</div>
            <div className="kpi-sub">Strictly separated from {synthCount} synthetic fixture runs</div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">Saved Baselines</div>
            <div className="kpi-value">{health?.counts.baselines ?? 0}</div>
            <div className="kpi-sub">Torch {health?.software.torch ?? "—"} • Safetensors {health?.software.safetensors ?? "—"}</div>
          </div>
        </div>
      </div>

      {activeBundle && (
        <div className="panel" data-testid="active-bundle-summary">
          <div className="panel-header">
            <h3 className="panel-title">
              Active Loaded Comparison:{" "}
              <span className="mono">{activeBundle.comparison_id}</span>
            </h3>
            <span className="badge badge-info">{activeBundle.experiment_type}</span>
          </div>
          <div className="grid-2">
            <div>
              <strong>Model A (Baseline):</strong>{" "}
              <span className="mono">{activeBundle.model_a.model_id}</span> (
              {activeBundle.model_a.total_parameters.toLocaleString()} params)
            </div>
            <div>
              <strong>Model B (Candidate):</strong>{" "}
              <span className="mono">{activeBundle.model_b.model_id}</span> (
              {activeBundle.model_b.total_parameters.toLocaleString()} params)
            </div>
          </div>
          <div style={{ marginTop: 12, display: "flex", flexWrap: "wrap", gap: 8 }}>
            <button className="btn" onClick={() => onNavigate("tensors")}>
              Tensor Explorer
            </button>
            <button className="btn" onClick={() => onNavigate("structural")}>
              Structural Diff
            </button>
            <button className="btn" onClick={() => onNavigate("layers")}>
              Layer Heatmap &amp; Heads
            </button>
            <button className="btn" onClick={() => onNavigate("activations")}>
              Activation Diff
            </button>
            <button className="btn" onClick={() => onNavigate("behavioral")}>
              Behavioral Diff
            </button>
            <button className="btn" onClick={() => onNavigate("regression")}>
              Regression Gate
            </button>
            <button className="btn" onClick={() => onNavigate("reports")}>
              Export Reports
            </button>
          </div>
        </div>
      )}

      <div className="panel">
        <div className="panel-header">
          <h3 className="panel-title">Recent Comparisons</h3>
          <button className="btn" onClick={() => onNavigate("history")}>
            Full History
          </button>
        </div>
        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Type</th>
                <th>Model A → Model B</th>
                <th>Modules</th>
                <th>Changed %</th>
                <th>Verdict</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {experiments.length === 0 ? (
                <tr>
                  <td colSpan={7} style={{ color: "var(--text-muted)" }}>
                    No comparisons recorded yet. Launch a comparison to inspect live measurements.
                  </td>
                </tr>
              ) : (
                experiments.slice(0, 8).map((exp) => (
                  <tr key={exp.comparison_id}>
                    <td className="mono">{exp.comparison_id}</td>
                    <td>
                      <span
                        className={`badge ${
                          exp.experiment_type === "REAL MODEL EXPERIMENT"
                            ? "badge-ok"
                            : "badge-purple"
                        }`}
                      >
                        {exp.experiment_type === "REAL MODEL EXPERIMENT" ? "REAL MODEL" : "SYNTHETIC"}
                      </span>
                    </td>
                    <td className="mono">
                      {exp.model_a_id} → {exp.model_b_id}
                    </td>
                    <td>{exp.selected_modules.join(", ")}</td>
                    <td className="num">
                      {exp.summary?.global_changed_pct != null
                        ? `${exp.summary.global_changed_pct.toFixed(2)}%`
                        : "—"}
                    </td>
                    <td>
                      {exp.verdict ? (
                        <span
                          className={`badge ${
                            exp.verdict === "PASSED"
                              ? "badge-ok"
                              : exp.verdict === "WARNING"
                                ? "badge-warn"
                                : "badge-danger"
                          }`}
                        >
                          {exp.verdict}
                        </span>
                      ) : (
                        <span className="badge badge-info">{exp.status}</span>
                      )}
                    </td>
                    <td>
                      {exp.status === "completed" && (
                        <button
                          className="btn"
                          onClick={() => onSelectExperiment(exp.comparison_id)}
                        >
                          Inspect
                        </button>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="panel">
        <h3 className="panel-title" style={{ marginBottom: 10 }}>
          Epistemic Verification Legend
        </h3>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 10, fontSize: 12.5 }}>
          <span className="badge badge-ok">OBSERVED — Direct tensor/activation/output measurement</span>
          <span className="badge badge-purple">INFERRED — Heuristic structural rename match</span>
          <span className="badge badge-warn">CORRELATED — Co-occurring internal &amp; output drift</span>
          <span className="badge badge-info">CAUSALLY VERIFIED — Requires counterfactual patching</span>
          <span className="badge">NOT VERIFIED — Unexecuted claim boundary</span>
        </div>
      </div>
    </div>
  );
};
