import React from "react";
import type { ExperimentSummaryItem } from "../types/api";

interface Props {
  experiments: ExperimentSummaryItem[];
  activeComparisonId: string | null;
  onSelectExperiment: (comparisonId: string) => void;
  onRefresh: () => void;
}

export const ExperimentHistoryView: React.FC<Props> = ({
  experiments,
  activeComparisonId,
  onSelectExperiment,
  onRefresh,
}) => {
  return (
    <div className="panel" data-testid="view-history">
      <div className="panel-header">
        <h2 className="panel-title">
          Persisted Experiment History ({experiments.length} runs)
        </h2>
        <button className="btn" onClick={onRefresh}>
          Refresh List
        </button>
      </div>
      <div className="data-table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Comparison ID</th>
              <th>Timestamp (UTC)</th>
              <th>Provenance</th>
              <th>Model A → Model B</th>
              <th>Modules</th>
              <th>Changed %</th>
              <th>Behav Δ</th>
              <th>Verdict / Status</th>
              <th>Duration</th>
              <th>Load</th>
            </tr>
          </thead>
          <tbody>
            {experiments.length === 0 ? (
              <tr>
                <td colSpan={10} style={{ color: "var(--text-muted)" }}>
                  No historical runs stored in SQLite yet.
                </td>
              </tr>
            ) : (
              experiments.map((exp) => (
                <tr
                  key={exp.comparison_id}
                  style={{
                    background:
                      exp.comparison_id === activeComparisonId
                        ? "rgba(255, 255, 255, 0.055)"
                        : undefined,
                  }}
                >
                  <td className="mono">{exp.comparison_id}</td>
                  <td className="mono">{exp.created_at_utc.slice(0, 19)}Z</td>
                  <td>
                    <span
                      className={`badge ${
                        exp.experiment_type === "REAL MODEL EXPERIMENT"
                          ? "badge-ok"
                          : "badge-purple"
                      }`}
                    >
                      {exp.experiment_type === "REAL MODEL EXPERIMENT"
                        ? "REAL MODEL"
                        : "SYNTHETIC"}
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
                  <td className="num">
                    {exp.summary?.behavioral_score_delta != null
                      ? `${exp.summary.behavioral_score_delta >= 0 ? "+" : ""}${exp.summary.behavioral_score_delta.toFixed(3)}`
                      : "—"}
                  </td>
                  <td>
                    <span
                      className={`badge ${
                        exp.verdict === "PASSED"
                          ? "badge-ok"
                          : exp.verdict === "FAILED_REGRESSION" || exp.status === "failed"
                            ? "badge-danger"
                            : "badge-info"
                      }`}
                    >
                      {exp.verdict ?? exp.status}
                    </span>
                  </td>
                  <td className="num">
                    {exp.summary?.duration_seconds != null
                      ? `${exp.summary.duration_seconds.toFixed(2)}s`
                      : "—"}
                  </td>
                  <td>
                    {exp.status === "completed" && (
                      <button
                        className="btn"
                        onClick={() => onSelectExperiment(exp.comparison_id)}
                      >
                        Load
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
  );
};

