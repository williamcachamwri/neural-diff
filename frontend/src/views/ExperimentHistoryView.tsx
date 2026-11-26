import React from "react";
import type { ExperimentSummaryItem } from "../types/api";
import { formatStatus, formatVerdict } from "../utils/format";

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
        <div>
          <h2 className="panel-title">Experiment Ledger ({experiments.length})</h2>
          <div className="panel-subtitle" style={{ marginTop: 2 }}>
            Persisted model comparisons and runtime telemetry
          </div>
        </div>
        <button type="button" className="btn" onClick={onRefresh}>
          Refresh
        </button>
      </div>
      <div className="data-table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Run ID</th>
              <th>Source</th>
              <th>Checkpoint Pair</th>
              <th className="num">Modified</th>
              <th className="num">Score Δ</th>
              <th>Outcome</th>
              <th className="num">Runtime</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {experiments.length === 0 ? (
              <tr>
                <td colSpan={8} style={{ color: "var(--text-muted)" }}>
                  No historical runs recorded yet.
                </td>
              </tr>
            ) : (
              experiments.map((exp) => {
                const isReal = exp.experiment_type === "REAL MODEL EXPERIMENT";
                const isActive = exp.comparison_id === activeComparisonId;
                return (
                  <tr
                    key={exp.comparison_id}
                    className={isActive ? "is-active-row" : ""}
                  >
                    <td className="mono" style={{ color: "var(--text-main)" }}>
                      {exp.comparison_id}
                    </td>
                    <td>
                      <span className="status-inline">
                        <span className={`dot ${isReal ? "dot-ivory" : ""}`} />
                        <span style={{ color: isReal ? "var(--text-secondary)" : "var(--text-muted)" }}>
                          {isReal ? "Real model" : "Fixture"}
                        </span>
                      </span>
                    </td>
                    <td className="mono" style={{ color: "var(--text-secondary)" }}>
                      {exp.model_a_id} → {exp.model_b_id}
                    </td>
                    <td className="num">
                      {exp.summary?.global_changed_pct != null
                        ? `${exp.summary.global_changed_pct.toFixed(2)}%`
                        : "—"}
                    </td>
                    <td
                      className="num"
                      style={{
                        color:
                          exp.summary?.behavioral_score_delta != null &&
                          exp.summary.behavioral_score_delta > 0.01
                            ? "var(--ok)"
                            : exp.summary?.behavioral_score_delta != null &&
                                exp.summary.behavioral_score_delta < -0.01
                              ? "var(--danger)"
                              : "var(--text-muted)",
                      }}
                    >
                      {exp.summary?.behavioral_score_delta != null
                        ? `${exp.summary.behavioral_score_delta >= 0 ? "+" : ""}${exp.summary.behavioral_score_delta.toFixed(3)}`
                        : "—"}
                    </td>
                    <td>
                      <span className="status-inline">
                        <span
                          className={`dot ${
                            exp.verdict === "PASSED"
                              ? "dot-ok"
                              : exp.verdict === "FAILED_REGRESSION" || exp.status === "failed"
                                ? "dot-danger"
                                : ""
                          }`}
                        />
                        <span>
                          {exp.verdict ? formatVerdict(exp.verdict) : formatStatus(exp.status)}
                        </span>
                      </span>
                    </td>
                    <td className="num" style={{ color: "var(--text-muted)" }}>
                      {exp.summary?.duration_seconds != null
                        ? `${exp.summary.duration_seconds.toFixed(2)}s`
                        : "—"}
                    </td>
                    <td className="num">
                      {exp.status === "completed" && (
                        <button
                          type="button"
                          className="btn"
                          style={{ padding: "4px 10px", fontSize: 11.5 }}
                          onClick={() => onSelectExperiment(exp.comparison_id)}
                        >
                          Open
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
