import React, { useState } from "react";
import { apiClient } from "../api/client";
import type { BaselineRecord, ComparisonBundle } from "../types/api";

interface Props {
  activeBundle: ComparisonBundle | null;
  baselines: BaselineRecord[];
  onRefreshBaselines: () => void;
  onError: (err: Error) => void;
}

export const RegressionDashboardView: React.FC<Props> = ({
  activeBundle,
  baselines,
  onRefreshBaselines,
  onError,
}) => {
  const [newBaselineId, setNewBaselineId] = useState("");
  const [side, setSide] = useState<"a" | "b">("a");

  const reg = activeBundle?.regression ?? null;

  const handleSaveBaseline = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!activeBundle) return;
    if (!newBaselineId.trim()) {
      onError(new Error("Provide an alphanumeric baseline ID (e.g. qwen25-05b-base-v1)."));
      return;
    }
    try {
      await apiClient.createBaseline(activeBundle.comparison_id, newBaselineId.trim(), side);
      setNewBaselineId("");
      onRefreshBaselines();
    } catch (err) {
      onError(err instanceof Error ? err : new Error(String(err)));
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await apiClient.deleteBaseline(id);
      onRefreshBaselines();
    } catch (err) {
      onError(err instanceof Error ? err : new Error(String(err)));
    }
  };

  return (
    <div data-testid="view-regression">
      {reg ? (
        <>
          <div className="kpi-grid">
            <div className="kpi-card">
              <div className="kpi-label">Gate Verdict</div>
              <div
                className="kpi-value"
                style={{
                  color:
                    reg.verdict === "PASSED"
                      ? "var(--ok)"
                      : reg.verdict === "WARNING"
                        ? "var(--warn)"
                        : "var(--danger)",
                }}
              >
                {reg.verdict}
              </div>
              <div className="kpi-sub">Highest Severity: {reg.highest_severity}</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Critical / High Findings</div>
              <div className="kpi-value">
                {(reg.finding_counts.CRITICAL ?? 0) + (reg.finding_counts.HIGH ?? 0)}
              </div>
              <div className="kpi-sub">
                Medium: {reg.finding_counts.MEDIUM ?? 0} • Low: {reg.finding_counts.LOW ?? 0}
              </div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Failure Reproductions</div>
              <div className="kpi-value">{reg.failure_reproductions.length}</div>
              <div className="kpi-sub">Deterministic CLI reproduction commands</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Active Policy</div>
              <div className="kpi-value mono" style={{ fontSize: 15 }}>
                {String(reg.thresholds.policy_name ?? "default")}
              </div>
              <div className="kpi-sub">Saved baselines: {baselines.length}</div>
            </div>
          </div>

          <div className="panel">
            <h3 className="panel-title" style={{ marginBottom: 10 }}>
              Regression Findings ({reg.findings.length})
            </h3>
            {reg.findings.length === 0 ? (
              <div style={{ color: "var(--ok)" }}>
                Zero regressions triggered under active policy thresholds.
              </div>
            ) : (
              <div className="data-table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Severity</th>
                      <th>Epistemic</th>
                      <th>Domain</th>
                      <th>Finding</th>
                      <th>Baseline → Candidate (Threshold)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {reg.findings.map((f) => (
                      <tr key={f.finding_id}>
                        <td className="mono">{f.finding_id}</td>
                        <td>
                          <span
                            className={`badge ${
                              f.severity === "CRITICAL" || f.severity === "HIGH"
                                ? "badge-danger"
                                : "badge-warn"
                            }`}
                          >
                            {f.severity}
                          </span>
                        </td>
                        <td>
                          <span className="badge badge-info">{f.epistemic_status}</span>
                        </td>
                        <td>{f.domain}</td>
                        <td>
                          <strong>{f.title}</strong>
                          <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                            {f.description}
                          </div>
                        </td>
                        <td className="mono">
                          {JSON.stringify(f.baseline_value)} → {JSON.stringify(f.candidate_value)}{" "}
                          (lim: {JSON.stringify(f.threshold_value)})
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {reg.failure_reproductions.length > 0 && (
            <div className="panel">
              <h3 className="panel-title" style={{ marginBottom: 10 }}>
                Regressed Case Reproduction Commands
              </h3>
              {reg.failure_reproductions.map((rep) => (
                <div key={rep.case_id} className="diff-box" style={{ marginBottom: 10 }}>
                  <div>
                    <strong>{rep.case_id}</strong> ({rep.category}) — Score{" "}
                    {rep.baseline_score.toFixed(2)} → {rep.candidate_score.toFixed(2)}
                  </div>
                  <div className="mono" style={{ marginTop: 6, color: "var(--accent-teal)" }}>
                    {rep.cli_command}
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      ) : (
        <div className="panel">
          <h2 className="panel-title">Regression Detection Dashboard</h2>
          <p style={{ color: "var(--text-muted)" }}>
            Select or execute a comparison with <code>regression</code> enabled to view gate verdicts.
          </p>
        </div>
      )}

      <div className="panel">
        <h3 className="panel-title" style={{ marginBottom: 10 }}>
          Versioned Regression Baselines ({baselines.length})
        </h3>
        {activeBundle?.behavioral && (
          <form
            onSubmit={handleSaveBaseline}
            style={{ display: "flex", flexWrap: "wrap", gap: 10, marginBottom: 14 }}
          >
            <input
              type="text"
              placeholder="Baseline ID (e.g. qwen25-05b-instruct-v1)"
              value={newBaselineId}
              onChange={(e) => setNewBaselineId(e.target.value)}
            />
            <select value={side} onChange={(e) => setSide(e.target.value as "a" | "b")}>
              <option value="a">Snapshot Model A ({activeBundle.model_a.model_id})</option>
              <option value="b">Snapshot Model B ({activeBundle.model_b.model_id})</option>
            </select>
            <button type="submit" className="btn btn-primary">
              Save Versioned Baseline
            </button>
          </form>
        )}
        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Baseline ID</th>
                <th>Model ID</th>
                <th>Suite</th>
                <th>Pass Rate</th>
                <th>Mean Score</th>
                <th>Created</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {baselines.length === 0 ? (
                <tr>
                  <td colSpan={7} style={{ color: "var(--text-muted)" }}>
                    No persisted baselines yet. Run a behavioral comparison to snapshot a baseline.
                  </td>
                </tr>
              ) : (
                baselines.map((b) => (
                  <tr key={b.baseline_id}>
                    <td className="mono">{b.baseline_id}</td>
                    <td className="mono">{b.model_id}</td>
                    <td className="mono">
                      {b.suite_id}@{b.suite_version}
                    </td>
                    <td className="num">{(b.overall_pass_rate * 100).toFixed(1)}%</td>
                    <td className="num">{b.overall_mean_score.toFixed(4)}</td>
                    <td className="mono">{b.created_at_utc.slice(0, 19)}Z</td>
                    <td>
                      <button
                        className="btn btn-danger"
                        onClick={() => handleDelete(b.baseline_id)}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
