import React, { useState } from "react";
import { Check, Copy } from "lucide-react";
import { apiClient } from "../api/client";
import type { BaselineRecord, ComparisonBundle } from "../types/api";
import {
  formatCategory,
  formatMetricNumber,
  formatSeverity,
  formatVerdict,
} from "../utils/format";

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
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const reg = activeBundle?.regression ?? null;

  const handleSaveBaseline = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!activeBundle) return;
    if (!newBaselineId.trim()) {
      onError(new Error("Enter an identifier for the baseline (e.g. qwen25-05b-v1)."));
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

  const handleCopy = (id: string, cmd: string) => {
    void navigator.clipboard?.writeText(cmd);
    setCopiedId(id);
    setTimeout(() => setCopiedId((prev) => (prev === id ? null : prev)), 1800);
  };

  return (
    <div data-testid="view-regression">
      {reg ? (
        <>
          <div className="kpi-grid">
            <div className="kpi-card">
              <div className="kpi-label">Gate Status</div>
              <div
                className="kpi-value"
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 9,
                  fontSize: 19,
                }}
              >
                <span
                  className={`dot ${
                    reg.verdict === "PASSED"
                      ? "dot-ok"
                      : reg.verdict === "WARNING"
                        ? "dot-warn"
                        : "dot-danger"
                  }`}
                  style={{ width: 8, height: 8 }}
                />
                <span>{formatVerdict(reg.verdict)}</span>
              </div>
              <div className="kpi-sub">Peak severity · {formatSeverity(reg.highest_severity)}</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Triggered Findings</div>
              <div className="kpi-value">
                {(reg.finding_counts.CRITICAL ?? 0) + (reg.finding_counts.HIGH ?? 0)}
              </div>
              <div className="kpi-sub">
                {reg.finding_counts.MEDIUM ?? 0} medium · {reg.finding_counts.LOW ?? 0} low
              </div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Reproducers</div>
              <div className="kpi-value">{reg.failure_reproductions.length}</div>
              <div className="kpi-sub">Isolated single-case commands</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Threshold Policy</div>
              <div className="kpi-value mono" style={{ fontSize: 15, marginTop: 9 }}>
                {String(reg.thresholds.policy_name ?? "default")}
              </div>
              <div className="kpi-sub">{baselines.length} saved baseline(s)</div>
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <h3 className="panel-title">Findings ({reg.findings.length})</h3>
            </div>
            {reg.findings.length === 0 ? (
              <div style={{ color: "var(--text-secondary)", fontSize: 13 }}>
                All evaluated domains remained within threshold tolerances.
              </div>
            ) : (
              <div className="data-table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Severity</th>
                      <th>Domain</th>
                      <th>Summary</th>
                      <th className="num">Baseline → Candidate</th>
                      <th className="num">Max Drop</th>
                    </tr>
                  </thead>
                  <tbody>
                    {reg.findings.map((f) => {
                      const cleanTitle = f.title.replace(
                        /'([a-z_]+)'/g,
                        (_m, g1: string) => formatCategory(g1)
                      );
                      const isCrit = f.severity === "CRITICAL" || f.severity === "HIGH";
                      return (
                        <tr key={f.finding_id}>
                          <td style={{ width: 110 }}>
                            <span className="status-inline">
                              <span className={`dot ${isCrit ? "dot-danger" : "dot-warn"}`} />
                              <span style={{ color: isCrit ? "var(--text-main)" : "var(--text-secondary)" }}>
                                {formatSeverity(f.severity)}
                              </span>
                            </span>
                          </td>
                          <td style={{ color: "var(--text-muted)", textTransform: "capitalize" }}>
                            {f.domain}
                          </td>
                          <td>
                            <div style={{ color: "var(--text-main)", fontWeight: 500 }}>
                              {cleanTitle}
                            </div>
                          </td>
                          <td className="num">
                            <span style={{ color: "var(--text-muted)" }}>
                              {formatMetricNumber(f.baseline_value)}
                            </span>
                            <span style={{ color: "var(--text-faint)", margin: "0 6px" }}>→</span>
                            <span style={{ color: "var(--danger)" }}>
                              {formatMetricNumber(f.candidate_value)}
                            </span>
                          </td>
                          <td className="num" style={{ color: "var(--text-muted)" }}>
                            {formatMetricNumber(f.threshold_value)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {reg.failure_reproductions.length > 0 && (
            <div className="panel">
              <div className="panel-header">
                <h3 className="panel-title">CLI Reproduction</h3>
                <span className="panel-subtitle">Click any row to copy command</span>
              </div>
              <div style={{ borderTop: "1px solid var(--border-subtle)" }}>
                {reg.failure_reproductions.map((rep) => (
                  <div
                    key={rep.case_id}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "155px 96px 1fr auto",
                      alignItems: "center",
                      gap: 12,
                      padding: "10px 4px",
                      borderBottom: "1px solid var(--border-faint)",
                    }}
                  >
                    <div>
                      <span className="mono" style={{ color: "var(--text-main)", fontWeight: 500 }}>
                        {rep.case_id}
                      </span>
                      <span style={{ color: "var(--text-muted)", fontSize: 11.5, marginLeft: 6 }}>
                        {formatCategory(rep.category)}
                      </span>
                    </div>
                    <div className="mono" style={{ fontSize: 11.5, color: "var(--text-muted)" }}>
                      {rep.baseline_score.toFixed(2)} →{" "}
                      <span style={{ color: "var(--danger)" }}>
                        {rep.candidate_score.toFixed(2)}
                      </span>
                    </div>
                    <code
                      className="mono"
                      style={{
                        fontSize: 11.5,
                        color: "var(--text-secondary)",
                        overflowX: "auto",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {rep.cli_command}
                    </code>
                    <button
                      type="button"
                      className="btn"
                      style={{ padding: "4px 8px", fontSize: 11 }}
                      onClick={() => handleCopy(rep.case_id, rep.cli_command)}
                    >
                      {copiedId === rep.case_id ? <Check size={12} /> : <Copy size={12} />}
                      <span>{copiedId === rep.case_id ? "Copied" : "Copy"}</span>
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      ) : (
        <div className="panel">
          <h2 className="panel-title">Regression Gate</h2>
          <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
            Select a comparison with regression checking enabled.
          </p>
        </div>
      )}

      <div className="panel">
        <div className="panel-header">
          <h3 className="panel-title">Saved Baselines ({baselines.length})</h3>
        </div>
        {activeBundle?.behavioral && (
          <form
            onSubmit={handleSaveBaseline}
            style={{ display: "flex", flexWrap: "wrap", gap: 10, marginBottom: 14 }}
          >
            <input
              type="text"
              placeholder="Baseline identifier (e.g. qwen25-05b-v1)"
              value={newBaselineId}
              onChange={(e) => setNewBaselineId(e.target.value)}
              style={{ minWidth: 240 }}
            />
            <select value={side} onChange={(e) => setSide(e.target.value as "a" | "b")}>
              <option value="a">Model A · {activeBundle.model_a.model_id}</option>
              <option value="b">Model B · {activeBundle.model_b.model_id}</option>
            </select>
            <button type="submit" className="btn btn-primary">
              Save Baseline
            </button>
          </form>
        )}
        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Baseline</th>
                <th>Model</th>
                <th>Suite</th>
                <th className="num">Pass Rate</th>
                <th className="num">Mean Score</th>
                <th>Created</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {baselines.length === 0 ? (
                <tr>
                  <td colSpan={7} style={{ color: "var(--text-muted)" }}>
                    No baselines saved yet.
                  </td>
                </tr>
              ) : (
                baselines.map((b) => (
                  <tr key={b.baseline_id}>
                    <td className="mono" style={{ color: "var(--text-main)" }}>
                      {b.baseline_id}
                    </td>
                    <td className="mono">{b.model_id}</td>
                    <td className="mono" style={{ color: "var(--text-muted)" }}>
                      {b.suite_id}@{b.suite_version}
                    </td>
                    <td className="num">{(b.overall_pass_rate * 100).toFixed(1)}%</td>
                    <td className="num">{b.overall_mean_score.toFixed(3)}</td>
                    <td className="mono" style={{ color: "var(--text-muted)" }}>
                      {b.created_at_utc.slice(0, 10)}
                    </td>
                    <td className="num">
                      <button
                        type="button"
                        className="btn btn-danger"
                        style={{ padding: "4px 9px", fontSize: 11 }}
                        onClick={() => handleDelete(b.baseline_id)}
                      >
                        Remove
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
