import React, { useState } from "react";
import type { ComparisonBundle } from "../types/api";

interface Props {
  activeBundle: ComparisonBundle | null;
}

export const BehavioralDiffView: React.FC<Props> = ({ activeBundle }) => {
  const beh = activeBundle?.behavioral;
  const [catFilter, setCatFilter] = useState("");

  if (!activeBundle || !beh) {
    return (
      <div className="panel" data-testid="view-behavioral">
        <h2 className="panel-title">Behavioral Intelligence Diff</h2>
        <p style={{ color: "var(--text-muted)" }}>
          Run a comparison with the <code>behavioral</code> module enabled to execute the 8 deterministic evaluation suites.
        </p>
      </div>
    );
  }

  const st = beh.overall_statistics;
  const visibleCases = catFilter
    ? beh.cases.filter((c) => c.case.category === catFilter)
    : beh.cases;

  return (
    <div data-testid="view-behavioral">
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">Pass Rate (A → B)</div>
          <div className="kpi-value">
            {(st.pass_rate_a * 100).toFixed(1)}% → {(st.pass_rate_b * 100).toFixed(1)}%
          </div>
          <div className="kpi-sub">
            Delta: {st.pass_rate_delta >= 0 ? "+" : ""}
            {(st.pass_rate_delta * 100).toFixed(1)}% ({st.n_cases} paired cases)
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Mean Score Delta (B − A)</div>
          <div className="kpi-value">
            {st.mean_score_delta >= 0 ? "+" : ""}
            {st.mean_score_delta.toFixed(4)}
          </div>
          <div className="kpi-sub">
            95% Bootstrap CI: [{st.bootstrap_ci95_low.toFixed(3)}, {st.bootstrap_ci95_high.toFixed(3)}]
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Paired Wins / Ties / Losses</div>
          <div className="kpi-value">
            {st.b_wins}W / {st.ties}T / {st.a_wins}L
          </div>
          <div className="kpi-sub">McNemar two-sided p = {st.mcnemar_p_value.toFixed(4)}</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Judge Methodology</div>
          <div className="kpi-value" style={{ fontSize: 16 }}>
            100% DETERMINISTIC
          </div>
          <div className="kpi-sub">Zero external LLM judge calls</div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-header">
          <h3 className="panel-title">8-Suite Domain Breakdown ({beh.suite_id}@{beh.suite_version})</h3>
          <span className="badge badge-ok">OBSERVED</span>
        </div>
        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Domain Suite</th>
                <th>Cases</th>
                <th>Pass Rate A</th>
                <th>Pass Rate B</th>
                <th>Mean Score A</th>
                <th>Mean Score B</th>
                <th>Delta</th>
                <th>Improved / Regressed</th>
              </tr>
            </thead>
            <tbody>
              {beh.category_summaries.map((cs) => (
                <tr
                  key={cs.category}
                  style={{ cursor: "pointer" }}
                  onClick={() =>
                    setCatFilter((prev) => (prev === cs.category ? "" : cs.category))
                  }
                >
                  <td className="mono">{cs.category}</td>
                  <td className="num">{cs.case_count}</td>
                  <td className="num">{(cs.pass_rate_a * 100).toFixed(0)}%</td>
                  <td className="num">{(cs.pass_rate_b * 100).toFixed(0)}%</td>
                  <td className="num">{cs.mean_score_a.toFixed(3)}</td>
                  <td className="num">{cs.mean_score_b.toFixed(3)}</td>
                  <td className="num">
                    <span
                      className={`badge ${
                        cs.mean_score_delta > 0.01
                          ? "badge-ok"
                          : cs.mean_score_delta < -0.01
                            ? "badge-danger"
                            : "badge-info"
                      }`}
                    >
                      {cs.mean_score_delta >= 0 ? "+" : ""}
                      {cs.mean_score_delta.toFixed(3)}
                    </span>
                  </td>
                  <td className="num">
                    +{cs.improved_cases} / -{cs.regressed_cases}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="panel">
        <div className="panel-header">
          <h3 className="panel-title">
            Side-by-Side Paired Output Diff ({visibleCases.length} cases)
          </h3>
          <select value={catFilter} onChange={(e) => setCatFilter(e.target.value)}>
            <option value="">All 8 Categories</option>
            {beh.category_summaries.map((c) => (
              <option key={c.category} value={c.category}>
                {c.category}
              </option>
            ))}
          </select>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {visibleCases.map((pc) => (
            <div key={pc.case.case_id} className="panel" style={{ marginBottom: 0 }}>
              <div className="panel-header" style={{ marginBottom: 8 }}>
                <div>
                  <span className="mono" style={{ fontWeight: 700, marginRight: 8 }}>
                    {pc.case.case_id}
                  </span>
                  <span className="badge badge-info" style={{ marginRight: 8 }}>
                    {pc.case.category}
                  </span>
                  <span style={{ color: "var(--text-muted)", fontSize: 12.5 }}>
                    {pc.case.description}
                  </span>
                </div>
                <span
                  className={`badge ${
                    pc.status_change === "improved" || pc.status_change === "both_passed"
                      ? "badge-ok"
                      : pc.status_change === "regressed"
                        ? "badge-danger"
                        : "badge-warn"
                  }`}
                >
                  {pc.status_change} ({pc.eval_a.score.toFixed(2)} → {pc.eval_b.score.toFixed(2)})
                </span>
              </div>
              <div className="mono" style={{ fontSize: 12.5, color: "var(--accent)" }}>
                Prompt: {pc.case.prompt}
              </div>
              <div className="diff-pair">
                <div className="diff-box">
                  <strong>
                    Model A ({pc.eval_a.passed ? "PASS" : "FAIL"} • {pc.latency_ms_a.toFixed(0)} ms)
                  </strong>
                  <div style={{ marginTop: 6 }}>{pc.output_a || "(empty output)"}</div>
                  <div style={{ marginTop: 6, fontSize: 11.5, color: "var(--text-muted)" }}>
                    {pc.eval_a.rationale}
                  </div>
                </div>
                <div className="diff-box">
                  <strong>
                    Model B ({pc.eval_b.passed ? "PASS" : "FAIL"} • {pc.latency_ms_b.toFixed(0)} ms)
                  </strong>
                  <div style={{ marginTop: 6 }}>{pc.output_b || "(empty output)"}</div>
                  <div style={{ marginTop: 6, fontSize: 11.5, color: "var(--text-muted)" }}>
                    {pc.eval_b.rationale}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
