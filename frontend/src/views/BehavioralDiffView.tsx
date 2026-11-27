import React, { useMemo, useState } from "react";
import { DivergingDeltaBars, PolarRadarChart } from "../components/charts/StudioVisuals";
import type { ComparisonBundle } from "../types/api";
import { formatCategory, formatStatus } from "../utils/format";

interface Props {
  activeBundle: ComparisonBundle | null;
}

export const BehavioralDiffView: React.FC<Props> = ({ activeBundle }) => {
  const beh = activeBundle?.behavioral;
  const [catFilter, setCatFilter] = useState("");
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);

  const visibleCases = useMemo(() => {
    if (!beh) return [];
    return catFilter
      ? beh.cases.filter((c) => c.case.category === catFilter)
      : beh.cases;
  }, [beh, catFilter]);

  if (!activeBundle || !beh) {
    return (
      <div className="panel" data-testid="view-behavioral">
        <h2 className="panel-title">Behavioral Evaluation</h2>
        <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
          Run a comparison with behavioral evaluation enabled to inspect paired prompt responses.
        </p>
      </div>
    );
  }

  const st = beh.overall_statistics;
  const meanLatA =
    beh.cases.length > 0
      ? beh.cases.reduce((acc, c) => acc + c.latency_ms_a, 0) / beh.cases.length
      : 0;
  const meanLatB =
    beh.cases.length > 0
      ? beh.cases.reduce((acc, c) => acc + c.latency_ms_b, 0) / beh.cases.length
      : 0;

  const activeCase =
    visibleCases.find((c) => c.case.case_id === selectedCaseId) ?? visibleCases[0] ?? null;

  return (
    <div data-testid="view-behavioral">
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">Pass Rate (A → B)</div>
          <div className="kpi-value">
            {(st.pass_rate_a * 100).toFixed(1)}% → {(st.pass_rate_b * 100).toFixed(1)}%
          </div>
          <div className="kpi-sub">
            {st.pass_rate_delta >= 0 ? "+" : ""}
            {(st.pass_rate_delta * 100).toFixed(1)} pts · {st.n_cases} paired prompts
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Mean Score Δ</div>
          <div className="kpi-value">
            {st.mean_score_delta >= 0 ? "+" : ""}
            {st.mean_score_delta.toFixed(3)}
          </div>
          <div className="kpi-sub">
            95% CI [{st.bootstrap_ci95_low.toFixed(3)}, {st.bootstrap_ci95_high.toFixed(3)}]
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Paired Outcomes</div>
          <div className="kpi-value">
            {st.b_wins}W · {st.ties}T · {st.a_wins}L
          </div>
          <div className="kpi-sub">McNemar p = {st.mcnemar_p_value.toFixed(3)}</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Mean Generation Latency</div>
          <div className="kpi-value">
            {Math.round(meanLatA)} → {Math.round(meanLatB)} ms
          </div>
          <div className="kpi-sub">Greedy decoding · seed 42</div>
        </div>
      </div>

      {/* Capability Polygon & Suite Delta Visual Panel */}
      {beh.category_summaries.length >= 3 && (
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">Capability Geometry &amp; Suite Deltas</h3>
            <span className="panel-subtitle mono">Model A vs Model B across {beh.category_summaries.length} suites</span>
          </div>
          <div className="grid-2" style={{ alignItems: "center" }}>
            <PolarRadarChart
              axes={beh.category_summaries.map((c) => formatCategory(c.category))}
              series={[
                {
                  label: "Model A",
                  values: beh.category_summaries.map((c) => c.mean_score_a),
                  color: "#71717a",
                },
                {
                  label: "Model B",
                  values: beh.category_summaries.map((c) => c.mean_score_b),
                  color: "#d6c5a9",
                },
              ]}
            />
            <DivergingDeltaBars
              items={beh.category_summaries.map((c) => ({
                label: formatCategory(c.category),
                delta: c.mean_score_delta,
              }))}
            />
          </div>
        </div>
      )}

      {/* Domain Suite Breakdown with Visual Paired Bars */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <h3 className="panel-title">Evaluation Suites</h3>
            <div className="panel-subtitle" style={{ marginTop: 2 }}>
              Click any suite row to filter the prompt inspector below
            </div>
          </div>
          <span className="mono" style={{ fontSize: 11, color: "var(--text-muted)" }}>
            {beh.suite_id} · v{beh.suite_version}
          </span>
        </div>

        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Suite</th>
                <th className="num">Cases</th>
                <th style={{ width: 170 }}>Score Profile (A vs B)</th>
                <th className="num">Mean A</th>
                <th className="num">Mean B</th>
                <th className="num">Delta</th>
                <th className="num">Won / Lost</th>
              </tr>
            </thead>
            <tbody>
              {beh.category_summaries.map((cs) => {
                const isSelected = catFilter === cs.category;
                return (
                  <tr
                    key={cs.category}
                    className={isSelected ? "is-active-row" : ""}
                    style={{ cursor: "pointer" }}
                    onClick={() =>
                      setCatFilter((prev) => (prev === cs.category ? "" : cs.category))
                    }
                  >
                    <td style={{ fontWeight: 500, color: "var(--text-main)" }}>
                      {formatCategory(cs.category)}
                    </td>
                    <td className="num" style={{ color: "var(--text-muted)" }}>
                      {cs.case_count}
                    </td>
                    <td>
                      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                        <div
                          style={{
                            height: 3.5,
                            background: "rgba(255,255,255,0.05)",
                            borderRadius: 99,
                            overflow: "hidden",
                          }}
                        >
                          <div
                            style={{
                              width: `${Math.round(cs.mean_score_a * 100)}%`,
                              height: "100%",
                              background: "#71717a",
                            }}
                          />
                        </div>
                        <div
                          style={{
                            height: 3.5,
                            background: "rgba(255,255,255,0.05)",
                            borderRadius: 99,
                            overflow: "hidden",
                          }}
                        >
                          <div
                            style={{
                              width: `${Math.round(cs.mean_score_b * 100)}%`,
                              height: "100%",
                              background:
                                cs.mean_score_delta < -0.01 ? "var(--danger)" : "#e4e4e7",
                            }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="num" style={{ color: "var(--text-muted)" }}>
                      {cs.mean_score_a.toFixed(3)}
                    </td>
                    <td className="num">{cs.mean_score_b.toFixed(3)}</td>
                    <td
                      className="num"
                      style={{
                        color:
                          cs.mean_score_delta > 0.01
                            ? "var(--ok)"
                            : cs.mean_score_delta < -0.01
                              ? "var(--danger)"
                              : "var(--text-muted)",
                      }}
                    >
                      {cs.mean_score_delta >= 0 ? "+" : ""}
                      {cs.mean_score_delta.toFixed(3)}
                    </td>
                    <td className="num" style={{ color: "var(--text-muted)" }}>
                      +{cs.improved_cases} / -{cs.regressed_cases}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Master-Detail Paired Response Workbench (Eliminates 24 Triple-Nested Boxes) */}
      <div className="panel" style={{ padding: 0, overflow: "hidden" }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            padding: "14px 20px",
            borderBottom: "1px solid var(--border-subtle)",
          }}
        >
          <h3 className="panel-title">Paired Prompt Responses ({visibleCases.length})</h3>
          <select
            value={catFilter}
            onChange={(e) => setCatFilter(e.target.value)}
            style={{ padding: "5px 10px", fontSize: 12 }}
          >
            <option value="">All suites ({beh.cases.length})</option>
            {beh.category_summaries.map((c) => (
              <option key={c.category} value={c.category}>
                {formatCategory(c.category)}
              </option>
            ))}
          </select>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "250px 1fr",
            minHeight: 340,
          }}
        >
          {/* Left Master List */}
          <div
            style={{
              borderRight: "1px solid var(--border-subtle)",
              maxHeight: 420,
              overflowY: "auto",
            }}
          >
            {visibleCases.map((pc) => {
              const selected = activeCase?.case.case_id === pc.case.case_id;
              const delta = pc.eval_b.score - pc.eval_a.score;
              return (
                <button
                  key={pc.case.case_id}
                  type="button"
                  onClick={() => setSelectedCaseId(pc.case.case_id)}
                  style={{
                    width: "100%",
                    textAlign: "left",
                    border: "none",
                    borderBottom: "1px solid var(--border-faint)",
                    background: selected ? "rgba(255,255,255,0.055)" : "transparent",
                    padding: "10px 14px",
                    cursor: "pointer",
                    color: "var(--text-main)",
                    display: "flex",
                    flexDirection: "column",
                    gap: 3,
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                    }}
                  >
                    <span className="mono" style={{ fontSize: 11.5, fontWeight: 500 }}>
                      {pc.case.case_id}
                    </span>
                    <span
                      className="mono"
                      style={{
                        fontSize: 11,
                        color:
                          delta > 0.01
                            ? "var(--ok)"
                            : delta < -0.01
                              ? "var(--danger)"
                              : "var(--text-muted)",
                      }}
                    >
                      {delta >= 0 ? "+" : ""}
                      {delta.toFixed(2)}
                    </span>
                  </div>
                  <div
                    style={{
                      fontSize: 11.5,
                      color: "var(--text-muted)",
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {formatCategory(pc.case.category)} · {formatStatus(pc.status_change)}
                  </div>
                </button>
              );
            })}
          </div>

          {/* Right Detail Pane */}
          {activeCase && (
            <div style={{ padding: "18px 22px", display: "flex", flexDirection: "column", gap: 16 }}>
              <div>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "baseline",
                    marginBottom: 6,
                  }}
                >
                  <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text-main)" }}>
                    {activeCase.case.description}
                  </span>
                  <span className="mono" style={{ fontSize: 11.5, color: "var(--text-muted)" }}>
                    Score {activeCase.eval_a.score.toFixed(2)} → {activeCase.eval_b.score.toFixed(2)}
                  </span>
                </div>
                <div
                  style={{
                    fontSize: 12.5,
                    color: "var(--text-secondary)",
                    padding: "10px 12px",
                    background: "var(--bg-surface-subtle)",
                    borderRadius: 6,
                    border: "1px solid var(--border-faint)",
                  }}
                >
                  {activeCase.case.prompt}
                </div>
              </div>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: 16,
                  flex: 1,
                }}
              >
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      fontSize: 11.5,
                      color: "var(--text-muted)",
                    }}
                  >
                    <span className="status-inline">
                      <span className={`dot ${activeCase.eval_a.passed ? "dot-ok" : ""}`} />
                      <span>Model A</span>
                    </span>
                    <span className="mono">
                      {activeCase.eval_a.score.toFixed(2)} · {Math.round(activeCase.latency_ms_a)} ms
                    </span>
                  </div>
                  <div
                    className="mono"
                    style={{
                      flex: 1,
                      fontSize: 12,
                      lineHeight: 1.6,
                      color: "var(--text-secondary)",
                      padding: "12px 14px",
                      background: "var(--bg-surface-subtle)",
                      border: "1px solid var(--border-faint)",
                      borderRadius: 6,
                      whiteSpace: "pre-wrap",
                      maxHeight: 240,
                      overflowY: "auto",
                    }}
                  >
                    {activeCase.output_a || "—"}
                  </div>
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      fontSize: 11.5,
                      color: "var(--text-muted)",
                    }}
                  >
                    <span className="status-inline">
                      <span
                        className={`dot ${
                          activeCase.eval_b.passed ? "dot-ok" : "dot-danger"
                        }`}
                      />
                      <span>Model B</span>
                    </span>
                    <span className="mono">
                      {activeCase.eval_b.score.toFixed(2)} · {Math.round(activeCase.latency_ms_b)} ms
                    </span>
                  </div>
                  <div
                    className="mono"
                    style={{
                      flex: 1,
                      fontSize: 12,
                      lineHeight: 1.6,
                      color: "var(--text-main)",
                      padding: "12px 14px",
                      background: "var(--bg-surface-subtle)",
                      border: "1px solid var(--border-subtle)",
                      borderRadius: 6,
                      whiteSpace: "pre-wrap",
                      maxHeight: 240,
                      overflowY: "auto",
                    }}
                  >
                    {activeCase.output_b || "—"}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
