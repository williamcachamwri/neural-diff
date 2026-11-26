import React, { useEffect, useState } from "react";
import { apiClient } from "../api/client";
import type { ComparisonBundle, TensorRecord } from "../types/api";
import { formatMetricNumber, formatRole, formatStatus } from "../utils/format";

interface Props {
  activeBundle: ComparisonBundle | null;
  onError: (err: Error) => void;
}

export const TensorExplorerView: React.FC<Props> = ({ activeBundle, onError }) => {
  const [search, setSearch] = useState("");
  const [role, setRole] = useState("");
  const [status, setStatus] = useState("");
  const [sortBy, setSortBy] = useState("relative_l2");
  const [records, setRecords] = useState<TensorRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [selected, setSelected] = useState<TensorRecord | null>(null);

  useEffect(() => {
    if (!activeBundle?.tensor_diff) {
      setRecords([]);
      setTotal(0);
      setSelected(null);
      return;
    }
    let cancelled = false;
    apiClient
      .queryTensors(activeBundle.comparison_id, {
        search,
        role,
        status,
        sort_by: sortBy,
        limit: 250,
      })
      .then((res) => {
        if (!cancelled) {
          setRecords(res.records);
          setTotal(res.total);
          setSelected(
            (prev) => res.records.find((r) => r.name === prev?.name) ?? res.records[0] ?? null
          );
        }
      })
      .catch((err) => {
        if (!cancelled) onError(err instanceof Error ? err : new Error(String(err)));
      });
    return () => {
      cancelled = true;
    };
  }, [activeBundle, search, role, status, sortBy, onError]);

  if (!activeBundle || !activeBundle.tensor_diff) {
    return (
      <div className="panel" data-testid="view-tensors">
        <h2 className="panel-title">Tensor Explorer</h2>
        <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
          Select a comparison with tensor diffing enabled to inspect parameter metrics.
        </p>
      </div>
    );
  }

  const sum = activeBundle.tensor_diff.summary;

  return (
    <div data-testid="view-tensors">
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">Compared Tensors</div>
          <div className="kpi-value">{sum.compared_tensors}</div>
          <div className="kpi-sub">
            {sum.modified_tensors} modified · {sum.identical_tensors} identical
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Modified Parameters</div>
          <div className="kpi-value">{sum.global_changed_pct.toFixed(2)}%</div>
          <div className="kpi-sub">
            {sum.total_changed_parameters.toLocaleString()} /{" "}
            {sum.total_compared_parameters.toLocaleString()}
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Mean Cosine Similarity</div>
          <div className="kpi-value">
            {sum.global_mean_cosine != null ? sum.global_mean_cosine.toFixed(6) : "—"}
          </div>
          <div className="kpi-sub">
            Mean rel L2 · {sum.global_mean_relative_l2?.toFixed(5) ?? "—"}
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Peak |Δ|</div>
          <div className="kpi-value">
            {sum.global_max_abs_diff != null ? sum.global_max_abs_diff.toFixed(4) : "—"}
          </div>
          <div className="kpi-sub">
            {sum.has_any_nan_or_inf ? "NaN / Inf present" : "Zero NaN or Inf"}
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-header">
          <h2 className="panel-title">Parameters ({total})</h2>
          <span className="panel-subtitle">Float64 chunked accumulation</span>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(175px, 1fr))",
            gap: 10,
            marginBottom: 14,
          }}
        >
          <input
            type="text"
            placeholder="Filter parameter name..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <select value={role} onChange={(e) => setRole(e.target.value)}>
            <option value="">All roles</option>
            <option value="embedding">Embedding</option>
            <option value="attn_q">Query (Q)</option>
            <option value="attn_k">Key (K)</option>
            <option value="attn_v">Value (V)</option>
            <option value="attn_o">Attn Out (O)</option>
            <option value="mlp_gate">MLP Gate</option>
            <option value="mlp_up">MLP Up</option>
            <option value="mlp_down">MLP Down</option>
            <option value="norm">Norm</option>
            <option value="lm_head">LM Head</option>
          </select>
          <select value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="">All states</option>
            <option value="modified">Modified</option>
            <option value="identical">Identical</option>
            <option value="within_tolerance">Within tolerance</option>
            <option value="shape_mismatch">Shape mismatch</option>
          </select>
          <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
            <option value="relative_l2">Sort · Relative L2</option>
            <option value="max_abs_diff">Sort · Max |Δ|</option>
            <option value="cosine_similarity">Sort · Cosine</option>
            <option value="changed_pct">Sort · Changed %</option>
            <option value="name">Sort · Name</option>
          </select>
        </div>

        <div className="data-table-wrap" style={{ maxHeight: 380, overflowY: "auto" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Parameter</th>
                <th>Role</th>
                <th>Shape</th>
                <th>State</th>
                <th className="num">Cosine</th>
                <th className="num">Rel L2</th>
                <th className="num">RMSE</th>
                <th className="num">Max |Δ|</th>
              </tr>
            </thead>
            <tbody>
              {records.map((rec) => {
                const isSel = selected?.name === rec.name;
                const shapeArr = rec.shape_a ?? rec.shape_b;
                return (
                  <tr
                    key={rec.name}
                    className={isSel ? "is-active-row" : ""}
                    style={{ cursor: "pointer" }}
                    onClick={() => setSelected(rec)}
                  >
                    <td className="mono" style={{ color: "var(--text-main)" }}>
                      {rec.name}
                    </td>
                    <td style={{ color: "var(--text-muted)" }}>
                      {formatRole(rec.canonical_role)}
                    </td>
                    <td className="mono" style={{ color: "var(--text-muted)" }}>
                      {shapeArr ? shapeArr.join(" × ") : "—"}
                    </td>
                    <td style={{ color: "var(--text-secondary)" }}>
                      {formatStatus(rec.status)}
                    </td>
                    <td className="num">
                      {rec.metrics?.cosine_similarity != null
                        ? rec.metrics.cosine_similarity.toFixed(6)
                        : "—"}
                    </td>
                    <td className="num" style={{ color: "var(--text-main)" }}>
                      {rec.metrics?.relative_l2 != null ? rec.metrics.relative_l2.toFixed(5) : "—"}
                    </td>
                    <td className="num" style={{ color: "var(--text-muted)" }}>
                      {rec.metrics?.rmse != null ? rec.metrics.rmse.toExponential(2) : "—"}
                    </td>
                    <td className="num" style={{ color: "var(--text-muted)" }}>
                      {rec.metrics?.max_abs_diff != null
                        ? rec.metrics.max_abs_diff.toFixed(4)
                        : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {selected && (
        <div className="panel" data-testid="tensor-detail-inspector">
          <div className="panel-header">
            <h3 className="panel-title mono" style={{ fontSize: 13 }}>
              {selected.name}
            </h3>
            <span className="mono" style={{ color: "var(--text-muted)", fontSize: 11.5 }}>
              {selected.dtype_a ?? "—"} → {selected.dtype_b ?? "—"} ·{" "}
              {selected.metrics ? `${selected.metrics.changed_pct.toFixed(2)}% changed` : "—"}
            </span>
          </div>

          <div className="grid-2">
            <table className="data-table">
              <tbody>
                <tr>
                  <td style={{ color: "var(--text-muted)" }}>L2 Norm (A → B)</td>
                  <td className="num">
                    {formatMetricNumber(selected.metrics?.l2_norm_a, 4)} →{" "}
                    {formatMetricNumber(selected.metrics?.l2_norm_b, 4)}
                  </td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text-muted)" }}>Mean |Δ|</td>
                  <td className="num">{formatMetricNumber(selected.metrics?.mean_abs_diff, 6)}</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text-muted)" }}>Zero ↔ Nonzero Transitions</td>
                  <td className="num">
                    +{selected.metrics?.zero_to_nonzero ?? 0} / -
                    {selected.metrics?.nonzero_to_zero ?? 0}
                  </td>
                </tr>
              </tbody>
            </table>

            <table className="data-table">
              <tbody>
                <tr>
                  <td style={{ color: "var(--text-muted)" }}>Mean (A → B)</td>
                  <td className="num">
                    {formatMetricNumber(selected.distribution_a?.mean, 5)} →{" "}
                    {formatMetricNumber(selected.distribution_b?.mean, 5)}
                  </td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text-muted)" }}>Std Dev (A → B)</td>
                  <td className="num">
                    {formatMetricNumber(selected.distribution_a?.std, 5)} →{" "}
                    {formatMetricNumber(selected.distribution_b?.std, 5)}
                  </td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text-muted)" }}>Wasserstein-1 Shift</td>
                  <td className="num">
                    {formatMetricNumber(selected.distribution_shift?.wasserstein_1d_approx, 5)}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
