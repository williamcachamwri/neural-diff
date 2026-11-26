import React, { useEffect, useState } from "react";
import { apiClient } from "../api/client";
import type { ComparisonBundle, TensorRecord } from "../types/api";

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
          setSelected((prev) => res.records.find((r) => r.name === prev?.name) ?? res.records[0] ?? null);
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
        <h2 className="panel-title">Tensor Diff Explorer</h2>
        <p style={{ color: "var(--text-muted)" }}>
          Load or run a comparison with the <code>tensor</code> module enabled to inspect parameter-level math metrics.
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
            Modified: {sum.modified_tensors} • Identical: {sum.identical_tensors}
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Global Changed Elements</div>
          <div className="kpi-value">{sum.global_changed_pct.toFixed(3)}%</div>
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
          <div className="kpi-sub">Mean Rel L2: {sum.global_mean_relative_l2?.toFixed(6) ?? "—"}</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Max Absolute Diff</div>
          <div className="kpi-value">
            {sum.global_max_abs_diff != null ? sum.global_max_abs_diff.toExponential(3) : "—"}
          </div>
          <div className="kpi-sub">
            NaN/Inf Detected: {sum.has_any_nan_or_inf ? "YES" : "NO"}
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-header">
          <h2 className="panel-title">Tensor Explorer ({total} matching tensors)</h2>
          <span className="badge badge-ok">OBSERVED • Float64 Accumulated</span>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            gap: 10,
            marginBottom: 14,
          }}
        >
          <input
            type="text"
            placeholder="Filter by tensor name..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <select value={role} onChange={(e) => setRole(e.target.value)}>
            <option value="">All Canonical Roles</option>
            <option value="embedding">embedding</option>
            <option value="attn_q">attn_q</option>
            <option value="attn_k">attn_k</option>
            <option value="attn_v">attn_v</option>
            <option value="attn_o">attn_o</option>
            <option value="mlp_gate">mlp_gate</option>
            <option value="mlp_up">mlp_up</option>
            <option value="mlp_down">mlp_down</option>
            <option value="norm">norm</option>
            <option value="lm_head">lm_head</option>
          </select>
          <select value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="">All Statuses</option>
            <option value="modified">modified</option>
            <option value="identical">identical</option>
            <option value="within_tolerance">within_tolerance</option>
            <option value="shape_mismatch">shape_mismatch</option>
            <option value="missing_in_b">missing_in_b</option>
            <option value="added_in_b">added_in_b</option>
          </select>
          <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
            <option value="relative_l2">Sort by Relative L2 (Desc)</option>
            <option value="max_abs_diff">Sort by Max Abs Diff (Desc)</option>
            <option value="cosine_similarity">Sort by Cosine Sim (Asc)</option>
            <option value="changed_pct">Sort by Changed % (Desc)</option>
            <option value="name">Sort by Name (A-Z)</option>
          </select>
        </div>

        <div className="data-table-wrap" style={{ maxHeight: 420, overflowY: "auto" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Tensor Name</th>
                <th>Role</th>
                <th>Shape</th>
                <th>Status</th>
                <th>Cosine</th>
                <th>Rel L2</th>
                <th>RMSE</th>
                <th>Max Abs</th>
                <th>Changed %</th>
              </tr>
            </thead>
            <tbody>
              {records.map((rec) => (
                <tr
                  key={rec.name}
                  style={{
                    cursor: "pointer",
                    background: selected?.name === rec.name ? "rgba(255, 255, 255, 0.055)" : undefined,
                  }}
                  onClick={() => setSelected(rec)}
                >
                  <td className="mono">{rec.name}</td>
                  <td>{rec.canonical_role}</td>
                  <td className="mono">{JSON.stringify(rec.shape_a ?? rec.shape_b)}</td>
                  <td>
                    <span
                      className={`badge ${
                        rec.status === "identical"
                          ? "badge-ok"
                          : rec.status === "modified"
                            ? "badge-warn"
                            : "badge-danger"
                      }`}
                    >
                      {rec.status}
                    </span>
                  </td>
                  <td className="num">
                    {rec.metrics?.cosine_similarity != null
                      ? rec.metrics.cosine_similarity.toFixed(6)
                      : "—"}
                  </td>
                  <td className="num">
                    {rec.metrics?.relative_l2 != null ? rec.metrics.relative_l2.toFixed(6) : "—"}
                  </td>
                  <td className="num">
                    {rec.metrics?.rmse != null ? rec.metrics.rmse.toExponential(3) : "—"}
                  </td>
                  <td className="num">
                    {rec.metrics?.max_abs_diff != null
                      ? rec.metrics.max_abs_diff.toExponential(3)
                      : "—"}
                  </td>
                  <td className="num">
                    {rec.metrics ? `${rec.metrics.changed_pct.toFixed(2)}%` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {selected && (
        <div className="panel" data-testid="tensor-detail-inspector">
          <h3 className="panel-title">
            Parameter Detail Inspector: <span className="mono">{selected.name}</span>
          </h3>
          <div className="grid-2" style={{ marginTop: 12 }}>
            <div className="diff-box">
              <strong>Numerical Diff Metrics</strong>
              <div>Dtype: {selected.dtype_a ?? "—"} → {selected.dtype_b ?? "—"}</div>
              <div>Exact Equal: {String(selected.metrics?.exact_equal ?? false)}</div>
              <div>Mean Abs Diff: {selected.metrics?.mean_abs_diff ?? "—"}</div>
              <div>L2 Norm A / B: {selected.metrics?.l2_norm_a?.toFixed(4) ?? "—"} / {selected.metrics?.l2_norm_b?.toFixed(4) ?? "—"}</div>
              <div>Zero → Nonzero: {selected.metrics?.zero_to_nonzero ?? 0}</div>
              <div>Nonzero → Zero: {selected.metrics?.nonzero_to_zero ?? 0}</div>
              <div>NaN (A / B): {selected.metrics?.nan_count_a ?? 0} / {selected.metrics?.nan_count_b ?? 0}</div>
            </div>
            <div className="diff-box">
              <strong>Distribution Shift (A → B)</strong>
              <div>Mean: {String(selected.distribution_a?.mean ?? "—")} → {String(selected.distribution_b?.mean ?? "—")}</div>
              <div>Std: {String(selected.distribution_a?.std ?? "—")} → {String(selected.distribution_b?.std ?? "—")}</div>
              <div>Median (p50): {String(selected.distribution_a?.p50 ?? "—")} → {String(selected.distribution_b?.p50 ?? "—")}</div>
              <div>p05 / p95 (B): {String(selected.distribution_b?.p05 ?? "—")} / {String(selected.distribution_b?.p95 ?? "—")}</div>
              <div>Wasserstein-1 Approx: {String(selected.distribution_shift?.wasserstein_1d_approx ?? "—")}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

