import React from "react";
import type { ComparisonBundle } from "../types/api";
import { formatRole } from "../utils/format";

interface Props {
  activeBundle: ComparisonBundle | null;
}

export const StructuralDiffView: React.FC<Props> = ({ activeBundle }) => {
  const st = activeBundle?.structural;
  if (!activeBundle || !st) {
    return (
      <div className="panel" data-testid="view-structural">
        <h2 className="panel-title">Structural Topology</h2>
        <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
          Select a comparison with structural analysis enabled.
        </p>
      </div>
    );
  }

  const hasChanges =
    st.added_tensors.length > 0 ||
    st.removed_tensors.length > 0 ||
    st.inferred_renames.length > 0 ||
    st.shape_changes.length > 0 ||
    st.dtype_changes.length > 0;

  return (
    <div data-testid="view-structural">
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">Parameter Delta</div>
          <div className="kpi-value">
            {st.delta_parameters >= 0 ? "+" : ""}
            {st.delta_parameters.toLocaleString()}
          </div>
          <div className="kpi-sub">
            {(st.model_a_parameters / 1e6).toFixed(1)}M → {(st.model_b_parameters / 1e6).toFixed(1)}M
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Key Additions / Removals</div>
          <div className="kpi-value">
            +{st.added_tensors.length} / -{st.removed_tensors.length}
          </div>
          <div className="kpi-sub">{st.shared_tensors.length} shared keys</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Matched Renames</div>
          <div className="kpi-value">{st.inferred_renames.length}</div>
          <div className="kpi-sub">Cosine similarity ≥ 0.985</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Shape / Dtype Shifts</div>
          <div className="kpi-value">
            {st.shape_changes.length} · {st.dtype_changes.length}
          </div>
          <div className="kpi-sub">{st.layer_deltas.length} indexed blocks</div>
        </div>
      </div>

      {hasChanges && (
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">Topology Delta Log</h3>
          </div>
          <div className="data-table-wrap" style={{ maxHeight: 320, overflowY: "auto" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th style={{ width: 95 }}>Change</th>
                  <th>Parameter</th>
                  <th>Specification</th>
                </tr>
              </thead>
              <tbody>
                {st.added_tensors.slice(0, 60).map((t) => (
                  <tr key={`add-${t.name}`}>
                    <td style={{ color: "var(--ok)" }}>+ Added</td>
                    <td className="mono">{t.name}</td>
                    <td className="mono" style={{ color: "var(--text-muted)" }}>
                      {t.shape.join(" × ")} · {t.dtype} · {formatRole(t.canonical_role)}
                    </td>
                  </tr>
                ))}
                {st.removed_tensors.slice(0, 60).map((t) => (
                  <tr key={`rem-${t.name}`}>
                    <td style={{ color: "var(--danger)" }}>− Removed</td>
                    <td className="mono">{t.name}</td>
                    <td className="mono" style={{ color: "var(--text-muted)" }}>
                      {t.shape.join(" × ")} · {t.dtype} · {formatRole(t.canonical_role)}
                    </td>
                  </tr>
                ))}
                {st.inferred_renames.map((r) => (
                  <tr key={`ren-${r.old_name}`}>
                    <td style={{ color: "var(--champagne)" }}>~ Renamed</td>
                    <td className="mono">
                      {r.old_name} → {r.new_name}
                    </td>
                    <td className="mono" style={{ color: "var(--text-muted)" }}>
                      {r.shape.join(" × ")} · cos {r.cosine_similarity.toFixed(5)}
                    </td>
                  </tr>
                ))}
                {st.shape_changes.slice(0, 60).map((s) => (
                  <tr key={`shp-${s.name}`}>
                    <td style={{ color: "var(--warn)" }}>Δ Shape</td>
                    <td className="mono">{s.name}</td>
                    <td className="mono" style={{ color: "var(--text-muted)" }}>
                      {s.shape_a.join(" × ")} → {s.shape_b.join(" × ")}
                    </td>
                  </tr>
                ))}
                {st.dtype_changes.slice(0, 40).map((d) => (
                  <tr key={`dt-${d.name}`}>
                    <td style={{ color: "var(--text-secondary)" }}>Δ Dtype</td>
                    <td className="mono">{d.name}</td>
                    <td className="mono" style={{ color: "var(--text-muted)" }}>
                      {d.dtype_a} → {d.dtype_b}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="panel">
        <div className="panel-header">
          <h3 className="panel-title">Block Inventory ({st.layer_deltas.length} layers)</h3>
        </div>
        <div className="data-table-wrap" style={{ maxHeight: 360, overflowY: "auto" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Block</th>
                <th>Presence</th>
                <th className="num">Tensors A</th>
                <th className="num">Tensors B</th>
                <th className="num">Parameters A</th>
                <th className="num">Parameters B</th>
                <th className="num">Delta</th>
              </tr>
            </thead>
            <tbody>
              {st.layer_deltas.map((ld) => (
                <tr key={ld.layer_index}>
                  <td className="mono">Layer {ld.layer_index}</td>
                  <td style={{ color: "var(--text-muted)" }}>
                    {ld.present_in_a && ld.present_in_b
                      ? "Both"
                      : ld.present_in_a
                        ? "Model A only"
                        : "Model B only"}
                  </td>
                  <td className="num">{ld.tensor_count_a}</td>
                  <td className="num">{ld.tensor_count_b}</td>
                  <td className="num">{ld.parameters_a.toLocaleString()}</td>
                  <td className="num">{ld.parameters_b.toLocaleString()}</td>
                  <td className="num" style={{ color: "var(--text-muted)" }}>
                    {ld.delta_parameters >= 0 ? "+" : ""}
                    {ld.delta_parameters.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
