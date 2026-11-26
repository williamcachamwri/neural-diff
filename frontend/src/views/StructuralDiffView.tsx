import React from "react";
import type { ComparisonBundle } from "../types/api";

interface Props {
  activeBundle: ComparisonBundle | null;
}

export const StructuralDiffView: React.FC<Props> = ({ activeBundle }) => {
  const st = activeBundle?.structural;
  if (!activeBundle || !st) {
    return (
      <div className="panel" data-testid="view-structural">
        <h2 className="panel-title">Structural Diff</h2>
        <p style={{ color: "var(--text-muted)" }}>
          Load or run a comparison with the <code>structural</code> module enabled.
        </p>
      </div>
    );
  }

  return (
    <div data-testid="view-structural">
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">Parameter Count Delta</div>
          <div className="kpi-value">
            {st.delta_parameters >= 0 ? "+" : ""}
            {st.delta_parameters.toLocaleString()}
          </div>
          <div className="kpi-sub">
            {st.model_a_parameters.toLocaleString()} → {st.model_b_parameters.toLocaleString()}
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Added / Removed Keys</div>
          <div className="kpi-value">
            +{st.added_tensors.length} / -{st.removed_tensors.length}
          </div>
          <div className="kpi-sub">Shared keys: {st.shared_tensors.length}</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Inferred Renames</div>
          <div className="kpi-value">{st.inferred_renames.length}</div>
          <div className="kpi-sub">Epistemic status: INFERRED (cos &ge; 0.985)</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Shape / Dtype Changes</div>
          <div className="kpi-value">
            {st.shape_changes.length} / {st.dtype_changes.length}
          </div>
          <div className="kpi-sub">{st.layer_deltas.length} transformer layers indexed</div>
        </div>
      </div>

      {(st.added_tensors.length > 0 ||
        st.removed_tensors.length > 0 ||
        st.inferred_renames.length > 0 ||
        st.shape_changes.length > 0 ||
        st.dtype_changes.length > 0) && (
        <div className="panel">
          <h3 className="panel-title" style={{ marginBottom: 12 }}>
            Git-Style Structural Change Log
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {st.added_tensors.map((t) => (
              <div key={`add-${t.name}`} className="diff-box mono" style={{ borderColor: "var(--ok)" }}>
                <span className="badge badge-ok" style={{ marginRight: 8 }}>+ ADDED</span>
                {t.name} ({JSON.stringify(t.shape)}, {t.dtype}, role={t.canonical_role})
              </div>
            ))}
            {st.removed_tensors.map((t) => (
              <div
                key={`rem-${t.name}`}
                className="diff-box mono"
                style={{ borderColor: "var(--danger)" }}
              >
                <span className="badge badge-danger" style={{ marginRight: 8 }}>- REMOVED</span>
                {t.name} ({JSON.stringify(t.shape)}, {t.dtype}, role={t.canonical_role})
              </div>
            ))}
            {st.inferred_renames.map((r) => (
              <div
                key={`ren-${r.old_name}`}
                className="diff-box mono"
                style={{ borderColor: "var(--purple)" }}
              >
                <span className="badge badge-purple" style={{ marginRight: 8 }}>
                  ~ RENAME [INFERRED]
                </span>
                {r.old_name} → {r.new_name} (shape={JSON.stringify(r.shape)}, cos=
                {r.cosine_similarity.toFixed(6)})
              </div>
            ))}
            {st.shape_changes.map((s) => (
              <div
                key={`shp-${s.name}`}
                className="diff-box mono"
                style={{ borderColor: "var(--warn)" }}
              >
                <span className="badge badge-warn" style={{ marginRight: 8 }}>Δ SHAPE</span>
                {s.name}: {JSON.stringify(s.shape_a)} → {JSON.stringify(s.shape_b)} (
                {s.delta_elements >= 0 ? "+" : ""}
                {s.delta_elements.toLocaleString()} elems)
              </div>
            ))}
            {st.dtype_changes.map((d) => (
              <div key={`dt-${d.name}`} className="diff-box mono">
                <span className="badge badge-info" style={{ marginRight: 8 }}>Δ DTYPE</span>
                {d.name}: {d.dtype_a} → {d.dtype_b}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="panel">
        <div className="panel-header">
          <h3 className="panel-title">Layer-by-Layer Structural Topology</h3>
          <span className="badge badge-ok">OBSERVED</span>
        </div>
        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Layer</th>
                <th>Present (A / B)</th>
                <th>Tensors A</th>
                <th>Tensors B</th>
                <th>Params A</th>
                <th>Params B</th>
                <th>Delta Params</th>
              </tr>
            </thead>
            <tbody>
              {st.layer_deltas.map((ld) => (
                <tr key={ld.layer_index}>
                  <td className="mono">Layer {ld.layer_index}</td>
                  <td>
                    {ld.present_in_a ? "Yes" : "No"} / {ld.present_in_b ? "Yes" : "No"}
                  </td>
                  <td className="num">{ld.tensor_count_a}</td>
                  <td className="num">{ld.tensor_count_b}</td>
                  <td className="num">{ld.parameters_a.toLocaleString()}</td>
                  <td className="num">{ld.parameters_b.toLocaleString()}</td>
                  <td className="num">
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
