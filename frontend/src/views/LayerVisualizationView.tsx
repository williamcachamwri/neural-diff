import React, { useState } from "react";
import type { ComparisonBundle } from "../types/api";

interface Props {
  activeBundle: ComparisonBundle | null;
}

const HEATMAP_ROLES = [
  "attn_q",
  "attn_k",
  "attn_v",
  "attn_o",
  "mlp_gate",
  "mlp_up",
  "mlp_down",
  "norm",
];

export const LayerVisualizationView: React.FC<Props> = ({ activeBundle }) => {
  const layers = activeBundle?.tensor_diff?.layer_summaries ?? [];
  const na = activeBundle?.neuron_attention ?? null;
  const [selectedLayer, setSelectedLayer] = useState<number>(0);

  if (!activeBundle || (layers.length === 0 && !na)) {
    return (
      <div className="panel" data-testid="view-layers">
        <h2 className="panel-title">Layer &amp; Neuron/Attention Visualization</h2>
        <p style={{ color: "var(--text-muted)" }}>
          Run a comparison with <code>tensor</code> and/or <code>neuron</code> modules enabled.
        </p>
      </div>
    );
  }

  let maxRelL2 = 1e-6;
  for (const ls of layers) {
    for (const r of HEATMAP_ROLES) {
      const v = ls.role_breakdown[r]?.mean_relative_l2;
      if (v != null && v > maxRelL2) maxRelL2 = v;
    }
  }

  const colorForRelL2 = (val: number | null | undefined): string => {
    if (val == null) return "#111116";
    const t = Math.min(1, Math.max(0, val / maxRelL2));
    // Obsidian -> Deep Espresso Bronze -> Warm Champagne Gold -> Luminous Ivory
    const r = Math.round(20 + 215 * Math.pow(t, 0.85));
    const g = Math.round(20 + 168 * Math.pow(t, 0.95));
    const b = Math.round(26 + 85 * Math.pow(t, 1.35));
    return `rgb(${r}, ${g}, ${b})`;
  };

  const layerAttn =
    na?.attention_projections.filter((a) => a.layer_index === selectedLayer) ?? [];
  const layerMlp = na?.mlp_projections.filter((m) => m.layer_index === selectedLayer) ?? [];

  return (
    <div data-testid="view-layers">
      {layers.length > 0 && (
        <div className="panel">
          <div className="panel-header">
            <h2 className="panel-title">
              Layer-by-Submodule Relative L2 Heatmap (Click Layer Column to Inspect Heads &amp; MLP)
            </h2>
            <span className="badge badge-ok">OBSERVED • Max Rel L2 = {maxRelL2.toFixed(5)}</span>
          </div>
          <div style={{ overflowX: "auto" }}>
            <table className="data-table" style={{ fontSize: 11.5 }}>
              <thead>
                <tr>
                  <th>Submodule Role</th>
                  {layers.map((ls) => (
                    <th
                      key={ls.layer_index}
                      onClick={() => setSelectedLayer(ls.layer_index)}
                      style={{
                        cursor: "pointer",
                        textAlign: "center",
                        color:
                          selectedLayer === ls.layer_index ? "#ffffff" : "var(--text-muted)",
                        borderBottom:
                          selectedLayer === ls.layer_index
                            ? "2px solid var(--accent)"
                            : undefined,
                      }}
                    >
                      L{ls.layer_index}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {HEATMAP_ROLES.map((roleName) => (
                  <tr key={roleName}>
                    <td className="mono">{roleName}</td>
                    {layers.map((ls) => {
                      const cell = ls.role_breakdown[roleName];
                      const val = cell?.mean_relative_l2 ?? null;
                      return (
                        <td
                          key={`${roleName}-${ls.layer_index}`}
                          onClick={() => setSelectedLayer(ls.layer_index)}
                          title={`Layer ${ls.layer_index} ${roleName}: rel_L2=${val ?? "N/A"}, cos=${cell?.mean_cosine ?? "N/A"}`}
                          style={{
                            cursor: "pointer",
                            background: colorForRelL2(val),
                            textAlign: "center",
                            padding: "7px 4px",
                            color: "#fff",
                            fontWeight: 600,
                          }}
                        >
                          {val != null ? val.toFixed(3) : "—"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {na && (
        <>
          <div className="panel" style={{ borderColor: "var(--warn)" }}>
            <div style={{ fontSize: 12.5, color: "var(--warn)" }}>
              <strong>Epistemic Caution ({na.epistemic_status}):</strong> {na.epistemic_caution}
            </div>
          </div>

          <div className="grid-2">
            <div className="panel">
              <h3 className="panel-title" style={{ marginBottom: 10 }}>
                Layer {selectedLayer} Attention Head Slicing ({layerAttn.length} projections)
              </h3>
              {layerAttn.length === 0 ? (
                <div style={{ color: "var(--text-muted)" }}>
                  No sliced attention projections for Layer {selectedLayer}.
                </div>
              ) : (
                layerAttn.map((proj) => (
                  <div key={proj.tensor_name} className="diff-box" style={{ marginBottom: 10 }}>
                    <div className="mono" style={{ fontWeight: 700 }}>
                      {proj.tensor_name} ({proj.num_heads} heads × {proj.head_dim}d)
                    </div>
                    <div>
                      Mean Head Cosine: {proj.mean_head_cosine.toFixed(6)} • Max Head Rel L2:{" "}
                      {proj.max_head_relative_l2.toFixed(6)}
                    </div>
                    <div style={{ marginTop: 6, fontSize: 12, color: "var(--text-muted)" }}>
                      Most diverged heads:{" "}
                      {proj.most_diverged_heads
                        .map(
                          (h) => `H${h.head_index} (relL2=${h.relative_l2.toFixed(4)}, cos=${h.cosine_similarity.toFixed(4)})`
                        )
                        .join(", ")}
                    </div>
                  </div>
                ))
              )}
            </div>

            <div className="panel">
              <h3 className="panel-title" style={{ marginBottom: 10 }}>
                Layer {selectedLayer} MLP Basis Coordinate Analysis ({layerMlp.length} matrices)
              </h3>
              {layerMlp.length === 0 ? (
                <div style={{ color: "var(--text-muted)" }}>
                  No MLP basis projections for Layer {selectedLayer}.
                </div>
              ) : (
                layerMlp.map((mlp) => (
                  <div key={mlp.tensor_name} className="diff-box" style={{ marginBottom: 10 }}>
                    <div className="mono" style={{ fontWeight: 700 }}>
                      {mlp.tensor_name} ({mlp.basis_dimension} coordinates)
                    </div>
                    <div>
                      Mean Basis Cosine: {mlp.mean_basis_cosine.toFixed(6)} • p95 Rel L2:{" "}
                      {mlp.p95_relative_l2.toFixed(6)}
                    </div>
                    <div style={{ marginTop: 6, fontSize: 12, color: "var(--text-muted)" }}>
                      Top shifted basis indices:{" "}
                      {mlp.top_shifted_indices
                        .slice(0, 5)
                        .map((idx) => `#${idx.basis_index} (${idx.relative_l2.toFixed(4)})`)
                        .join(", ")}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {na.embeddings_and_heads.length > 0 && (
            <div className="panel">
              <h3 className="panel-title" style={{ marginBottom: 10 }}>
                Embedding &amp; Output Projection Drift
              </h3>
              <div className="data-table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Tensor</th>
                      <th>Role</th>
                      <th>Vocab × Hidden</th>
                      <th>Mean Token Cosine</th>
                      <th>Changed Token Rows %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {na.embeddings_and_heads.map((emb) => (
                      <tr key={emb.tensor_name}>
                        <td className="mono">{emb.tensor_name}</td>
                        <td>{emb.role}</td>
                        <td className="mono">
                          {emb.vocab_size.toLocaleString()} × {emb.hidden_size}
                        </td>
                        <td className="num">{emb.mean_token_cosine.toFixed(6)}</td>
                        <td className="num">{emb.changed_token_pct.toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};
