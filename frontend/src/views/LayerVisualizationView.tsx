import React, { useState } from "react";
import type { ComparisonBundle } from "../types/api";
import { formatRole, formatShortTensorName } from "../utils/format";

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
        <h2 className="panel-title">Layer &amp; Head Analysis</h2>
        <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
          Select a comparison with tensor or head analysis enabled.
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

  const tileColor = (val: number | null | undefined): string => {
    if (val == null) return "#141418";
    const t = Math.min(1, Math.max(0, val / maxRelL2));
    const r = Math.round(22 + 196 * Math.pow(t, 0.88));
    const g = Math.round(22 + 178 * Math.pow(t, 0.94));
    const b = Math.round(27 + 138 * Math.pow(t, 1.15));
    return `rgb(${r}, ${g}, ${b})`;
  };

  const tileTextColor = (val: number | null | undefined): string => {
    if (val == null) return "var(--text-faint)";
    const t = Math.min(1, Math.max(0, val / maxRelL2));
    return t > 0.58 ? "#09090b" : "#e4e4e7";
  };

  const layerAttn =
    na?.attention_projections.filter((a) => a.layer_index === selectedLayer) ?? [];
  const layerMlp = na?.mlp_projections.filter((m) => m.layer_index === selectedLayer) ?? [];

  const maxHeadL2 = Math.max(
    1e-5,
    ...layerAttn.flatMap((p) => p.most_diverged_heads.map((h) => h.relative_l2))
  );

  return (
    <div data-testid="view-layers">
      {layers.length > 0 && (
        <div className="panel">
          <div className="panel-header">
            <div>
              <h2 className="panel-title">Submodule Relative L2 Matrix</h2>
              <div className="panel-subtitle" style={{ marginTop: 2 }}>
                Select any layer column to inspect per-head and MLP projection shifts
              </div>
            </div>
            <div className="mono" style={{ color: "var(--text-muted)", fontSize: 11.5 }}>
              Peak rel L2 · {maxRelL2.toFixed(4)}
            </div>
          </div>

          <div style={{ overflowX: "auto", paddingBottom: 4 }}>
            <table
              style={{
                width: "100%",
                borderCollapse: "separate",
                borderSpacing: "3px",
                fontSize: 11,
              }}
            >
              <thead>
                <tr>
                  <th
                    style={{
                      textAlign: "left",
                      color: "var(--text-muted)",
                      fontWeight: 500,
                      padding: "4px 8px 6px 0",
                      minWidth: 96,
                    }}
                  >
                    Submodule
                  </th>
                  {layers.map((ls) => {
                    const active = selectedLayer === ls.layer_index;
                    return (
                      <th
                        key={ls.layer_index}
                        onClick={() => setSelectedLayer(ls.layer_index)}
                        style={{
                          cursor: "pointer",
                          textAlign: "center",
                          padding: "4px 2px 6px",
                          fontWeight: active ? 600 : 500,
                          color: active ? "#f4f4f5" : "var(--text-muted)",
                          borderBottom: active ? "1.5px solid #f4f4f5" : "1.5px solid transparent",
                        }}
                      >
                        L{ls.layer_index}
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody>
                {HEATMAP_ROLES.map((roleName) => (
                  <tr key={roleName}>
                    <td
                      style={{
                        color: "var(--text-secondary)",
                        padding: "4px 8px 4px 0",
                        whiteSpace: "nowrap",
                        fontSize: 11.5,
                      }}
                    >
                      {formatRole(roleName)}
                    </td>
                    {layers.map((ls) => {
                      const cell = ls.role_breakdown[roleName];
                      const val = cell?.mean_relative_l2 ?? null;
                      const active = selectedLayer === ls.layer_index;
                      return (
                        <td
                          key={`${roleName}-${ls.layer_index}`}
                          onClick={() => setSelectedLayer(ls.layer_index)}
                          title={`Layer ${ls.layer_index} · ${formatRole(roleName)} · rel L2 ${val?.toFixed(5) ?? "—"}`}
                          className="mono"
                          style={{
                            cursor: "pointer",
                            background: tileColor(val),
                            color: tileTextColor(val),
                            textAlign: "center",
                            padding: "6px 3px",
                            borderRadius: 4,
                            fontSize: 10.5,
                            fontWeight: 500,
                            outline: active ? "1px solid rgba(255,255,255,0.28)" : "none",
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
        <div className="grid-2">
          {/* Left: Clean Hairline Attention Head Breakdown with Proportional Bars */}
          <div className="panel">
            <div className="panel-header">
              <h3 className="panel-title">Layer {selectedLayer} · Attention Heads</h3>
              <span className="panel-subtitle">{layerAttn.length} projections</span>
            </div>

            {layerAttn.length === 0 ? (
              <div style={{ color: "var(--text-muted)", fontSize: 12.5 }}>
                No head-sliced projections recorded for Layer {selectedLayer}.
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                {layerAttn.map((proj, pIdx) => (
                  <div
                    key={proj.tensor_name}
                    style={{
                      paddingTop: pIdx === 0 ? 0 : 14,
                      borderTop: pIdx === 0 ? "none" : "1px solid var(--border-faint)",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "baseline",
                        marginBottom: 8,
                      }}
                    >
                      <span className="mono" style={{ color: "var(--text-main)", fontWeight: 500 }}>
                        {formatShortTensorName(proj.tensor_name)}
                      </span>
                      <span className="mono" style={{ fontSize: 11, color: "var(--text-muted)" }}>
                        {proj.num_heads} × {proj.head_dim}d · cos {proj.mean_head_cosine.toFixed(4)}
                      </span>
                    </div>

                    <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                      {proj.most_diverged_heads.slice(0, 4).map((h) => {
                        const widthPct = Math.min(100, Math.max(4, (h.relative_l2 / maxHeadL2) * 100));
                        return (
                          <div
                            key={h.head_index}
                            style={{
                              display: "grid",
                              gridTemplateColumns: "42px 1fr 62px 68px",
                              alignItems: "center",
                              gap: 8,
                              fontSize: 11,
                            }}
                          >
                            <span className="mono" style={{ color: "var(--text-muted)" }}>
                              H{h.head_index}
                            </span>
                            <div
                              style={{
                                height: 4,
                                background: "rgba(255,255,255,0.05)",
                                borderRadius: 99,
                                overflow: "hidden",
                              }}
                            >
                              <div
                                style={{
                                  width: `${widthPct}%`,
                                  height: "100%",
                                  background: "#d6c5a9",
                                  borderRadius: 99,
                                }}
                              />
                            </div>
                            <span className="num" style={{ color: "var(--text-secondary)" }}>
                              {h.relative_l2.toFixed(4)}
                            </span>
                            <span className="num" style={{ color: "var(--text-faint)" }}>
                              {h.cosine_similarity.toFixed(4)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Right: Clean Hairline MLP Coordinate Table without Nested Dark Boxes */}
          <div className="panel">
            <div className="panel-header">
              <h3 className="panel-title">Layer {selectedLayer} · MLP Projections</h3>
              <span className="panel-subtitle">{layerMlp.length} matrices</span>
            </div>

            {layerMlp.length === 0 ? (
              <div style={{ color: "var(--text-muted)", fontSize: 12.5 }}>
                No MLP projections recorded for Layer {selectedLayer}.
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                {layerMlp.map((mlp, mIdx) => (
                  <div
                    key={mlp.tensor_name}
                    style={{
                      paddingTop: mIdx === 0 ? 0 : 14,
                      borderTop: mIdx === 0 ? "none" : "1px solid var(--border-faint)",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "baseline",
                        marginBottom: 8,
                      }}
                    >
                      <span className="mono" style={{ color: "var(--text-main)", fontWeight: 500 }}>
                        {formatShortTensorName(mlp.tensor_name)}
                      </span>
                      <span className="mono" style={{ fontSize: 11, color: "var(--text-muted)" }}>
                        {mlp.basis_dimension.toLocaleString()} dim · p95 {mlp.p95_relative_l2.toFixed(4)}
                      </span>
                    </div>

                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                      {mlp.top_shifted_indices.slice(0, 5).map((idx) => (
                        <div
                          key={idx.basis_index}
                          className="mono"
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: 6,
                            padding: "4px 8px",
                            borderRadius: 5,
                            background: "rgba(255,255,255,0.03)",
                            border: "1px solid var(--border-faint)",
                            fontSize: 11,
                          }}
                        >
                          <span style={{ color: "var(--text-muted)" }}>#{idx.basis_index}</span>
                          <span style={{ color: "var(--text-secondary)" }}>
                            {idx.relative_l2.toFixed(4)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {na && na.embeddings_and_heads.length > 0 && (
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">Vocabulary Embeddings &amp; Unembedding</h3>
          </div>
          <div className="data-table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Parameter</th>
                  <th>Role</th>
                  <th>Dimensions</th>
                  <th className="num">Mean Token Cosine</th>
                  <th className="num">Modified Rows</th>
                </tr>
              </thead>
              <tbody>
                {na.embeddings_and_heads.map((emb) => (
                  <tr key={emb.tensor_name}>
                    <td className="mono">{emb.tensor_name}</td>
                    <td>{formatRole(emb.role)}</td>
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
    </div>
  );
};
