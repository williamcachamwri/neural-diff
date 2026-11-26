import React, { useState } from "react";
import type { ComparisonBundle } from "../types/api";

interface Props {
  activeBundle: ComparisonBundle | null;
}

export const ActivationComparisonView: React.FC<Props> = ({ activeBundle }) => {
  const act = activeBundle?.activation;
  const [promptIdx, setPromptIdx] = useState(0);

  if (!activeBundle || !act) {
    return (
      <div className="panel" data-testid="view-activations">
        <h2 className="panel-title">Activation &amp; Linear CKA</h2>
        <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
          Select a comparison with activation analysis enabled to inspect hidden-state geometry.
        </p>
      </div>
    );
  }

  const layerIndices = Object.keys(act.summary.mean_hidden_cosine_by_layer).sort(
    (a, b) => Number(a) - Number(b)
  );
  const firstLayerKey = layerIndices[0];
  const lastLayerKey = layerIndices[layerIndices.length - 1];
  const firstCka = firstLayerKey != null ? act.summary.mean_hidden_cka_by_layer[firstLayerKey] : null;
  const lastCka = lastLayerKey != null ? act.summary.mean_hidden_cka_by_layer[lastLayerKey] : null;
  const ckaDrift =
    firstCka != null && lastCka != null ? lastCka - firstCka : null;

  const currentPrompt = act.prompt_diffs[promptIdx] ?? act.prompt_diffs[0];

  return (
    <div data-testid="view-activations">
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">Controlled Prompts</div>
          <div className="kpi-value">{act.summary.prompt_count}</div>
          <div className="kpi-sub">{layerIndices.length} hooked transformer blocks</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Final-Layer Pooled CKA</div>
          <div className="kpi-value">
            {act.summary.cross_prompt_pooled_cka_final_layer != null
              ? act.summary.cross_prompt_pooled_cka_final_layer.toFixed(5)
              : "—"}
          </div>
          <div className="kpi-sub">Orthogonal &amp; scale-invariant</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Depth CKA Drift (L0 → Final)</div>
          <div className="kpi-value">
            {ckaDrift != null ? `${ckaDrift >= 0 ? "+" : ""}${ckaDrift.toFixed(4)}` : "—"}
          </div>
          <div className="kpi-sub">
            {firstCka != null && lastCka != null
              ? `${firstCka.toFixed(4)} → ${lastCka.toFixed(4)}`
              : "Cross-layer representation shift"}
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Next-Token JS Divergence</div>
          <div className="kpi-value">
            {act.summary.mean_final_token_js_divergence != null
              ? act.summary.mean_final_token_js_divergence.toFixed(5)
              : "—"}
          </div>
          <div className="kpi-sub">Final prompt token logits</div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-header">
          <h3 className="panel-title">Layer Representation Alignment</h3>
          <span className="panel-subtitle">Hidden state cosine, Linear CKA, attention &amp; MLP</span>
        </div>
        <div className="data-table-wrap" style={{ maxHeight: 330, overflowY: "auto" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Block</th>
                <th className="num">Hidden Cosine</th>
                <th className="num">Linear CKA</th>
                <th className="num">Attention Cosine</th>
                <th className="num">MLP Cosine</th>
                <th style={{ width: 170 }}>CKA Profile</th>
              </tr>
            </thead>
            <tbody>
              {layerIndices.map((lk) => {
                const hCos = act.summary.mean_hidden_cosine_by_layer[lk];
                const hCka = act.summary.mean_hidden_cka_by_layer[lk];
                const aCos = act.summary.mean_attn_cosine_by_layer[lk];
                const mCos = act.summary.mean_mlp_cosine_by_layer[lk];
                const pct = Math.max(0, Math.min(100, (hCka ?? hCos ?? 0) * 100));
                return (
                  <tr key={lk}>
                    <td className="mono" style={{ color: "var(--text-main)" }}>
                      Layer {lk}
                    </td>
                    <td className="num">{hCos != null ? hCos.toFixed(6) : "—"}</td>
                    <td className="num" style={{ color: "var(--text-main)" }}>
                      {hCka != null ? hCka.toFixed(6) : "—"}
                    </td>
                    <td className="num" style={{ color: "var(--text-muted)" }}>
                      {aCos != null ? aCos.toFixed(6) : "—"}
                    </td>
                    <td className="num" style={{ color: "var(--text-muted)" }}>
                      {mCos != null ? mCos.toFixed(6) : "—"}
                    </td>
                    <td>
                      <div
                        style={{
                          background: "rgba(255,255,255,0.05)",
                          borderRadius: 99,
                          height: 4,
                          overflow: "hidden",
                        }}
                      >
                        <div
                          style={{
                            width: `${pct}%`,
                            height: "100%",
                            background: "#d4d4d8",
                            borderRadius: 99,
                          }}
                        />
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {currentPrompt && (
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">Next-Token Distribution</h3>
            <select
              value={promptIdx}
              onChange={(e) => setPromptIdx(Number(e.target.value))}
              style={{ maxWidth: 380, padding: "5px 10px", fontSize: 12 }}
            >
              {act.prompt_diffs.map((p, idx) => (
                <option key={idx} value={idx}>
                  Prompt {idx + 1} · {p.prompt.slice(0, 52)}...
                </option>
              ))}
            </select>
          </div>

          <div
            style={{
              fontSize: 12.5,
              color: "var(--text-secondary)",
              padding: "10px 12px",
              background: "var(--bg-surface-subtle)",
              border: "1px solid var(--border-faint)",
              borderRadius: 6,
              marginBottom: 16,
            }}
          >
            {currentPrompt.prompt}
          </div>

          <div className="grid-2">
            <div>
              <div style={{ fontSize: 12, fontWeight: 500, color: "var(--text-muted)", marginBottom: 10 }}>
                Model A · Top-5 Tokens
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
                {currentPrompt.top_tokens_a.map((t) => (
                  <div
                    key={t.token_id}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "110px 1fr 56px",
                      alignItems: "center",
                      gap: 10,
                    }}
                  >
                    <span
                      className="mono"
                      style={{
                        color: "var(--text-main)",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {JSON.stringify(t.token)}
                    </span>
                    <div
                      style={{
                        height: 5,
                        background: "rgba(255,255,255,0.05)",
                        borderRadius: 99,
                        overflow: "hidden",
                      }}
                    >
                      <div
                        style={{
                          width: `${Math.min(100, t.prob * 100)}%`,
                          height: "100%",
                          background: "#71717a",
                          borderRadius: 99,
                        }}
                      />
                    </div>
                    <span className="num" style={{ color: "var(--text-secondary)" }}>
                      {(t.prob * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <div style={{ fontSize: 12, fontWeight: 500, color: "var(--text-muted)", marginBottom: 10 }}>
                Model B · Top-5 Tokens
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
                {currentPrompt.top_tokens_b.map((t) => (
                  <div
                    key={t.token_id}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "110px 1fr 56px",
                      alignItems: "center",
                      gap: 10,
                    }}
                  >
                    <span
                      className="mono"
                      style={{
                        color: "var(--text-main)",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {JSON.stringify(t.token)}
                    </span>
                    <div
                      style={{
                        height: 5,
                        background: "rgba(255,255,255,0.05)",
                        borderRadius: 99,
                        overflow: "hidden",
                      }}
                    >
                      <div
                        style={{
                          width: `${Math.min(100, t.prob * 100)}%`,
                          height: "100%",
                          background: "#e4e4e7",
                          borderRadius: 99,
                        }}
                      />
                    </div>
                    <span className="num" style={{ color: "var(--text-main)" }}>
                      {(t.prob * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
