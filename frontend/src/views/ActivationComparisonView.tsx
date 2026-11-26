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
        <h2 className="panel-title">Activation Diff &amp; Linear CKA</h2>
        <p style={{ color: "var(--text-muted)" }}>
          Run a comparison with the <code>activation</code> module enabled to inspect forward-hook hidden states, MLP activations, and next-token JS divergence.
        </p>
      </div>
    );
  }

  const layerIndices = Object.keys(act.summary.mean_hidden_cosine_by_layer).sort(
    (a, b) => Number(a) - Number(b)
  );
  const currentPrompt = act.prompt_diffs[promptIdx] ?? act.prompt_diffs[0];

  return (
    <div data-testid="view-activations">
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">Controlled Prompts</div>
          <div className="kpi-value">{act.summary.prompt_count}</div>
          <div className="kpi-sub">Deterministic forward hooks (seed=42)</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Final-Layer Pooled CKA</div>
          <div className="kpi-value">
            {act.summary.cross_prompt_pooled_cka_final_layer != null
              ? act.summary.cross_prompt_pooled_cka_final_layer.toFixed(5)
              : "—"}
          </div>
          <div className="kpi-sub">Orthogonal &amp; scale invariant</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Next-Token JS Divergence</div>
          <div className="kpi-value">
            {act.summary.mean_final_token_js_divergence != null
              ? act.summary.mean_final_token_js_divergence.toFixed(5)
              : "—"}
          </div>
          <div className="kpi-sub">Final prompt token output logits</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Epistemic Classification</div>
          <div className="kpi-value" style={{ fontSize: 17 }}>
            {act.epistemic_status}
          </div>
          <div className="kpi-sub">Observational forward-pass capture</div>
        </div>
      </div>

      <div className="panel">
        <h3 className="panel-title" style={{ marginBottom: 12 }}>
          Layer-by-Layer Activation Alignment (Hidden Cosine, Linear CKA, Attention &amp; MLP)
        </h3>
        <div className="data-table-wrap" style={{ maxHeight: 340, overflowY: "auto" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Layer</th>
                <th>Hidden Cosine</th>
                <th>Hidden Linear CKA</th>
                <th>Attention Cosine</th>
                <th>MLP Cosine</th>
                <th>Alignment Bar</th>
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
                    <td className="mono">Layer {lk}</td>
                    <td className="num">{hCos != null ? hCos.toFixed(6) : "—"}</td>
                    <td className="num">{hCka != null ? hCka.toFixed(6) : "—"}</td>
                    <td className="num">{aCos != null ? aCos.toFixed(6) : "—"}</td>
                    <td className="num">{mCos != null ? mCos.toFixed(6) : "—"}</td>
                    <td style={{ width: 180 }}>
                      <div
                        style={{
                          background: "#1b1b21",
                          borderRadius: 4,
                          height: 10,
                          overflow: "hidden",
                        }}
                      >
                        <div
                          style={{
                            width: `${pct}%`,
                            height: "100%",
                            background: "#d4d4d8",
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
            <h3 className="panel-title">Per-Prompt Logit &amp; Token Activation Inspector</h3>
            <select
              value={promptIdx}
              onChange={(e) => setPromptIdx(Number(e.target.value))}
            >
              {act.prompt_diffs.map((p, idx) => (
                <option key={idx} value={idx}>
                  Prompt #{idx + 1}: {p.prompt.slice(0, 56)}...
                </option>
              ))}
            </select>
          </div>
          <div className="diff-box mono" style={{ marginBottom: 12 }}>
            {currentPrompt.prompt}
          </div>
          <div className="diff-pair">
            <div className="diff-box">
              <strong>Model A Top-5 Next-Token Probabilities</strong>
              {currentPrompt.top_tokens_a.map((t) => (
                <div key={t.token_id} className="mono">
                  {JSON.stringify(t.token)} (id={t.token_id}): {(t.prob * 100).toFixed(2)}%
                </div>
              ))}
            </div>
            <div className="diff-box">
              <strong>Model B Top-5 Next-Token Probabilities</strong>
              {currentPrompt.top_tokens_b.map((t) => (
                <div key={t.token_id} className="mono">
                  {JSON.stringify(t.token)} (id={t.token_id}): {(t.prob * 100).toFixed(2)}%
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

