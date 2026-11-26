import React, { useState } from "react";
import { apiClient } from "../api/client";
import type {
  CompatibilityReport,
  ComparisonBundle,
  ModelMetadata,
} from "../types/api";
import { formatSeverity } from "../utils/format";

interface Props {
  modelA: string;
  modelB: string;
  activeBundle: ComparisonBundle | null;
  onChangeModelA: (val: string) => void;
  onChangeModelB: (val: string) => void;
  onError: (err: Error) => void;
}

export const CompatibilityView: React.FC<Props> = ({
  modelA,
  modelB,
  activeBundle,
  onChangeModelA,
  onChangeModelB,
  onError,
}) => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{
    model_a: ModelMetadata;
    model_b: ModelMetadata;
    compatibility: CompatibilityReport;
  } | null>(
    activeBundle?.structural
      ? {
          model_a: activeBundle.model_a,
          model_b: activeBundle.model_b,
          compatibility: activeBundle.structural.compatibility,
        }
      : null
  );

  const handleValidate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!modelA.trim() || !modelB.trim()) {
      onError(new Error("Enter both Model A and Model B identifiers."));
      return;
    }
    setLoading(true);
    try {
      const res = await apiClient.validateCompatibility(modelA.trim(), modelB.trim());
      setResult(res);
    } catch (err) {
      onError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  };

  const comp = result?.compatibility ?? activeBundle?.structural?.compatibility ?? null;
  const metaA = result?.model_a ?? activeBundle?.model_a ?? null;
  const metaB = result?.model_b ?? activeBundle?.model_b ?? null;

  return (
    <div data-testid="view-compatibility">
      <div className="panel">
        <div className="panel-header">
          <h2 className="panel-title">Checkpoint Compatibility</h2>
        </div>
        <form onSubmit={handleValidate}>
          <div className="grid-2">
            <div className="field-group">
              <label htmlFor="compat-model-a">Model A</label>
              <input
                id="compat-model-a"
                data-testid="compat-model-a"
                type="text"
                value={modelA}
                onChange={(e) => onChangeModelA(e.target.value)}
              />
            </div>
            <div className="field-group">
              <label htmlFor="compat-model-b">Model B</label>
              <input
                id="compat-model-b"
                data-testid="compat-model-b"
                type="text"
                value={modelB}
                onChange={(e) => onChangeModelB(e.target.value)}
              />
            </div>
          </div>
          <button
            type="submit"
            className="btn btn-primary"
            data-testid="btn-validate-compatibility"
            disabled={loading}
          >
            {loading ? "Validating..." : "Validate Checkpoints"}
          </button>
        </form>
      </div>

      {comp && metaA && metaB && (
        <div data-testid="compatibility-result">
          <div className="kpi-grid">
            <div className="kpi-card">
              <div className="kpi-label">Tensor Alignment</div>
              <div className="kpi-value">
                {comp.shape_matched_tensor_count}/{comp.shared_tensor_count}
              </div>
              <div className="kpi-sub">
                {comp.compatible_for_direct_tensor_diff ? "Direct diff ready" : "Shape mismatch"}
              </div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Key Overlap</div>
              <div className="kpi-value">{(comp.jaccard_key_overlap * 100).toFixed(1)}%</div>
              <div className="kpi-sub">Jaccard index</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Head Slicing</div>
              <div className="kpi-value" style={{ fontSize: 18 }}>
                {comp.compatible_for_head_neuron_analysis ? "Compatible" : "Partial"}
              </div>
              <div className="kpi-sub">Attention &amp; MLP geometry</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Forward Hooks &amp; Eval</div>
              <div className="kpi-value" style={{ fontSize: 18 }}>
                {comp.compatible_for_behavioral_diff ? "Compatible" : "Weights only"}
              </div>
              <div className="kpi-sub">Tokenizer &amp; config check</div>
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <h3 className="panel-title">Architecture Metadata</h3>
              <span className="panel-subtitle">{comp.summary}</span>
            </div>
            <div className="data-table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Property</th>
                    <th>Model A ({metaA.model_id})</th>
                    <th>Model B ({metaB.model_id})</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={{ color: "var(--text-muted)" }}>Format</td>
                    <td className="mono">{metaA.format}</td>
                    <td className="mono">{metaB.format}</td>
                  </tr>
                  <tr>
                    <td style={{ color: "var(--text-muted)" }}>Parameters</td>
                    <td className="mono">{metaA.total_parameters.toLocaleString()}</td>
                    <td className="mono">{metaB.total_parameters.toLocaleString()}</td>
                  </tr>
                  <tr>
                    <td style={{ color: "var(--text-muted)" }}>Tensors · Shards</td>
                    <td className="mono">
                      {metaA.tensor_count} · {metaA.shard_count} shard(s)
                    </td>
                    <td className="mono">
                      {metaB.tensor_count} · {metaB.shard_count} shard(s)
                    </td>
                  </tr>
                  <tr>
                    <td style={{ color: "var(--text-muted)" }}>Precision</td>
                    <td className="mono">{metaA.dtypes_present.join(", ")}</td>
                    <td className="mono">{metaB.dtypes_present.join(", ")}</td>
                  </tr>
                  <tr>
                    <td style={{ color: "var(--text-muted)" }}>Architecture</td>
                    <td className="mono">
                      {metaA.architecture.architecture_class ?? "—"} (
                      {metaA.architecture.num_hidden_layers ?? "—"}L ×{" "}
                      {metaA.architecture.hidden_size ?? "—"}d)
                    </td>
                    <td className="mono">
                      {metaB.architecture.architecture_class ?? "—"} (
                      {metaB.architecture.num_hidden_layers ?? "—"}L ×{" "}
                      {metaB.architecture.hidden_size ?? "—"}d)
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {comp.issues.length > 0 && (
            <div className="panel">
              <div className="panel-header">
                <h3 className="panel-title">Notices ({comp.issues.length})</h3>
              </div>
              <div className="data-table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Severity</th>
                      <th>Field</th>
                      <th>Model A</th>
                      <th>Model B</th>
                      <th>Detail</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comp.issues.map((iss, idx) => (
                      <tr key={`${iss.code}-${idx}`}>
                        <td>
                          <span className="status-inline">
                            <span
                              className={`dot ${
                                iss.severity === "CRITICAL" || iss.severity === "HIGH"
                                  ? "dot-danger"
                                  : "dot-warn"
                              }`}
                            />
                            <span>{formatSeverity(iss.severity)}</span>
                          </span>
                        </td>
                        <td className="mono">{iss.field}</td>
                        <td className="mono">{String(iss.value_a ?? "—")}</td>
                        <td className="mono">{String(iss.value_b ?? "—")}</td>
                        <td style={{ color: "var(--text-secondary)" }}>{iss.message}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
