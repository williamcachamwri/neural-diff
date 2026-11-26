import React, { useState } from "react";
import { apiClient } from "../api/client";
import type {
  CompatibilityReport,
  ComparisonBundle,
  ModelMetadata,
} from "../types/api";

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
      onError(new Error("Enter both Model A and Model B identifiers to validate compatibility."));
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
        <h2 className="panel-title" style={{ marginBottom: 12 }}>
          Model Compatibility Validation
        </h2>
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
            {loading ? "Validating Checkpoints..." : "Run Compatibility Check"}
          </button>
        </form>
      </div>

      {comp && metaA && metaB && (
        <>
          <div className="panel" data-testid="compatibility-result">
            <div className="panel-header">
              <h3 className="panel-title">{comp.summary}</h3>
              <span className="badge badge-ok">{comp.epistemic_status}</span>
            </div>
            <div className="kpi-grid">
              <div className="kpi-card">
                <div className="kpi-label">Direct Tensor Diff</div>
                <div className="kpi-value">
                  {comp.compatible_for_direct_tensor_diff ? "SUPPORTED" : "INCOMPATIBLE"}
                </div>
                <div className="kpi-sub">
                  {comp.shape_matched_tensor_count} shape-aligned tensors
                </div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">Key Jaccard Overlap</div>
                <div className="kpi-value">{(comp.jaccard_key_overlap * 100).toFixed(1)}%</div>
                <div className="kpi-sub">{comp.shared_tensor_count} shared keys</div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">Head / Neuron Slicing</div>
                <div className="kpi-value">
                  {comp.compatible_for_head_neuron_analysis ? "SUPPORTED" : "LIMITED"}
                </div>
                <div className="kpi-sub">Requires matching architecture family</div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">Activation &amp; Behavioral</div>
                <div className="kpi-value">
                  {comp.compatible_for_behavioral_diff ? "SUPPORTED" : "WEIGHTS ONLY"}
                </div>
                <div className="kpi-sub">Requires HF config &amp; tokenizer</div>
              </div>
            </div>

            <div className="grid-2">
              <div className="diff-box">
                <strong>Model A Metadata ({metaA.model_id})</strong>
                <div>Format: {metaA.format}</div>
                <div>Parameters: {metaA.total_parameters.toLocaleString()}</div>
                <div>Tensors: {metaA.tensor_count} ({metaA.shard_count} shard(s))</div>
                <div>Dtypes: {metaA.dtypes_present.join(", ")}</div>
                <div>
                  Architecture: {metaA.architecture.architecture_class ?? "—"} (type:{" "}
                  {metaA.architecture.model_type ?? "—"}, layers:{" "}
                  {metaA.architecture.num_hidden_layers ?? "—"}, hidden:{" "}
                  {metaA.architecture.hidden_size ?? "—"})
                </div>
              </div>
              <div className="diff-box">
                <strong>Model B Metadata ({metaB.model_id})</strong>
                <div>Format: {metaB.format}</div>
                <div>Parameters: {metaB.total_parameters.toLocaleString()}</div>
                <div>Tensors: {metaB.tensor_count} ({metaB.shard_count} shard(s))</div>
                <div>Dtypes: {metaB.dtypes_present.join(", ")}</div>
                <div>
                  Architecture: {metaB.architecture.architecture_class ?? "—"} (type:{" "}
                  {metaB.architecture.model_type ?? "—"}, layers:{" "}
                  {metaB.architecture.num_hidden_layers ?? "—"}, hidden:{" "}
                  {metaB.architecture.hidden_size ?? "—"})
                </div>
              </div>
            </div>
          </div>

          <div className="panel">
            <h3 className="panel-title" style={{ marginBottom: 10 }}>
              Compatibility Diagnostic Findings ({comp.issues.length})
            </h3>
            {comp.issues.length === 0 ? (
              <div style={{ color: "var(--ok)" }}>
                Zero structural or architectural mismatches detected.
              </div>
            ) : (
              <div className="data-table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Severity</th>
                      <th>Code</th>
                      <th>Field</th>
                      <th>Model A</th>
                      <th>Model B</th>
                      <th>Diagnostic Message</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comp.issues.map((iss, idx) => (
                      <tr key={`${iss.code}-${idx}`}>
                        <td>
                          <span
                            className={`badge ${
                              iss.severity === "CRITICAL" || iss.severity === "HIGH"
                                ? "badge-danger"
                                : iss.severity === "MEDIUM"
                                  ? "badge-warn"
                                  : "badge-info"
                            }`}
                          >
                            {iss.severity}
                          </span>
                        </td>
                        <td className="mono">{iss.code}</td>
                        <td className="mono">{iss.field}</td>
                        <td className="mono">{JSON.stringify(iss.value_a)}</td>
                        <td className="mono">{JSON.stringify(iss.value_b)}</td>
                        <td>{iss.message}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};
