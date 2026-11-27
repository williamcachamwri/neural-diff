import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Download } from "lucide-react";
import { evolutionApi } from "../../api/evolutionClient";
import type { EvolutionRunBundle, TransitionTensorRecord } from "../../types/evolution";
import { formatCategoryShort, formatRole } from "../../utils/format";
import {
  DualTrajectoryChart,
  PixelMatrixHeatmap,
  PolarRadarChart,
  humanizeEnum,
} from "../charts/StudioVisuals";

const fmtFloat = (v: number | null | undefined, digits = 6): string =>
  v == null || Number.isNaN(v) ? "—" : v.toFixed(digits);

const fmtPct = (v: number | null | undefined, digits = 1): string =>
  v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

/* --------------------------------------------------------------------------
 * 1. Multi-Generation & Cumulative Weight Evolution Tab
 * -------------------------------------------------------------------------- */
export const WeightEvolutionSection: React.FC<{
  bundle: EvolutionRunBundle;
}> = ({ bundle }) => {
  const [scopeFilter, setScopeFilter] = useState<string>("all");
  const transitions = bundle.tensor_transitions ?? [];
  const filtered =
    scopeFilter === "all" ? transitions : transitions.filter((t) => t.scope === scopeFilter);

  const baseSeries = transitions.filter((t) => t.scope === "base_to_current");
  const maxBaseL2 = Math.max(1e-6, ...baseSeries.map((t) => t.l2_distance ?? 0));

  const selectedTopSource: TransitionTensorRecord | undefined =
    transitions.find((t) => t.scope === "consecutive" && t.top_k_modified_tensors.length > 0) ??
    transitions[0];

  return (
    <div data-testid="evo-section-weights">
      <div className="panel">
        <div className="panel-header">
          <div>
            <h3 className="panel-title">
              Direct Cumulative Weight Distance (Base → Current) vs Step Summation
            </h3>
            <div className="panel-subtitle">
              Computed directly against the original base checkpoint tensors (never summing step L2)
            </div>
          </div>
        </div>

        {baseSeries.length > 0 ? (
          <>
            <DualTrajectoryChart
              xLabels={baseSeries.map((r) => r.target_label)}
              primary={baseSeries.map((r) => r.l2_distance)}
              primaryLabel="Direct ||W_current − W_base||_2"
              secondary={baseSeries.map((r) => r.sum_of_step_l2_along_path ?? r.l2_distance)}
              secondaryLabel="Sum of Step L2"
              yMin={0}
              yMax={
                Math.max(
                  1,
                  ...baseSeries.map((r) =>
                    Math.max(r.l2_distance ?? 0, r.sum_of_step_l2_along_path ?? 0)
                  )
                ) * 1.08
              }
              height={186}
            />
            <svg
              viewBox={`0 0 920 ${Math.max(70, baseSeries.length * 32 + 24)}`}
              width="100%"
              style={{ display: "block", marginTop: 12, marginBottom: 14 }}
            >
            {baseSeries.map((row, idx) => {
              const l2 = row.l2_distance ?? 0;
              const sumStep = row.sum_of_step_l2_along_path ?? l2;
              const scaleMax = Math.max(maxBaseL2, sumStep, 1e-6);
              const wDirect = Math.max(3, Math.round((l2 / scaleMax) * 320));
              const wSum = Math.max(3, Math.round((sumStep / scaleMax) * 320));
              const y = 14 + idx * 32;
              return (
                <g key={row.target_checkpoint_id}>
                  <text
                    x={4}
                    y={y + 13}
                    fill="#e4e4e7"
                    fontSize={11.5}
                    fontFamily="JetBrains Mono, monospace"
                  >
                    {row.target_label.slice(0, 22)}
                  </text>
                  <rect
                    x={178}
                    y={y + 1}
                    width={wSum}
                    height={7}
                    fill="rgba(255,255,255,0.16)"
                    rx={2}
                  />
                  <rect x={178} y={y + 10} width={wDirect} height={9} fill="#d6c5a9" rx={2} />
                  <text
                    x={186 + Math.max(wDirect, wSum)}
                    y={y + 14}
                    fill="#a1a1aa"
                    fontSize={11}
                    fontFamily="JetBrains Mono, monospace"
                  >
                    Direct L2={l2.toFixed(5)} · Cosine={fmtFloat(row.global_cosine_similarity, 5)}
                    {row.path_cancellation_ratio != null && row.source_checkpoint_id !== row.target_checkpoint_id
                      ? ` · Direct/Sum=${row.path_cancellation_ratio.toFixed(3)}`
                      : ""}
                  </text>
                </g>
              );
            })}
            </svg>
          </>
        ) : (
          <div className="panel-subtitle">No base-to-current transitions available.</div>
        )}
      </div>

      <div className="panel">
        <div className="panel-header">
          <div>
            <h3 className="panel-title">Multi-Generation Tensor Transitions</h3>
            <div className="panel-subtitle">
              Consecutive edges, direct base-to-current distances, and branch-to-branch diffs
            </div>
          </div>
          <select
            aria-label="Transition Scope Filter"
            className="studio-run-select"
            value={scopeFilter}
            onChange={(e) => setScopeFilter(e.target.value)}
          >
            <option value="all">All scopes ({transitions.length})</option>
            <option value="consecutive">Consecutive (parent → child)</option>
            <option value="base_to_current">Base → Current (direct)</option>
            <option value="branch_to_branch">Branch ↔ Branch</option>
          </select>
        </div>

        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Scope</th>
                <th>Transition</th>
                <th>Stage</th>
                <th className="num">L2 Dist</th>
                <th className="num">Rel L2</th>
                <th className="num">RMSE</th>
                <th className="num">Max |Δ|</th>
                <th className="num">Cosine</th>
                <th className="num">Mod %</th>
                <th className="num">0→≠0 / ≠0→0</th>
                <th className="num">Sparsity Δ</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((t, i) => (
                <tr key={`${t.scope}-${t.source_checkpoint_id}-${t.target_checkpoint_id}-${i}`}>
                  <td style={{ color: "var(--text-secondary)", whiteSpace: "nowrap" }}>
                    {humanizeEnum(t.scope)}
                  </td>
                  <td className="mono">
                    {t.source_label} → {t.target_label}
                  </td>
                  <td>{t.transformation_type}</td>
                  <td className="num">{fmtFloat(t.l2_distance, 5)}</td>
                  <td className="num">{fmtFloat(t.relative_l2_distance, 6)}</td>
                  <td className="num">{fmtFloat(t.rmse, 6)}</td>
                  <td className="num">{fmtFloat(t.max_abs_diff, 5)}</td>
                  <td className="num">{fmtFloat(t.global_cosine_similarity, 6)}</td>
                  <td className="num">{t.modified_parameter_pct.toFixed(2)}%</td>
                  <td className="num">
                    {t.zero_to_nonzero_transitions} / {t.nonzero_to_zero_transitions}
                  </td>
                  <td className="num">{fmtFloat(t.sparsity_delta, 5)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {selectedTopSource && selectedTopSource.top_k_modified_tensors.length > 0 && (
        <div className="panel">
          <div className="panel-header">
            <div>
              <h3 className="panel-title">
                Top-K Most Modified Tensors ({selectedTopSource.source_label} →{" "}
                {selectedTopSource.target_label})
              </h3>
              <div className="panel-subtitle">Ranked by exact float64 Relative L2 distance</div>
            </div>
          </div>
          <div className="data-table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Tensor</th>
                  <th>Role</th>
                  <th className="num">Rel L2</th>
                  <th className="num">Max |Δ|</th>
                  <th className="num">Cosine</th>
                  <th className="num">Changed %</th>
                </tr>
              </thead>
              <tbody>
                {selectedTopSource.top_k_modified_tensors.map((rec) => (
                  <tr key={rec.name}>
                    <td className="mono">{rec.name}</td>
                    <td style={{ color: "var(--text-secondary)" }}>
                      {formatRole(rec.canonical_role)}
                    </td>
                    <td className="num">{fmtFloat(rec.relative_l2, 6)}</td>
                    <td className="num">{fmtFloat(rec.max_abs_diff, 5)}</td>
                    <td className="num">{fmtFloat(rec.cosine_similarity, 6)}</td>
                    <td className="num">{rec.changed_pct.toFixed(2)}%</td>
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

/* --------------------------------------------------------------------------
 * 2. Dedicated Layer Evolution Heatmap Tab
 * -------------------------------------------------------------------------- */
export const LayerEvolutionSection: React.FC<{ bundle: EvolutionRunBundle }> = ({ bundle }) => {
  const le = bundle.layer_evolution;
  if (!le || le.layer_indices.length === 0) {
    return (
      <div className="panel" data-testid="evo-section-layers">
        <div className="panel-subtitle">No layered transformer parameters recorded in this run.</div>
      </div>
    );
  }

  const rows = le.heatmap.rows;
  const lids = le.heatmap.layer_indices;
  const maxHeat = Math.max(
    1e-6,
    ...rows.flatMap((r) => r.relative_l2_by_layer.map((v) => v ?? 0))
  );

  return (
    <div data-testid="evo-section-layers">
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">Most Modified Layers</div>
          <div className="kpi-value mono">
            {le.most_modified_layers.length > 0
              ? le.most_modified_layers.map((l) => `L${l}`).join(", ")
              : "—"}
          </div>
          <div className="kpi-sub">Highest peak transition Relative L2</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Least Modified Layers</div>
          <div className="kpi-value mono">
            {le.least_modified_layers.length > 0
              ? le.least_modified_layers.map((l) => `L${l}`).join(", ")
              : "—"}
          </div>
          <div className="kpi-sub">Lowest mean Relative L2</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Anomalous Outliers</div>
          <div className="kpi-value mono">
            {le.unusually_large_change_layers.length > 0
              ? le.unusually_large_change_layers.map((x) => `L${x.layer_index}`).join(", ")
              : "None"}
          </div>
          <div className="kpi-sub">Statistical z-score &gt; 1.25σ</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Consistently Drifting</div>
          <div className="kpi-value mono">
            {le.consistently_changing_weight_layers.length > 0
              ? le.consistently_changing_weight_layers.map((l) => `L${l}`).join(", ")
              : "—"}
          </div>
          <div className="kpi-sub">Modified across ≥75% transitions</div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-header">
          <div>
            <h3 className="panel-title">Layer Weight Evolution Matrix (Transitions × Layers)</h3>
            <div className="panel-subtitle">
              Peak transition Relative L2 = {maxHeat.toFixed(6)}
            </div>
          </div>
        </div>

        <PixelMatrixHeatmap
          rowLabels={rows.map((r) => r.transition_label)}
          colLabels={lids.map((lid) => `L${lid}`)}
          matrix={rows.map((r) => r.relative_l2_by_layer)}
          minVal={0}
          maxVal={maxHeat}
        />

        <div className="data-table-wrap" style={{ marginTop: 14 }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Transition</th>
                {lids.map((lid) => (
                  <th key={lid} className="num">
                    L{lid}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={`${r.source_checkpoint_id}-${r.target_checkpoint_id}`}>
                  <td className="mono">{r.transition_label}</td>
                  {r.relative_l2_by_layer.map((val, idx) => {
                    const alpha = Math.min(0.65, Math.max(0.03, (val / maxHeat) * 0.65));
                    return (
                      <td
                        key={lids[idx]}
                        className="num"
                        style={{
                          background: `rgba(214, 197, 169, ${alpha.toFixed(3)})`,
                          color: "#f4f4f5",
                        }}
                      >
                        {val.toFixed(4)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="panel">
        <div className="panel-header">
          <div>
            <h3 className="panel-title">Canonical Component Breakdown Across Transitions</h3>
            <div className="panel-subtitle">
              Attention projections, MLP blocks, normalization, embeddings, and LM head
            </div>
          </div>
        </div>
        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Scope</th>
                <th>Transition</th>
                <th className="num">Attention Rel L2</th>
                <th className="num">MLP Rel L2</th>
                <th className="num">Norm Rel L2</th>
                <th className="num">Embedding Rel L2</th>
                <th className="num">Output Head Rel L2</th>
              </tr>
            </thead>
            <tbody>
              {le.transition_profiles.map((p, idx) => (
                <tr key={`${p.scope}-${p.source_checkpoint_id}-${p.target_checkpoint_id}-${idx}`}>
                  <td style={{ color: "var(--text-secondary)" }}>{humanizeEnum(p.scope)}</td>
                  <td className="mono">
                    {p.source_label} → {p.target_label}
                  </td>
                  <td className="num">
                    {fmtFloat(p.attention_projection_changes.mean_relative_l2 as number | null, 5)}
                  </td>
                  <td className="num">
                    {fmtFloat(p.mlp_changes.mean_relative_l2 as number | null, 5)}
                  </td>
                  <td className="num">
                    {fmtFloat(p.normalization_changes.mean_relative_l2 as number | null, 5)}
                  </td>
                  <td className="num">
                    {fmtFloat(p.embedding_changes.mean_relative_l2 as number | null, 5)}
                  </td>
                  <td className="num">
                    {fmtFloat(p.output_head_changes.mean_relative_l2 as number | null, 5)}
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

/* --------------------------------------------------------------------------
 * 3. Behavioral Evolution Chart & Capability Ledger Tab
 * -------------------------------------------------------------------------- */
export const BehavioralEvolutionSection: React.FC<{ bundle: EvolutionRunBundle }> = ({
  bundle,
}) => {
  const beh = bundle.behavioral_evolution;
  if (!beh || beh.timeline.length === 0) {
    return (
      <div className="panel" data-testid="evo-section-behavioral">
        <div className="panel-subtitle">
          Behavioral evaluation was skipped or checkpoints lacked generation config/tokenizers.
        </div>
      </div>
    );
  }

  const allCategories = Array.from(
    new Set(beh.timeline.flatMap((t) => Object.keys(t.category_scores)))
  ).sort();

  const radarSeries = (() => {
    if (allCategories.length < 3 || beh.timeline.length === 0) return [];
    const picks = [beh.timeline[0]];
    if (beh.timeline.length > 2) {
      const peakMid = beh.timeline
        .slice(1, -1)
        .reduce((best, cur) => (cur.pass_rate > best.pass_rate ? cur : best), beh.timeline[1]);
      picks.push(peakMid);
    }
    if (beh.timeline.length > 1) {
      picks.push(beh.timeline[beh.timeline.length - 1]);
    }
    const palette = ["#71717a", "#4ade80", "#d6c5a9"];
    return picks.map((item, idx) => ({
      label: item.version_label,
      values: allCategories.map((c) => item.category_scores[c] ?? 0),
      color: palette[idx % palette.length],
    }));
  })();

  return (
    <div data-testid="evo-section-behavioral">
      <div className="panel">
        <div className="panel-header">
          <div>
            <h3 className="panel-title">
              Multi-Generation Capability &amp; Pass-Rate Trajectory ({beh.suite_id}@{beh.suite_version})
            </h3>
            <div className="panel-subtitle">
              Deterministic programmatic suites · n={beh.sample_size} paired cases · 95% Wilson CI
            </div>
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: radarSeries.length > 0 ? "1.45fr 1fr" : "1fr",
            gap: 16,
            alignItems: "center",
            marginBottom: 16,
          }}
        >
          <DualTrajectoryChart
            xLabels={beh.timeline.map((t) => t.version_label)}
            primary={beh.timeline.map((t) => t.pass_rate)}
            primaryLabel="Pass Rate"
            secondary={beh.timeline.map((t) => t.mean_score)}
            secondaryLabel="Mean Rubric Score"
            ciBands={beh.timeline.map((t) => t.pass_rate_wilson_ci95)}
            yMin={0}
            yMax={1}
            height={210}
          />
          {radarSeries.length > 0 && (
            <PolarRadarChart
              axes={allCategories.map((c) => humanizeEnum(c))}
              series={radarSeries}
              size={258}
            />
          )}
        </div>

        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Version</th>
                <th>Stage</th>
                <th className="num">Pass Rate</th>
                <th className="num">Mean Score</th>
                {allCategories.map((cat) => (
                  <th key={cat} className="num">
                    {formatCategoryShort(cat)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {beh.timeline.map((row) => (
                <tr key={row.checkpoint_id}>
                  <td className="mono">{row.version_label}</td>
                  <td>{row.transformation_type}</td>
                  <td className="num">{fmtPct(row.pass_rate)}</td>
                  <td className="num">{row.mean_score.toFixed(3)}</td>
                  {allCategories.map((cat) => (
                    <td key={cat} className="num">
                      {row.category_scores[cat] != null
                        ? row.category_scores[cat].toFixed(2)
                        : "—"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="panel">
        <div className="panel-header">
          <div>
            <h3 className="panel-title">Paired Transition Deltas, Consistency & Surviving Regressions</h3>
            <div className="panel-subtitle">
              Exact McNemar p-values, output consistency, newly introduced failures, and surviving drops
            </div>
          </div>
        </div>
        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Transition</th>
                <th>Stage</th>
                <th className="num">Pass Δ</th>
                <th className="num">Score Δ</th>
                <th className="num">McNemar p</th>
                <th className="num">Output Consistency</th>
                <th>Newly Failed Cases</th>
                <th>Newly Passed Cases</th>
              </tr>
            </thead>
            <tbody>
              {beh.consecutive_transitions.map((tr) => (
                <tr key={`${tr.source_checkpoint_id}-${tr.target_checkpoint_id}`}>
                  <td className="mono">
                    {tr.source_label} → {tr.target_label}
                  </td>
                  <td>{tr.transformation_type}</td>
                  <td
                    className="num"
                    style={{
                      color:
                        tr.paired_statistics.pass_rate_delta < 0
                          ? "var(--danger)"
                          : tr.paired_statistics.pass_rate_delta > 0
                            ? "var(--ok)"
                            : undefined,
                    }}
                  >
                    {tr.paired_statistics.pass_rate_delta >= 0 ? "+" : ""}
                    {(tr.paired_statistics.pass_rate_delta * 100).toFixed(1)}%
                  </td>
                  <td className="num">
                    {tr.paired_statistics.mean_score_delta >= 0 ? "+" : ""}
                    {tr.paired_statistics.mean_score_delta.toFixed(3)}
                  </td>
                  <td className="num">{tr.paired_statistics.mcnemar_p_value.toFixed(4)}</td>
                  <td className="num">{fmtPct(tr.output_consistency_rate)}</td>
                  <td className="mono">{tr.newly_introduced_failures.join(", ") || "—"}</td>
                  <td className="mono">{tr.newly_passed_cases.join(", ") || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

/* --------------------------------------------------------------------------
 * 4. Representation Evolution Matrix (CKA & Cosine) Tab
 * -------------------------------------------------------------------------- */
export const RepresentationEvolutionSection: React.FC<{ bundle: EvolutionRunBundle }> = ({
  bundle,
}) => {
  const rep = bundle.representation_evolution;
  if (!rep || rep.evolution_matrix.rows.length === 0) {
    return (
      <div className="panel" data-testid="evo-section-representations">
        <div className="panel-subtitle">
          Representation capture was skipped or no compatible forward-hook checkpoints were present.
        </div>
      </div>
    );
  }

  const lids = rep.evolution_matrix.layer_indices;
  const rows = rep.evolution_matrix.rows;

  return (
    <div data-testid="evo-section-representations">
      <div className="panel">
        <div className="panel-header">
          <div>
            <h3 className="panel-title">
              Representation Evolution Matrix (Centered Linear CKA vs Base Model)
            </h3>
            <div className="panel-subtitle">
              Pooled across {rep.prompt_count} diagnostic prompts · scale &amp; orthogonal invariant
            </div>
          </div>
        </div>

        <div style={{ marginBottom: 14 }}>
          <PixelMatrixHeatmap
            rowLabels={rows.map((r) => r.version_label)}
            colLabels={lids.map((lid) => `L${lid}`)}
            matrix={rows.map((r) => r.layer_cka)}
            minVal={0}
            maxVal={1}
          />
        </div>

        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Version</th>
                <th>Stage</th>
                <th className="num">Mean CKA</th>
                <th className="num">Drift (1−CKA)</th>
                <th className="num">Logit JS Div</th>
                {lids.map((lid) => (
                  <th key={lid} className="num">
                    L{lid}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.checkpoint_id}>
                  <td className="mono">{r.version_label}</td>
                  <td>{r.transformation_type}</td>
                  <td className="num">{fmtFloat(r.mean_cka_to_base, 4)}</td>
                  <td className="num">{fmtFloat(r.representation_drift, 4)}</td>
                  <td className="num">{fmtFloat(r.mean_logit_js_to_base, 5)}</td>
                  {r.layer_cka.map((val, idx) => {
                    const drift = val != null ? Math.max(0, 1 - val) : 0;
                    const bg =
                      val == null
                        ? "transparent"
                        : `rgba(214, 197, 169, ${Math.min(0.6, drift * 2.2).toFixed(3)})`;
                    return (
                      <td key={lids[idx]} className="num" style={{ background: bg }}>
                        {val != null ? val.toFixed(4) : "N/A"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="panel">
        <div className="panel-header">
          <div>
            <h3 className="panel-title">
              Step-to-Step Representation Drift &amp; Cross-Checkpoint Causal Patching
            </h3>
            <div className="panel-subtitle">
              Single-layer residual stream intervention across aligned checkpoints
            </div>
          </div>
        </div>
        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Transition</th>
                <th>Stage</th>
                <th className="num">Step CKA</th>
                <th className="num">Hidden Cosine</th>
                <th className="num">Logit JS</th>
                <th>Causal Patching Top Site</th>
              </tr>
            </thead>
            <tbody>
              {rep.consecutive_transitions.map((tr) => (
                <tr key={`${tr.source_checkpoint_id}-${tr.target_checkpoint_id}`}>
                  <td className="mono">
                    {tr.source_label} → {tr.target_label}
                  </td>
                  <td>{tr.transformation_type}</td>
                  <td className="num">{fmtFloat(tr.mean_cka, 5)}</td>
                  <td className="num">{fmtFloat(tr.mean_hidden_cosine, 5)}</td>
                  <td className="num">{fmtFloat(tr.mean_logit_js_divergence, 5)}</td>
                  <td className="mono">
                    {tr.causal_patching && tr.causal_patching.most_causal_layer != null
                      ? `L${tr.causal_patching.most_causal_layer} (recovery=${(
                          tr.causal_patching.peak_recovery_fraction * 100
                        ).toFixed(1)}%)`
                      : tr.compatibility_warnings[0] ?? "—"}
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

/* --------------------------------------------------------------------------
 * 5. Automatic Regression Origin Analysis Tab
 * -------------------------------------------------------------------------- */
export const RegressionOriginSection: React.FC<{ bundle: EvolutionRunBundle }> = ({ bundle }) => {
  const events = bundle.regression_events ?? [];
  return (
    <div data-testid="evo-section-regressions">
      {events.length === 0 ? (
        <div className="panel">
          <div className="panel-subtitle">
            Zero regressions exceeded configured pass-rate or category score drop thresholds.
          </div>
        </div>
      ) : (
        events.map((ev) => (
          <div key={ev.event_id} className="panel" data-testid={`reg-origin-card-${ev.event_id}`}>
            <div className="panel-header">
              <div>
                <h3 className="panel-title">
                  <span className="status-inline" style={{ marginRight: 10 }}>
                    <span className="dot dot-danger" />
                    <span>{humanizeEnum(ev.affected_category)}</span>
                  </span>
                  {ev.previous_label} → {ev.first_affected_label} ({ev.transformation_type})
                </h3>
                <div className="panel-subtitle">{ev.interpretation_note}</div>
                <div
                  className="mono"
                  style={{
                    marginTop: 6,
                    display: "flex",
                    flexWrap: "wrap",
                    gap: 14,
                    fontSize: 11.5,
                    color: "var(--text-secondary)",
                  }}
                >
                  <span>
                    behavioral_regression_observed:{" "}
                    <strong style={{ color: ev.behavioral_regression_observed ? "var(--danger)" : "var(--text-muted)" }}>
                      {String(Boolean(ev.behavioral_regression_observed))}
                    </strong>
                  </span>
                  <span>
                    score_regression_observed:{" "}
                    <strong style={{ color: ev.score_regression_observed ? "var(--text-primary)" : "var(--text-muted)" }}>
                      {String(Boolean(ev.score_regression_observed))}
                    </strong>
                  </span>
                  <span>
                    causal_intervention_verified:{" "}
                    <strong style={{ color: ev.causal_intervention_verified ? "var(--ok)" : "var(--text-muted)" }}>
                      {String(Boolean(ev.causal_intervention_verified))}
                    </strong>
                  </span>
                </div>
              </div>
              <span className="status-inline">
                <span
                  className={`dot ${
                    ev.causal_verification.causal_verification_exists ? "dot-ok" : "dot-ivory"
                  }`}
                />
                {humanizeEnum(ev.epistemic_status)}
              </span>
            </div>

            <div className="kpi-grid" style={{ marginBottom: 12 }}>
              <div className="kpi-card">
                <div className="kpi-label">
                  {ev.regression_kind ? humanizeEnum(ev.regression_kind) : "Pass Rate Shift"}
                </div>
                <div className="kpi-value mono" style={{ color: "var(--danger)", fontSize: 18 }}>
                  {fmtPct(ev.pass_rate_before)} → {fmtPct(ev.pass_rate_after)}
                </div>
                <div className="kpi-sub">
                  Score {ev.score_before.toFixed(3)} → {ev.score_after.toFixed(3)}
                </div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">Failed Test Cases</div>
                <div className="kpi-value mono" style={{ fontSize: 15 }}>
                  {ev.failed_test_case_ids.join(", ") || "Rubric score drop"}
                </div>
                <div className="kpi-sub">
                  Survives to leaf: {ev.survives_to_leaf ? "Yes" : "Recovered downstream"}
                </div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">Statistical Power (Holm-Bonferroni)</div>
                <div className="kpi-value mono" style={{ fontSize: 15 }}>
                  n={ev.statistical_confidence.sample_size} · p=
                  {(
                    ev.statistical_confidence.holm_bonferroni_p_value ??
                    ev.statistical_confidence.mcnemar_p_value
                  ).toFixed(3)}
                </div>
                <div className="kpi-sub">
                  {humanizeEnum(ev.statistical_confidence.confidence_level)}
                </div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">Mechanistic Attribution</div>
                <div className="kpi-value mono" style={{ fontSize: 15 }}>
                  {ev.causal_verification.causal_verification_exists
                    ? `Verified (L${ev.causal_verification.most_causal_layer})`
                    : ev.causal_verification.causal_support_exists
                      ? `Causal Support (L${ev.causal_verification.most_causal_layer})`
                      : "Correlational Only"}
                </div>
                <div className="kpi-sub">
                  Weight Rel L2: {fmtFloat(ev.weight_changes.relative_l2_distance, 5)} · CKA:{" "}
                  {fmtFloat(ev.activation_changes.mean_linear_cka, 4)}
                </div>
              </div>
            </div>

            {ev.correlated_layers.length > 0 && (
              <div className="data-table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Layer</th>
                      <th className="num">Weight Mean Rel L2</th>
                      <th className="num">Modified %</th>
                      <th className="num">Step Linear CKA</th>
                      <th>Epistemic Relationship</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ev.correlated_layers.map((cl) => (
                      <tr key={cl.layer_index}>
                        <td className="mono">Layer {cl.layer_index}</td>
                        <td className="num">{fmtFloat(cl.weight_mean_relative_l2, 6)}</td>
                        <td className="num">
                          {cl.weight_changed_pct != null
                            ? `${cl.weight_changed_pct.toFixed(2)}%`
                            : "—"}
                        </td>
                        <td className="num">{fmtFloat(cl.linear_cka, 5)}</td>
                        <td style={{ color: "var(--text-secondary)" }}>
                          {humanizeEnum(cl.evidence_relationship)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        ))
      )}
    </div>
  );
};

/* --------------------------------------------------------------------------
 * 6. Experiment Provenance, Reproducibility & Multi-Format Export Tab
 * -------------------------------------------------------------------------- */
export const ExperimentExportSection: React.FC<{
  bundle: EvolutionRunBundle;
  onRerun: (force: boolean) => void;
  running: boolean;
}> = ({ bundle, onRerun, running }) => {
  const [previewFmt, setPreviewFmt] = useState<"html" | "markdown" | "csv" | "json">("html");
  const [previewText, setPreviewText] = useState<string>("");
  const [loadingPreview, setLoadingPreview] = useState<boolean>(false);

  const handleLoadPreview = async (fmt: "html" | "markdown" | "csv" | "json") => {
    setPreviewFmt(fmt);
    setLoadingPreview(true);
    try {
      const txt = await evolutionApi.fetchExportPreview(bundle.run_id, fmt);
      setPreviewText(fmt === "html" ? txt : txt.slice(0, 12000));
    } catch {
      setPreviewText("");
    } finally {
      setLoadingPreview(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    setLoadingPreview(true);
    evolutionApi
      .fetchExportPreview(bundle.run_id, "html")
      .then((txt) => {
        if (!cancelled) {
          setPreviewFmt("html");
          setPreviewText(txt);
        }
      })
      .catch(() => {
        if (!cancelled) setPreviewText("");
      })
      .finally(() => {
        if (!cancelled) setLoadingPreview(false);
      });
    return () => {
      cancelled = true;
    };
  }, [bundle.run_id]);

  const prov = bundle.provenance;

  return (
    <div data-testid="evo-section-experiments">
      <div className="panel">
        <div className="panel-header">
          <div>
            <h3 className="panel-title">Experiment Reproducibility &amp; Provenance</h3>
            <div className="panel-subtitle">
              Content-addressed SHA-256 cache invalidates automatically on any checkpoint, edge,
              dataset, or tolerance mutation
            </div>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button
              type="button"
              className="btn"
              disabled={running}
              onClick={() => onRerun(false)}
            >
              Verify Cache Run
            </button>
            <button
              type="button"
              className="btn-studio-primary"
              disabled={running}
              onClick={() => onRerun(true)}
            >
              Force Recompute
            </button>
          </div>
        </div>

        <div className="data-table-wrap">
          <table className="data-table">
            <tbody>
              <tr>
                <th>Run ID</th>
                <td className="mono">{bundle.run_id}</td>
                <th>Selected Device</th>
                <td className="mono">{prov.selected_device}</td>
              </tr>
              <tr>
                <th>Content Cache Key (SHA-256)</th>
                <td className="mono">{bundle.cache_key}</td>
                <th>Random Seed</th>
                <td className="mono">{prov.random_seed}</td>
              </tr>
              <tr>
                <th>Dataset Fingerprint</th>
                <td className="mono">{prov.evaluation_dataset_fingerprint ?? "N/A"}</td>
                <th>Toolchain</th>
                <td className="mono">
                  PyTorch {prov.software_versions.torch} · Transformers{" "}
                  {prov.software_versions.transformers} · Safetensors{" "}
                  {prov.software_versions.safetensors}
                </td>
              </tr>
              <tr>
                <th>Started / Completed (UTC)</th>
                <td className="mono">
                  {prov.started_at_utc} → {prov.completed_at_utc ?? "running"}
                </td>
                <th>Warnings</th>
                <td className="mono">{bundle.warnings.length}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="panel" data-testid="evo-export-panel">
        <div className="panel-header">
          <div>
            <h3 className="panel-title">Report Artifacts (HTML · MD · PDF · CSV · JSON)</h3>
            <div className="panel-subtitle">
              Minimalist multi-format exports with embedded vector DAG, heatmaps, CKA matrices, and
              Mermaid graphs
            </div>
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {(
              [
                { fmt: "html", ext: "html", label: "HTML" },
                { fmt: "markdown", ext: "md", label: "MD" },
                { fmt: "pdf", ext: "pdf", label: "PDF" },
                { fmt: "csv", ext: "csv", label: "CSV" },
                { fmt: "json", ext: "json", label: "JSON" },
              ] as const
            ).map(({ fmt, ext, label }) => (
              <a
                key={fmt}
                className="btn"
                data-testid={`btn-evo-download-${fmt}`}
                href={evolutionApi.getExportUrl(bundle.run_id, fmt)}
                download={`${bundle.run_id}.${ext}`}
              >
                <Download size={12} />
                <span className="mono">{label}</span>
              </a>
            ))}
          </div>
        </div>

        <div className="segmented-bar" style={{ marginBottom: 12 }}>
          {(
            [
              { fmt: "html", label: "Visual HTML" },
              { fmt: "markdown", label: "Markdown" },
              { fmt: "csv", label: "CSV" },
              { fmt: "json", label: "JSON" },
            ] as const
          ).map(({ fmt, label }) => {
            const active = previewFmt === fmt;
            return (
              <button
                key={fmt}
                type="button"
                data-testid={`btn-evo-preview-${fmt}`}
                className={`segmented-btn ${active ? "active" : ""}`}
                style={{ background: "transparent", boxShadow: "none" }}
                onClick={() => void handleLoadPreview(fmt)}
              >
                {active && (
                  <motion.span
                    layoutId="evo-export-seg-pill"
                    className="studio-nav-pill"
                    transition={{ type: "spring", stiffness: 520, damping: 38 }}
                  />
                )}
                <span className="studio-nav-label">{label}</span>
              </button>
            );
          })}
        </div>

        {loadingPreview && <div className="panel-subtitle">Loading report preview…</div>}
        {previewText && !loadingPreview && previewFmt === "html" && (
          <iframe
            data-testid="evo-export-preview-content"
            title="Evolution Visual HTML Report Preview"
            sandbox=""
            srcDoc={previewText}
            style={{
              width: "100%",
              height: 540,
              border: "1px solid var(--border-subtle)",
              borderRadius: 12,
              background: "#09090b",
            }}
          />
        )}
        {previewText && !loadingPreview && previewFmt !== "html" && (
          <pre
            data-testid="evo-export-preview-content"
            className="mono"
            style={{
              maxHeight: 420,
              overflow: "auto",
              padding: 14,
              background: "#09090b",
              border: "1px solid var(--border-subtle)",
              borderRadius: 12,
              color: "var(--text-secondary)",
              fontSize: 11.5,
              margin: 0,
            }}
          >
            {previewText}
          </pre>
        )}
      </div>
    </div>
  );
};
