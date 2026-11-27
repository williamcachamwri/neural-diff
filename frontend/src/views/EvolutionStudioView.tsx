import React, { useCallback, useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { ArrowLeftRight, ChevronLeft, ChevronRight, Play, Plus } from "lucide-react";
import type { ApiClientError } from "../api/client";
import { evolutionApi } from "../api/evolutionClient";
import { DualTrajectoryChart, humanizeEnum } from "../components/charts/StudioVisuals";
import {
  BehavioralEvolutionSection,
  ExperimentExportSection,
  LayerEvolutionSection,
  RegressionOriginSection,
  RepresentationEvolutionSection,
  WeightEvolutionSection,
} from "../components/evolution/EvolutionAnalyticsTabs";
import { EvolutionDagCanvas } from "../components/evolution/EvolutionDagCanvas";
import type { DiscoveredModelsResponse } from "../types/api";
import { formatCategory } from "../utils/format";
import type {
  EvolutionCheckpoint,
  EvolutionOverviewResponse,
  EvolutionProjectDetailResponse,
  EvolutionProjectSummary,
  TransitionTensorRecord,
} from "../types/evolution";

type EvoTabId =
  | "overview"
  | "graph"
  | "registry"
  | "weights"
  | "layers"
  | "behavioral"
  | "representations"
  | "regressions"
  | "experiments";

const EVO_TABS: Array<{ id: EvoTabId; label: string }> = [
  { id: "overview", label: "Dashboard" },
  { id: "graph", label: "Timeline & DAG" },
  { id: "registry", label: "Checkpoints" },
  { id: "weights", label: "Weights" },
  { id: "layers", label: "Layers" },
  { id: "behavioral", label: "Behavior" },
  { id: "representations", label: "Representations" },
  { id: "regressions", label: "Regressions" },
  { id: "experiments", label: "Provenance & Export" },
];

function formatMetaSummary(meta: Record<string, unknown>, emptyLabel: string): string {
  const entries = Object.entries(meta).filter(
    ([, v]) => v !== null && v !== undefined && v !== ""
  );
  if (entries.length === 0) return emptyLabel;
  return entries
    .slice(0, 4)
    .map(([k, v]) => {
      const cleanKey = k.replace(/_/g, " ");
      if (Array.isArray(v)) return `${cleanKey}: ${v.join(", ")}`;
      if (typeof v === "object") return `${cleanKey}: ${Object.keys(v as object).length} fields`;
      return `${cleanKey}: ${String(v)}`;
    })
    .join(" · ");
}

export interface EvolutionStudioViewProps {
  discovered: DiscoveredModelsResponse | null;
  onError: (err: Error | ApiClientError) => void;
}

export const EvolutionStudioView: React.FC<EvolutionStudioViewProps> = ({
  discovered,
  onError,
}) => {
  const [activeTab, setActiveTab] = useState<EvoTabId>("overview");
  const [overview, setOverview] = useState<EvolutionOverviewResponse | null>(null);
  const [projects, setProjects] = useState<EvolutionProjectSummary[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string>("");
  const [detail, setDetail] = useState<EvolutionProjectDetailResponse | null>(null);
  const [selectedCheckpointId, setSelectedCheckpointId] = useState<string | null>(null);

  // Project creation form
  const [newProjName, setNewProjName] = useState<string>("");
  const [newProjDesc, setNewProjDesc] = useState<string>("");

  // Checkpoint registration form
  const [ckptSource, setCkptSource] = useState<string>("");
  const [ckptLabel, setCkptLabel] = useState<string>("");
  const [ckptTransform, setCkptTransform] = useState<string>("Base Model");
  const [ckptParents, setCkptParents] = useState<string[]>([]);
  const [ckptNotes, setCkptNotes] = useState<string>("");

  // Rename / edit selected checkpoint
  const [editLabel, setEditLabel] = useState<string>("");
  const [editNotes, setEditNotes] = useState<string>("");

  // DAG interactive controls & arbitrary 2-version comparator
  const [transformFilter, setTransformFilter] = useState<string>("");
  const [categoryFilter, setCategoryFilter] = useState<string>("");
  const [highlightRegressions, setHighlightRegressions] = useState<boolean>(true);
  const [highlightMajorWeights, setHighlightMajorWeights] = useState<boolean>(true);
  const [compareAId, setCompareAId] = useState<string>("");
  const [compareBId, setCompareBId] = useState<string>("");
  const [pairComparison, setPairComparison] = useState<{
    lcaLabel: string | null;
    result: TransitionTensorRecord;
  } | null>(null);

  const [busy, setBusy] = useState<boolean>(false);

  const loadOverviewAndProjects = useCallback(
    async (preferProjectId?: string) => {
      try {
        const [ov, projResp] = await Promise.all([
          evolutionApi.getOverview(),
          evolutionApi.listProjects(50, 0),
        ]);
        setOverview(ov);
        setProjects(projResp.projects);

        const targetId =
          preferProjectId ||
          selectedProjectId ||
          (projResp.projects.length > 0 ? projResp.projects[0].project_id : "");
        if (targetId) {
          setSelectedProjectId(targetId);
          const det = await evolutionApi.getProjectDetail(targetId);
          setDetail(det);
          if (det.checkpoints.length > 0) {
            const validIds = new Set(det.checkpoints.map((c) => c.checkpoint_id));
            setSelectedCheckpointId((prev) =>
              prev && validIds.has(prev) ? prev : det.checkpoints[0].checkpoint_id
            );
            setCompareAId((prev) =>
              prev && validIds.has(prev) ? prev : det.checkpoints[0].checkpoint_id
            );
            setCompareBId((prev) =>
              prev && validIds.has(prev) && prev !== det.checkpoints[0].checkpoint_id
                ? prev
                : det.checkpoints[det.checkpoints.length - 1].checkpoint_id
            );
          } else {
            setSelectedCheckpointId(null);
            setCompareAId("");
            setCompareBId("");
          }
        }
      } catch (err) {
        onError(err instanceof Error ? err : new Error(String(err)));
      }
    },
    [onError, selectedProjectId]
  );

  useEffect(() => {
    void loadOverviewAndProjects();
  }, [loadOverviewAndProjects]);

  const selectedCheckpoint: EvolutionCheckpoint | null = useMemo(() => {
    if (!detail || !selectedCheckpointId) return null;
    return detail.checkpoints.find((c) => c.checkpoint_id === selectedCheckpointId) ?? null;
  }, [detail, selectedCheckpointId]);

  useEffect(() => {
    if (selectedCheckpoint) {
      setEditLabel(selectedCheckpoint.version_label);
      setEditNotes(selectedCheckpoint.experiment_notes);
    }
  }, [selectedCheckpoint]);

  const handleSelectProject = async (pid: string) => {
    setSelectedProjectId(pid);
    setPairComparison(null);
    if (!pid) {
      setDetail(null);
      return;
    }
    try {
      const det = await evolutionApi.getProjectDetail(pid);
      setDetail(det);
      if (det.checkpoints.length > 0) {
        setSelectedCheckpointId(det.checkpoints[0].checkpoint_id);
        setCompareAId(det.checkpoints[0].checkpoint_id);
        setCompareBId(det.checkpoints[det.checkpoints.length - 1].checkpoint_id);
      } else {
        setSelectedCheckpointId(null);
      }
    } catch (err) {
      onError(err instanceof Error ? err : new Error(String(err)));
    }
  };

  const handleCreateProject = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newProjName.trim()) return;
    setBusy(true);
    try {
      const created = await evolutionApi.createProject({
        name: newProjName.trim(),
        description: newProjDesc.trim(),
      });
      setNewProjName("");
      setNewProjDesc("");
      await loadOverviewAndProjects(created.project_id);
    } catch (err) {
      onError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setBusy(false);
    }
  };

  const handleRegisterCheckpoint = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedProjectId || !ckptSource.trim()) return;
    setBusy(true);
    try {
      const saved = await evolutionApi.registerCheckpoint(selectedProjectId, {
        source: ckptSource.trim(),
        version_label: ckptLabel.trim() || undefined,
        parent_checkpoint_ids: ckptParents,
        transformation_type: ckptTransform,
        experiment_notes: ckptNotes.trim(),
      });
      setCkptLabel("");
      setCkptNotes("");
      setSelectedCheckpointId(saved.checkpoint_id);
      await loadOverviewAndProjects(selectedProjectId);
    } catch (err) {
      onError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setBusy(false);
    }
  };

  const handleUpdateSelectedCheckpoint = async () => {
    if (!selectedProjectId || !selectedCheckpoint) return;
    setBusy(true);
    try {
      await evolutionApi.updateCheckpoint(selectedProjectId, selectedCheckpoint.checkpoint_id, {
        version_label: editLabel.trim() || selectedCheckpoint.version_label,
        experiment_notes: editNotes,
      });
      await loadOverviewAndProjects(selectedProjectId);
    } catch (err) {
      onError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setBusy(false);
    }
  };

  const handleRemoveCheckpoint = async (cid: string) => {
    if (!selectedProjectId) return;
    setBusy(true);
    try {
      await evolutionApi.removeCheckpoint(selectedProjectId, cid);
      await loadOverviewAndProjects(selectedProjectId);
    } catch (err) {
      onError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setBusy(false);
    }
  };

  const handleRunAnalysis = async (forceRerun = false) => {
    if (!selectedProjectId) return;
    setBusy(true);
    try {
      await evolutionApi.runAnalysis(selectedProjectId, {
        include_tensor_evolution: true,
        include_layer_evolution: true,
        include_behavioral_evolution: true,
        include_representation_evolution: true,
        enable_causal_patching: true,
        force_rerun: forceRerun,
        allow_cache: !forceRerun,
      });
      await loadOverviewAndProjects(selectedProjectId);
    } catch (err) {
      onError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setBusy(false);
    }
  };

  const handleCompareSelectedVersions = async () => {
    if (!selectedProjectId || !compareAId || !compareBId) return;
    setBusy(true);
    try {
      const res = await evolutionApi.compareVersions(selectedProjectId, compareAId, compareBId);
      setPairComparison({
        lcaLabel: res.lowest_common_ancestor_label,
        result: res.comparison,
      });
    } catch (err) {
      onError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setBusy(false);
    }
  };

  const stepGeneration = (delta: -1 | 1) => {
    if (!detail || detail.graph.topological_order.length === 0) return;
    const order = detail.graph.topological_order;
    const idx = selectedCheckpointId ? order.indexOf(selectedCheckpointId) : 0;
    const nextIdx = Math.max(0, Math.min(order.length - 1, idx + delta));
    setSelectedCheckpointId(order[nextIdx]);
  };

  const activeRun = detail?.latest_analysis ?? null;
  const supportedTransforms = overview?.supported_transformations ?? [
    "Base Model",
    "SFT",
    "DPO",
    "RLHF",
    "LoRA",
    "QLoRA",
    "Model Merging",
    "Quantization",
    "Pruning",
    "Distillation",
    "Fine-Tuning",
    "Instruction Tuning",
    "Weight Modification",
    "Unknown Transformation",
  ];

  const evalCategories = useMemo(() => {
    const timeline = activeRun?.behavioral_evolution?.timeline ?? [];
    return Array.from(new Set(timeline.flatMap((t) => Object.keys(t.category_scores)))).sort();
  }, [activeRun]);

  const baseWeightSeries = useMemo(
    () => (activeRun?.tensor_transitions ?? []).filter((t) => t.scope === "base_to_current"),
    [activeRun]
  );
  const behTimeline = activeRun?.behavioral_evolution?.timeline ?? [];
  const repRows = activeRun?.representation_evolution?.evolution_matrix?.rows ?? [];

  return (
    <div data-testid="evolution-studio-view">
      {/* Project Toolbar & Sub-Tab Navigation */}
      <div
        className="panel"
        style={{
          padding: "12px 16px",
          marginBottom: 16,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <span className="kpi-label">Evolution Project</span>
          <select
            aria-label="Evolution Project Selector"
            data-testid="evo-project-selector"
            className="studio-run-select"
            value={selectedProjectId}
            onChange={(e) => void handleSelectProject(e.target.value)}
          >
            {projects.length === 0 && <option value="">No projects yet</option>}
            {projects.map((p) => (
              <option key={p.project_id} value={p.project_id}>
                {p.name} ({p.checkpoint_count} ckpts · {p.branch_count} branches)
              </option>
            ))}
          </select>

          {selectedProjectId && (
            <button
              type="button"
              data-testid="btn-run-evolution-analysis"
              className="btn-studio-primary"
              disabled={busy || (detail?.checkpoints.length ?? 0) === 0}
              onClick={() => void handleRunAnalysis(false)}
            >
              <Play size={11} fill="currentColor" />
              <span>{busy ? "Analyzing…" : "Run Evolution Analysis"}</span>
            </button>
          )}
        </div>

        {/* Sub-Tab Ribbon */}
        <div
          role="tablist"
          aria-label="Evolution Analytics Sections"
          style={{ display: "flex", alignItems: "center", gap: 4, flexWrap: "wrap" }}
        >
          {EVO_TABS.map((tab) => {
            const isAct = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                role="tab"
                aria-selected={isAct}
                data-testid={`evo-tab-${tab.id}`}
                className={`studio-nav-item ${isAct ? "active" : ""}`}
                onClick={() => setActiveTab(tab.id)}
              >
                {isAct && (
                  <motion.span
                    layoutId="evo-subtab-pill"
                    className="studio-nav-pill"
                    transition={{ type: "spring", stiffness: 500, damping: 38 }}
                  />
                )}
                <span className="studio-nav-label">{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* 1. OVERVIEW & INSIGHTS DASHBOARD */}
      {activeTab === "overview" && (
        <div data-testid="evo-section-overview">
          <div className="kpi-grid">
            <div className="kpi-card">
              <div className="kpi-label">Evolution Projects</div>
              <div className="kpi-value">{overview?.summary.total_projects ?? 0}</div>
              <div className="kpi-sub">
                {overview?.summary.completed_analysis_runs ?? 0} reproducible runs
              </div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Registered Checkpoints</div>
              <div className="kpi-value">
                {detail?.checkpoints.length ?? overview?.summary.total_checkpoints ?? 0}
              </div>
              <div className="kpi-sub">SHA-256 content-fingerprinted</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Transformations</div>
              <div className="kpi-value">
                {detail?.graph.edges.length ?? overview?.summary.total_transformations ?? 0}
              </div>
              <div className="kpi-sub">Parent → child &amp; merge edges</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">DAG Branches</div>
              <div className="kpi-value">
                {detail?.graph.branch_count ?? overview?.summary.total_branches ?? 0}
              </div>
              <div className="kpi-sub">Distinct root-to-leaf lineages</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Detected Regressions</div>
              <div
                className="kpi-value"
                style={{
                  color:
                    (activeRun?.regression_events.length ?? 0) > 0 ? "var(--danger)" : undefined,
                }}
              >
                {activeRun?.regression_events.length ??
                  overview?.summary.total_detected_regressions ??
                  0}
              </div>
              <div className="kpi-sub">Pinpointed to first transition</div>
            </div>
          </div>

          {/* 3-Panel Visual Trajectory Triptych */}
          {activeRun && (baseWeightSeries.length > 0 || behTimeline.length > 0) && (
            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  behTimeline.length > 0 && repRows.length > 0 ? "repeat(3, minmax(0, 1fr))" : "1fr",
                gap: 14,
                marginBottom: 16,
              }}
            >
              {baseWeightSeries.length > 0 && (
                <div className="panel" style={{ marginBottom: 0 }}>
                  <div className="panel-header">
                    <h3 className="panel-title">Direct Cumulative Weight Δ</h3>
                    <span className="panel-subtitle mono">||W_current − W_base||_2</span>
                  </div>
                  <DualTrajectoryChart
                    xLabels={baseWeightSeries.map((t) => t.target_label)}
                    primary={baseWeightSeries.map((t) => t.l2_distance)}
                    primaryLabel="Direct L2"
                    secondary={baseWeightSeries.map(
                      (t) => t.sum_of_step_l2_along_path ?? t.l2_distance
                    )}
                    secondaryLabel="Step Sum"
                    yMin={0}
                    yMax={
                      Math.max(
                        1,
                        ...baseWeightSeries.map((t) =>
                          Math.max(t.l2_distance ?? 0, t.sum_of_step_l2_along_path ?? 0)
                        )
                      ) * 1.08
                    }
                    height={176}
                  />
                </div>
              )}

              {behTimeline.length > 0 && (
                <div className="panel" style={{ marginBottom: 0 }}>
                  <div className="panel-header">
                    <h3 className="panel-title">Capability Trajectory</h3>
                    <span className="panel-subtitle mono">Pass rate + 95% CI</span>
                  </div>
                  <DualTrajectoryChart
                    xLabels={behTimeline.map((b) => b.version_label)}
                    primary={behTimeline.map((b) => b.pass_rate)}
                    primaryLabel="Pass Rate"
                    secondary={behTimeline.map((b) => b.mean_score)}
                    secondaryLabel="Mean Score"
                    ciBands={behTimeline.map((b) => b.pass_rate_wilson_ci95)}
                    yMin={0}
                    yMax={1}
                    height={176}
                  />
                </div>
              )}

              {repRows.length > 0 && (
                <div className="panel" style={{ marginBottom: 0 }}>
                  <div className="panel-header">
                    <h3 className="panel-title">Representation Alignment</h3>
                    <span className="panel-subtitle mono">Centered Linear CKA</span>
                  </div>
                  <DualTrajectoryChart
                    xLabels={repRows.map((r) => r.version_label)}
                    primary={repRows.map((r) => r.mean_cka_to_base)}
                    primaryLabel="Mean CKA"
                    secondary={repRows.map((r) => r.representation_drift)}
                    secondaryLabel="Drift (1−CKA)"
                    yMin={0}
                    yMax={1}
                    height={176}
                  />
                </div>
              )}
            </div>
          )}

          {/* Evidence-Backed Evolution Insights */}
          <div className="panel">
            <div className="panel-header">
              <div>
                <h3 className="panel-title">Empirical Evolution Findings</h3>
                <div className="panel-subtitle">
                  Threshold-verified weight, representation, and rubric signals
                </div>
              </div>
            </div>

            {activeRun && activeRun.insights.length > 0 ? (
              <div className="data-table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Domain</th>
                      <th>Evidence</th>
                      <th>Observation</th>
                      <th>Scope</th>
                    </tr>
                  </thead>
                  <tbody>
                    {activeRun.insights.map((ins) => (
                      <tr key={ins.insight_id}>
                        <td style={{ color: "var(--text-secondary)", whiteSpace: "nowrap" }}>
                          {humanizeEnum(ins.domain)}
                        </td>
                        <td style={{ whiteSpace: "nowrap" }}>
                          <span className="status-inline">
                            <span
                              className={`dot ${
                                ins.domain === "regression_origin" ? "dot-danger" : "dot-ivory"
                              }`}
                            />
                            {humanizeEnum(ins.epistemic_status)}
                          </span>
                        </td>
                        <td>
                          {ins.statement.replace(/'([a-z_]+)'/g, (_m, k: string) =>
                            formatCategory(k)
                          )}
                        </td>
                        <td className="mono" style={{ color: "var(--text-muted)" }}>
                          {ins.evaluation_cases.length > 0
                            ? ins.evaluation_cases.join(", ")
                            : `${ins.checkpoints.length} ckpts`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="panel-subtitle">
                Select or create an evolution project and execute analysis to populate empirical
                insights.
              </div>
            )}
          </div>

          {/* Create New Evolution Project */}
          <div className="panel">
            <div className="panel-header">
              <div>
                <h3 className="panel-title">Create Evolution Project</h3>
                <div className="panel-subtitle">
                  Organize Base → SFT → DPO → Quantization / Pruning / Merging lifecycles
                </div>
              </div>
            </div>
            <form
              onSubmit={(e) => void handleCreateProject(e)}
              style={{ display: "grid", gridTemplateColumns: "1fr 1.6fr auto", gap: 10 }}
            >
              <input
                type="text"
                data-testid="input-new-project-name"
                placeholder="Project name (e.g. SmolLM2 Post-Training DAG)"
                value={newProjName}
                onChange={(e) => setNewProjName(e.target.value)}
                className="studio-run-select"
                style={{ width: "100%" }}
              />
              <input
                type="text"
                data-testid="input-new-project-desc"
                placeholder="Lifecycle hypothesis or description"
                value={newProjDesc}
                onChange={(e) => setNewProjDesc(e.target.value)}
                className="studio-run-select"
                style={{ width: "100%" }}
              />
              <button
                type="submit"
                data-testid="btn-create-project"
                className="btn-studio-primary"
                disabled={busy || !newProjName.trim()}
              >
                <Plus size={12} />
                <span>Create Project</span>
              </button>
            </form>
          </div>
        </div>
      )}

      {/* 2. INTERACTIVE EVOLUTION GRAPH, TIMELINE & ARBITRARY PAIR COMPARISON */}
      {activeTab === "graph" && detail && (
        <div data-testid="evo-section-graph">
          <div className="panel">
            <div className="panel-header">
              <div>
                <h3 className="panel-title">
                  Evolution Lineage Graph ({detail.project.name})
                </h3>
                <div className="panel-subtitle">
                  Branch topology, merge weights, and cross-generation pair selection
                </div>
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                <select
                  aria-label="Filter by Transformation Type"
                  data-testid="evo-filter-transformation"
                  className="studio-run-select"
                  value={transformFilter}
                  onChange={(e) => setTransformFilter(e.target.value)}
                >
                  <option value="">All transformations</option>
                  {supportedTransforms.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>

                <select
                  aria-label="Filter by Evaluation Category"
                  data-testid="evo-filter-category"
                  className="studio-run-select"
                  value={categoryFilter}
                  onChange={(e) => setCategoryFilter(e.target.value)}
                >
                  <option value="">Overall pass rate</option>
                  {evalCategories.map((c) => (
                    <option key={c} value={c}>
                      {formatCategory(c)}
                    </option>
                  ))}
                </select>

                <button
                  type="button"
                  className="btn"
                  onClick={() => setHighlightRegressions((v) => !v)}
                  style={{
                    borderColor: highlightRegressions ? "rgba(248,113,113,0.45)" : undefined,
                  }}
                >
                  Regressions: {highlightRegressions ? "On" : "Off"}
                </button>

                <button
                  type="button"
                  className="btn"
                  onClick={() => setHighlightMajorWeights((v) => !v)}
                  style={{
                    borderColor: highlightMajorWeights ? "rgba(214,197,169,0.45)" : undefined,
                  }}
                >
                  Peak ΔW: {highlightMajorWeights ? "On" : "Off"}
                </button>

                <button
                  type="button"
                  className="btn"
                  title="Previous generation"
                  onClick={() => stepGeneration(-1)}
                >
                  <ChevronLeft size={13} />
                </button>
                <button
                  type="button"
                  className="btn"
                  title="Next generation"
                  onClick={() => stepGeneration(1)}
                >
                  <ChevronRight size={13} />
                </button>
              </div>
            </div>

            <EvolutionDagCanvas
              graph={detail.graph}
              tensorTransitions={activeRun?.tensor_transitions ?? []}
              behavioralTimeline={activeRun?.behavioral_evolution?.timeline ?? []}
              regressions={activeRun?.regression_events ?? []}
              selectedCheckpointId={selectedCheckpointId}
              compareAId={compareAId}
              compareBId={compareBId}
              transformationFilter={transformFilter}
              categoryFilter={categoryFilter}
              highlightRegressions={highlightRegressions}
              highlightMajorWeights={highlightMajorWeights}
              onSelectCheckpoint={(cid) => setSelectedCheckpointId(cid)}
              onSelectCompareA={setCompareAId}
              onSelectCompareB={setCompareBId}
            />
          </div>

          {/* Arbitrary Version Pair Comparator */}
          <div className="panel" data-testid="evo-arbitrary-compare-panel">
            <div className="panel-header">
              <div>
                <h3 className="panel-title">Compare Arbitrary or Cross-Branch Versions (A ↔ B)</h3>
                <div className="panel-subtitle">
                  Direct chunked float64 comparison across consecutive, base-to-current, or
                  diverging branch checkpoints
                </div>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <select
                  aria-label="Version A"
                  data-testid="evo-compare-select-a"
                  className="studio-run-select"
                  value={compareAId}
                  onChange={(e) => setCompareAId(e.target.value)}
                >
                  {detail.checkpoints.map((c) => (
                    <option key={c.checkpoint_id} value={c.checkpoint_id}>
                      A: {c.version_label}
                    </option>
                  ))}
                </select>
                <select
                  aria-label="Version B"
                  data-testid="evo-compare-select-b"
                  className="studio-run-select"
                  value={compareBId}
                  onChange={(e) => setCompareBId(e.target.value)}
                >
                  {detail.checkpoints.map((c) => (
                    <option key={c.checkpoint_id} value={c.checkpoint_id}>
                      B: {c.version_label}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  data-testid="btn-evo-compare-pair"
                  className="btn-studio-primary"
                  disabled={busy || !compareAId || !compareBId}
                  onClick={() => void handleCompareSelectedVersions()}
                >
                  <ArrowLeftRight size={12} />
                  <span>Compare Pair</span>
                </button>
              </div>
            </div>

            {pairComparison && (
              <div className="kpi-grid" style={{ marginBottom: 0 }}>
                <div className="kpi-card">
                  <div className="kpi-label">Scope · Common Ancestor</div>
                  <div className="kpi-value" style={{ fontSize: 15 }}>
                    {humanizeEnum(pairComparison.result.scope)}
                  </div>
                  <div className="kpi-sub">LCA: {pairComparison.lcaLabel ?? "None"}</div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label">Direct L2 · Rel L2</div>
                  <div className="kpi-value mono" style={{ fontSize: 16 }}>
                    {pairComparison.result.l2_distance?.toFixed(5) ?? "—"}
                  </div>
                  <div className="kpi-sub">
                    Rel L2: {pairComparison.result.relative_l2_distance?.toFixed(6) ?? "—"}
                  </div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label">Global Cosine</div>
                  <div className="kpi-value mono" style={{ fontSize: 16 }}>
                    {pairComparison.result.global_cosine_similarity?.toFixed(6) ?? "—"}
                  </div>
                  <div className="kpi-sub">
                    RMSE: {pairComparison.result.rmse?.toFixed(6) ?? "—"}
                  </div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label">Modified Params</div>
                  <div className="kpi-value mono" style={{ fontSize: 16 }}>
                    {pairComparison.result.modified_parameter_pct.toFixed(2)}%
                  </div>
                  <div className="kpi-sub">
                    +{pairComparison.result.added_tensors.length} / −
                    {pairComparison.result.removed_tensors.length} tensors
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Complete Branch Lineage Timeline Table */}
          <div className="panel" data-testid="evo-timeline-table">
            <div className="panel-header">
              <div>
                <h3 className="panel-title">Evolution Branch Lineages &amp; Generation Timeline</h3>
                <div className="panel-subtitle">
                  {detail.graph.branch_count} root-to-leaf branch paths across{" "}
                  {detail.graph.nodes.length} checkpoints
                </div>
              </div>
            </div>
            <div className="data-table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Branch</th>
                    <th>Lineage Sequence</th>
                    <th>Merge Node</th>
                    <th className="num">Stages</th>
                  </tr>
                </thead>
                <tbody>
                  {detail.graph.branches.map((b) => (
                    <tr key={b.branch_id}>
                      <td className="mono">{b.branch_id}</td>
                      <td className="mono">{b.version_labels.join("  →  ")}</td>
                      <td>{b.contains_merge ? "Multi-parent merge" : "Single-parent chain"}</td>
                      <td className="num">{b.checkpoint_ids.length}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* 3. CHECKPOINT REGISTRY & DETAILS INSPECTOR */}
      {activeTab === "registry" && detail && (
        <div data-testid="evo-section-registry">
          <div className="panel">
            <div className="panel-header">
              <div>
                <h3 className="panel-title">Register Checkpoint into Evolution DAG</h3>
                <div className="panel-subtitle">
                  Supports Safetensors, safe PyTorch weights, Hugging Face directories, and PEFT
                  LoRA adapters with automatic SHA-256 duplicate &amp; cycle checks
                </div>
              </div>
            </div>

            <form
              onSubmit={(e) => void handleRegisterCheckpoint(e)}
              style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10 }}
            >
              <input
                type="text"
                list="evo-model-suggestions"
                data-testid="input-ckpt-source"
                placeholder="Local checkpoint path or Hugging Face ID"
                value={ckptSource}
                onChange={(e) => setCkptSource(e.target.value)}
                className="studio-run-select"
                style={{ width: "100%" }}
              />
              <datalist id="evo-model-suggestions">
                {discovered?.local_models.map((m) => (
                  <option key={m.source} value={m.source}>
                    {m.label}
                  </option>
                ))}
                {discovered?.hub_presets
                  .filter((h) => !h.gated)
                  .map((h) => (
                    <option key={h.id} value={h.id}>
                      {h.label}
                    </option>
                  ))}
              </datalist>

              <input
                type="text"
                data-testid="input-ckpt-label"
                placeholder="Version label (e.g. v2-dpo, merged-ab)"
                value={ckptLabel}
                onChange={(e) => setCkptLabel(e.target.value)}
                className="studio-run-select"
                style={{ width: "100%" }}
              />

              <select
                aria-label="Transformation Type"
                data-testid="select-ckpt-transform"
                className="studio-run-select"
                value={ckptTransform}
                onChange={(e) => setCkptTransform(e.target.value)}
                style={{ width: "100%" }}
              >
                {supportedTransforms.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>

              <select
                multiple
                aria-label="Parent Checkpoints"
                data-testid="select-ckpt-parents"
                className="studio-run-select"
                value={ckptParents}
                onChange={(e) =>
                  setCkptParents(Array.from(e.target.selectedOptions).map((o) => o.value))
                }
                style={{ width: "100%", height: 58 }}
              >
                {detail.checkpoints.map((c) => (
                  <option key={c.checkpoint_id} value={c.checkpoint_id}>
                    Parent: {c.version_label} ({c.checkpoint_id})
                  </option>
                ))}
              </select>

              <input
                type="text"
                data-testid="input-ckpt-notes"
                placeholder="Experiment notes (hyperparameters, dataset)"
                value={ckptNotes}
                onChange={(e) => setCkptNotes(e.target.value)}
                className="studio-run-select"
                style={{ width: "100%" }}
              />

              <button
                type="submit"
                data-testid="btn-register-checkpoint"
                className="btn-studio-primary"
                disabled={busy || !ckptSource.trim()}
              >
                <Plus size={12} />
                <span>Register Checkpoint</span>
              </button>
            </form>
          </div>

          <div className="panel">
            <div className="panel-header">
              <div>
                <h3 className="panel-title">
                  Checkpoint Registry ({detail.checkpoints.length} versions)
                </h3>
                <div className="panel-subtitle">Click any row to inspect full provenance</div>
              </div>
            </div>
            <div className="data-table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Label</th>
                    <th>Transformation</th>
                    <th>Architecture</th>
                    <th className="num">Params</th>
                    <th className="num">Layers · Hidden</th>
                    <th className="num">Tensors</th>
                    <th>SHA-256</th>
                    <th>Status</th>
                    <th className="num">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {detail.checkpoints.map((c) => (
                    <tr
                      key={c.checkpoint_id}
                      data-testid={`ckpt-row-${c.checkpoint_id}`}
                      className={
                        selectedCheckpointId === c.checkpoint_id ? "is-active-row" : undefined
                      }
                      style={{ cursor: "pointer" }}
                      onClick={() => setSelectedCheckpointId(c.checkpoint_id)}
                    >
                      <td className="mono">{c.version_label}</td>
                      <td>{c.transformation_type}</td>
                      <td className="mono">{c.architecture ?? c.model_family ?? "—"}</td>
                      <td className="num">{c.parameter_count.toLocaleString()}</td>
                      <td className="num">
                        {c.layer_count ?? "—"} · {c.hidden_size ?? "—"}
                      </td>
                      <td className="num">{c.tensor_count}</td>
                      <td className="mono">{c.sha256_fingerprint.slice(0, 14)}…</td>
                      <td>
                        <span className="status-inline">
                          <span
                            className={`dot ${
                              c.validation_status === "VERIFIED" ? "dot-ok" : "dot-warn"
                            }`}
                          />
                          <span>{humanizeEnum(c.validation_status)}</span>
                        </span>
                      </td>
                      <td className="num">
                        <button
                          type="button"
                          className="btn"
                          onClick={(e) => {
                            e.stopPropagation();
                            void handleRemoveCheckpoint(c.checkpoint_id);
                          }}
                        >
                          Remove
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Checkpoint Details Inspector */}
          {selectedCheckpoint && (
            <div className="panel" data-testid="evo-checkpoint-details">
              <div className="panel-header">
                <div>
                  <h3 className="panel-title">
                    Checkpoint Details · {selectedCheckpoint.version_label} (
                    <span className="mono">{selectedCheckpoint.checkpoint_id}</span>)
                  </h3>
                  <div className="panel-subtitle">
                    Content hashes, architecture dimensions, and lineage metadata
                  </div>
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <input
                    type="text"
                    aria-label="Rename Version Label"
                    value={editLabel}
                    onChange={(e) => setEditLabel(e.target.value)}
                    className="studio-run-select"
                  />
                  <input
                    type="text"
                    aria-label="Update Experiment Notes"
                    value={editNotes}
                    onChange={(e) => setEditNotes(e.target.value)}
                    placeholder="Experiment notes"
                    className="studio-run-select"
                  />
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() => void handleUpdateSelectedCheckpoint()}
                  >
                    Save Metadata
                  </button>
                </div>
              </div>

              <div className="data-table-wrap">
                <table className="data-table">
                  <tbody>
                    <tr>
                      <th>Model Name</th>
                      <td className="mono">{selectedCheckpoint.model_name}</td>
                      <th>Architecture / Family</th>
                      <td className="mono">
                        {selectedCheckpoint.architecture ?? "—"} /{" "}
                        {selectedCheckpoint.model_family ?? "—"}
                      </td>
                    </tr>
                    <tr>
                      <th>SHA-256 Fingerprint</th>
                      <td className="mono">{selectedCheckpoint.sha256_fingerprint}</td>
                      <th>Tensor Structure Hash</th>
                      <td className="mono">{selectedCheckpoint.tensor_metadata_hash}</td>
                    </tr>
                    <tr>
                      <th>Parameters / Tensors</th>
                      <td className="mono">
                        {selectedCheckpoint.parameter_count.toLocaleString()} params ·{" "}
                        {selectedCheckpoint.tensor_count} tensors
                      </td>
                      <th>Dimensions</th>
                      <td className="mono">
                        layers={selectedCheckpoint.layer_count ?? "—"} · hidden=
                        {selectedCheckpoint.hidden_size ?? "—"} · vocab=
                        {selectedCheckpoint.vocab_size ?? "—"}
                      </td>
                    </tr>
                    <tr>
                      <th>Format / File Size</th>
                      <td>
                        {humanizeEnum(selectedCheckpoint.checkpoint_format)} ·{" "}
                        <span className="mono">
                          {(selectedCheckpoint.file_size_bytes / (1024 * 1024)).toFixed(2)} MiB (
                          {selectedCheckpoint.dtypes_present.join(", ")})
                        </span>
                      </td>
                      <th>Parent IDs</th>
                      <td className="mono">
                        {selectedCheckpoint.parent_checkpoint_ids.join(", ") || "Root (Base Model)"}
                      </td>
                    </tr>
                    <tr>
                      <th>Tokenizer / Quantization</th>
                      <td>
                        {formatMetaSummary(selectedCheckpoint.tokenizer_metadata, "Default")} ·{" "}
                        {formatMetaSummary(
                          selectedCheckpoint.quantization_metadata,
                          "Unquantized"
                        )}
                      </td>
                      <th>Adapter / Training</th>
                      <td>
                        {formatMetaSummary(selectedCheckpoint.lora_metadata, "Full weights")} ·{" "}
                        {formatMetaSummary(selectedCheckpoint.training_metadata, "Standard")}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 4–8. ANALYTICAL & EXPORT TABS */}
      {activeTab === "weights" && activeRun && <WeightEvolutionSection bundle={activeRun} />}
      {activeTab === "layers" && activeRun && <LayerEvolutionSection bundle={activeRun} />}
      {activeTab === "behavioral" && activeRun && (
        <BehavioralEvolutionSection bundle={activeRun} />
      )}
      {activeTab === "representations" && activeRun && (
        <RepresentationEvolutionSection bundle={activeRun} />
      )}
      {activeTab === "regressions" && activeRun && (
        <RegressionOriginSection bundle={activeRun} />
      )}
      {activeTab === "experiments" && activeRun && (
        <ExperimentExportSection
          bundle={activeRun}
          onRerun={(force) => void handleRunAnalysis(force)}
          running={busy}
        />
      )}

      {["weights", "layers", "behavioral", "representations", "regressions", "experiments"].includes(
        activeTab
      ) &&
        !activeRun && (
          <div className="panel">
            <div className="panel-subtitle">
              No completed evolution analysis run found for this project yet. Click{" "}
              <strong>Run Evolution Analysis</strong> above to compute multi-generation measurements.
            </div>
          </div>
        )}
    </div>
  );
};
