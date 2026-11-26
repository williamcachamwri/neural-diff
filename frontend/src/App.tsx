import React, { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Columns2,
  Cpu,
  Database,
  FileSpreadsheet,
  GitBranch,
  History,
  Layers,
  Maximize2,
  Play,
  Settings,
  Workflow,
  X,
} from "lucide-react";
import { apiClient, ApiClientError } from "./api/client";
import { WorkflowCanvas } from "./components/canvas/WorkflowCanvas";
import type {
  BaselineRecord,
  ComparisonBundle,
  DiscoveredModelsResponse,
  ExperimentSummaryItem,
  HealthResponse,
  ViewId,
} from "./types/api";
import { ActivationComparisonView } from "./views/ActivationComparisonView";
import { BehavioralDiffView } from "./views/BehavioralDiffView";
import { CompatibilityView } from "./views/CompatibilityView";
import { DashboardView } from "./views/DashboardView";
import { ExperimentHistoryView } from "./views/ExperimentHistoryView";
import { LayerVisualizationView } from "./views/LayerVisualizationView";
import { ModelComparisonView } from "./views/ModelComparisonView";
import { RegressionDashboardView } from "./views/RegressionDashboardView";
import { ReportViewerView } from "./views/ReportViewerView";
import { SettingsView } from "./views/SettingsView";
import { StructuralDiffView } from "./views/StructuralDiffView";
import { TensorExplorerView } from "./views/TensorExplorerView";

const NAV_ITEMS: Array<{ id: ViewId; label: string; icon: React.ReactNode }> = [
  { id: "dashboard", label: "Node Graph Canvas", icon: <Workflow size={17} /> },
  { id: "compare", label: "Queue Comparison", icon: <Play size={16} /> },
  { id: "compatibility", label: "Compatibility Gate", icon: <CheckCircle2 size={16} /> },
  { id: "tensors", label: "Tensor Explorer", icon: <Database size={16} /> },
  { id: "structural", label: "Structural Topology", icon: <GitBranch size={16} /> },
  { id: "layers", label: "Layer & GQA Heatmap", icon: <Layers size={16} /> },
  { id: "activations", label: "Activation CKA Scope", icon: <Activity size={16} /> },
  { id: "behavioral", label: "Behavioral Harness", icon: <Cpu size={16} /> },
  { id: "regression", label: "Regression Gate", icon: <AlertTriangle size={16} /> },
  { id: "history", label: "Experiment History", icon: <History size={16} /> },
  { id: "reports", label: "Report Exporter", icon: <FileSpreadsheet size={16} /> },
  { id: "settings", label: "Studio Settings", icon: <Settings size={16} /> },
];

export const App: React.FC = () => {
  const [activeView, setActiveView] = useState<ViewId>("dashboard");
  const [drawerExpanded, setDrawerExpanded] = useState<boolean>(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [discovered, setDiscovered] = useState<DiscoveredModelsResponse | null>(null);
  const [experiments, setExperiments] = useState<ExperimentSummaryItem[]>([]);
  const [baselines, setBaselines] = useState<BaselineRecord[]>([]);
  const [activeBundle, setActiveBundle] = useState<ComparisonBundle | null>(null);

  const [modelA, setModelA] = useState<string>("Qwen/Qwen2.5-0.5B");
  const [modelB, setModelB] = useState<string>("Qwen/Qwen2.5-0.5B-Instruct");
  const [errorBanner, setErrorBanner] = useState<{
    code: string;
    message: string;
    status?: number;
  } | null>(null);

  const handleError = useCallback((err: Error | ApiClientError) => {
    if (err instanceof ApiClientError) {
      setErrorBanner({ code: err.code, message: err.message, status: err.httpStatus });
    } else {
      setErrorBanner({ code: "APPLICATION_ERROR", message: err.message });
    }
  }, []);

  const refreshCoreData = useCallback(async () => {
    try {
      const [h, disc, expList, baseList] = await Promise.all([
        apiClient.getHealth(),
        apiClient.discoverModels(),
        apiClient.listComparisons(),
        apiClient.listBaselines(),
      ]);
      setHealth(h);
      setDiscovered(disc);
      setExperiments(expList.experiments);
      setBaselines(baseList.baselines);

      setActiveBundle((prev) => {
        if (prev) return prev;
        const preferred =
          expList.experiments.find(
            (e) => e.comparison_id === "ndiff-real-qwen25-05b" && e.status === "completed"
          ) ?? expList.experiments.find((e) => e.status === "completed");
        if (preferred) {
          void apiClient.getComparison(preferred.comparison_id).then((full) => {
            if (full.bundle) setActiveBundle(full.bundle);
          });
        }
        return prev;
      });
    } catch (err) {
      handleError(err instanceof Error ? err : new Error(String(err)));
    }
  }, [handleError]);

  useEffect(() => {
    void refreshCoreData();
  }, [refreshCoreData]);

  const handleSelectExperiment = useCallback(
    async (comparisonId: string) => {
      try {
        const full = await apiClient.getComparison(comparisonId);
        if (full.bundle) {
          setActiveBundle(full.bundle);
        }
      } catch (err) {
        handleError(err instanceof Error ? err : new Error(String(err)));
      }
    },
    [handleError]
  );

  const handleComparisonComplete = useCallback(
    (exp: ExperimentSummaryItem) => {
      if (exp.bundle) {
        setActiveBundle(exp.bundle);
      }
      void refreshCoreData();
      setActiveView("tensors");
    },
    [refreshCoreData]
  );

  const activeNavMeta = NAV_ITEMS.find((n) => n.id === activeView);

  return (
    <div className="studio-viewport">
      {/* 1. Full-Viewport ComfyUI Interactive Node-Graph Canvas */}
      <div className="studio-canvas-layer">
        <WorkflowCanvas
          activeBundle={activeBundle}
          health={health}
          experiments={experiments}
          modelA={modelA}
          modelB={modelB}
          activeView={activeView}
          onSelectView={setActiveView}
        />
      </div>

      {/* 2. Left ComfyUI Compact Icon Rail with Liquid Spring Pill */}
      <aside className="comfy-left-rail" aria-label="Studio Node & Inspector Dock">
        <div className="rail-group">
          {NAV_ITEMS.slice(0, 9).map((item) => {
            const isActive = activeView === item.id;
            return (
              <button
                key={item.id}
                type="button"
                data-testid={`nav-${item.id}`}
                title={item.label}
                aria-label={item.label}
                className={`rail-btn ${isActive ? "active" : ""}`}
                onClick={() => setActiveView(item.id)}
              >
                {isActive && (
                  <motion.span
                    layoutId="comfy-rail-pill"
                    className="rail-active-pill"
                    transition={{ type: "spring", stiffness: 460, damping: 34, mass: 0.8 }}
                  />
                )}
                <span className="rail-icon-fg">{item.icon}</span>
              </button>
            );
          })}
        </div>

        <div className="rail-group" style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: 8 }}>
          {NAV_ITEMS.slice(9).map((item) => {
            const isActive = activeView === item.id;
            return (
              <button
                key={item.id}
                type="button"
                data-testid={`nav-${item.id}`}
                title={item.label}
                aria-label={item.label}
                className={`rail-btn ${isActive ? "active" : ""}`}
                onClick={() => setActiveView(item.id)}
              >
                {isActive && (
                  <motion.span
                    layoutId="comfy-rail-pill"
                    className="rail-active-pill"
                    transition={{ type: "spring", stiffness: 460, damping: 34, mass: 0.8 }}
                  />
                )}
                <span className="rail-icon-fg">{item.icon}</span>
              </button>
            );
          })}
        </div>
      </aside>

      {/* 3. Floating Top ComfyUI Action & Queue Bar */}
      <header className="comfy-topbar">
        <div className="comfy-bar-cluster">
          <div className="studio-brand">
            <div className="studio-logo-mark">
              <Workflow size={15} />
            </div>
            <div>
              <div className="studio-brand-title">NEURAL DIFF</div>
              <div className="studio-brand-sub">
                NODES 2.0 STUDIO • v{health?.version ?? "0.1.0"}
              </div>
            </div>
          </div>

          <span className="badge badge-ok">
            {health ? `BACKEND ONLINE (${health.device.toUpperCase()})` : "CONNECTING"}
          </span>

          <div className="mode-segmented">
            <button
              type="button"
              className={`mode-seg-btn ${activeView === "dashboard" ? "active" : ""}`}
              onClick={() => setActiveView("dashboard")}
            >
              {activeView === "dashboard" && (
                <motion.span layoutId="studio-mode-seg" className="mode-seg-pill" />
              )}
              <span style={{ position: "relative", zIndex: 1 }}>Graph Canvas</span>
            </button>
            <button
              type="button"
              className={`mode-seg-btn ${activeView !== "dashboard" ? "active" : ""}`}
              onClick={() => {
                if (activeView === "dashboard") setActiveView("tensors");
              }}
            >
              {activeView !== "dashboard" && (
                <motion.span layoutId="studio-mode-seg" className="mode-seg-pill" />
              )}
              <span style={{ position: "relative", zIndex: 1 }}>
                Inspector: {activeNavMeta?.label ?? "Studio"}
              </span>
            </button>
          </div>
        </div>

        <div className="comfy-bar-cluster">
          <label
            htmlFor="active-exp-selector"
            className="mono"
            style={{ fontSize: 11, color: "var(--text-muted)" }}
          >
            RUN:
          </label>
          <select
            id="active-exp-selector"
            data-testid="active-exp-selector"
            value={activeBundle?.comparison_id ?? ""}
            onChange={(e) => {
              if (e.target.value) void handleSelectExperiment(e.target.value);
            }}
            style={{ maxWidth: 310, padding: "5px 9px", fontSize: 11.5 }}
          >
            <option value="">— Select Executed Workflow —</option>
            {experiments
              .filter((e) => e.status === "completed")
              .map((e) => (
                <option key={e.comparison_id} value={e.comparison_id}>
                  {e.comparison_id} ({e.model_a_id} → {e.model_b_id})
                </option>
              ))}
          </select>

          <motion.button
            type="button"
            whileHover={{ y: -1 }}
            whileTap={{ scale: 0.97 }}
            className="btn-queue-gold"
            onClick={() => setActiveView("compare")}
          >
            <Play size={12} fill="currentColor" />
            <span>Queue Diff</span>
          </motion.button>
        </div>
      </header>

      {/* Global Floating Error Toast when on Canvas Mode */}
      <AnimatePresence>
        {errorBanner && activeView === "dashboard" && (
          <motion.div
            initial={{ opacity: 0, y: -16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            style={{ position: "absolute", top: 72, right: 20, zIndex: 45, maxWidth: 560 }}
          >
            <div className="error-banner" role="alert" data-testid="error-banner">
              <div>
                <strong>
                  [{errorBanner.code}
                  {errorBanner.status ? ` • HTTP ${errorBanner.status}` : ""}]
                </strong>{" "}
                {errorBanner.message}
              </div>
              <button
                className="btn"
                data-testid="btn-dismiss-error"
                onClick={() => setErrorBanner(null)}
              >
                Dismiss
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 4. Floating HUD Ribbon on Graph Canvas OR Spring-Physics Glass Inspector Dock */}
      <AnimatePresence>
        {activeView === "dashboard" ? (
          <DashboardView
            key="hud-dashboard"
            health={health}
            experiments={experiments}
            activeBundle={activeBundle}
            onNavigate={setActiveView}
            onSelectExperiment={(cid) => {
              void handleSelectExperiment(cid);
              setActiveView("tensors");
            }}
          />
        ) : (
          <motion.section
            key="inspector-dock"
            aria-label="Node Inspector Workbench"
            className={`inspector-drawer ${drawerExpanded ? "is-expanded" : ""}`}
            initial={{ opacity: 0, x: 54, scale: 0.985 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 42, scale: 0.985 }}
            transition={{ type: "spring", stiffness: 390, damping: 34, mass: 0.85 }}
          >
            <div className="inspector-drawer-header">
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ color: "var(--gold)", display: "grid" }}>
                  {activeNavMeta?.icon}
                </span>
                <span style={{ fontWeight: 700, fontSize: 14, letterSpacing: "-0.01em" }}>
                  {activeNavMeta?.label}
                </span>
                {activeBundle && (
                  <span
                    className={`badge ${
                      activeBundle.experiment_type === "REAL MODEL EXPERIMENT"
                        ? "badge-ok"
                        : "badge-purple"
                    }`}
                  >
                    {activeBundle.experiment_type === "REAL MODEL EXPERIMENT"
                      ? "REAL MODEL"
                      : "SYNTHETIC"}
                  </span>
                )}
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                <button
                  type="button"
                  className="btn"
                  style={{ padding: "5px 9px", fontSize: 11.5 }}
                  onClick={() => setDrawerExpanded((v) => !v)}
                  title="Toggle Split Canvas vs Expanded Studio Sheet"
                >
                  {drawerExpanded ? <Columns2 size={13} /> : <Maximize2 size={13} />}
                  <span>{drawerExpanded ? "Split Graph" : "Expand"}</span>
                </button>
                <button
                  type="button"
                  className="btn"
                  style={{ padding: "5px 9px", fontSize: 11.5 }}
                  onClick={() => setActiveView("dashboard")}
                  title="Return to unobstructed Node Graph Canvas"
                >
                  <X size={13} />
                  <span>Canvas</span>
                </button>
              </div>
            </div>

            <div className="inspector-drawer-scroll">
              {errorBanner && (
                <div className="error-banner" role="alert" data-testid="error-banner">
                  <div>
                    <strong>
                      [{errorBanner.code}
                      {errorBanner.status ? ` • HTTP ${errorBanner.status}` : ""}]
                    </strong>{" "}
                    {errorBanner.message}
                  </div>
                  <button
                    className="btn"
                    data-testid="btn-dismiss-error"
                    onClick={() => setErrorBanner(null)}
                  >
                    Dismiss
                  </button>
                </div>
              )}

              <motion.div
                key={activeView}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
              >
                {activeView === "compare" && (
                  <ModelComparisonView
                    discovered={discovered}
                    baselines={baselines}
                    modelA={modelA}
                    modelB={modelB}
                    onChangeModelA={setModelA}
                    onChangeModelB={setModelB}
                    onComparisonComplete={handleComparisonComplete}
                    onError={handleError}
                  />
                )}
                {activeView === "compatibility" && (
                  <CompatibilityView
                    modelA={modelA}
                    modelB={modelB}
                    activeBundle={activeBundle}
                    onChangeModelA={setModelA}
                    onChangeModelB={setModelB}
                    onError={handleError}
                  />
                )}
                {activeView === "tensors" && (
                  <TensorExplorerView activeBundle={activeBundle} onError={handleError} />
                )}
                {activeView === "structural" && (
                  <StructuralDiffView activeBundle={activeBundle} />
                )}
                {activeView === "layers" && (
                  <LayerVisualizationView activeBundle={activeBundle} />
                )}
                {activeView === "activations" && (
                  <ActivationComparisonView activeBundle={activeBundle} />
                )}
                {activeView === "behavioral" && (
                  <BehavioralDiffView activeBundle={activeBundle} />
                )}
                {activeView === "regression" && (
                  <RegressionDashboardView
                    activeBundle={activeBundle}
                    baselines={baselines}
                    onRefreshBaselines={() => void refreshCoreData()}
                    onError={handleError}
                  />
                )}
                {activeView === "history" && (
                  <ExperimentHistoryView
                    experiments={experiments}
                    activeComparisonId={activeBundle?.comparison_id ?? null}
                    onSelectExperiment={(cid) => {
                      void handleSelectExperiment(cid);
                      setActiveView("tensors");
                    }}
                    onRefresh={() => void refreshCoreData()}
                  />
                )}
                {activeView === "reports" && (
                  <ReportViewerView activeBundle={activeBundle} onError={handleError} />
                )}
                {activeView === "settings" && (
                  <SettingsView
                    onError={handleError}
                    onSettingsSaved={() => void refreshCoreData()}
                  />
                )}
              </motion.div>
            </div>
          </motion.section>
        )}
      </AnimatePresence>
    </div>
  );
};
