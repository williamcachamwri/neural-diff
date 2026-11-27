import React, { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Columns2, Maximize2, Play, Workflow, X } from "lucide-react";
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
import { EvolutionStudioView } from "./views/EvolutionStudioView";
import { ExperimentHistoryView } from "./views/ExperimentHistoryView";
import { LayerVisualizationView } from "./views/LayerVisualizationView";
import { ModelComparisonView } from "./views/ModelComparisonView";
import { RegressionDashboardView } from "./views/RegressionDashboardView";
import { ReportViewerView } from "./views/ReportViewerView";
import { SettingsView } from "./views/SettingsView";
import { StructuralDiffView } from "./views/StructuralDiffView";
import { TensorExplorerView } from "./views/TensorExplorerView";

const NAV_ITEMS: Array<{ id: ViewId; short: string; title: string }> = [
  { id: "dashboard", short: "Canvas", title: "Workflow Graph" },
  { id: "evolution", short: "Evolution", title: "Neural Evolution Tracking" },
  { id: "compare", short: "Compare", title: "Configure Comparison" },
  { id: "compatibility", short: "Validate", title: "Compatibility Check" },
  { id: "tensors", short: "Tensors", title: "Tensor Explorer" },
  { id: "structural", short: "Topology", title: "Structural Topology" },
  { id: "layers", short: "Layers", title: "Layer & Head Analysis" },
  { id: "activations", short: "Activations", title: "Activation & Linear CKA" },
  { id: "behavioral", short: "Behavior", title: "Behavioral Evaluation" },
  { id: "regression", short: "Regression", title: "Regression Gate" },
  { id: "history", short: "History", title: "Experiment Ledger" },
  { id: "reports", short: "Export", title: "Reports & Artifacts" },
  { id: "settings", short: "Settings", title: "Runtime Settings" },
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
      {/* Unified Single-Line Studio Header */}
      <header className="studio-header">
        <div className="studio-header-left">
          <button
            type="button"
            className="studio-brand"
            onClick={() => setActiveView("dashboard")}
            title="Return to Node Graph"
          >
            <span className="studio-mark">
              <Workflow size={13} strokeWidth={2} />
            </span>
            <span className="studio-brand-name">Neural Diff</span>
          </button>

          <span className="studio-divider-v" />

          <div className="studio-telemetry-quiet">
            <span className={`dot ${health ? "dot-ok" : ""}`} />
            <span>{health ? "Backend online" : "Connecting"}</span>
            <span style={{ color: "var(--text-faint)" }}>·</span>
            <span className="mono" style={{ fontSize: 11, color: "var(--text-secondary)" }}>
              {health ? health.device.toUpperCase() : "CUDA:0"}
            </span>
          </div>
        </div>

        {/* Unified Navigation Ribbon with Smooth Spring Pill */}
        <nav className="studio-nav-ribbon" aria-label="Studio Workspace">
          {NAV_ITEMS.map((item) => {
            const isActive = activeView === item.id;
            return (
              <button
                key={item.id}
                type="button"
                data-testid={`nav-${item.id}`}
                title={item.title}
                className={`studio-nav-item ${isActive ? "active" : ""}`}
                onClick={() => setActiveView(item.id)}
              >
                {isActive && (
                  <motion.span
                    layoutId="studio-nav-pill"
                    className="studio-nav-pill"
                    transition={{ type: "spring", stiffness: 500, damping: 38, mass: 0.7 }}
                  />
                )}
                <span className="studio-nav-label">{item.short}</span>
              </button>
            );
          })}
        </nav>

        <div className="studio-header-right">
          <select
            id="active-exp-selector"
            aria-label="Active Experiment Run"
            data-testid="active-exp-selector"
            className="studio-run-select"
            value={activeBundle?.comparison_id ?? ""}
            onChange={(e) => {
              if (e.target.value) void handleSelectExperiment(e.target.value);
            }}
          >
            <option value="">Select comparison run...</option>
            {experiments
              .filter((e) => e.status === "completed")
              .map((e) => (
                <option key={e.comparison_id} value={e.comparison_id}>
                  {e.comparison_id} · {e.model_a_id} → {e.model_b_id}
                </option>
              ))}
          </select>

          <button
            type="button"
            className="btn-studio-primary"
            onClick={() => setActiveView("compare")}
          >
            <Play size={11} fill="currentColor" />
            <span>New Diff</span>
          </button>
        </div>
      </header>

      {/* Main ComfyUI Node Graph + Sliding Editorial Sheet */}
      <div className="studio-body">
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

        {/* Canvas-mode error toast */}
        <AnimatePresence>
          {errorBanner && activeView === "dashboard" && (
            <motion.div
              initial={{ opacity: 0, y: -12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
              style={{ position: "absolute", top: 16, right: 20, zIndex: 45, maxWidth: 520 }}
            >
              <div className="error-banner" role="alert" data-testid="error-banner">
                <div>
                  <strong className="mono">
                    {errorBanner.code}
                    {errorBanner.status ? ` · ${errorBanner.status}` : ""}
                  </strong>{" "}
                  — {errorBanner.message}
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

        {/* Either Quiet Canvas Status Strip OR Smooth Sliding Inspector Sheet */}
        <AnimatePresence>
          {activeView === "dashboard" ? (
            <DashboardView
              key="canvas-status"
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
            <React.Fragment key="inspector-layer">
              <motion.div
                className="inspector-backdrop"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.22 }}
                onClick={() => setActiveView("dashboard")}
              />
              <motion.section
                aria-label="Inspector Sheet"
                className={`inspector-sheet ${
                  drawerExpanded || activeView === "evolution" ? "is-expanded" : ""
                }`}
                initial={{ opacity: 0, x: 48 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 36 }}
                transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
              >
                <div className="inspector-sheet-header">
                  <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                    <span className="inspector-sheet-title">
                      {activeNavMeta?.title ?? "Inspector"}
                    </span>
                    {activeBundle && (
                      <span className="inspector-sheet-meta">
                        <span
                          className={`dot ${
                            activeBundle.experiment_type === "REAL MODEL EXPERIMENT"
                              ? "dot-ivory"
                              : ""
                          }`}
                        />
                        <span>
                          {activeBundle.experiment_type === "REAL MODEL EXPERIMENT"
                            ? "Real model"
                            : "Synthetic fixture"}
                        </span>
                        <span style={{ color: "var(--text-faint)" }}>·</span>
                        <span className="mono" style={{ fontSize: 11 }}>
                          {activeBundle.model_a.model_id} → {activeBundle.model_b.model_id}
                        </span>
                      </span>
                    )}
                  </div>

                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <button
                      type="button"
                      className="btn"
                      onClick={() => setDrawerExpanded((v) => !v)}
                      title="Toggle full-width view"
                    >
                      {drawerExpanded ? <Columns2 size={13} /> : <Maximize2 size={13} />}
                      <span>{drawerExpanded ? "Split" : "Full width"}</span>
                    </button>
                    <button
                      type="button"
                      className="btn"
                      onClick={() => setActiveView("dashboard")}
                      title="Close inspector"
                    >
                      <X size={13} />
                      <span>Close</span>
                    </button>
                  </div>
                </div>

                <div className="inspector-sheet-scroll">
                  {errorBanner && (
                    <div className="error-banner" role="alert" data-testid="error-banner">
                      <div>
                        <strong className="mono">
                          {errorBanner.code}
                          {errorBanner.status ? ` · ${errorBanner.status}` : ""}
                        </strong>{" "}
                        — {errorBanner.message}
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
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
                  >
                    {activeView === "evolution" && (
                      <EvolutionStudioView discovered={discovered} onError={handleError} />
                    )}
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
            </React.Fragment>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};
