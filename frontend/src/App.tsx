import React, { useCallback, useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Database,
  FileSpreadsheet,
  GitBranch,
  History,
  Layers,
  LayoutDashboard,
  Play,
  Settings,
  Sliders,
} from "lucide-react";
import { apiClient, ApiClientError } from "./api/client";
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
  { id: "dashboard", label: "Dashboard", icon: <LayoutDashboard size={16} /> },
  { id: "compare", label: "Model Comparison", icon: <Play size={16} /> },
  { id: "compatibility", label: "Compatibility Check", icon: <CheckCircle2 size={16} /> },
  { id: "tensors", label: "Tensor Explorer", icon: <Database size={16} /> },
  { id: "structural", label: "Structural Diff", icon: <GitBranch size={16} /> },
  { id: "layers", label: "Layer Visualization", icon: <Layers size={16} /> },
  { id: "activations", label: "Activation Diff", icon: <Activity size={16} /> },
  { id: "behavioral", label: "Behavioral Diff", icon: <Cpu size={16} /> },
  { id: "regression", label: "Regression Gate", icon: <AlertTriangle size={16} /> },
  { id: "history", label: "Experiment History", icon: <History size={16} /> },
  { id: "reports", label: "Report Viewer", icon: <FileSpreadsheet size={16} /> },
  { id: "settings", label: "Settings", icon: <Settings size={16} /> },
];

export const App: React.FC = () => {
  const [activeView, setActiveView] = useState<ViewId>("dashboard");
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
        const firstCompleted = expList.experiments.find((e) => e.status === "completed");
        if (firstCompleted) {
          void apiClient.getComparison(firstCompleted.comparison_id).then((full) => {
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
          setActiveView("tensors");
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

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-box">
          <div className="brand-title">
            <Sliders size={18} color="#38bdf8" />
            <span>NEURAL DIFF</span>
          </div>
          <div className="brand-sub">Git for AI Brains • v{health?.version ?? "0.1.0"}</div>
        </div>

        <nav className="nav-list" aria-label="Main Navigation">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              data-testid={`nav-${item.id}`}
              className={`nav-btn ${activeView === item.id ? "active" : ""}`}
              onClick={() => setActiveView(item.id)}
            >
              {item.icon}
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
      </aside>

      <div className="main-area">
        <header className="topbar">
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span className="badge badge-ok">
              {health ? `BACKEND ONLINE (${health.device.toUpperCase()})` : "CONNECTING"}
            </span>
            {activeBundle && (
              <span className="mono" style={{ fontSize: 12.5, color: "var(--text-muted)" }}>
                Active: <strong>{activeBundle.comparison_id}</strong> ({activeBundle.model_a.model_id}{" "}
                → {activeBundle.model_b.model_id})
              </span>
            )}
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <label htmlFor="active-exp-selector" style={{ fontSize: 12, color: "var(--text-muted)" }}>
              Active Run:
            </label>
            <select
              id="active-exp-selector"
              data-testid="active-exp-selector"
              value={activeBundle?.comparison_id ?? ""}
              onChange={(e) => {
                if (e.target.value) void handleSelectExperiment(e.target.value);
              }}
            >
              <option value="">— Select Completed Experiment —</option>
              {experiments
                .filter((e) => e.status === "completed")
                .map((e) => (
                  <option key={e.comparison_id} value={e.comparison_id}>
                    {e.comparison_id} ({e.model_a_id} → {e.model_b_id})
                  </option>
                ))}
            </select>
          </div>
        </header>

        <main className="content-wrap">
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

          {activeView === "dashboard" && (
            <DashboardView
              health={health}
              experiments={experiments}
              activeBundle={activeBundle}
              onNavigate={setActiveView}
              onSelectExperiment={handleSelectExperiment}
            />
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
          {activeView === "structural" && <StructuralDiffView activeBundle={activeBundle} />}
          {activeView === "layers" && <LayerVisualizationView activeBundle={activeBundle} />}
          {activeView === "activations" && (
            <ActivationComparisonView activeBundle={activeBundle} />
          )}
          {activeView === "behavioral" && <BehavioralDiffView activeBundle={activeBundle} />}
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
              onSelectExperiment={handleSelectExperiment}
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
        </main>
      </div>
    </div>
  );
};
