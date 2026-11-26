import React from "react";
import { motion } from "framer-motion";
import { Plus } from "lucide-react";
import type {
  ComparisonBundle,
  ExperimentSummaryItem,
  HealthResponse,
  ViewId,
} from "../types/api";

interface Props {
  health: HealthResponse | null;
  experiments: ExperimentSummaryItem[];
  activeBundle: ComparisonBundle | null;
  onNavigate: (view: ViewId) => void;
  onSelectExperiment: (comparisonId: string) => void;
}

export const DashboardView: React.FC<Props> = ({
  health,
  experiments,
  activeBundle,
  onNavigate,
}) => {
  const realCount = experiments.filter(
    (e) => e.experiment_type === "REAL MODEL EXPERIMENT" && e.status === "completed"
  ).length;
  const synthCount = experiments.filter(
    (e) => e.experiment_type === "SYNTHETIC FIXTURE EXPERIMENT" && e.status === "completed"
  ).length;

  return (
    <motion.div
      data-testid="view-dashboard"
      className="canvas-footer-bar"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 6 }}
      transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
    >
      <span style={{ color: "var(--text-secondary)", fontWeight: 500 }}>
        NEURAL DIFF Engineering Workbench
      </span>
      <span className="studio-divider-v" style={{ height: 12 }} />
      <span>
        {health && health.hardware.cuda_total_vram_mb > 0
          ? `${(health.hardware.cuda_total_vram_mb / 1024).toFixed(1)} GB VRAM`
          : "Local compute"}
      </span>
      <span style={{ color: "var(--text-faint)" }}>·</span>
      <span>
        {realCount} real · {synthCount} fixture runs
      </span>
      {activeBundle && (
        <>
          <span className="studio-divider-v" style={{ height: 12 }} />
          <span className="mono" style={{ fontSize: 11, color: "var(--text-secondary)" }}>
            {activeBundle.model_a.model_id} → {activeBundle.model_b.model_id}
          </span>
        </>
      )}
      <span className="studio-divider-v" style={{ height: 12 }} />
      <button
        type="button"
        data-testid="cta-new-comparison"
        className="canvas-footer-btn"
        onClick={() => onNavigate("compare")}
      >
        <Plus size={12} />
        <span>Configure</span>
      </button>
    </motion.div>
  );
};
