import React from "react";
import { motion } from "framer-motion";
import { Play, ShieldCheck, Sparkles } from "lucide-react";
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
  activeBundle,
  onNavigate,
}) => {
  return (
    <motion.div
      data-testid="view-dashboard"
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -6 }}
      transition={{ type: "spring", stiffness: 420, damping: 32 }}
      style={{
        position: "absolute",
        top: 66,
        left: 78,
        zIndex: 18,
        display: "inline-flex",
        alignItems: "center",
        flexWrap: "wrap",
        gap: 8,
        background: "rgba(14, 14, 18, 0.84)",
        backdropFilter: "blur(22px) saturate(150%)",
        border: "1px solid var(--border)",
        borderRadius: 12,
        padding: "6px 11px",
        boxShadow: "var(--shadow-node)",
        pointerEvents: "auto",
      }}
    >
      <div style={{ display: "inline-flex", alignItems: "center", gap: 6, paddingRight: 6, borderRight: "1px solid var(--border-subtle)" }}>
        <Sparkles size={12} color="#e5b869" />
        <span style={{ fontSize: 11.5, fontWeight: 700, letterSpacing: "0.01em", color: "#fff" }}>
          NEURAL DIFF Engineering Workbench
        </span>
        <span className="mono" style={{ fontSize: 10.5, color: "var(--text-muted)" }}>
          {activeBundle
            ? `${activeBundle.model_a.model_id} → ${activeBundle.model_b.model_id}`
            : "Click any node to inspect"}
        </span>
      </div>

      <span className="badge badge-ok">OBSERVED</span>
      <span className="badge badge-purple">INFERRED</span>
      <span className="badge badge-warn">CORRELATED</span>
      <span className="badge badge-info">
        {health ? health.device.toUpperCase() : "CUDA:0"}
      </span>
      {activeBundle && (
        <span className="badge">
          {activeBundle.experiment_type === "REAL MODEL EXPERIMENT" ? "REAL MODEL" : "SYNTHETIC"}
        </span>
      )}

      <div style={{ display: "inline-flex", gap: 5, marginLeft: 2 }}>
        <button
          type="button"
          data-testid="cta-new-comparison"
          className="btn-queue-gold"
          style={{ padding: "4px 9px", fontSize: 10.5, borderRadius: 8 }}
          onClick={() => onNavigate("compare")}
        >
          <Play size={10} fill="currentColor" />
          <span>New Run</span>
        </button>
        <button
          type="button"
          className="btn"
          style={{ padding: "4px 8px", fontSize: 10.5, borderRadius: 8 }}
          onClick={() => onNavigate("compatibility")}
          title="Validate Checkpoint Compatibility"
        >
          <ShieldCheck size={11} color="#e5b869" />
        </button>
      </div>
    </motion.div>
  );
};
