import React, { useEffect, useMemo, useState } from "react";
import {
  Background,
  BackgroundVariant,
  type Edge,
  Handle,
  MiniMap,
  type Node,
  type NodeProps,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import { ChevronDown, ChevronRight, GitBranch, GitMerge } from "lucide-react";
import type {
  BehavioralTimelineEntry,
  EvolutionGraphNode,
  EvolutionGraphPayload,
  RegressionOriginEvent,
  TransitionTensorRecord,
} from "../../types/evolution";

interface EvoNodeData extends Record<string, unknown> {
  node: EvolutionGraphNode;
  passRate: number | null;
  categoryScore: number | null;
  categoryFilter: string;
  cumulativeL2: number | null;
  isRegressionOrigin: boolean;
  isMajorWeightStep: boolean;
  isCollapsedBranchRoot: boolean;
  isSelectedInspect: boolean;
  isSelectedA: boolean;
  isSelectedB: boolean;
  onInspect: (id: string) => void;
  onToggleCollapse: (id: string) => void;
  onSelectA: (id: string) => void;
  onSelectB: (id: string) => void;
}

const EvolutionCheckpointNode: React.FC<NodeProps> = ({ data }) => {
  const d = data as EvoNodeData;
  const n = d.node;

  const borderColor = d.isRegressionOrigin
    ? "rgba(248, 113, 113, 0.72)"
    : d.isSelectedInspect
      ? "rgba(244, 244, 245, 0.55)"
      : d.isMajorWeightStep
        ? "rgba(214, 197, 169, 0.60)"
        : "rgba(255, 255, 255, 0.11)";

  const metricLabel = d.categoryFilter ? `${d.categoryFilter}` : "Pass rate";
  const metricVal = d.categoryFilter
    ? d.categoryScore != null
      ? `${(d.categoryScore * 100).toFixed(1)}%`
      : "—"
    : d.passRate != null
      ? `${(d.passRate * 100).toFixed(1)}%`
      : "—";

  return (
    <div
      data-testid={`evo-node-${n.checkpoint_id}`}
      onClick={() => d.onInspect(n.checkpoint_id)}
      style={{
        width: 210,
        background: "var(--bg-node)",
        border: `1px solid ${borderColor}`,
        borderRadius: 10,
        boxShadow: "var(--shadow-node)",
        color: "var(--text-main)",
        cursor: "pointer",
        overflow: "hidden",
      }}
    >
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: "#a1a1aa", width: 7, height: 7, border: "1px solid #09090b" }}
      />

      {/* Node Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "8px 11px",
          background: "var(--bg-node-header)",
          borderBottom: "1px solid var(--border-faint)",
          fontSize: 11.5,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 0 }}>
          <span
            className={`dot ${
              d.isRegressionOrigin ? "dot-danger" : n.is_base_model ? "dot-ok" : "dot-ivory"
            }`}
          />
          <span className="mono" style={{ color: "var(--text-muted)", fontSize: 10.5 }}>
            G{n.generation_depth}
          </span>
          <strong
            style={{
              fontWeight: 600,
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {n.version_label}
          </strong>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          {n.is_merge_node && <GitMerge size={12} color="#d6c5a9" />}
          {n.is_branch_point && (
            <button
              type="button"
              data-testid={`btn-collapse-${n.checkpoint_id}`}
              onClick={(e) => {
                e.stopPropagation();
                d.onToggleCollapse(n.checkpoint_id);
              }}
              title={d.isCollapsedBranchRoot ? "Expand branch" : "Collapse branch"}
              style={{
                background: "transparent",
                border: "none",
                color: "var(--text-secondary)",
                padding: 2,
                cursor: "pointer",
                display: "grid",
                placeItems: "center",
              }}
            >
              {d.isCollapsedBranchRoot ? <ChevronRight size={13} /> : <ChevronDown size={13} />}
            </button>
          )}
        </div>
      </div>

      {/* Node Body */}
      <div style={{ padding: "9px 11px", fontSize: 11.5 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginBottom: 5,
            color: "var(--text-secondary)",
          }}
        >
          <span>Stage</span>
          <span className="mono">{n.transformation_type}</span>
        </div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginBottom: 5,
            color: "var(--text-secondary)",
          }}
        >
          <span>{metricLabel}</span>
          <span
            className="mono"
            style={{
              color: d.isRegressionOrigin ? "var(--danger)" : "var(--text-main)",
              fontWeight: 500,
            }}
          >
            {metricVal}
          </span>
        </div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginBottom: 7,
            color: "var(--text-secondary)",
          }}
        >
          <span>Base L2</span>
          <span className="mono" style={{ color: d.isMajorWeightStep ? "#d6c5a9" : undefined }}>
            {d.cumulativeL2 != null ? d.cumulativeL2.toFixed(4) : "—"}
          </span>
        </div>

        {/* Quick A / B comparison selectors */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            paddingTop: 6,
            borderTop: "1px solid var(--border-faint)",
            gap: 6,
          }}
        >
          <span className="mono" style={{ fontSize: 10, color: "var(--text-faint)" }}>
            {(n.parameter_count / 1e6).toFixed(1)}M · {n.sha256_fingerprint.slice(0, 7)}
          </span>
          <div style={{ display: "flex", gap: 4 }}>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                d.onSelectA(n.checkpoint_id);
              }}
              style={{
                fontSize: 10,
                padding: "1px 6px",
                borderRadius: 4,
                border: "1px solid var(--border-subtle)",
                background: d.isSelectedA ? "rgba(244,244,245,0.18)" : "transparent",
                color: d.isSelectedA ? "#f4f4f5" : "var(--text-muted)",
                cursor: "pointer",
              }}
            >
              A
            </button>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                d.onSelectB(n.checkpoint_id);
              }}
              style={{
                fontSize: 10,
                padding: "1px 6px",
                borderRadius: 4,
                border: "1px solid var(--border-subtle)",
                background: d.isSelectedB ? "rgba(214,197,169,0.22)" : "transparent",
                color: d.isSelectedB ? "#d6c5a9" : "var(--text-muted)",
                cursor: "pointer",
              }}
            >
              B
            </button>
          </div>
        </div>
      </div>

      <Handle
        type="source"
        position={Position.Right}
        style={{ background: "#e4e4e7", width: 7, height: 7, border: "1px solid #09090b" }}
      />
    </div>
  );
};

const EVO_NODE_TYPES = { evoCheckpoint: EvolutionCheckpointNode };

const DagAutoFit: React.FC<{ triggerKey: string }> = ({ triggerKey }) => {
  const { fitView } = useReactFlow();
  useEffect(() => {
    const t = setTimeout(() => {
      void fitView({ padding: 0.12, duration: 0 });
    }, 80);
    return () => clearTimeout(t);
  }, [fitView, triggerKey]);
  return null;
};

export interface EvolutionDagCanvasProps {
  graph: EvolutionGraphPayload;
  tensorTransitions: TransitionTensorRecord[];
  behavioralTimeline: BehavioralTimelineEntry[];
  regressions: RegressionOriginEvent[];
  selectedCheckpointId: string | null;
  compareAId: string;
  compareBId: string;
  transformationFilter: string;
  categoryFilter: string;
  highlightRegressions: boolean;
  highlightMajorWeights: boolean;
  onSelectCheckpoint: (checkpointId: string) => void;
  onSelectCompareA: (checkpointId: string) => void;
  onSelectCompareB: (checkpointId: string) => void;
}

export const EvolutionDagCanvas: React.FC<EvolutionDagCanvasProps> = ({
  graph,
  tensorTransitions,
  behavioralTimeline,
  regressions,
  selectedCheckpointId,
  compareAId,
  compareBId,
  transformationFilter,
  categoryFilter,
  highlightRegressions,
  highlightMajorWeights,
  onSelectCheckpoint,
  onSelectCompareA,
  onSelectCompareB,
}) => {
  const [collapsedRoots, setCollapsedRoots] = useState<Set<string>>(new Set());

  useEffect(() => {
    setCollapsedRoots(new Set());
  }, [graph.base_checkpoint_id, graph.nodes.length]);

  const toggleCollapse = (cid: string) => {
    setCollapsedRoots((prev) => {
      const next = new Set(prev);
      if (next.has(cid)) next.delete(cid);
      else next.add(cid);
      return next;
    });
  };

  // Compute hidden nodes downstream of collapsed branch points
  const hiddenIds = useMemo(() => {
    const hidden = new Set<string>();
    if (collapsedRoots.size === 0) return hidden;
    const childMap = new Map<string, string[]>();
    for (const n of graph.nodes) childMap.set(n.checkpoint_id, n.child_ids);

    const queue = [...collapsedRoots];
    while (queue.length > 0) {
      const cur = queue.shift()!;
      for (const ch of childMap.get(cur) ?? []) {
        if (!hidden.has(ch)) {
          hidden.add(ch);
          queue.push(ch);
        }
      }
    }
    return hidden;
  }, [collapsedRoots, graph.nodes]);

  const behById = useMemo(() => {
    const m = new Map<string, BehavioralTimelineEntry>();
    for (const item of behavioralTimeline) m.set(item.checkpoint_id, item);
    return m;
  }, [behavioralTimeline]);

  const baseL2ById = useMemo(() => {
    const m = new Map<string, number>();
    for (const tr of tensorTransitions) {
      if (tr.scope === "base_to_current" && tr.l2_distance != null) {
        m.set(tr.target_checkpoint_id, tr.l2_distance);
      }
    }
    return m;
  }, [tensorTransitions]);

  const peakConsecutiveTargetId = useMemo(() => {
    const consec = tensorTransitions.filter(
      (t) => t.scope === "consecutive" && t.relative_l2_distance != null
    );
    if (consec.length === 0) return null;
    const best = consec.reduce((a, b) =>
      (b.relative_l2_distance ?? 0) > (a.relative_l2_distance ?? 0) ? b : a
    );
    return best.target_checkpoint_id;
  }, [tensorTransitions]);

  const regressionTargetIds = useMemo(() => {
    const s = new Set<string>();
    for (const r of regressions) {
      if (!categoryFilter || r.affected_category === categoryFilter || r.affected_category === "overall") {
        s.add(r.first_affected_checkpoint_id);
      }
    }
    return s;
  }, [regressions, categoryFilter]);

  const visibleNodes = useMemo(() => {
    return graph.nodes.filter((n) => {
      if (hiddenIds.has(n.checkpoint_id)) return false;
      if (transformationFilter && n.transformation_type !== transformationFilter) return false;
      return true;
    });
  }, [graph.nodes, hiddenIds, transformationFilter]);

  const visibleIdSet = useMemo(
    () => new Set(visibleNodes.map((n) => n.checkpoint_id)),
    [visibleNodes]
  );

  const rfNodes: Node[] = useMemo(() => {
    return visibleNodes.map((n) => {
      const beh = behById.get(n.checkpoint_id);
      const yOffset = (n.lane_index - (n.lane_count - 1) / 2) * 176;
      return {
        id: n.checkpoint_id,
        type: "evoCheckpoint",
        position: {
          x: 24 + n.generation_depth * 252,
          y: 165 + yOffset,
        },
        data: {
          node: n,
          passRate: beh?.pass_rate ?? null,
          categoryScore: categoryFilter ? (beh?.category_scores?.[categoryFilter] ?? null) : null,
          categoryFilter,
          cumulativeL2: baseL2ById.get(n.checkpoint_id) ?? null,
          isRegressionOrigin: highlightRegressions && regressionTargetIds.has(n.checkpoint_id),
          isMajorWeightStep: highlightMajorWeights && peakConsecutiveTargetId === n.checkpoint_id,
          isCollapsedBranchRoot: collapsedRoots.has(n.checkpoint_id),
          isSelectedInspect: selectedCheckpointId === n.checkpoint_id,
          isSelectedA: compareAId === n.checkpoint_id,
          isSelectedB: compareBId === n.checkpoint_id,
          onInspect: onSelectCheckpoint,
          onToggleCollapse: toggleCollapse,
          onSelectA: onSelectCompareA,
          onSelectB: onSelectCompareB,
        } satisfies EvoNodeData,
      };
    });
  }, [
    visibleNodes,
    behById,
    categoryFilter,
    baseL2ById,
    highlightRegressions,
    regressionTargetIds,
    highlightMajorWeights,
    peakConsecutiveTargetId,
    collapsedRoots,
    selectedCheckpointId,
    compareAId,
    compareBId,
    onSelectCheckpoint,
    onSelectCompareA,
    onSelectCompareB,
  ]);

  const rfEdges: Edge[] = useMemo(() => {
    return graph.edges
      .filter(
        (e) => visibleIdSet.has(e.parent_checkpoint_id) && visibleIdSet.has(e.child_checkpoint_id)
      )
      .map((e) => {
        const isRegEdge = highlightRegressions && regressionTargetIds.has(e.child_checkpoint_id);
        const isMajorWt = highlightMajorWeights && peakConsecutiveTargetId === e.child_checkpoint_id;
        const stroke = isRegEdge ? "#f87171" : isMajorWt ? "#d6c5a9" : "rgba(228,228,231,0.38)";
        return {
          id: e.relationship_id,
          source: e.parent_checkpoint_id,
          target: e.child_checkpoint_id,
          label: e.merge_weight != null
            ? `${e.transformation_type} (w=${e.merge_weight})`
            : e.transformation_type,
          animated: isRegEdge || isMajorWt,
          labelStyle: { fill: "#a1a1aa", fontSize: 10, fontFamily: "monospace" },
          labelBgStyle: { fill: "#0d0d10", fillOpacity: 0.9 },
          style: { stroke, strokeWidth: isRegEdge || isMajorWt ? 2.1 : 1.35 },
        };
      });
  }, [
    graph.edges,
    visibleIdSet,
    highlightRegressions,
    regressionTargetIds,
    highlightMajorWeights,
    peakConsecutiveTargetId,
  ]);

  if (typeof globalThis.ResizeObserver === "undefined") {
    return (
      <div
        data-testid="evo-dag-fallback"
        style={{
          padding: 16,
          background: "var(--bg-surface-subtle)",
          border: "1px solid var(--border-subtle)",
          borderRadius: 10,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
          <GitBranch size={14} />
          <span className="mono">{visibleNodes.length} checkpoints · {graph.branch_count} branches</span>
        </div>
        {visibleNodes.map((n) => (
          <div key={n.checkpoint_id} className="mono" style={{ fontSize: 12 }}>
            G{n.generation_depth}: {n.version_label} ({n.transformation_type})
          </div>
        ))}
      </div>
    );
  }

  return (
    <div
      data-testid="evolution-dag-canvas"
      style={{
        width: "100%",
        height: 440,
        background: "#09090b",
        border: "1px solid var(--border-subtle)",
        borderRadius: 10,
        overflow: "hidden",
        position: "relative",
      }}
    >
      <ReactFlowProvider>
        <ReactFlow
          nodes={rfNodes}
          edges={rfEdges}
          nodeTypes={EVO_NODE_TYPES}
          fitView
          fitViewOptions={{ padding: 0.12 }}
          minZoom={0.2}
          maxZoom={1.8}
          className="dark"
          proOptions={{ hideAttribution: true }}
        >
          <DagAutoFit triggerKey={`${rfNodes.length}-${rfEdges.length}`} />
          <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#1e1e24" />
          <MiniMap
            pannable
            zoomable
            nodeColor="#52525b"
            maskColor="rgba(9, 9, 11, 0.82)"
            style={{ bottom: 10, right: 10, width: 140, height: 86 }}
          />
        </ReactFlow>
      </ReactFlowProvider>
    </div>
  );
};
