import React, { useEffect, useMemo, useState } from "react";
import {
  Background,
  BackgroundVariant,
  type Edge,
  EdgeLabelRenderer,
  type EdgeProps,
  getBezierPath,
  Handle,
  MiniMap,
  type Node,
  type NodeProps,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import { ChevronDown, ChevronRight, GitBranch, GitMerge, Maximize2, Minus, Plus } from "lucide-react";
import type {
  BehavioralTimelineEntry,
  EvolutionGraphNode,
  EvolutionGraphPayload,
  RegressionOriginEvent,
  TransitionTensorRecord,
} from "../../types/evolution";
import { formatCategoryShort } from "../../utils/format";

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

interface EvoEdgeData extends Record<string, unknown> {
  stroke: string;
  active: boolean;
  mergeWeight: number | null;
}

const EvoOpticalEdge: React.FC<EdgeProps> = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}) => {
  const d = (data ?? {}) as EvoEdgeData;
  const stroke = d.stroke ?? "#a1a1aa";
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    curvature: 0.36,
  });

  return (
    <>
      <g className="evo-optical-edge">
        <path
          d={edgePath}
          fill="none"
          stroke="rgba(9, 9, 11, 0.78)"
          strokeWidth={4.2}
          strokeLinecap="round"
        />
        <path
          d={edgePath}
          fill="none"
          stroke={stroke}
          strokeOpacity={d.active ? 0.55 : 0.34}
          strokeWidth={d.active ? 1.85 : 1.45}
          strokeLinecap="round"
        />
        {d.active && (
          <path
            id={id}
            d={edgePath}
            fill="none"
            stroke={stroke}
            strokeWidth={2}
            strokeOpacity={0.95}
            strokeLinecap="round"
            strokeDasharray="24 190"
          >
            <animate
              attributeName="stroke-dashoffset"
              from="214"
              to="0"
              dur="2.8s"
              repeatCount="indefinite"
            />
          </path>
        )}
      </g>
      {d.mergeWeight != null && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: "absolute",
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: "none",
              padding: "2px 6px",
              borderRadius: 6,
              background: "rgba(13, 13, 17, 0.92)",
              border: "1px solid rgba(214, 197, 169, 0.28)",
              color: "#d6c5a9",
              fontSize: 10,
              fontFamily: "JetBrains Mono, monospace",
              boxShadow: "0 4px 12px rgba(0, 0, 0, 0.55)",
            }}
          >
            w={d.mergeWeight}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
};

const EvolutionCheckpointNode: React.FC<NodeProps> = ({ data }) => {
  const d = data as EvoNodeData;
  const n = d.node;

  const borderColor = d.isRegressionOrigin
    ? "rgba(248, 113, 113, 0.68)"
    : d.isSelectedInspect
      ? "rgba(244, 244, 245, 0.52)"
      : d.isMajorWeightStep
        ? "rgba(214, 197, 169, 0.55)"
        : "rgba(255, 255, 255, 0.1)";

  const metricLabel = d.categoryFilter ? formatCategoryShort(d.categoryFilter) : "Pass rate";
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
        width: 218,
        background: "var(--bg-node)",
        backdropFilter: "blur(20px)",
        border: `1px solid ${borderColor}`,
        borderRadius: 14,
        boxShadow: d.isSelectedInspect
          ? "0 22px 46px -12px rgba(0,0,0,0.92), 0 0 0 1px rgba(244,244,245,0.14), inset 0 1px 0 rgba(255,255,255,0.1)"
          : "var(--shadow-node)",
        color: "var(--text-main)",
        cursor: "pointer",
        overflow: "hidden",
        transition:
          "border-color 0.22s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.22s cubic-bezier(0.22, 1, 0.36, 1), transform 0.22s cubic-bezier(0.22, 1, 0.36, 1)",
      }}
    >
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: "#a1a1aa", width: 8, height: 8, border: "2px solid #09090b" }}
      />

      {/* Node Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "8.5px 12px",
          background: "var(--bg-node-header)",
          borderBottom: "1px solid var(--border-faint)",
          fontSize: 11.5,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 6.5, minWidth: 0 }}>
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
              letterSpacing: "-0.01em",
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
                borderRadius: 6,
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
      <div style={{ padding: "10px 12px", fontSize: 11.5 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginBottom: 5.5,
            color: "var(--text-muted)",
          }}
        >
          <span>Stage</span>
          <span className="mono" style={{ color: "var(--text-secondary)" }}>
            {n.transformation_type}
          </span>
        </div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginBottom: 5.5,
            color: "var(--text-muted)",
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
            marginBottom: 8,
            color: "var(--text-muted)",
          }}
        >
          <span>Base L2</span>
          <span
            className="mono"
            style={{ color: d.isMajorWeightStep ? "#d6c5a9" : "var(--text-secondary)" }}
          >
            {d.cumulativeL2 != null ? d.cumulativeL2.toFixed(4) : "—"}
          </span>
        </div>

        {/* Quick A / B comparison selectors */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            paddingTop: 6.5,
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
                fontWeight: 600,
                padding: "2px 7px",
                borderRadius: 6,
                border: d.isSelectedA
                  ? "1px solid rgba(244,244,245,0.38)"
                  : "1px solid var(--border-subtle)",
                background: d.isSelectedA ? "rgba(244,244,245,0.16)" : "transparent",
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
                fontWeight: 600,
                padding: "2px 7px",
                borderRadius: 6,
                border: d.isSelectedB
                  ? "1px solid rgba(214,197,169,0.42)"
                  : "1px solid var(--border-subtle)",
                background: d.isSelectedB ? "rgba(214,197,169,0.2)" : "transparent",
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
        style={{ background: "#e4e4e7", width: 8, height: 8, border: "2px solid #09090b" }}
      />
    </div>
  );
};

const EVO_NODE_TYPES = { evoCheckpoint: EvolutionCheckpointNode };
const EVO_EDGE_TYPES = { evoOptical: EvoOpticalEdge };

const DagViewportControls: React.FC<{ triggerKey: string }> = ({ triggerKey }) => {
  const { fitView, zoomIn, zoomOut } = useReactFlow();
  useEffect(() => {
    const t = setTimeout(() => {
      void fitView({ padding: 0.14, duration: 0 });
    }, 80);
    return () => clearTimeout(t);
  }, [fitView, triggerKey]);

  return (
    <div
      style={{
        position: "absolute",
        left: 12,
        bottom: 12,
        zIndex: 15,
        display: "inline-flex",
        alignItems: "center",
        gap: 2,
        background: "rgba(12, 12, 15, 0.86)",
        backdropFilter: "blur(16px)",
        border: "1px solid var(--border-subtle)",
        borderRadius: 10,
        padding: "3.5px 6px",
        boxShadow: "0 8px 24px rgba(0, 0, 0, 0.6)",
      }}
    >
      <button
        type="button"
        className="canvas-footer-btn"
        onClick={() => void zoomOut({ duration: 200 })}
        title="Zoom out"
        aria-label="Zoom out DAG"
      >
        <Minus size={11} />
      </button>
      <button
        type="button"
        className="canvas-footer-btn"
        onClick={() => void zoomIn({ duration: 200 })}
        title="Zoom in"
        aria-label="Zoom in DAG"
      >
        <Plus size={11} />
      </button>
      <span className="studio-divider-v" style={{ height: 11 }} />
      <button
        type="button"
        className="canvas-footer-btn"
        onClick={() => void fitView({ padding: 0.14, duration: 320 })}
        title="Fit graph"
      >
        <Maximize2 size={10} />
        <span>Fit</span>
      </button>
    </div>
  );
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
      if (
        !categoryFilter ||
        r.affected_category === categoryFilter ||
        r.affected_category === "overall"
      ) {
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
      const yOffset = (n.lane_index - (n.lane_count - 1) / 2) * 190;
      return {
        id: n.checkpoint_id,
        type: "evoCheckpoint",
        position: {
          x: 36 + n.generation_depth * 304,
          y: 168 + yOffset,
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
        const stroke = isRegEdge ? "#f87171" : isMajorWt ? "#d6c5a9" : "#a1a1aa";
        return {
          id: e.relationship_id,
          source: e.parent_checkpoint_id,
          target: e.child_checkpoint_id,
          type: "evoOptical",
          data: {
            stroke,
            active: isRegEdge || isMajorWt,
            mergeWeight: e.merge_weight,
          } satisfies EvoEdgeData,
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
          borderRadius: 12,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
          <GitBranch size={14} />
          <span className="mono">
            {visibleNodes.length} checkpoints · {graph.branch_count} branches
          </span>
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
        height: 452,
        background:
          "radial-gradient(760px 380px at 50% 45%, rgba(255,255,255,0.02), transparent 72%), #09090b",
        border: "1px solid var(--border-subtle)",
        borderRadius: 14,
        overflow: "hidden",
        position: "relative",
      }}
    >
      <ReactFlowProvider>
        <ReactFlow
          nodes={rfNodes}
          edges={rfEdges}
          nodeTypes={EVO_NODE_TYPES}
          edgeTypes={EVO_EDGE_TYPES}
          fitView
          fitViewOptions={{ padding: 0.14 }}
          minZoom={0.2}
          maxZoom={1.8}
          className="dark"
          proOptions={{ hideAttribution: true }}
        >
          <DagViewportControls triggerKey={`${rfNodes.length}-${rfEdges.length}`} />
          <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#1e1e24" />
          <MiniMap
            pannable
            zoomable
            nodeColor="#52525b"
            maskColor="rgba(9, 9, 11, 0.82)"
            style={{ bottom: 12, right: 12, width: 144, height: 88 }}
          />
        </ReactFlow>
      </ReactFlowProvider>
    </div>
  );
};
