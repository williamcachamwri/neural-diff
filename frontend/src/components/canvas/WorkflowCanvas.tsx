import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Background,
  BackgroundVariant,
  type Edge,
  MiniMap,
  type Node,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
} from "@xyflow/react";
import { GitCommitHorizontal, Maximize2, Spline } from "lucide-react";
import type {
  ComparisonBundle,
  ExperimentSummaryItem,
  HealthResponse,
  ViewId,
} from "../../types/api";
import { StudioModuleNode, type StudioNodePayload } from "../nodes/StudioNodes";
import { SignalEdge, type SignalEdgeData } from "./SignalEdge";

const NODE_TYPES = { studioModule: StudioModuleNode };
const EDGE_TYPES = { signal: SignalEdge };

const HEATMAP_ROLES = [
  "attn_q",
  "attn_k",
  "attn_v",
  "attn_o",
  "mlp_gate",
  "mlp_up",
  "mlp_down",
];

interface WorkflowCanvasProps {
  activeBundle: ComparisonBundle | null;
  health: HealthResponse | null;
  experiments: ExperimentSummaryItem[];
  modelA: string;
  modelB: string;
  activeView: ViewId;
  onSelectView: (view: ViewId) => void;
}

function buildStudioNodes(
  bundle: ComparisonBundle | null,
  modelA: string,
  modelB: string,
  activeView: ViewId,
  onInspect: (view: ViewId) => void
): Node[] {
  const metaA = bundle?.model_a;
  const metaB = bundle?.model_b;
  const comp = bundle?.structural?.compatibility;
  const st = bundle?.structural;
  const td = bundle?.tensor_diff;
  const na = bundle?.neuron_attention;
  const act = bundle?.activation;
  const beh = bundle?.behavioral;
  const reg = bundle?.regression;

  const layers = td?.layer_summaries ?? [];
  let maxHeat = 1e-6;
  const heatMatrix: number[][] = HEATMAP_ROLES.map((r) =>
    layers.map((ls) => {
      const v = ls.role_breakdown[r]?.mean_relative_l2 ?? 0;
      if (v > maxHeat) maxHeat = v;
      return v;
    })
  );

  const layerKeys = act
    ? Object.keys(act.summary.mean_hidden_cka_by_layer).sort((a, b) => Number(a) - Number(b))
    : [];
  const ckaSeries = layerKeys.map((k) => act?.summary.mean_hidden_cka_by_layer[k] ?? 1);
  const cosSeries = layerKeys.map((k) => act?.summary.mean_hidden_cosine_by_layer[k] ?? 1);

  const behBars = (beh?.category_summaries ?? []).map((cs) => ({
    label: cs.category,
    a: cs.mean_score_a,
    b: cs.mean_score_b,
  }));

  const verdictStr = reg?.verdict ?? "READY";
  const verdictDot =
    verdictStr === "PASSED" ? "#4ade80" : verdictStr === "READY" ? "#a1a1aa" : "#f87171";

  const defs: Array<{
    id: string;
    pos: { x: number; y: number };
    matchViews: ViewId[];
    data: StudioNodePayload;
  }> = [
    {
      id: "node-loader-a",
      pos: { x: 48, y: 64 },
      matchViews: ["compare"],
      data: {
        nodeIndex: "01",
        title: "Checkpoint A",
        accentColor: "#e4e4e7",
        targetView: "compare",
        outSocket: [{ id: "out-a", label: "model_a", color: "#a1a1aa" }],
        rows: [
          { key: "Model", value: (metaA?.model_id ?? modelA).slice(0, 24), highlight: "#f4f4f5" },
          {
            key: "Parameters",
            value: metaA ? `${(metaA.total_parameters / 1e6).toFixed(1)}M` : "—",
          },
          {
            key: "Precision",
            value: metaA
              ? `${metaA.shard_count} shard · ${metaA.dtypes_present[0] ?? "fp32"}`
              : "Safetensors",
          },
          {
            key: "Revision",
            value: metaA?.revision ? metaA.revision.slice(0, 10) : "local",
          },
        ],
        onInspect,
      },
    },
    {
      id: "node-loader-b",
      pos: { x: 48, y: 318 },
      matchViews: ["compare"],
      data: {
        nodeIndex: "02",
        title: "Checkpoint B",
        accentColor: "#e4e4e7",
        targetView: "compare",
        outSocket: [{ id: "out-b", label: "model_b", color: "#a1a1aa" }],
        rows: [
          { key: "Model", value: (metaB?.model_id ?? modelB).slice(0, 24), highlight: "#f4f4f5" },
          {
            key: "Parameters",
            value: metaB ? `${(metaB.total_parameters / 1e6).toFixed(1)}M` : "—",
          },
          {
            key: "Precision",
            value: metaB
              ? `${metaB.shard_count} shard · ${metaB.dtypes_present[0] ?? "fp32"}`
              : "Safetensors",
          },
          {
            key: "Revision",
            value: metaB?.revision ? metaB.revision.slice(0, 10) : "local",
          },
        ],
        onInspect,
      },
    },
    {
      id: "node-compat",
      pos: { x: 386, y: 64 },
      matchViews: ["compatibility"],
      data: {
        nodeIndex: "03",
        title: "Compatibility",
        accentColor: "#a1a1aa",
        targetView: "compatibility",
        inSocket: [
          { id: "in-a", label: "a", color: "#a1a1aa" },
          { id: "in-b", label: "b", color: "#a1a1aa" },
        ],
        outSocket: [{ id: "out-compat", label: "aligned", color: "#a1a1aa" }],
        rows: [
          {
            key: "Aligned tensors",
            value: comp
              ? `${comp.shape_matched_tensor_count} / ${comp.shared_tensor_count}`
              : "—",
            highlight: "#f4f4f5",
          },
          {
            key: "Key overlap",
            value: comp ? `${(comp.jaccard_key_overlap * 100).toFixed(1)}%` : "—",
          },
        ],
        visual: {
          kind: "status-leds",
          leds: [
            { label: "Tensor diff", ok: comp?.compatible_for_direct_tensor_diff ?? true },
            { label: "Head slicing", ok: comp?.compatible_for_head_neuron_analysis ?? true },
            { label: "Linear CKA", ok: comp?.compatible_for_activation_diff ?? true },
            { label: "Behavioral", ok: comp?.compatible_for_behavioral_diff ?? true },
          ],
        },
        onInspect,
      },
    },
    {
      id: "node-structural",
      pos: { x: 386, y: 318 },
      matchViews: ["structural"],
      data: {
        nodeIndex: "04",
        title: "Topology",
        accentColor: "#a1a1aa",
        targetView: "structural",
        inSocket: [{ id: "in-st", label: "models", color: "#a1a1aa" }],
        outSocket: [{ id: "out-st", label: "graph", color: "#a1a1aa" }],
        rows: [
          {
            key: "Param delta",
            value: st
              ? `${st.delta_parameters >= 0 ? "+" : ""}${st.delta_parameters.toLocaleString()}`
              : "0",
          },
          {
            key: "Key changes",
            value: st
              ? `+${st.added_tensors.length} / -${st.removed_tensors.length} / ~${st.inferred_renames.length}`
              : "+0 / -0 / ~0",
          },
          {
            key: "Shape · Dtype",
            value: st ? `${st.shape_changes.length} · ${st.dtype_changes.length}` : "0 · 0",
          },
          {
            key: "Blocks",
            value: st ? `${st.layer_deltas.length} layers` : "—",
          },
        ],
        onInspect,
      },
    },
    {
      id: "node-tensor",
      pos: { x: 724, y: 24 },
      matchViews: ["tensors"],
      data: {
        nodeIndex: "05",
        title: "Tensor Math",
        accentColor: "#d6c5a9",
        targetView: "tensors",
        inSocket: [{ id: "in-td", label: "aligned", color: "#a1a1aa" }],
        outSocket: [{ id: "out-td", label: "metrics", color: "#d6c5a9" }],
        rows: [
          {
            key: "Changed weights",
            value: td ? `${td.summary.global_changed_pct.toFixed(2)}%` : "—",
            highlight: "#f4f4f5",
          },
          {
            key: "Mean cosine",
            value:
              td?.summary.global_mean_cosine != null
                ? td.summary.global_mean_cosine.toFixed(6)
                : "—",
          },
          {
            key: "Mean rel L2",
            value:
              td?.summary.global_mean_relative_l2 != null
                ? td.summary.global_mean_relative_l2.toFixed(6)
                : "—",
          },
          {
            key: "Max |Δ| · Numerics",
            value: td
              ? `${td.summary.global_max_abs_diff?.toFixed(4) ?? "0"} · ${
                  td.summary.has_any_nan_or_inf ? "NaN" : "Clean"
                }`
              : "—",
          },
        ],
        onInspect,
      },
    },
    {
      id: "node-layers",
      pos: { x: 724, y: 254 },
      matchViews: ["layers"],
      data: {
        nodeIndex: "06",
        title: "Layer & Head Scope",
        accentColor: "#d6c5a9",
        targetView: "layers",
        inSocket: [{ id: "in-ly", label: "metrics", color: "#d6c5a9" }],
        outSocket: [{ id: "out-ly", label: "heatmap", color: "#d6c5a9" }],
        rows: [
          {
            key: "Head projections",
            value: na ? `${na.attention_projections.length} sliced` : "—",
          },
          {
            key: "Peak rel L2",
            value: layers.length > 0 ? maxHeat.toFixed(5) : "—",
            highlight: "#d6c5a9",
          },
        ],
        visual: {
          kind: "heatmap",
          matrix: heatMatrix,
          maxVal: maxHeat,
        },
        onInspect,
      },
    },
    {
      id: "node-activations",
      pos: { x: 724, y: 472 },
      matchViews: ["activations"],
      data: {
        nodeIndex: "07",
        title: "Activation CKA",
        accentColor: "#e4e4e7",
        targetView: "activations",
        inSocket: [{ id: "in-act", label: "graph", color: "#a1a1aa" }],
        outSocket: [{ id: "out-act", label: "cka", color: "#e4e4e7" }],
        rows: [
          {
            key: "Final-layer CKA",
            value:
              act?.summary.cross_prompt_pooled_cka_final_layer != null
                ? act.summary.cross_prompt_pooled_cka_final_layer.toFixed(5)
                : "—",
            highlight: "#f4f4f5",
          },
          {
            key: "Logit JS div",
            value:
              act?.summary.mean_final_token_js_divergence != null
                ? act.summary.mean_final_token_js_divergence.toFixed(5)
                : "—",
          },
        ],
        visual: {
          kind: "oscilloscope",
          cka: ckaSeries,
          cos: cosSeries,
        },
        onInspect,
      },
    },
    {
      id: "node-behavioral",
      pos: { x: 1062, y: 48 },
      matchViews: ["behavioral"],
      data: {
        nodeIndex: "08",
        title: "Behavioral Suite",
        accentColor: "#e4e4e7",
        targetView: "behavioral",
        inSocket: [{ id: "in-beh", label: "aligned", color: "#a1a1aa" }],
        outSocket: [{ id: "out-beh", label: "scores", color: "#e4e4e7" }],
        rows: [
          {
            key: "Pass rate",
            value: beh
              ? `${(beh.overall_statistics.pass_rate_a * 100).toFixed(1)}% → ${(
                  beh.overall_statistics.pass_rate_b * 100
                ).toFixed(1)}%`
              : "—",
            highlight:
              beh && beh.overall_statistics.pass_rate_delta >= 0 ? "#4ade80" : "#f87171",
          },
          {
            key: "Mean score Δ",
            value: beh
              ? `${beh.overall_statistics.mean_score_delta >= 0 ? "+" : ""}${beh.overall_statistics.mean_score_delta.toFixed(3)}`
              : "—",
          },
        ],
        visual: {
          kind: "bars",
          items: behBars,
        },
        onInspect,
      },
    },
    {
      id: "node-regression",
      pos: { x: 1062, y: 368 },
      matchViews: ["regression", "reports"],
      data: {
        nodeIndex: "09",
        title: "Regression Gate",
        accentColor: verdictDot,
        targetView: "regression",
        inSocket: [
          { id: "in-reg-td", label: "metrics", color: "#d6c5a9" },
          { id: "in-reg-act", label: "cka", color: "#e4e4e7" },
          { id: "in-reg-beh", label: "scores", color: "#e4e4e7" },
        ],
        rows: [
          {
            key: "Policy",
            value: reg ? String(reg.thresholds.policy_name ?? "default") : "default-v1",
          },
          {
            key: "Repro commands",
            value: reg ? `${reg.failure_reproductions.length}` : "0",
          },
        ],
        visual: {
          kind: "verdict",
          verdict: verdictStr,
          findings: reg?.findings.length ?? 0,
          comparisonId: bundle?.comparison_id ?? null,
        },
        onInspect,
      },
    },
    {
      id: "node-evolution",
      pos: { x: 386, y: 534 },
      matchViews: ["evolution"],
      data: {
        nodeIndex: "10",
        title: "Evolution Lineage",
        accentColor: "#d6c5a9",
        targetView: "evolution",
        inSocket: [{ id: "in-evo", label: "dag", color: "#d6c5a9" }],
        rows: [
          { key: "Topology", value: "Branch & Merge DAG", highlight: "#f4f4f5" },
          { key: "Scopes", value: "Step · Base · Branch" },
          { key: "Origin gate", value: "CKA · Behavioral" },
        ],
        onInspect,
      },
    },
  ];

  return defs.map((d) => ({
    id: d.id,
    type: "studioModule",
    position: d.pos,
    selected: d.matchViews.includes(activeView),
    data: d.data,
  }));
}

function buildStudioEdges(
  routing: "bezier" | "pcb",
  bundle: ComparisonBundle | null
): Edge[] {
  const pulse = bundle !== null;
  const specs: Array<{
    id: string;
    source: string;
    sourceHandle: string;
    target: string;
    targetHandle: string;
    color: string;
  }> = [
    {
      id: "e-a-compat",
      source: "node-loader-a",
      sourceHandle: "out-a",
      target: "node-compat",
      targetHandle: "in-a",
      color: "#a1a1aa",
    },
    {
      id: "e-b-compat",
      source: "node-loader-b",
      sourceHandle: "out-b",
      target: "node-compat",
      targetHandle: "in-b",
      color: "#a1a1aa",
    },
    {
      id: "e-b-struct",
      source: "node-loader-b",
      sourceHandle: "out-b",
      target: "node-structural",
      targetHandle: "in-st",
      color: "#71717a",
    },
    {
      id: "e-compat-td",
      source: "node-compat",
      sourceHandle: "out-compat",
      target: "node-tensor",
      targetHandle: "in-td",
      color: "#d6c5a9",
    },
    {
      id: "e-td-layers",
      source: "node-tensor",
      sourceHandle: "out-td",
      target: "node-layers",
      targetHandle: "in-ly",
      color: "#d6c5a9",
    },
    {
      id: "e-st-act",
      source: "node-structural",
      sourceHandle: "out-st",
      target: "node-activations",
      targetHandle: "in-act",
      color: "#a1a1aa",
    },
    {
      id: "e-compat-beh",
      source: "node-compat",
      sourceHandle: "out-compat",
      target: "node-behavioral",
      targetHandle: "in-beh",
      color: "#a1a1aa",
    },
    {
      id: "e-td-reg",
      source: "node-tensor",
      sourceHandle: "out-td",
      target: "node-regression",
      targetHandle: "in-reg-td",
      color: "#d6c5a9",
    },
    {
      id: "e-act-reg",
      source: "node-activations",
      sourceHandle: "out-act",
      target: "node-regression",
      targetHandle: "in-reg-act",
      color: "#a1a1aa",
    },
    {
      id: "e-beh-reg",
      source: "node-behavioral",
      sourceHandle: "out-beh",
      target: "node-regression",
      targetHandle: "in-reg-beh",
      color: "#e4e4e7",
    },
    {
      id: "e-st-evo",
      source: "node-structural",
      sourceHandle: "out-st",
      target: "node-evolution",
      targetHandle: "in-evo",
      color: "#d6c5a9",
    },
  ];

  return specs.map((s) => ({
    id: s.id,
    source: s.source,
    sourceHandle: s.sourceHandle,
    target: s.target,
    targetHandle: s.targetHandle,
    type: "signal",
    data: {
      color: s.color,
      routing,
      animatedPulse: pulse,
    } satisfies SignalEdgeData,
  }));
}

const InnerCanvas: React.FC<WorkflowCanvasProps> = ({
  activeBundle,
  modelA,
  modelB,
  activeView,
  onSelectView,
}) => {
  const [routingMode, setRoutingMode] = useState<"bezier" | "pcb">("bezier");
  const { fitView } = useReactFlow();

  const onInspect = useCallback(
    (view: ViewId) => {
      onSelectView(view);
    },
    [onSelectView]
  );

  const computedNodes = useMemo(
    () => buildStudioNodes(activeBundle, modelA, modelB, activeView, onInspect),
    [activeBundle, modelA, modelB, activeView, onInspect]
  );
  const computedEdges = useMemo(
    () => buildStudioEdges(routingMode, activeBundle),
    [routingMode, activeBundle]
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(computedNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(computedEdges);

  useEffect(() => {
    setNodes((prev) => {
      const posMap = new Map(prev.map((n) => [n.id, n.position]));
      return computedNodes.map((cn) => ({
        ...cn,
        position: posMap.get(cn.id) ?? cn.position,
      }));
    });
  }, [computedNodes, setNodes]);

  useEffect(() => {
    setEdges(computedEdges);
  }, [computedEdges, setEdges]);

  const handleResetLayout = useCallback(() => {
    setNodes(computedNodes);
    setTimeout(() => {
      void fitView({ padding: 0.12, duration: 420 });
    }, 30);
  }, [computedNodes, setNodes, fitView]);

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={NODE_TYPES}
        edgeTypes={EDGE_TYPES}
        onNodeClick={(_evt, node) => {
          const payload = node.data as StudioNodePayload;
          if (payload?.targetView) {
            onSelectView(payload.targetView);
          }
        }}
        fitView
        fitViewOptions={{ padding: 0.12 }}
        minZoom={0.35}
        maxZoom={1.75}
        className="dark"
        proOptions={{ hideAttribution: true }}
      >
        <Background
          id="minor-dots"
          variant={BackgroundVariant.Dots}
          gap={24}
          size={1}
          color="#1e1e24"
        />
        <MiniMap
          pannable
          zoomable
          nodeColor={(n) => {
            const p = n.data as StudioNodePayload | undefined;
            return p?.accentColor ?? "#52525b";
          }}
          maskColor="rgba(9, 9, 11, 0.8)"
          style={{ bottom: 14, right: 16, width: 170, height: 108 }}
        />
      </ReactFlow>

      {/* Subtle Cable & Viewport Controls integrated into Bottom-Right beside MiniMap */}
      <div
        style={{
          position: "absolute",
          right: 196,
          bottom: 14,
          zIndex: 20,
          display: "inline-flex",
          alignItems: "center",
          gap: 4,
          background: "var(--bg-chrome)",
          backdropFilter: "blur(20px)",
          border: "1px solid var(--border-subtle)",
          borderRadius: 9,
          padding: "4px 8px",
        }}
      >
        <button
          type="button"
          className="canvas-footer-btn"
          onClick={() => setRoutingMode((m) => (m === "bezier" ? "pcb" : "bezier"))}
          title="Switch wire routing geometry"
        >
          {routingMode === "bezier" ? <Spline size={12} /> : <GitCommitHorizontal size={12} />}
          <span>Cables: {routingMode === "bezier" ? "Bezier Spline" : "Orthogonal PCB"}</span>
        </button>
        <span className="studio-divider-v" style={{ height: 12 }} />
        <button
          type="button"
          className="canvas-footer-btn"
          onClick={handleResetLayout}
          title="Reset node layout"
        >
          <Maximize2 size={11} />
          <span>Fit</span>
        </button>
      </div>
    </div>
  );
};

export const WorkflowCanvas: React.FC<WorkflowCanvasProps> = (props) => {
  if (typeof globalThis.ResizeObserver === "undefined") {
    return (
      <div
        data-testid="comfy-canvas-fallback"
        style={{ width: "100%", height: "100%", background: "var(--bg-void)" }}
      />
    );
  }
  return (
    <ReactFlowProvider>
      <InnerCanvas {...props} />
    </ReactFlowProvider>
  );
};
