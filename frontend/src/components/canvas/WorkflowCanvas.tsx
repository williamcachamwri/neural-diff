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
import { SignalEdge } from "./SignalEdge";

const NODE_TYPES = { studioModule: StudioModuleNode };
const EDGE_TYPES = { signal: SignalEdge };

const HEAT_ROLES = ["attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_down"];

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
  onInspect: (v: ViewId) => void
): Node[] {
  const metaA = bundle?.model_a;
  const metaB = bundle?.model_b;
  const comp = bundle?.structural?.compatibility;
  const st = bundle?.structural;
  const td = bundle?.tensor_diff;
  const act = bundle?.activation;
  const beh = bundle?.behavioral;
  const reg = bundle?.regression;

  // Build 6xN matrix for Layer Probe inline micro-heatmap
  const layerList = td?.layer_summaries ?? [];
  let maxRel = 1e-5;
  const matrix: number[][] = HEAT_ROLES.map((role) =>
    layerList.slice(0, 24).map((ls) => {
      const v = ls.role_breakdown[role]?.mean_relative_l2 ?? 0;
      if (v > maxRel) maxRel = v;
      return v;
    })
  );

  // Build CKA & Cosine arrays for Activation Oscilloscope
  const actKeys = act
    ? Object.keys(act.summary.mean_hidden_cka_by_layer).sort((a, b) => Number(a) - Number(b))
    : [];
  const ckaSeries = actKeys.map((k) => act?.summary.mean_hidden_cka_by_layer[k] ?? 1);
  const cosSeries = actKeys.map((k) => act?.summary.mean_hidden_cosine_by_layer[k] ?? 1);

  const behBars = (beh?.category_summaries ?? []).map((cs) => ({
    label: cs.category,
    a: cs.mean_score_a,
    b: cs.mean_score_b,
  }));

  const verdictStr = reg?.verdict ?? "READY";
  const verdictColor =
    verdictStr === "PASSED" ? "#34d399" : verdictStr === "READY" ? "#e5b869" : "#fb7185";

  const defs: Array<{
    id: string;
    pos: { x: number; y: number };
    matchViews: ViewId[];
    data: StudioNodePayload;
  }> = [
    {
      id: "node-loader-a",
      pos: { x: 75, y: 82 },
      matchViews: ["compare"],
      data: {
        nodeIndex: "01",
        title: "Checkpoint Loader A",
        accentColor: "#e5b869",
        targetView: "compare",
        outSocket: [{ id: "out-a", label: "MODEL_A", color: "#e5b869" }],
        rows: [
          { key: "Model ID", value: (metaA?.model_id ?? modelA).slice(0, 24), highlight: "#f5efe4" },
          {
            key: "Parameters",
            value: metaA ? `${(metaA.total_parameters / 1e6).toFixed(1)}M` : "Pending",
          },
          {
            key: "Format / Dtype",
            value: metaA
              ? `${metaA.shard_count}sh • ${metaA.dtypes_present[0] ?? "fp32"}`
              : "Safetensors",
          },
          {
            key: "Revision",
            value: metaA?.revision ? metaA.revision.slice(0, 10) : "local/verified",
          },
        ],
        onInspect,
      },
    },
    {
      id: "node-loader-b",
      pos: { x: 75, y: 332 },
      matchViews: ["compare"],
      data: {
        nodeIndex: "02",
        title: "Checkpoint Loader B",
        accentColor: "#e5b869",
        targetView: "compare",
        outSocket: [{ id: "out-b", label: "MODEL_B", color: "#e5b869" }],
        rows: [
          { key: "Model ID", value: (metaB?.model_id ?? modelB).slice(0, 24), highlight: "#f5efe4" },
          {
            key: "Parameters",
            value: metaB ? `${(metaB.total_parameters / 1e6).toFixed(1)}M` : "Pending",
          },
          {
            key: "Format / Dtype",
            value: metaB
              ? `${metaB.shard_count}sh • ${metaB.dtypes_present[0] ?? "fp32"}`
              : "Safetensors",
          },
          {
            key: "Revision",
            value: metaB?.revision ? metaB.revision.slice(0, 10) : "local/verified",
          },
        ],
        onInspect,
      },
    },
    {
      id: "node-compat",
      pos: { x: 420, y: 82 },
      matchViews: ["compatibility"],
      data: {
        nodeIndex: "03",
        title: "Compatibility Gate",
        accentColor: "#d4d4d8",
        targetView: "compatibility",
        inSocket: [
          { id: "in-a", label: "MODEL_A", color: "#e5b869" },
          { id: "in-b", label: "MODEL_B", color: "#e5b869" },
        ],
        outSocket: [{ id: "out-compat", label: "ALIGNED", color: "#d4d4d8" }],
        rows: [
          {
            key: "Shape Aligned",
            value: comp
              ? `${comp.shape_matched_tensor_count} / ${comp.shared_tensor_count}`
              : "—",
          },
          {
            key: "Jaccard Overlap",
            value: comp ? `${(comp.jaccard_key_overlap * 100).toFixed(1)}%` : "—",
            highlight: "#34d399",
          },
        ],
        visual: {
          kind: "status-leds",
          leds: [
            { label: "TENSORS", ok: comp?.compatible_for_direct_tensor_diff ?? true },
            { label: "GQA HEADS", ok: comp?.compatible_for_head_neuron_analysis ?? true },
            { label: "HOOK CKA", ok: comp?.compatible_for_activation_diff ?? true },
            { label: "BEHAVIOR", ok: comp?.compatible_for_behavioral_diff ?? true },
          ],
        },
        onInspect,
      },
    },
    {
      id: "node-structural",
      pos: { x: 420, y: 338 },
      matchViews: ["structural"],
      data: {
        nodeIndex: "04",
        title: "Structural Topology",
        accentColor: "#a78bfa",
        targetView: "structural",
        inSocket: [{ id: "in-st", label: "MODELS", color: "#e5b869" }],
        outSocket: [{ id: "out-st", label: "TOPOLOGY", color: "#a78bfa" }],
        rows: [
          {
            key: "Param Delta",
            value: st
              ? `${st.delta_parameters >= 0 ? "+" : ""}${st.delta_parameters.toLocaleString()}`
              : "0",
          },
          {
            key: "+Add / -Rem / ~Ren",
            value: st
              ? `+${st.added_tensors.length} / -${st.removed_tensors.length} / ~${st.inferred_renames.length}`
              : "+0 / -0 / ~0",
          },
          {
            key: "Shape / Dtype Δ",
            value: st ? `${st.shape_changes.length} / ${st.dtype_changes.length}` : "0 / 0",
          },
          {
            key: "Indexed Layers",
            value: st ? `${st.layer_deltas.length} blocks` : "—",
          },
        ],
        onInspect,
      },
    },
    {
      id: "node-tensor",
      pos: { x: 765, y: 42 },
      matchViews: ["tensors"],
      data: {
        nodeIndex: "05",
        title: "Float64 Tensor Diff",
        accentColor: "#34d399",
        targetView: "tensors",
        inSocket: [{ id: "in-td", label: "ALIGNED", color: "#d4d4d8" }],
        outSocket: [{ id: "out-td", label: "METRICS", color: "#34d399" }],
        rows: [
          {
            key: "Changed Params",
            value: td ? `${td.summary.global_changed_pct.toFixed(2)}%` : "—",
            highlight: "#e5b869",
          },
          {
            key: "Mean Cosine",
            value:
              td?.summary.global_mean_cosine != null
                ? td.summary.global_mean_cosine.toFixed(6)
                : "—",
          },
          {
            key: "Mean Rel L2",
            value:
              td?.summary.global_mean_relative_l2 != null
                ? td.summary.global_mean_relative_l2.toFixed(6)
                : "—",
          },
          {
            key: "Max |Δ| / NaN",
            value: td
              ? `${td.summary.global_max_abs_diff?.toFixed(4) ?? "0"} • ${
                  td.summary.has_any_nan_or_inf ? "WARN" : "CLEAN"
                }`
              : "—",
          },
        ],
        onInspect,
      },
    },
    {
      id: "node-layers",
      pos: { x: 765, y: 242 },
      matchViews: ["layers"],
      data: {
        nodeIndex: "06",
        title: "Layer & GQA Head Probe",
        accentColor: "#fb923c",
        targetView: "layers",
        inSocket: [{ id: "in-ly", label: "METRICS", color: "#34d399" }],
        outSocket: [{ id: "out-ly", label: "HEATMAP", color: "#fb923c" }],
        rows: [
          {
            key: "Head Projections",
            value: bundle?.neuron_attention
              ? `${bundle.neuron_attention.attention_projections.length} sliced`
              : `${layerList.length} layers`,
          },
          {
            key: "Peak Submodule L2",
            value: layerList.length > 0 ? maxRel.toFixed(5) : "—",
            highlight: "#fb923c",
          },
        ],
        visual:
          layerList.length > 0
            ? { kind: "heatmap", matrix, maxVal: maxRel }
            : undefined,
        onInspect,
      },
    },
    {
      id: "node-activation",
      pos: { x: 765, y: 452 },
      matchViews: ["activations"],
      data: {
        nodeIndex: "07",
        title: "Activation CKA Scope",
        accentColor: "#7dd3fc",
        targetView: "activations",
        inSocket: [{ id: "in-act", label: "TOPOLOGY", color: "#a78bfa" }],
        outSocket: [{ id: "out-act", label: "CKA_TRACE", color: "#7dd3fc" }],
        rows: [
          {
            key: "Pooled Final CKA",
            value:
              act?.summary.cross_prompt_pooled_cka_final_layer != null
                ? act.summary.cross_prompt_pooled_cka_final_layer.toFixed(5)
                : "Not selected",
            highlight: "#7dd3fc",
          },
          {
            key: "Next-Tok JS Div",
            value:
              act?.summary.mean_final_token_js_divergence != null
                ? act.summary.mean_final_token_js_divergence.toFixed(5)
                : "—",
          },
        ],
        visual:
          ckaSeries.length > 1
            ? { kind: "oscilloscope", cka: ckaSeries, cos: cosSeries }
            : undefined,
        onInspect,
      },
    },
    {
      id: "node-behavior",
      pos: { x: 1115, y: 65 },
      matchViews: ["behavioral"],
      data: {
        nodeIndex: "08",
        title: "8-Suite Behavioral Eval",
        accentColor: "#2dd4bf",
        targetView: "behavioral",
        inSocket: [{ id: "in-beh", label: "ALIGNED", color: "#d4d4d8" }],
        outSocket: [{ id: "out-beh", label: "SCORES", color: "#2dd4bf" }],
        rows: [
          {
            key: "Pass Rate A → B",
            value: beh
              ? `${(beh.overall_statistics.pass_rate_a * 100).toFixed(1)}% → ${(
                  beh.overall_statistics.pass_rate_b * 100
                ).toFixed(1)}%`
              : "Not executed",
            highlight: "#34d399",
          },
          {
            key: "Mean Δ / Judge",
            value: beh
              ? `${beh.overall_statistics.mean_score_delta >= 0 ? "+" : ""}${beh.overall_statistics.mean_score_delta.toFixed(3)} • Determ.`
              : "Zero-LLM-Judge",
          },
        ],
        visual: behBars.length > 0 ? { kind: "bars", items: behBars } : undefined,
        onInspect,
      },
    },
    {
      id: "node-regression",
      pos: { x: 1115, y: 368 },
      matchViews: ["regression", "reports"],
      data: {
        nodeIndex: "09",
        title: "Regression & Report Sink",
        accentColor: verdictColor,
        targetView: "regression",
        inSocket: [
          { id: "in-r1", label: "METRICS", color: "#34d399" },
          { id: "in-r2", label: "CKA_TRACE", color: "#7dd3fc" },
          { id: "in-r3", label: "SCORES", color: "#2dd4bf" },
        ],
        outSocket: [{ id: "out-rep", label: "BUNDLE", color: verdictColor }],
        rows: [
          {
            key: "Policy",
            value: String(reg?.thresholds.policy_name ?? "production-default-v1").slice(0, 22),
          },
          {
            key: "CLI Repros",
            value: reg ? `${reg.failure_reproductions.length} generated` : "0",
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
  ];

  return defs.map((d) => ({
    id: d.id,
    type: "studioModule",
    position: d.pos,
    selected: d.matchViews.includes(activeView),
    data: d.data,
  }));
}

function buildStudioEdges(routingMode: "bezier" | "pcb", bundle: ComparisonBundle | null): Edge[] {
  const hasAct = Boolean(bundle?.activation);
  const hasBeh = Boolean(bundle?.behavioral);
  const specs: Array<{
    id: string;
    source: string;
    sourceHandle: string;
    target: string;
    targetHandle: string;
    color: string;
    active: boolean;
  }> = [
    {
      id: "e-a-compat",
      source: "node-loader-a",
      sourceHandle: "out-a",
      target: "node-compat",
      targetHandle: "in-a",
      color: "#e5b869",
      active: true,
    },
    {
      id: "e-b-compat",
      source: "node-loader-b",
      sourceHandle: "out-b",
      target: "node-compat",
      targetHandle: "in-b",
      color: "#e5b869",
      active: true,
    },
    {
      id: "e-b-struct",
      source: "node-loader-b",
      sourceHandle: "out-b",
      target: "node-structural",
      targetHandle: "in-st",
      color: "#a78bfa",
      active: true,
    },
    {
      id: "e-compat-tensor",
      source: "node-compat",
      sourceHandle: "out-compat",
      target: "node-tensor",
      targetHandle: "in-td",
      color: "#34d399",
      active: true,
    },
    {
      id: "e-tensor-layers",
      source: "node-tensor",
      sourceHandle: "out-td",
      target: "node-layers",
      targetHandle: "in-ly",
      color: "#fb923c",
      active: true,
    },
    {
      id: "e-struct-act",
      source: "node-structural",
      sourceHandle: "out-st",
      target: "node-activation",
      targetHandle: "in-act",
      color: "#7dd3fc",
      active: hasAct,
    },
    {
      id: "e-compat-beh",
      source: "node-compat",
      sourceHandle: "out-compat",
      target: "node-behavior",
      targetHandle: "in-beh",
      color: "#2dd4bf",
      active: hasBeh,
    },
    {
      id: "e-tensor-reg",
      source: "node-tensor",
      sourceHandle: "out-td",
      target: "node-regression",
      targetHandle: "in-r1",
      color: "#34d399",
      active: true,
    },
    {
      id: "e-act-reg",
      source: "node-activation",
      sourceHandle: "out-act",
      target: "node-regression",
      targetHandle: "in-r2",
      color: "#7dd3fc",
      active: hasAct,
    },
    {
      id: "e-beh-reg",
      source: "node-behavior",
      sourceHandle: "out-beh",
      target: "node-regression",
      targetHandle: "in-r3",
      color: "#2dd4bf",
      active: hasBeh,
    },
  ];

  return specs.map((s) => ({
    id: s.id,
    source: s.source,
    sourceHandle: s.sourceHandle,
    target: s.target,
    targetHandle: s.targetHandle,
    type: "signal",
    data: { color: s.color, routingMode, active: s.active },
  }));
}

const InnerCanvas: React.FC<WorkflowCanvasProps> = ({
  activeBundle,
  health,
  experiments,
  modelA,
  modelB,
  activeView,
  onSelectView,
}) => {
  const [routingMode, setRoutingMode] = useState<"bezier" | "pcb">("bezier");
  const { fitView } = useReactFlow();

  const computedNodes = useMemo(
    () => buildStudioNodes(activeBundle, modelA, modelB, activeView, onSelectView),
    [activeBundle, modelA, modelB, activeView, onSelectView]
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
      void fitView({ padding: 0.12, duration: 480 });
    }, 30);
  }, [computedNodes, setNodes, fitView]);

  const realCount = experiments.filter(
    (e) => e.experiment_type === "REAL MODEL EXPERIMENT" && e.status === "completed"
  ).length;
  const synthCount = experiments.filter(
    (e) => e.experiment_type === "SYNTHETIC FIXTURE EXPERIMENT" && e.status === "completed"
  ).length;

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
          gap={22}
          size={1.25}
          color="#24242c"
        />
        <MiniMap
          pannable
          zoomable
          nodeColor={(n) => {
            const p = n.data as StudioNodePayload | undefined;
            return p?.accentColor ?? "#e5b869";
          }}
          maskColor="rgba(7, 7, 10, 0.76)"
          style={{ bottom: 16, right: 16, width: 185, height: 118 }}
        />
      </ReactFlow>

      {/* Bottom-Left Studio Telemetry & Cable Routing Bar */}
      <div className="canvas-bottom-hud">
        <div className="hud-pill mono">
          <span
            style={{
              width: 7,
              height: 7,
              borderRadius: 99,
              background: "var(--ok)",
              boxShadow: "0 0 8px #34d399",
            }}
          />
          <span>{health ? health.device.toUpperCase() : "CUDA:0"}</span>
          <span style={{ color: "var(--text-muted)" }}>|</span>
          <span>
            {health
              ? `${(health.hardware.cuda_total_vram_mb / 1024).toFixed(1)} GB VRAM`
              : "RTX 5050"}
          </span>
          <span style={{ color: "var(--text-muted)" }}>|</span>
          <span style={{ color: "var(--gold)" }}>{realCount} Real</span>
          <span>•</span>
          <span>{synthCount} Synth</span>
        </div>

        <button
          type="button"
          className="hud-pill"
          style={{ cursor: "pointer" }}
          onClick={() => setRoutingMode((m) => (m === "bezier" ? "pcb" : "bezier"))}
          title="Toggle ComfyUI Bezier Spline vs Nodes 2.0 Orthogonal PCB Cable Routing"
        >
          {routingMode === "bezier" ? <Spline size={13} color="#e5b869" /> : <GitCommitHorizontal size={13} color="#e5b869" />}
          <span className="mono">
            Cables: {routingMode === "bezier" ? "Bezier Spline" : "Orthogonal PCB"}
          </span>
        </button>

        <button
          type="button"
          className="hud-pill"
          style={{ cursor: "pointer" }}
          onClick={handleResetLayout}
          title="Snap DAG nodes to studio grid and fit viewport"
        >
          <Maximize2 size={12} color="#d4d4d8" />
          <span className="mono">Fit DAG</span>
        </button>
      </div>
    </div>
  );
};

export const WorkflowCanvas: React.FC<WorkflowCanvasProps> = (props) => {
  // Ensure ResizeObserver exists in headless unit-test environments (e.g. Vitest jsdom)
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
