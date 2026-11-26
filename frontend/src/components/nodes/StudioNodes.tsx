import React, { useState } from "react";
import { Handle, type NodeProps, Position } from "@xyflow/react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowUpRight, ChevronDown } from "lucide-react";
import type { ViewId } from "../../types/api";

export interface StudioNodePayload extends Record<string, unknown> {
  nodeIndex: string;
  title: string;
  accentColor: string;
  targetView: ViewId;
  inSocket?: { id: string; label: string; color: string }[];
  outSocket?: { id: string; label: string; color: string }[];
  rows: Array<{ key: string; value: string; highlight?: string }>;
  visual?:
    | { kind: "heatmap"; matrix: number[][]; maxVal: number }
    | { kind: "oscilloscope"; cka: number[]; cos: number[] }
    | { kind: "bars"; items: Array<{ label: string; a: number; b: number }> }
    | { kind: "status-leds"; leds: Array<{ label: string; ok: boolean }> }
    | { kind: "verdict"; verdict: string; findings: number; comparisonId: string | null };
  onInspect: (view: ViewId) => void;
}

export const StudioModuleNode: React.FC<NodeProps> = ({ data, selected }) => {
  const payload = data as StudioNodePayload;
  const [collapsed, setCollapsed] = useState(false);

  const inPins = payload.inSocket ?? [];
  const outPins = payload.outSocket ?? [];

  return (
    <div className={`comfy-node ${selected ? "is-selected" : ""}`}>
      {/* Input Handles */}
      {inPins.map((pin, idx) => (
        <Handle
          key={pin.id}
          id={pin.id}
          type="target"
          position={Position.Left}
          style={{
            top: `${38 + idx * 22}px`,
            background: pin.color,
            boxShadow: `0 0 10px ${pin.color}66`,
          }}
          title={pin.label}
        />
      ))}

      {/* Header */}
      <div className="comfy-node-header">
        <div className="comfy-node-title-wrap">
          <span
            className="comfy-node-led"
            style={{
              background: payload.accentColor,
              boxShadow: `0 0 10px ${payload.accentColor}`,
            }}
          />
          <span className="comfy-node-title">{payload.title}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <span className="comfy-node-id">{payload.nodeIndex}</span>
          <button
            type="button"
            className="nodrag"
            aria-label={`Collapse ${payload.title}`}
            onClick={(e) => {
              e.stopPropagation();
              setCollapsed((c) => !c);
            }}
            style={{
              background: "transparent",
              border: "none",
              color: "var(--text-muted)",
              cursor: "pointer",
              padding: 2,
              display: "grid",
              placeItems: "center",
            }}
          >
            <motion.span
              animate={{ rotate: collapsed ? -90 : 0 }}
              transition={{ duration: 0.18 }}
              style={{ display: "grid" }}
            >
              <ChevronDown size={13} />
            </motion.span>
          </button>
          <button
            type="button"
            className="nodrag"
            aria-label={`Inspect ${payload.title}`}
            onClick={(e) => {
              e.stopPropagation();
              payload.onInspect(payload.targetView);
            }}
            style={{
              background: "rgba(255,255,255,0.05)",
              border: "1px solid var(--border-subtle)",
              borderRadius: 5,
              color: "var(--gold)",
              cursor: "pointer",
              padding: "2px 4px",
              display: "grid",
              placeItems: "center",
            }}
          >
            <ArrowUpRight size={12} />
          </button>
        </div>
      </div>

      {/* Collapsible Telemetry & Inline Micro-Scope Body */}
      <AnimatePresence initial={false}>
        {!collapsed && (
          <motion.div
            key="body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
            style={{ overflow: "hidden" }}
          >
            <div className="comfy-node-body">
              {payload.rows.map((r) => (
                <div key={r.key} className="comfy-kv-row">
                  <span className="comfy-kv-key">{r.key}</span>
                  <span
                    className="comfy-kv-val"
                    style={r.highlight ? { color: r.highlight, fontWeight: 600 } : undefined}
                  >
                    {r.value}
                  </span>
                </div>
              ))}

              {/* Inline Visual 1: 24-Layer Pixel Matrix Heatmap */}
              {payload.visual?.kind === "heatmap" && payload.visual.matrix.length > 0 && (
                <div
                  style={{
                    background: "#09090c",
                    border: "1px solid var(--border-subtle)",
                    borderRadius: 8,
                    padding: 6,
                    marginTop: 2,
                  }}
                >
                  <div
                    style={{
                      display: "grid",
                      gridTemplateRows: `repeat(${payload.visual.matrix.length}, 6px)`,
                      gap: 2,
                    }}
                  >
                    {payload.visual.matrix.map((rowVals, rIdx) => (
                      <div
                        key={rIdx}
                        style={{
                          display: "grid",
                          gridTemplateColumns: `repeat(${rowVals.length}, 1fr)`,
                          gap: 1.5,
                        }}
                      >
                        {rowVals.map((v, cIdx) => {
                          const norm = Math.min(
                            1,
                            Math.max(0, v / Math.max(payload.visual && payload.visual.kind === "heatmap" ? payload.visual.maxVal : 1, 1e-6))
                          );
                          const alpha = (0.14 + norm * 0.86).toFixed(2);
                          return (
                            <span
                              key={cIdx}
                              style={{
                                borderRadius: 1.5,
                                background:
                                  norm > 0.68
                                    ? `rgba(229, 184, 105, ${alpha})`
                                    : `rgba(52, 211, 153, ${alpha})`,
                              }}
                            />
                          );
                        })}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Inline Visual 2: Layer CKA & Cosine Oscilloscope */}
              {payload.visual?.kind === "oscilloscope" && payload.visual.cka.length > 1 && (
                <div
                  style={{
                    background: "#09090c",
                    border: "1px solid var(--border-subtle)",
                    borderRadius: 8,
                    padding: "6px 8px 4px",
                  }}
                >
                  <OscilloscopeMini
                    cka={payload.visual.cka}
                    cos={payload.visual.cos}
                  />
                </div>
              )}

              {/* Inline Visual 3: 8-Suite Behavioral Meter Tracks */}
              {payload.visual?.kind === "bars" && payload.visual.items.length > 0 && (
                <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 2 }}>
                  {payload.visual.items.slice(0, 6).map((it) => (
                    <div
                      key={it.label}
                      style={{ display: "grid", gridTemplateColumns: "58px 1fr 38px", alignItems: "center", gap: 6 }}
                    >
                      <span className="mono" style={{ fontSize: 9.5, color: "var(--text-muted)" }}>
                        {it.label.slice(0, 7)}
                      </span>
                      <div style={{ height: 5, background: "#09090c", borderRadius: 99, overflow: "hidden", position: "relative" }}>
                        <div
                          style={{
                            width: `${Math.round(it.b * 100)}%`,
                            height: "100%",
                            background: it.b >= it.a ? "var(--ok)" : "var(--danger)",
                            borderRadius: 99,
                          }}
                        />
                      </div>
                      <span className="mono" style={{ fontSize: 9.5, textAlign: "right", color: "var(--text-secondary)" }}>
                        {Math.round(it.b * 100)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* Inline Visual 4: Compatibility Hardware LEDs */}
              {payload.visual?.kind === "status-leds" && (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 5, marginTop: 2 }}>
                  {payload.visual.leds.map((l) => (
                    <div
                      key={l.label}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        background: "#0b0b0f",
                        border: "1px solid var(--border-subtle)",
                        borderRadius: 6,
                        padding: "4px 7px",
                        fontSize: 10,
                      }}
                    >
                      <span
                        style={{
                          width: 6,
                          height: 6,
                          borderRadius: 99,
                          background: l.ok ? "var(--ok)" : "var(--warn)",
                          boxShadow: `0 0 6px ${l.ok ? "#34d399" : "#f59e0b"}`,
                        }}
                      />
                      <span className="mono" style={{ fontSize: 9.5 }}>{l.label}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Inline Visual 5: Regression Verdict & Direct Artifact Export */}
              {payload.visual?.kind === "verdict" && (
                <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 2 }}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      padding: "6px 9px",
                      borderRadius: 8,
                      background:
                        payload.visual.verdict === "PASSED"
                          ? "rgba(52, 211, 153, 0.1)"
                          : "rgba(251, 113, 133, 0.1)",
                      border: `1px solid ${
                        payload.visual.verdict === "PASSED"
                          ? "rgba(52, 211, 153, 0.35)"
                          : "rgba(251, 113, 133, 0.35)"
                      }`,
                    }}
                  >
                    <span
                      className="mono"
                      style={{
                        fontSize: 10.5,
                        fontWeight: 700,
                        color:
                          payload.visual.verdict === "PASSED" ? "var(--ok)" : "var(--danger)",
                      }}
                    >
                      {payload.visual.verdict}
                    </span>
                    <span className="mono" style={{ fontSize: 10, color: "var(--text-muted)" }}>
                      {payload.visual.findings} finding(s)
                    </span>
                  </div>
                  {payload.visual.comparisonId && (
                    <div style={{ display: "flex", gap: 5 }} className="nodrag">
                      {(["html", "markdown", "json"] as const).map((fmt) => (
                        <a
                          key={fmt}
                          href={`/api/comparisons/${encodeURIComponent(payload.visual && payload.visual.kind === "verdict" && payload.visual.comparisonId ? payload.visual.comparisonId : "")}/export/${fmt}`}
                          download
                          onClick={(e) => e.stopPropagation()}
                          style={{
                            flex: 1,
                            textAlign: "center",
                            padding: "4px 0",
                            borderRadius: 6,
                            background: "#0b0b0f",
                            border: "1px solid var(--border)",
                            color: "var(--ivory)",
                            fontSize: 10,
                            fontFamily: "JetBrains Mono, monospace",
                            textDecoration: "none",
                          }}
                        >
                          .{fmt === "markdown" ? "md" : fmt}
                        </a>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Footer Typed Socket Labels */}
      <div className="comfy-socket-bar">
        <span>{inPins.map((p) => `● ${p.label}`).join("  ") || "SOURCE"}</span>
        <span style={{ color: payload.accentColor }}>
          {outPins.map((p) => `${p.label} ●`).join("  ") || "SINK"}
        </span>
      </div>

      {/* Output Handles */}
      {outPins.map((pin, idx) => (
        <Handle
          key={pin.id}
          id={pin.id}
          type="source"
          position={Position.Right}
          style={{
            top: `${38 + idx * 22}px`,
            background: pin.color,
            boxShadow: `0 0 10px ${pin.color}66`,
          }}
          title={pin.label}
        />
      ))}
    </div>
  );
};

const OscilloscopeMini: React.FC<{ cka: number[]; cos: number[] }> = ({ cka, cos }) => {
  const w = 244;
  const h = 42;
  const toPath = (arr: number[]) => {
    if (arr.length === 0) return "";
    const minV = Math.min(...arr, 0.85);
    const span = Math.max(1 - minV, 0.02);
    return arr
      .map((v, i) => {
        const x = (i / Math.max(1, arr.length - 1)) * w;
        const y = h - 4 - ((v - minV) / span) * (h - 8);
        return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  };

  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={h}>
      <line x1="0" y1={h / 2} x2={w} y2={h / 2} stroke="rgba(255,255,255,0.06)" strokeDasharray="2 3" />
      <path d={toPath(cos)} fill="none" stroke="#e5b869" strokeWidth="1.4" strokeOpacity="0.65" />
      <path d={toPath(cka)} fill="none" stroke="#7dd3fc" strokeWidth="1.9" />
    </svg>
  );
};
