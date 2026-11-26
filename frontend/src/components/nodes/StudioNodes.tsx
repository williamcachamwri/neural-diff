import React, { useState } from "react";
import { Handle, type NodeProps, Position } from "@xyflow/react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowUpRight, ChevronDown } from "lucide-react";
import type { ViewId } from "../../types/api";
import { formatCategoryShort, formatVerdict } from "../../utils/format";

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
          style={{ top: `${36 + idx * 20}px` }}
          title={pin.label}
        />
      ))}

      {/* Minimal Editorial Node Header */}
      <div className="comfy-node-header">
        <div className="comfy-node-title-wrap">
          <span className="comfy-node-dot" style={{ background: payload.accentColor }} />
          <span className="comfy-node-title">{payload.title}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 3 }}>
          <span className="comfy-node-id" style={{ marginRight: 3 }}>
            {payload.nodeIndex}
          </span>
          <button
            type="button"
            className="nodrag comfy-node-icon-btn"
            aria-label={`Collapse ${payload.title}`}
            onClick={(e) => {
              e.stopPropagation();
              setCollapsed((c) => !c);
            }}
          >
            <motion.span
              animate={{ rotate: collapsed ? -90 : 0 }}
              transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1] }}
              style={{ display: "grid" }}
            >
              <ChevronDown size={13} />
            </motion.span>
          </button>
          <button
            type="button"
            className="nodrag comfy-node-icon-btn"
            aria-label={`Inspect ${payload.title}`}
            onClick={(e) => {
              e.stopPropagation();
              payload.onInspect(payload.targetView);
            }}
          >
            <ArrowUpRight size={13} />
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
                    style={r.highlight ? { color: r.highlight, fontWeight: 500 } : undefined}
                  >
                    {r.value}
                  </span>
                </div>
              ))}

              {/* Inline Visual 1: Monochrome + Warm Champagne 24-Layer Pixel Matrix */}
              {payload.visual?.kind === "heatmap" && payload.visual.matrix.length > 0 && (
                <div
                  style={{
                    background: "#0b0b0e",
                    border: "1px solid var(--border-faint)",
                    borderRadius: 7,
                    padding: 6,
                    marginTop: 3,
                  }}
                >
                  <div
                    style={{
                      display: "grid",
                      gridTemplateRows: `repeat(${payload.visual.matrix.length}, 5.5px)`,
                      gap: 1.5,
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
                          const maxVal =
                            payload.visual && payload.visual.kind === "heatmap"
                              ? payload.visual.maxVal
                              : 1;
                          const norm = Math.min(1, Math.max(0, v / Math.max(maxVal, 1e-6)));
                          const lum = Math.round(28 + norm * 195);
                          const warmB = Math.round(32 + norm * 145);
                          return (
                            <span
                              key={cIdx}
                              style={{
                                borderRadius: 1,
                                background: `rgb(${lum}, ${Math.round(lum * 0.94)}, ${warmB})`,
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
                    background: "#0b0b0e",
                    border: "1px solid var(--border-faint)",
                    borderRadius: 7,
                    padding: "6px 8px 4px",
                    marginTop: 3,
                  }}
                >
                  <OscilloscopeMini cka={payload.visual.cka} cos={payload.visual.cos} />
                </div>
              )}

              {/* Inline Visual 3: Behavioral Domain Meter Tracks */}
              {payload.visual?.kind === "bars" && payload.visual.items.length > 0 && (
                <div style={{ display: "flex", flexDirection: "column", gap: 4.5, marginTop: 3 }}>
                  {payload.visual.items.slice(0, 6).map((it) => (
                    <div
                      key={it.label}
                      style={{
                        display: "grid",
                        gridTemplateColumns: "68px 1fr 34px",
                        alignItems: "center",
                        gap: 6,
                      }}
                    >
                      <span style={{ fontSize: 10.5, color: "var(--text-muted)" }}>
                        {formatCategoryShort(it.label)}
                      </span>
                      <div
                        style={{
                          height: 4,
                          background: "#1b1b21",
                          borderRadius: 99,
                          overflow: "hidden",
                        }}
                      >
                        <div
                          style={{
                            width: `${Math.round(it.b * 100)}%`,
                            height: "100%",
                            background: it.b >= it.a ? "#d4d4d8" : "var(--danger)",
                            borderRadius: 99,
                          }}
                        />
                      </div>
                      <span
                        className="mono"
                        style={{
                          fontSize: 9.5,
                          textAlign: "right",
                          color: "var(--text-secondary)",
                        }}
                      >
                        {Math.round(it.b * 100)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* Inline Visual 4: Clean Minimal Compatibility Indicators */}
              {payload.visual?.kind === "status-leds" && (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: "5px 10px",
                    marginTop: 4,
                    paddingTop: 6,
                    borderTop: "1px solid var(--border-faint)",
                  }}
                >
                  {payload.visual.leds.map((l) => (
                    <div
                      key={l.label}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        fontSize: 10.5,
                        color: "var(--text-secondary)",
                      }}
                    >
                      <span
                        style={{
                          width: 5,
                          height: 5,
                          borderRadius: 99,
                          background: l.ok ? "var(--ok)" : "var(--text-faint)",
                        }}
                      />
                      <span>{l.label}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Inline Visual 5: Quiet Regression Verdict & Export Links */}
              {payload.visual?.kind === "verdict" && (
                <div style={{ display: "flex", flexDirection: "column", gap: 7, marginTop: 3 }}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      paddingTop: 5,
                      borderTop: "1px solid var(--border-faint)",
                    }}
                  >
                    <span className="status-inline" style={{ fontSize: 11 }}>
                      <span
                        className={`dot ${
                          payload.visual.verdict === "PASSED" ? "dot-ok" : "dot-danger"
                        }`}
                      />
                      <span style={{ color: "var(--text-main)", fontSize: 11.5, fontWeight: 500 }}>
                        {formatVerdict(payload.visual.verdict)}
                      </span>
                    </span>
                    <span style={{ fontSize: 10.5, color: "var(--text-muted)" }}>
                      {payload.visual.findings} finding{payload.visual.findings === 1 ? "" : "s"}
                    </span>
                  </div>
                  {payload.visual.comparisonId && (
                    <div style={{ display: "flex", gap: 5 }} className="nodrag">
                      {(["html", "markdown", "json"] as const).map((fmt) => (
                        <a
                          key={fmt}
                          href={`/api/comparisons/${encodeURIComponent(
                            payload.visual &&
                              payload.visual.kind === "verdict" &&
                              payload.visual.comparisonId
                              ? payload.visual.comparisonId
                              : ""
                          )}/export/${fmt}`}
                          download
                          onClick={(e) => e.stopPropagation()}
                          style={{
                            flex: 1,
                            textAlign: "center",
                            padding: "4px 0",
                            borderRadius: 5,
                            background: "rgba(255,255,255,0.03)",
                            border: "1px solid var(--border-subtle)",
                            color: "var(--text-secondary)",
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

      {/* Footer Socket Labels */}
      <div className="comfy-socket-bar">
        <span>{inPins.map((p) => p.label).join(" · ") || "input"}</span>
        <span>{outPins.map((p) => p.label).join(" · ") || "output"}</span>
      </div>

      {/* Output Handles */}
      {outPins.map((pin, idx) => (
        <Handle
          key={pin.id}
          id={pin.id}
          type="source"
          position={Position.Right}
          style={{ top: `${36 + idx * 20}px` }}
          title={pin.label}
        />
      ))}
    </div>
  );
};

const OscilloscopeMini: React.FC<{ cka: number[]; cos: number[] }> = ({ cka, cos }) => {
  const w = 240;
  const h = 38;
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
      <line
        x1="0"
        y1={h / 2}
        x2={w}
        y2={h / 2}
        stroke="rgba(255,255,255,0.05)"
        strokeDasharray="2 3"
      />
      <path d={toPath(cos)} fill="none" stroke="#71717a" strokeWidth="1.25" />
      <path d={toPath(cka)} fill="none" stroke="#e4e4e7" strokeWidth="1.65" />
    </svg>
  );
};
