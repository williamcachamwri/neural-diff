import React from "react";

export function humanizeEnum(raw: string | null | undefined): string {
  if (!raw) return "—";
  const map: Record<string, string> = {
    OBSERVED: "Observed",
    INFERRED: "Inferred",
    CORRELATED: "Correlated",
    "CAUSAL SUPPORT": "Causal support",
    "CAUSALLY VERIFIED": "Causally verified",
    "NOT VERIFIED": "Unverified",
    VERIFIED: "Verified",
    DUPLICATE_CONTENT: "Duplicate SHA-256",
    weight_evolution: "Weight geometry",
    layer_evolution: "Layer drift",
    behavioral_evolution: "Behavioral",
    representation_evolution: "Representations",
    regression_origin: "Regression origin",
    base_to_current: "Base → Current",
    ancestor_to_descendant: "Ancestor → Descendant",
    descendant_to_ancestor: "Descendant → Ancestor",
    consecutive: "Step → Step",
    branch_to_branch: "Branch ↔ Branch",
    unrelated: "Disjoint roots",
    identical: "Identical checkpoint",
    overall: "Overall suite",
    pass_rate_and_score: "Pass rate & score",
    pass_rate_only: "Pass rate drop",
    score_only: "Score drop",
    exploratory_small_sample: "Exploratory (small n)",
    statistically_significant: "Statistically significant",
    moderate_observed_effect: "Moderate effect",
    huggingface_local_dir: "Hugging Face Directory",
    safetensors_file: "Safetensors",
    pytorch_weights_only: "PyTorch Archive",
    lora_adapter: "LoRA Adapter",
  };
  if (map[raw]) return map[raw];
  const cleaned = raw.replace(/_/g, " ").trim();
  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1).toLowerCase();
}

export const DualTrajectoryChart: React.FC<{
  xLabels: string[];
  primary: Array<number | null | undefined>;
  primaryLabel: string;
  secondary?: Array<number | null | undefined>;
  secondaryLabel?: string;
  ciBands?: Array<[number, number] | null>;
  yMin?: number;
  yMax?: number;
  height?: number;
}> = ({
  xLabels,
  primary,
  primaryLabel,
  secondary,
  secondaryLabel,
  ciBands,
  yMin = 0,
  yMax = 1,
  height = 188,
}) => {
  const n = xLabels.length;
  if (n === 0) return null;

  const width = 760;
  const padL = 46;
  const padR = 18;
  const padT = 26;
  const padB = 26;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const spanY = Math.max(1e-9, yMax - yMin);

  const toXY = (idx: number, val: number): [number, number] => {
    const px = n > 1 ? padL + (idx / (n - 1)) * plotW : padL + plotW / 2;
    const clamped = Math.max(yMin, Math.min(yMax, val));
    const py = padT + (1 - (clamped - yMin) / spanY) * plotH;
    return [px, py];
  };

  const pPts: Array<{ idx: number; x: number; y: number; v: number }> = [];
  primary.forEach((v, idx) => {
    if (v != null && Number.isFinite(v)) {
      const [x, y] = toXY(idx, v);
      pPts.push({ idx, x, y, v });
    }
  });

  const sPts: Array<{ idx: number; x: number; y: number; v: number }> = [];
  (secondary ?? []).forEach((v, idx) => {
    if (v != null && Number.isFinite(v)) {
      const [x, y] = toXY(idx, v);
      sPts.push({ idx, x, y, v });
    }
  });

  const stride = Math.max(1, Math.floor(n / 12));
  const areaPath =
    pPts.length >= 2
      ? `M ${pPts[0].x.toFixed(1)} ${(padT + plotH).toFixed(1)} ` +
        pPts.map((p) => `L ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ") +
        ` L ${pPts[pPts.length - 1].x.toFixed(1)} ${(padT + plotH).toFixed(1)} Z`
      : "";

  let bandPath = "";
  if (ciBands && ciBands.length === n && n >= 2) {
    const upper: string[] = [];
    const lower: string[] = [];
    ciBands.forEach((b, i) => {
      if (b) {
        const [ux, uy] = toXY(i, b[1]);
        const [lx, ly] = toXY(i, b[0]);
        upper.push(`${ux.toFixed(1)},${uy.toFixed(1)}`);
        lower.unshift(`${lx.toFixed(1)},${ly.toFixed(1)}`);
      }
    });
    if (upper.length >= 2) {
      bandPath = `M ${upper.join(" L ")} L ${lower.join(" L ")} Z`;
    }
  }

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" style={{ display: "block" }}>
      <defs>
        <linearGradient id="studioAreaFade" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#f4f4f5" stopOpacity={0.16} />
          <stop offset="100%" stopColor="#f4f4f5" stopOpacity={0.0} />
        </linearGradient>
      </defs>

      {[0, 0.5, 1].map((step) => {
        const gv = yMin + step * spanY;
        const gy = padT + (1 - step) * plotH;
        return (
          <g key={step}>
            <line
              x1={padL}
              y1={gy}
              x2={width - padR}
              y2={gy}
              stroke="rgba(255,255,255,0.06)"
              strokeWidth={1}
            />
            <text
              x={padL - 6}
              y={gy + 3.5}
              fill="#71717a"
              fontSize={9.5}
              textAnchor="end"
              fontFamily="JetBrains Mono, monospace"
            >
              {gv >= 10 ? gv.toFixed(0) : gv.toFixed(2)}
            </text>
          </g>
        );
      })}

      {xLabels.map((lbl, idx) => {
        if (idx % stride !== 0 && idx !== n - 1) return null;
        const [px] = toXY(idx, yMin);
        return (
          <text
            key={`${lbl}-${idx}`}
            x={px}
            y={height - 7}
            fill="#71717a"
            fontSize={9.5}
            textAnchor="middle"
            fontFamily="JetBrains Mono, monospace"
          >
            {lbl.slice(0, 14)}
          </text>
        );
      })}

      {bandPath && <path d={bandPath} fill="rgba(214,197,169,0.11)" />}
      {areaPath && <path d={areaPath} fill="url(#studioAreaFade)" />}

      {pPts.length >= 2 && (
        <polyline
          points={pPts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ")}
          fill="none"
          stroke="#f4f4f5"
          strokeWidth={1.75}
        />
      )}
      {pPts.map((p) => (
        <circle key={`p-${p.idx}`} cx={p.x} cy={p.y} r={2.5} fill="#f4f4f5">
          <title>{`${xLabels[p.idx]}: ${p.v.toFixed(5)}`}</title>
        </circle>
      ))}

      {sPts.length >= 2 && (
        <polyline
          points={sPts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ")}
          fill="none"
          stroke="#d6c5a9"
          strokeWidth={1.5}
          strokeDasharray="4 3"
        />
      )}
      {sPts.map((p) => (
        <circle key={`s-${p.idx}`} cx={p.x} cy={p.y} r={2.2} fill="#d6c5a9">
          <title>{`${xLabels[p.idx]}: ${p.v.toFixed(5)}`}</title>
        </circle>
      ))}

      <line x1={padL} y1={11} x2={padL + 14} y2={11} stroke="#f4f4f5" strokeWidth={2} />
      <text
        x={padL + 20}
        y={14}
        fill="#e4e4e7"
        fontSize={10}
        fontFamily="JetBrains Mono, monospace"
      >
        {primaryLabel}
      </text>

      {secondaryLabel && sPts.length > 0 && (
        <>
          <line
            x1={padL + 36 + primaryLabel.length * 6.4}
            y1={11}
            x2={padL + 50 + primaryLabel.length * 6.4}
            y2={11}
            stroke="#d6c5a9"
            strokeWidth={2}
            strokeDasharray="3 2"
          />
          <text
            x={padL + 56 + primaryLabel.length * 6.4}
            y={14}
            fill="#d6c5a9"
            fontSize={10}
            fontFamily="JetBrains Mono, monospace"
          >
            {secondaryLabel}
          </text>
        </>
      )}
    </svg>
  );
};

export const PolarRadarChart: React.FC<{
  axes: string[];
  series: Array<{ label: string; values: number[]; color: string }>;
  size?: number;
}> = ({ axes, series, size = 268 }) => {
  const m = axes.length;
  if (m < 3 || series.length === 0) return null;

  const padX = 72;
  const totalW = size + padX * 2;
  const cx = totalW / 2;
  const cy = size / 2 + 2;
  const radius = size * 0.32;

  return (
    <svg
      viewBox={`0 0 ${totalW} ${size}`}
      width="100%"
      style={{ maxHeight: 268, display: "block" }}
    >
      {[0.25, 0.5, 0.75, 1].map((ring) => {
        const r = radius * ring;
        const pts = axes
          .map((_, i) => {
            const ang = -Math.PI / 2 + (2 * Math.PI * i) / m;
            return `${(cx + r * Math.cos(ang)).toFixed(1)},${(cy + r * Math.sin(ang)).toFixed(1)}`;
          })
          .join(" ");
        return (
          <polygon
            key={ring}
            points={pts}
            fill="none"
            stroke="rgba(255,255,255,0.07)"
            strokeWidth={1}
          />
        );
      })}

      {axes.map((lbl, i) => {
        const ang = -Math.PI / 2 + (2 * Math.PI * i) / m;
        const ex = cx + radius * Math.cos(ang);
        const ey = cy + radius * Math.sin(ang);
        const lx = cx + (radius + 16) * Math.cos(ang);
        const ly = cy + (radius + 13) * Math.sin(ang);
        const anchor =
          Math.cos(ang) > 0.28 ? "start" : Math.cos(ang) < -0.28 ? "end" : "middle";
        return (
          <g key={`${lbl}-${i}`}>
            <line
              x1={cx}
              y1={cy}
              x2={ex}
              y2={ey}
              stroke="rgba(255,255,255,0.06)"
              strokeWidth={1}
            />
            <text
              x={lx}
              y={ly + 3}
              fill="#a1a1aa"
              fontSize={9.5}
              textAnchor={anchor}
              fontFamily="JetBrains Mono, monospace"
            >
              {lbl.slice(0, 22)}
            </text>
          </g>
        );
      })}

      {series.map((s, idx) => {
        const pts = axes
          .map((_, i) => {
            const v = Math.max(0, Math.min(1, s.values[i] ?? 0));
            const ang = -Math.PI / 2 + (2 * Math.PI * i) / m;
            return `${(cx + radius * v * Math.cos(ang)).toFixed(1)},${(
              cy +
              radius * v * Math.sin(ang)
            ).toFixed(1)}`;
          })
          .join(" ");
        const legX = 12 + idx * 126;
        return (
          <g key={s.label}>
            <polygon
              points={pts}
              fill={s.color}
              fillOpacity={0.14}
              stroke={s.color}
              strokeWidth={1.6}
            />
            <circle cx={legX} cy={size - 8} r={3} fill={s.color} />
            <text
              x={legX + 8}
              y={size - 4.5}
              fill="#a1a1aa"
              fontSize={9.5}
              fontFamily="JetBrains Mono, monospace"
            >
              {s.label.slice(0, 18)}
            </text>
          </g>
        );
      })}
    </svg>
  );
};

export const PixelMatrixHeatmap: React.FC<{
  rowLabels: string[];
  colLabels: string[];
  matrix: Array<Array<number | null | undefined>>;
  minVal?: number;
  maxVal?: number;
  invert?: boolean;
}> = ({ rowLabels, colLabels, matrix, minVal, maxVal, invert = false }) => {
  if (rowLabels.length === 0 || colLabels.length === 0) return null;

  const flat = matrix
    .flat()
    .filter((v): v is number => v != null && Number.isFinite(v));
  if (flat.length === 0) return null;

  const lo = minVal ?? Math.min(...flat);
  const hi = maxVal ?? Math.max(...flat);
  const span = Math.max(1e-9, hi - lo);

  const leftPad = Math.max(115, Math.min(205, Math.max(...rowLabels.map((r) => r.length)) * 6 + 16));
  const cellW = Math.max(18, Math.min(34, Math.floor(690 / Math.max(1, colLabels.length))));
  const cellH = 22;
  const topPad = 22;
  const width = leftPad + colLabels.length * cellW + 14;
  const height = topPad + rowLabels.length * cellH + 10;
  const stride = Math.max(1, Math.floor(colLabels.length / 15));

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" style={{ display: "block" }}>
      {colLabels.map((cl, cIdx) => {
        if (cIdx % stride !== 0 && cIdx !== colLabels.length - 1) return null;
        return (
          <text
            key={`${cl}-${cIdx}`}
            x={leftPad + cIdx * cellW + cellW / 2}
            y={14}
            fill="#71717a"
            fontSize={9.5}
            textAnchor="middle"
            fontFamily="JetBrains Mono, monospace"
          >
            {cl}
          </text>
        );
      })}

      {rowLabels.map((rl, rIdx) => {
        const y = topPad + rIdx * cellH;
        const rowVals = matrix[rIdx] ?? [];
        return (
          <g key={`${rl}-${rIdx}`}>
            <text
              x={leftPad - 8}
              y={y + 14.5}
              fill="#a1a1aa"
              fontSize={10.5}
              textAnchor="end"
              fontFamily="JetBrains Mono, monospace"
            >
              {rl.slice(0, 28)}
            </text>
            {colLabels.map((cl, cIdx) => {
              const x = leftPad + cIdx * cellW;
              const v = rowVals[cIdx];
              if (v == null || !Number.isFinite(v)) {
                return (
                  <rect
                    key={cl}
                    x={x}
                    y={y}
                    width={cellW - 2}
                    height={cellH - 2}
                    rx={2}
                    fill="rgba(255,255,255,0.03)"
                  />
                );
              }
              const norm = Math.max(0, Math.min(1, (v - lo) / span));
              const intensity = invert ? 1 - norm : norm;
              const alpha = 0.06 + 0.78 * intensity;
              return (
                <rect
                  key={cl}
                  x={x}
                  y={y}
                  width={cellW - 2}
                  height={cellH - 2}
                  rx={2}
                  fill={`rgba(214, 197, 169, ${alpha.toFixed(3)})`}
                >
                  <title>{`${rl} · ${cl}: ${v.toFixed(5)}`}</title>
                </rect>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
};

export const DivergingDeltaBars: React.FC<{
  items: Array<{ label: string; delta: number }>;
}> = ({ items }) => {
  if (items.length === 0) return null;
  const maxAbs = Math.max(0.1, ...items.map((i) => Math.abs(i.delta)));
  const rowH = 25;
  const leftPad = 162;
  const halfW = 126;
  const center = leftPad + halfW;
  const width = leftPad + halfW * 2 + 74;
  const height = items.length * rowH + 16;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" style={{ display: "block" }}>
      <line
        x1={center}
        y1={6}
        x2={center}
        y2={height - 6}
        stroke="rgba(255,255,255,0.14)"
        strokeWidth={1}
      />
      {items.map((item, idx) => {
        const y = 10 + idx * rowH;
        const w = Math.max(1.5, (Math.abs(item.delta) / maxAbs) * (halfW - 6));
        const x = item.delta >= 0 ? center : center - w;
        const col =
          item.delta > 0.01 ? "#4ade80" : item.delta < -0.01 ? "#f87171" : "#71717a";
        return (
          <g key={item.label}>
            <text
              x={leftPad - 8}
              y={y + 11}
              fill="#a1a1aa"
              fontSize={10.5}
              textAnchor="end"
              fontFamily="JetBrains Mono, monospace"
            >
              {item.label.slice(0, 22)}
            </text>
            <rect x={x} y={y + 2} width={w} height={10} rx={2} fill={col} />
            <text
              x={leftPad + halfW * 2 + 8}
              y={y + 11}
              fill={col}
              fontSize={10.5}
              fontFamily="JetBrains Mono, monospace"
            >
              {item.delta >= 0 ? "+" : ""}
              {item.delta.toFixed(3)}
            </text>
          </g>
        );
      })}
    </svg>
  );
};
