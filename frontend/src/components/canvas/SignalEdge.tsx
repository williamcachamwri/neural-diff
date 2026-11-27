import React from "react";
import {
  type EdgeProps,
  getBezierPath,
  getSmoothStepPath,
} from "@xyflow/react";

export interface SignalEdgeData extends Record<string, unknown> {
  color?: string;
  routing?: "bezier" | "pcb";
  animatedPulse?: boolean;
}

export const SignalEdge: React.FC<EdgeProps> = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}) => {
  const edgeData = (data ?? {}) as SignalEdgeData;
  const stroke = edgeData.color ?? "#a1a1aa";
  const routing = edgeData.routing ?? "bezier";
  const pulse = edgeData.animatedPulse ?? true;

  const [edgePath] =
    routing === "pcb"
      ? getSmoothStepPath({
          sourceX,
          sourceY,
          sourcePosition,
          targetX,
          targetY,
          targetPosition,
          borderRadius: 14,
        })
      : getBezierPath({
          sourceX,
          sourceY,
          sourcePosition,
          targetX,
          targetY,
          targetPosition,
          curvature: 0.36,
        });

  return (
    <g className="comfy-signal-edge">
      {/* Layer 1: Insulated dark outer sheath for clean crossing separation */}
      <path
        id={`${id}-sheath`}
        d={edgePath}
        fill="none"
        stroke="rgba(9, 9, 11, 0.72)"
        strokeWidth={4.2}
        strokeLinecap="round"
      />
      {/* Layer 2: Continuous brushed-titanium signal conductor */}
      <path
        id={`${id}-core`}
        d={edgePath}
        fill="none"
        stroke={stroke}
        strokeOpacity={0.28}
        strokeWidth={1.5}
        strokeLinecap="round"
      />
      {/* Layer 3: Smooth gliding optical pulse packet (zero choppy marching ants) */}
      {pulse && (
        <path
          id={id}
          d={edgePath}
          fill="none"
          stroke={stroke}
          strokeWidth={1.85}
          strokeOpacity={0.88}
          strokeLinecap="round"
          strokeDasharray="28 220"
        >
          <animate
            attributeName="stroke-dashoffset"
            from="248"
            to="0"
            dur="3.1s"
            repeatCount="indefinite"
          />
        </path>
      )}
    </g>
  );
};
