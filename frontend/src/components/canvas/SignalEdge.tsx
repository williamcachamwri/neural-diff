import React from "react";
import {
  BaseEdge,
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
  const stroke = edgeData.color ?? "#71717a";
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
          borderRadius: 12,
        })
      : getBezierPath({
          sourceX,
          sourceY,
          sourcePosition,
          targetX,
          targetY,
          targetPosition,
          curvature: 0.34,
        });

  return (
    <g className="comfy-signal-edge">
      {/* Subtle architectural wire track */}
      <BaseEdge
        id={`${id}-track`}
        path={edgePath}
        style={{
          stroke: "rgba(255, 255, 255, 0.085)",
          strokeWidth: 1.5,
        }}
      />
      {/* Delicate flowing signal dash */}
      <path
        id={id}
        d={edgePath}
        fill="none"
        stroke={stroke}
        strokeWidth={1.35}
        strokeOpacity={pulse ? 0.62 : 0.32}
        strokeDasharray={pulse ? "5 14" : undefined}
      >
        {pulse && (
          <animate
            attributeName="stroke-dashoffset"
            from="38"
            to="0"
            dur="2.8s"
            repeatCount="indefinite"
          />
        )}
      </path>
    </g>
  );
};
