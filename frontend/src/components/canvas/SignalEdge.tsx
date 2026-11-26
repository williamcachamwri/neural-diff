import React from "react";
import {
  BaseEdge,
  type EdgeProps,
  getBezierPath,
  getSmoothStepPath,
} from "@xyflow/react";

export interface SignalEdgeData extends Record<string, unknown> {
  color?: string;
  routingMode?: "bezier" | "pcb";
  active?: boolean;
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
  selected,
}) => {
  const edgeData = (data ?? {}) as SignalEdgeData;
  const signalColor = edgeData.color ?? "#e5b869";
  const isPcb = edgeData.routingMode === "pcb";
  const isActive = edgeData.active !== false;

  const [edgePath] = isPcb
    ? getSmoothStepPath({
        sourceX,
        sourceY,
        sourcePosition,
        targetX,
        targetY,
        targetPosition,
        borderRadius: 10,
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
      {/* Outer dark cable sheath for crisp separation when wires cross */}
      <BaseEdge
        id={`${id}-sheath`}
        path={edgePath}
        style={{
          stroke: "#070709",
          strokeWidth: selected ? 6.2 : 5,
          strokeLinecap: "round",
        }}
      />
      {/* Base conductor wire */}
      <BaseEdge
        id={id}
        path={edgePath}
        style={{
          stroke: signalColor,
          strokeWidth: selected ? 2.4 : 1.55,
          strokeOpacity: selected ? 0.95 : 0.48,
          strokeLinecap: "round",
        }}
      />
      {/* Animated traveling telemetry pulse */}
      {isActive && (
        <path
          d={edgePath}
          fill="none"
          stroke={signalColor}
          strokeWidth={selected ? 2.6 : 1.9}
          strokeOpacity={0.95}
          strokeLinecap="round"
          className="comfy-edge-pulse"
        />
      )}
    </g>
  );
};
