/**
 * Human-centered editorial formatters for NEURAL DIFF Studio.
 * Eliminates raw Python/JSON snake_case enums, unrounded floats, and screaming ALL-CAPS labels.
 */

const CATEGORY_NAMES: Record<string, string> = {
  vietnamese: "Vietnamese",
  code_preservation: "Code Preservation",
  json_reliability: "JSON Schema",
  math_reasoning: "Math Reasoning",
  instruction_following: "Instruction Following",
  conciseness: "Conciseness",
  refusal_behavior: "Safety & Refusal",
  factual_accuracy: "Factual Grounding",
};

const CATEGORY_SHORT_NAMES: Record<string, string> = {
  vietnamese: "Vietnamese",
  code_preservation: "Code",
  json_reliability: "JSON",
  math_reasoning: "Math",
  instruction_following: "Instruct",
  conciseness: "Concise",
  refusal_behavior: "Safety",
  factual_accuracy: "Factual",
};

const ROLE_NAMES: Record<string, string> = {
  embedding: "Embedding",
  attn_q: "Query (Q)",
  attn_k: "Key (K)",
  attn_v: "Value (V)",
  attn_o: "Attn Out (O)",
  attn_qkv_fused: "QKV Fused",
  mlp_gate: "MLP Gate",
  mlp_up: "MLP Up",
  mlp_down: "MLP Down",
  mlp_fc: "MLP FC",
  norm: "Norm",
  lm_head: "LM Head",
  other: "Other",
};

export function formatCategory(raw: string): string {
  return CATEGORY_NAMES[raw] ?? raw.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function formatCategoryShort(raw: string): string {
  return CATEGORY_SHORT_NAMES[raw] ?? formatCategory(raw).slice(0, 9);
}

export function formatRole(raw: string): string {
  return ROLE_NAMES[raw] ?? raw.replace(/_/g, " ");
}

export function formatVerdict(raw: string | null | undefined): string {
  if (!raw) return "Ready";
  switch (raw) {
    case "PASSED":
      return "Passed";
    case "FAILED_REGRESSION":
      return "Regression detected";
    case "WARNING":
      return "Review advised";
    case "READY":
      return "Ready";
    default:
      return raw
        .toLowerCase()
        .replace(/_/g, " ")
        .replace(/^\w/, (c) => c.toUpperCase());
  }
}

export function formatSeverity(raw: string): string {
  switch (raw.toUpperCase()) {
    case "CRITICAL":
      return "Critical";
    case "HIGH":
      return "High";
    case "MEDIUM":
      return "Medium";
    case "LOW":
      return "Low";
    default:
      return "Info";
  }
}

export function formatStatus(raw: string): string {
  switch (raw) {
    case "modified":
      return "Modified";
    case "identical":
      return "Identical";
    case "within_tolerance":
      return "Within tol.";
    case "shape_mismatch":
      return "Shape Δ";
    case "missing_in_b":
      return "Removed";
    case "added_in_b":
      return "Added";
    case "improved":
      return "Improved";
    case "regressed":
      return "Regressed";
    case "both_passed":
      return "Maintained";
    case "both_failed":
      return "Unresolved";
    case "completed":
      return "Completed";
    case "running":
      return "Running";
    case "failed":
      return "Failed";
    default:
      return raw.replace(/_/g, " ");
  }
}

export function formatMetricNumber(val: unknown, digits = 3): string {
  if (val === null || val === undefined) return "—";
  if (typeof val === "number") {
    if (!Number.isFinite(val)) return String(val);
    if (val === 0) return "0.000";
    if (Math.abs(val) < 1e-3 || Math.abs(val) >= 1e5) {
      return val.toExponential(2);
    }
    return val.toFixed(digits);
  }
  return String(val);
}

export function formatShortTensorName(name: string): string {
  return name
    .replace(/^model\.layers\.\d+\./, "")
    .replace(/^transformer\.h\.\d+\./, "");
}
