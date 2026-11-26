import type {
  BaselineRecord,
  CompatibilityReport,
  DiscoveredModelsResponse,
  ExperimentSummaryItem,
  HealthResponse,
  ModelMetadata,
  RuntimeSettings,
  TensorRecord,
} from "../types/api";

export class ApiClientError extends Error {
  readonly code: string;
  readonly httpStatus: number;
  readonly details: Record<string, unknown>;

  constructor(code: string, message: string, httpStatus: number, details: Record<string, unknown> = {}) {
    super(message);
    this.name = "ApiClientError";
    this.code = code;
    this.httpStatus = httpStatus;
    this.details = details;
  }
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
      ...(init?.headers || {}),
    },
  });

  if (!resp.ok) {
    let payload: Record<string, unknown> = {};
    try {
      payload = (await resp.json()) as Record<string, unknown>;
    } catch {
      payload = {};
    }
    const code = typeof payload.error === "string" ? payload.error : `HTTP_${resp.status}`;
    const msg =
      typeof payload.message === "string"
        ? payload.message
        : typeof payload.detail === "string"
          ? payload.detail
          : Array.isArray(payload.detail)
            ? JSON.stringify(payload.detail)
            : `Request failed with status ${resp.status}`;
    const details =
      payload.details && typeof payload.details === "object"
        ? (payload.details as Record<string, unknown>)
        : {};
    throw new ApiClientError(code, msg, resp.status, details);
  }

  return (await resp.json()) as T;
}

export const apiClient = {
  getHealth: () => requestJson<HealthResponse>("/api/health"),

  discoverModels: () => requestJson<DiscoveredModelsResponse>("/api/models/discover"),

  validateCompatibility: (modelA: string, modelB: string, localFilesOnly = false) =>
    requestJson<{
      model_a: ModelMetadata;
      model_b: ModelMetadata;
      compatibility: CompatibilityReport;
    }>("/api/models/validate", {
      method: "POST",
      body: JSON.stringify({ model_a: modelA, model_b: modelB, local_files_only: localFilesOnly }),
    }),

  runComparison: (payload: Record<string, unknown>) =>
    requestJson<ExperimentSummaryItem>("/api/comparisons/run", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  cancelComparison: (comparisonId: string) =>
    requestJson<{ comparison_id: string; cancelled: boolean; status: string }>(
      `/api/comparisons/${encodeURIComponent(comparisonId)}/cancel`,
      { method: "POST" }
    ),

  listComparisons: () =>
    requestJson<{ experiments: ExperimentSummaryItem[] }>("/api/comparisons"),

  getComparison: (comparisonId: string) =>
    requestJson<ExperimentSummaryItem>(`/api/comparisons/${encodeURIComponent(comparisonId)}`),

  queryTensors: (
    comparisonId: string,
    params: { search?: string; role?: string; status?: string; sort_by?: string; limit?: number }
  ) => {
    const qs = new URLSearchParams();
    if (params.search) qs.set("search", params.search);
    if (params.role) qs.set("role", params.role);
    if (params.status) qs.set("status", params.status);
    if (params.sort_by) qs.set("sort_by", params.sort_by);
    qs.set("limit", String(params.limit ?? 200));
    return requestJson<{ total: number; records: TensorRecord[] }>(
      `/api/comparisons/${encodeURIComponent(comparisonId)}/tensors?${qs.toString()}`
    );
  },

  fetchReportText: async (comparisonId: string, fmt: "json" | "markdown" | "html"): Promise<string> => {
    const resp = await fetch(
      `/api/comparisons/${encodeURIComponent(comparisonId)}/export/${fmt}`
    );
    if (!resp.ok) {
      throw new ApiClientError(`EXPORT_${resp.status}`, `Failed to load ${fmt} report`, resp.status);
    }
    return await resp.text();
  },

  listBaselines: () => requestJson<{ baselines: BaselineRecord[] }>("/api/baselines"),

  createBaseline: (comparisonId: string, baselineId: string, side: "a" | "b") =>
    requestJson<BaselineRecord>("/api/baselines/from-comparison", {
      method: "POST",
      body: JSON.stringify({ comparison_id: comparisonId, baseline_id: baselineId, side }),
    }),

  deleteBaseline: (baselineId: string) =>
    requestJson<{ deleted: boolean }>(`/api/baselines/${encodeURIComponent(baselineId)}`, {
      method: "DELETE",
    }),

  getSettings: () => requestJson<RuntimeSettings>("/api/settings"),

  updateSettings: (settings: RuntimeSettings) =>
    requestJson<RuntimeSettings>("/api/settings", {
      method: "PUT",
      body: JSON.stringify(settings),
    }),
};
