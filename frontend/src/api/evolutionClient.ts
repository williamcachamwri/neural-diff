import { ApiClientError } from "./client";
import type {
  EvolutionCheckpoint,
  EvolutionOverviewResponse,
  EvolutionProjectDetailResponse,
  EvolutionProjectSummary,
  EvolutionRunBundle,
  EvolutionRunSummary,
  RegressionOriginEvent,
  TransitionTensorRecord,
} from "../types/evolution";

async function evoRequestJson<T>(url: string, init?: RequestInit): Promise<T> {
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
          : `Request failed with status ${resp.status}`;
    const details =
      payload.details && typeof payload.details === "object"
        ? (payload.details as Record<string, unknown>)
        : {};
    throw new ApiClientError(code, msg, resp.status, details);
  }

  return (await resp.json()) as T;
}

export const evolutionApi = {
  getOverview: () => evoRequestJson<EvolutionOverviewResponse>("/api/evolution/overview"),

  listProjects: (limit = 50, offset = 0) =>
    evoRequestJson<{
      total: number;
      limit: number;
      offset: number;
      projects: EvolutionProjectSummary[];
    }>(`/api/evolution/projects?limit=${limit}&offset=${offset}`),

  createProject: (payload: { name: string; description?: string; tags?: string[] }) =>
    evoRequestJson<{ project_id: string; name: string; description: string }>(
      "/api/evolution/projects",
      {
        method: "POST",
        body: JSON.stringify(payload),
      }
    ),

  getProjectDetail: (projectId: string) =>
    evoRequestJson<EvolutionProjectDetailResponse>(
      `/api/evolution/projects/${encodeURIComponent(projectId)}`
    ),

  deleteProject: (projectId: string) =>
    evoRequestJson<{ deleted: boolean }>(
      `/api/evolution/projects/${encodeURIComponent(projectId)}`,
      { method: "DELETE" }
    ),

  registerCheckpoint: (
    projectId: string,
    payload: {
      source: string;
      version_label?: string;
      model_name?: string;
      parent_checkpoint_ids?: string[];
      transformation_type?: string;
      experiment_notes?: string;
      local_files_only?: boolean;
    }
  ) =>
    evoRequestJson<EvolutionCheckpoint>(
      `/api/evolution/projects/${encodeURIComponent(projectId)}/checkpoints`,
      {
        method: "POST",
        body: JSON.stringify(payload),
      }
    ),

  updateCheckpoint: (
    projectId: string,
    checkpointId: string,
    payload: {
      version_label?: string;
      model_name?: string;
      transformation_type?: string;
      experiment_notes?: string;
      parent_checkpoint_ids?: string[];
    }
  ) =>
    evoRequestJson<EvolutionCheckpoint>(
      `/api/evolution/projects/${encodeURIComponent(projectId)}/checkpoints/${encodeURIComponent(checkpointId)}`,
      {
        method: "PATCH",
        body: JSON.stringify(payload),
      }
    ),

  removeCheckpoint: (projectId: string, checkpointId: string) =>
    evoRequestJson<{ deleted: boolean }>(
      `/api/evolution/projects/${encodeURIComponent(projectId)}/checkpoints/${encodeURIComponent(checkpointId)}`,
      { method: "DELETE" }
    ),

  createRelationship: (
    projectId: string,
    payload: {
      parent_checkpoint_id: string;
      child_checkpoint_id: string;
      transformation_type?: string;
      merge_weight?: number;
    }
  ) =>
    evoRequestJson<Record<string, unknown>>(
      `/api/evolution/projects/${encodeURIComponent(projectId)}/relationships`,
      {
        method: "POST",
        body: JSON.stringify(payload),
      }
    ),

  compareVersions: (
    projectId: string,
    checkpointAId: string,
    checkpointBId: string,
    localFilesOnly = false
  ) =>
    evoRequestJson<{
      project_id: string;
      lowest_common_ancestor_id: string | null;
      lowest_common_ancestor_label: string | null;
      directed_path: string[] | null;
      comparison: TransitionTensorRecord;
    }>(`/api/evolution/projects/${encodeURIComponent(projectId)}/compare`, {
      method: "POST",
      body: JSON.stringify({
        checkpoint_a_id: checkpointAId,
        checkpoint_b_id: checkpointBId,
        local_files_only: localFilesOnly,
      }),
    }),

  runAnalysis: (
    projectId: string,
    payload: {
      include_tensor_evolution?: boolean;
      include_layer_evolution?: boolean;
      include_behavioral_evolution?: boolean;
      include_representation_evolution?: boolean;
      enable_causal_patching?: boolean;
      behavioral_max_cases?: number | null;
      force_rerun?: boolean;
      allow_cache?: boolean;
      local_files_only?: boolean;
      async_job?: boolean;
    }
  ) =>
    evoRequestJson<EvolutionRunBundle>(
      `/api/evolution/projects/${encodeURIComponent(projectId)}/analyze`,
      {
        method: "POST",
        body: JSON.stringify(payload),
      }
    ),

  listRuns: (projectId: string, limit = 25) =>
    evoRequestJson<{ total: number; runs: EvolutionRunSummary[] }>(
      `/api/evolution/projects/${encodeURIComponent(projectId)}/runs?limit=${limit}`
    ),

  getRun: (runId: string) =>
    evoRequestJson<EvolutionRunBundle>(`/api/evolution/runs/${encodeURIComponent(runId)}`),

  getRegressions: (projectId: string, category?: string) => {
    const qs = category ? `?category=${encodeURIComponent(category)}` : "";
    return evoRequestJson<{ total: number; regressions: RegressionOriginEvent[] }>(
      `/api/evolution/projects/${encodeURIComponent(projectId)}/regressions${qs}`
    );
  },

  getExportUrl: (runId: string, fmt: "json" | "csv" | "html" | "pdf") =>
    `/api/evolution/runs/${encodeURIComponent(runId)}/export/${fmt}`,

  fetchExportPreview: async (runId: string, fmt: "json" | "csv" | "html"): Promise<string> => {
    const resp = await fetch(`/api/evolution/runs/${encodeURIComponent(runId)}/export/${fmt}`);
    if (!resp.ok) {
      throw new ApiClientError(`EXPORT_${resp.status}`, `Failed to load ${fmt} export`, resp.status);
    }
    return await resp.text();
  },
};
