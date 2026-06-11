import type { DesignCurve, JobState, Point, RouteSegment, SessionState } from "./types";

async function json<T>(res: Response): Promise<T> {
  const payload = await res.json().catch(() => ({}));
  if (!res.ok || payload.ok === false) {
    const detail = payload.detail || payload.error || payload.reason || res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload as T;
}

export async function uploadImage(params: {
  file: File;
  inputPreprocess: string;
  extractionMode: string;
  weakLineThreshold: string;
  parallelCollapse: string;
}): Promise<SessionState> {
  const body = new FormData();
  body.append("file", params.file);
  const res = await fetch("/api/upload", {
    method: "POST",
    headers: {
      "X-Input-Preprocess": params.inputPreprocess,
      "X-Extraction-Mode": params.extractionMode,
      "X-Weak-Line-Threshold": params.weakLineThreshold,
      "X-Parallel-Collapse": params.parallelCollapse
    },
    body
  });
  return json<SessionState>(res);
}

export async function reextractImage(sid: string, params: {
  inputPreprocess: string;
  extractionMode: string;
  weakLineThreshold: string;
  parallelCollapse: string;
}): Promise<SessionState> {
  const res = await fetch(`/api/sessions/${sid}/reextract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      input_preprocess: params.inputPreprocess,
      extraction_mode: params.extractionMode,
      weak_line_threshold: params.weakLineThreshold,
      parallel_collapse: params.parallelCollapse
    })
  });
  return json<SessionState>(res);
}

export async function openProject(project: File): Promise<SessionState> {
  const body = new FormData();
  body.append("project", project);
  return json<SessionState>(await fetch("/api/projects/open", { method: "POST", body }));
}

export async function downloadProject(
  sid: string,
  designCurves: DesignCurve[],
  editorState: Record<string, unknown>
): Promise<Blob> {
  const res = await fetch(`/api/sessions/${sid}/project`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ corrections: [], design_curves: designCurves, editor_state: editorState })
  });
  if (!res.ok) {
    const payload = await res.json().catch(() => ({}));
    throw new Error(payload.detail || payload.error || res.statusText);
  }
  return res.blob();
}

export async function routePoints(
  sid: string,
  points: Point[],
  closed: boolean,
  branchChoices: number[] = []
) {
  const res = await fetch(`/api/sessions/${sid}/route`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      points,
      closed,
      branch_choices: branchChoices,
      candidate_count: 3
    })
  });
  return json<{ ok: boolean; points: number[][]; segments: RouteSegment[]; point_count: number }>(res);
}

export async function snapPoint(sid: string, point: Point, maxDistance: number) {
  const res = await fetch(`/api/sessions/${sid}/snap`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ point, max_distance: maxDistance })
  });
  return json<{ ok: boolean; x: number; y: number; distance?: number; reason?: string }>(res);
}

export async function editSkeleton(
  sid: string,
  action: "add" | "delete",
  point: Point,
  radius: number
) {
  const res = await fetch(`/api/sessions/${sid}/skeleton-edit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      action,
      point,
      connect_radius: radius,
      delete_radius: radius,
      link_radius: radius * 4
    })
  });
  return json<{
    ok: boolean;
    reason?: string;
    full_skeleton_points?: number[][];
    skeleton_pixels?: number;
    last_skeleton_edit_index?: number | null;
    skeleton_edit_count?: number;
  }>(res);
}

export async function breakSkeletonChain(sid: string) {
  return json<{ ok: boolean; message: string }>(
    await fetch(`/api/sessions/${sid}/skeleton-edit/break`, { method: "POST" })
  );
}

export async function saveDesignCurves(sid: string, designCurves: DesignCurve[]) {
  const res = await fetch(`/api/sessions/${sid}/corrections`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ corrections: [], design_curves: designCurves })
  });
  return json<{ ok: boolean; corrections_path: string; design_curve_count: number }>(res);
}

export async function autoSegment(
  sid: string,
  mode: string,
  imageSize: { width: number; height: number } = { width: 0, height: 0 }
) {
  const diag = Math.max(1, Math.hypot(imageSize.width || 0, imageSize.height || 0));
  const payload =
    mode === "full"
      ? {
          mode,
          max_curves: 420,
          min_length: Math.max(2, diag * 0.0025),
          max_turn_deg: 68,
          max_junction_turn_deg: 44,
          max_chain_edges: 28,
          max_gap: Math.max(6, diag * 0.006),
          max_gap_turn_deg: 58
        }
      : mode === "detail"
        ? {
            mode,
            max_curves: 160,
            min_length: Math.max(5, diag * 0.006),
            max_turn_deg: 46,
            max_junction_turn_deg: 32,
            max_chain_edges: 14,
            max_gap: Math.max(4, diag * 0.004),
            max_gap_turn_deg: 42
          }
        : mode === "main"
          ? {
              mode,
              max_curves: 32,
              max_turn_deg: 28,
              max_junction_turn_deg: 18,
              max_chain_edges: 8,
              max_gap: 0
            }
          : {
              mode: "coverage",
              max_curves: 220,
              min_length: Math.max(4, diag * 0.005),
              max_turn_deg: 54,
              max_junction_turn_deg: 36,
              max_chain_edges: 36,
              max_gap: Math.max(8, diag * 0.009),
              max_gap_turn_deg: 50
            };
  const res = await fetch(`/api/sessions/${sid}/auto-segment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return json<{ ok: boolean; curve_count: number; curves: DesignCurve[] }>(res);
}

export async function startExport(sid: string, designCurves: DesignCurve[], degree: string, precisionFit: boolean) {
  const res = await fetch(`/api/sessions/${sid}/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      corrections: [],
      design_curves: designCurves,
      degree,
      fit_mode: precisionFit ? "precision" : "manual_class_a_g2",
      wire_export: true
    })
  });
  return json<{ ok: boolean; job_id: string }>(res);
}

export async function getJob(jobId: string): Promise<JobState> {
  return json<JobState>(await fetch(`/api/jobs/${jobId}`));
}

export function imageUrl(sid: string): string {
  return `/api/sessions/${sid}/image?t=${Date.now()}`;
}
