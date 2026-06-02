export type Point = {
  x: number;
  y: number;
  snap_source?: string;
  order?: number;
  anchor_curve_id?: string;
  anchor_point_order?: number;
  anchor_semantic?: string;
};

export type RouteSegment = {
  ok?: boolean;
  points?: number[][];
  segment_index?: number;
  selected_candidate?: number;
  alternatives?: RouteSegment[];
  length?: number;
};

export type DesignCurve = {
  id: string;
  type: string;
  semantic: string;
  manual_points: Point[];
  cut_points: Point[];
  closed: boolean;
  routed_points: number[][];
  route_segments: RouteSegment[];
  branch_choices: number[];
  route_ok: boolean;
  source?: string;
  created_at?: string;
};

export type ReviewGraph = {
  image_size?: { width: number; height: number };
  edge_count?: number;
  node_count?: number;
  extraction_mode?: string;
  parallel_collapse?: string;
  full_skeleton_points?: number[][];
  edges?: { id?: string; label?: string; points?: number[][]; length?: number }[];
  design_strokes?: { id?: string; label?: string; points?: number[][]; length?: number }[];
  [key: string]: unknown;
};

export type SessionState = {
  ok?: boolean;
  sid: string;
  graph: ReviewGraph;
  corrections: unknown[];
  design_curves: DesignCurve[];
  editor_state?: Record<string, unknown>;
  skeleton_edit_count?: number;
  message?: string;
  corrections_path?: string;
  exports?: Record<string, { path: string; url: string }>;
};

export type JobState = {
  ok: boolean;
  job_id: string;
  status: "queued" | "running" | "done" | "failed";
  progress: number;
  message?: string;
  error?: string;
  result?: {
    ok: boolean;
    curve_count: number;
    passed_count: number;
    skipped_count: number;
    out: string;
    exports: Record<string, { path: string; url: string }>;
    warnings: { label: string; warnings: string[] }[];
  };
};
