from __future__ import annotations

import heapq
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 直接运行本文件时（``python .../review/graph.py``），Python 不会自动把仓库的 ``src`` 加入
# sys.path；未 ``pip install -e .`` 时会报 ``No module named 'auto  alias'``。
_src = Path(__file__).resolve().parents[2]
if (_src / "autoalias").is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np

from autoalias.vision.extractor import (
    _collapse_parallel_strokes,
    _curve_length,
    _is_line_art,
    _is_white_on_black_sketch,
    _line_art_ink,
    _pencil_weak_line_ink,
    _prune_skeleton_artifacts,
    _require_cv2,
    _semantic_guess,
    _skeletonize_zhang_suen,
    _trace_skeleton_chains,
    _white_on_black_sketch_ink,
)

@dataclass(slots=True)
class ReviewGraphOptions:
    min_edge_length: float = 3.0
    endpoint_cluster_radius: float = 6.0
    max_points_per_edge: int = 320
    extraction_mode: str = "auto"
    parallel_collapse: str = "off"
    weak_line_threshold: float = 32.0


@dataclass(slots=True)
class SkeletonRouter:
    coords: np.ndarray
    adjacency: list[list[tuple[int, float]]]

    @classmethod
    def from_skeleton(cls, skeleton: np.ndarray) -> "SkeletonRouter":
        ys, xs = np.where(skeleton)
        coords = np.column_stack([xs, ys]).astype(float)
        index_by_pixel = {(int(x), int(y)): idx for idx, (x, y) in enumerate(coords)}
        adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(coords))]
        neighbors = (
            (-1, -1, 2**0.5),
            (0, -1, 1.0),
            (1, -1, 2**0.5),

            (-1, 0, 1.0),
            (1, 0, 1.0),
            (-1, 1, 2**0.5),
            (0, 1, 1.0),
            (1, 1, 2**0.5),
        )
        for idx, (x_f, y_f) in enumerate(coords):
            x = int(x_f)
            y = int(y_f)
            for dx, dy, weight in neighbors:
                other = index_by_pixel.get((x + dx, y + dy))
                if other is not None:
                    adjacency[idx].append((other, weight))
        return cls(coords=coords, adjacency=adjacency)

    def route(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        max_preview_points: int = 900,
    ) -> dict[str, Any]:
        if len(self.coords) == 0:
            return {
                "ok": False,
                "reason": "no skeleton pixels",
                "points": [list(start), list(end)],
            }
        start_idx, start_dist = self.nearest_index(start)
        end_idx, end_dist = self.nearest_index(end)
        path = self._shortest_path(start_idx, end_idx)
        if not path:
            return {
                "ok": False,
                "reason": "no connected skeleton path",
                "points": [list(start), list(end)],
                "snap_distance_start": round(float(start_dist), 3),
                "snap_distance_end": round(float(end_dist), 3),
            }
        routed = self.coords[path]
        routed = _smooth_route_points(routed)
        routed = _downsample_points(routed, max_preview_points)
        return {
            "ok": True,
            "points": _round_points(routed),
            "point_count": int(len(routed)),
            "length": round(float(_curve_length(routed)), 3),
            "snapped_start": _round_points(self.coords[[start_idx]])[0],
            "snapped_end": _round_points(self.coords[[end_idx]])[0],
            "snap_distance_start": round(float(start_dist), 3),
            "snap_distance_end": round(float(end_dist), 3),
        }

    def route_candidates(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        count: int = 3,
        max_preview_points: int = 900,
    ) -> list[dict[str, Any]]:
        if len(self.coords) == 0:
            return [
                {
                    "ok": False,
                    "reason": "no skeleton pixels",
                    "points": [list(start), list(end)],
                }
            ]
        start_idx, start_dist = self.nearest_index(start)
        end_idx, end_dist = self.nearest_index(end)
        candidates: list[dict[str, Any]] = []
        penalty: dict[int, float] = {}
        attempts = max(count * 2, count)
        for attempt in range(attempts):
            path = self._shortest_path(start_idx, end_idx, node_penalty=penalty)
            if not path:
                break
            if _is_distinct_path(path, [item["path_indices"] for item in candidates]):
                routed = self.coords[path]
                routed = _smooth_route_points(routed)
                routed = _downsample_points(routed, max_preview_points)
                candidates.append(
                    {
                        "ok": True,
                        "candidate_index": len(candidates),
                        "points": _round_points(routed),
                        "point_count": int(len(routed)),
                        "length": round(float(_curve_length(routed)), 3),
                        "snapped_start": _round_points(self.coords[[start_idx]])[0],
                        "snapped_end": _round_points(self.coords[[end_idx]])[0],
                        "snap_distance_start": round(float(start_dist), 3),
                        "snap_distance_end": round(float(end_dist), 3),
                        "path_indices": path,
                    }
                )
                if len(candidates) >= count:
                    break
            penalty_value = 20.0 + 20.0 * attempt
            for node in path[1:-1]:
                penalty[node] = penalty.get(node, 0.0) + penalty_value
        if not candidates:
            return [
                {
                    "ok": False,
                    "reason": "no connected skeleton path",
                    "points": [list(start), list(end)],
                    "snap_distance_start": round(float(start_dist), 3),
                    "snap_distance_end": round(float(end_dist), 3),
                }
            ]
        for item in candidates:
            item.pop("path_indices", None)
        return candidates

    def nearest_index(self, point: tuple[float, float]) -> tuple[int, float]:
        p = np.asarray(point, dtype=float)
        d2 = np.sum((self.coords - p) ** 2, axis=1)
        idx = int(np.argmin(d2))
        return idx, float(d2[idx] ** 0.5)

    def _shortest_path(
        self,
        start_idx: int,
        end_idx: int,
        node_penalty: dict[int, float] | None = None,
    ) -> list[int]:
        if start_idx == end_idx:
            return [start_idx]
        node_penalty = node_penalty or {}
        count = len(self.coords)
        distances = [float("inf")] * count
        previous = [-1] * count
        distances[start_idx] = 0.0
        heap: list[tuple[float, int]] = [(0.0, start_idx)]
        while heap:
            dist, current = heapq.heappop(heap)
            if current == end_idx:
                break
            if dist > distances[current]:
                continue
            for other, weight in self.adjacency[current]:
                new_dist = dist + weight + node_penalty.get(other, 0.0)
                if new_dist < distances[other]:
                    distances[other] = new_dist
                    previous[other] = current
                    heapq.heappush(heap, (new_dist, other))
        if previous[end_idx] < 0:
            return []
        path = [end_idx]
        current = end_idx
        while current != start_idx:
            current = previous[current]
            if current < 0:
                return []
            path.append(current)
        path.reverse()
        return path


def build_review_graph(
    image_path: str | Path,
    options: ReviewGraphOptions | None = None,
) -> dict[str, Any]:
    graph, _router = build_review_graph_bundle(image_path, options)
    return graph


def build_review_graph_bundle(
    image_path: str | Path,
    options: ReviewGraphOptions | None = None,
) -> tuple[dict[str, Any], SkeletonRouter]:
    """Build a stroke graph for human correction.

    This graph is deliberately earlier than the NURBS fitting stage. It preserves junction
    ambiguity so the user can teach the system which branches belong together.
    """
    options = options or ReviewGraphOptions()
    cv2 = _require_cv2()
    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {path}")

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    extraction_mode = _resolve_extraction_mode(image, options.extraction_mode)
    if extraction_mode == "pencil_weak_line_art":
        ink = _pencil_weak_line_ink(gray, options.weak_line_threshold)
    elif extraction_mode == "white_on_black_sketch":
        ink = _white_on_black_sketch_ink(gray)
    elif extraction_mode == "black_on_white_line_art":
        _gray, ink, extraction_mode = _line_art_ink(image)
        if options.extraction_mode == "black_on_white_line_art" and extraction_mode != "black_on_white_line_art":
            gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
            _, ink = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, small, iterations=1)
            ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, small, iterations=1)
            extraction_mode = "black_on_white_line_art"
    else:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        ink = cv2.Canny(gray, 55, 150)
        small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, small, iterations=1)
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, small, iterations=1)

    skeleton = _skeletonize_zhang_suen(ink > 0)
    if extraction_mode in {"white_on_black_sketch", "pencil_weak_line_art"}:
        skeleton = _prune_skeleton_artifacts(
            skeleton,
            max_spur_length=10.0 if extraction_mode == "pencil_weak_line_art" else 18.0,
            min_component_pixels=2 if extraction_mode == "pencil_weak_line_art" else 4,
            min_component_extent=2.0 if extraction_mode == "pencil_weak_line_art" else 4.0,
        )
    parallel_collapse = _clean_parallel_collapse(options.parallel_collapse)
    if parallel_collapse != "off":
        skeleton = _collapse_parallel_strokes(skeleton, parallel_collapse)
    router = SkeletonRouter.from_skeleton(skeleton)
    chains = _trace_skeleton_chains(skeleton)
    raw_edges: list[dict[str, Any]] = []
    coverage_fragments: list[list[list[float]]] = []
    image_diag = float(np.hypot(h, w))
    min_edge_length = options.min_edge_length
    if extraction_mode == "white_on_black_sketch":
        min_edge_length = max(min_edge_length, image_diag * 0.018)
    promote_short_fragments = extraction_mode == "pencil_weak_line_art"
    for chain in chains:
        points = _as_points3(chain)
        length = _curve_length(points)
        bbox = _bbox(points)
        is_fragment = length < min_edge_length or len(points) < 3 or max(bbox["width"], bbox["height"]) < 3
        if is_fragment and not _promote_fragment_as_edge(points, length, extraction_mode=extraction_mode):
            coverage_fragments.append(_round_points(_downsample_points(points, 12)))
            continue
        raw_edges.append(
            {
                "points": points,
                "length": float(length),
                "bbox": bbox,
                "label": _semantic_guess(points) if not is_fragment else "detail_line_fragment",
                "fragment_promoted": bool(is_fragment and promote_short_fragments),
            }
        )

    raw_edges.sort(key=lambda item: item["length"], reverse=True)
    node_ids = _cluster_endpoints(raw_edges, options.endpoint_cluster_radius)
    nodes = _make_nodes(raw_edges, node_ids)
    edges = []
    for idx, edge in enumerate(raw_edges):
        points = _downsample_points(edge["points"], options.max_points_per_edge)
        start_node, end_node = node_ids[idx]
        edges.append(
            {
                "id": f"edge_{idx:04d}",
                "label": edge["label"],
                "points": _round_points(points),
                "start_node": start_node,
                "end_node": end_node,
                "length": round(float(edge["length"]), 3),
                "bbox": edge["bbox"],
                "fragment_promoted": bool(edge.get("fragment_promoted")),
            }
        )

    node_list = []
    for node_id, node in nodes.items():
        node_list.append(
            {
                "id": node_id,
                "x": round(float(node["point"][0]), 3),
                "y": round(float(node["point"][1]), 3),
                "degree": len(node["edges"]),
                "edges": sorted(node["edges"]),
            }
        )
    node_list.sort(key=lambda item: (-int(item["degree"]), item["id"]))
    coverage = _edge_coverage_metrics(skeleton, raw_edges, coverage_fragments)
    junction_points = _make_router_junction_points(router, image_diag)
    design_strokes = _build_design_strokes(
        edges,
        router,
        image_diag,
        extraction_mode=extraction_mode,
    )

    return {
        "version": 1,
        "image": str(path),
        "image_name": path.name,
        "extraction_mode": extraction_mode,
        "parallel_collapse": parallel_collapse,
        "weak_line_threshold": round(float(options.weak_line_threshold), 3),
        "image_size": {"width": int(w), "height": int(h)},
        "edge_count": len(edges),
        "design_stroke_count": len(design_strokes),
        "junction_count": len(junction_points),
        "node_count": len(node_list),
        "coverage": coverage,
        "coverage_fragments": coverage_fragments,
        "edges": edges,
        "design_strokes": design_strokes,
        "junction_points": junction_points,
        "nodes": node_list,
    }, router


def graph_snapshot_for_training(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": graph.get("version", 1),
        "image": graph.get("image"),
        "image_name": graph.get("image_name"),
        "extraction_mode": graph.get("extraction_mode"),
        "parallel_collapse": graph.get("parallel_collapse"),
        "weak_line_threshold": graph.get("weak_line_threshold"),
        "image_size": graph.get("image_size"),
        "edge_count": graph.get("edge_count", 0),
        "design_stroke_count": graph.get("design_stroke_count", 0),
        "junction_count": graph.get("junction_count", 0),
        "node_count": graph.get("node_count", 0),
        "edges": [
            {
                "id": edge["id"],
                "label": edge.get("label", ""),
                "start_node": edge.get("start_node"),
                "end_node": edge.get("end_node"),
                "length": edge.get("length", 0.0),
                "bbox": edge.get("bbox", {}),
            }
            for edge in graph.get("edges", [])
        ],
        "design_strokes": [
            {
                "id": stroke["id"],
                "label": stroke.get("label", ""),
                "start_node": stroke.get("start_node"),
                "end_node": stroke.get("end_node"),
                "length": stroke.get("length", 0.0),
                "bbox": stroke.get("bbox", {}),
                "source_edge_count": stroke.get("source_edge_count", 0),
            }
            for stroke in graph.get("design_strokes", [])
        ],
        "junction_points": graph.get("junction_points", []),
        "nodes": graph.get("nodes", []),
    }


def _resolve_extraction_mode(image: np.ndarray, requested: str) -> str:
    requested = (requested or "auto").strip().lower()
    aliases = {
        "dark_sketch": "white_on_black_sketch",
        "white_on_black": "white_on_black_sketch",
        "bright_on_dark": "white_on_black_sketch",
        "pencil": "pencil_weak_line_art",
        "pencil_weak": "pencil_weak_line_art",
        "weak_line": "pencil_weak_line_art",
        "weak_pencil": "pencil_weak_line_art",
        "line_art": "black_on_white_line_art",
        "black_on_white": "black_on_white_line_art",
        "canny": "canny_edges",
        "edges": "canny_edges",
    }
    requested = aliases.get(requested, requested)
    if requested in {"white_on_black_sketch", "black_on_white_line_art", "canny_edges", "pencil_weak_line_art"}:
        return requested
    if _is_white_on_black_sketch(image):
        return "white_on_black_sketch"
    if _is_line_art(image):
        return "black_on_white_line_art"
    return "canny_edges"


def _clean_parallel_collapse(value: str) -> str:
    value = (value or "off").strip().lower()
    aliases = {
        "none": "off",
        "false": "off",
        "0": "off",
        "light": "soft",
        "normal": "medium",
        "default": "medium",
        "high": "strong",
        "true": "medium",
        "1": "medium",
    }
    value = aliases.get(value, value)
    return value if value in {"off", "soft", "medium", "strong"} else "off"


def _as_points3(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2:
        raise ValueError("points must be a 2D array")
    if arr.shape[1] == 2:
        arr = np.column_stack([arr, np.zeros(len(arr), dtype=float)])
    return arr


def _bbox(points: np.ndarray) -> dict[str, float]:
    x0 = float(np.min(points[:, 0]))
    y0 = float(np.min(points[:, 1]))
    x1 = float(np.max(points[:, 0]))
    y1 = float(np.max(points[:, 1]))
    return {
        "x": round(x0, 3),
        "y": round(y0, 3),
        "width": round(x1 - x0, 3),
        "height": round(y1 - y0, 3),
    }


def _promote_fragment_as_edge(
    points: np.ndarray,
    length: float,
    *,
    extraction_mode: str,
) -> bool:
    if extraction_mode != "pencil_weak_line_art":
        return False
    if len(points) < 2:
        return False
    return float(length) >= 0.75


def _make_router_junction_points(
    router: SkeletonRouter,
    image_diag: float,
) -> list[dict[str, Any]]:
    coords = np.asarray(router.coords, dtype=float)
    if len(coords) == 0:
        return []
    junction_indices = [
        idx for idx, neighbors in enumerate(router.adjacency) if len(neighbors) >= 3
    ]
    if not junction_indices:
        return []
    cluster_radius = max(2.2, min(6.0, float(image_diag) * 0.0016))
    cell_size = max(cluster_radius, 1.0)
    clusters: list[dict[str, Any]] = []
    grid: dict[tuple[int, int], list[int]] = {}
    for idx in junction_indices:
        point = coords[idx, :2]
        degree = len(router.adjacency[idx])
        cluster_idx = _assign_junction_cluster(clusters, grid, point, cluster_radius, cell_size)
        cluster = clusters[cluster_idx]
        cluster["points"].append(point)
        cluster["degrees"].append(degree)
    out: list[dict[str, Any]] = []
    for idx, cluster in enumerate(clusters):
        points = np.vstack(cluster["points"])
        point = np.mean(points, axis=0)
        out.append(
            {
                "id": f"junction_{idx:04d}",
                "x": round(float(point[0]), 3),
                "y": round(float(point[1]), 3),
                "degree": int(max(cluster["degrees"])),
                "pixel_count": int(len(cluster["points"])),
            }
        )
    out.sort(key=lambda item: (float(item["y"]), float(item["x"])))
    return out


def _assign_junction_cluster(
    clusters: list[dict[str, Any]],
    grid: dict[tuple[int, int], list[int]],
    point: np.ndarray,
    radius: float,
    cell_size: float,
) -> int:
    cell = _grid_cell(point, cell_size)
    best: tuple[float, int] | None = None
    for gy in range(cell[1] - 1, cell[1] + 2):
        for gx in range(cell[0] - 1, cell[0] + 2):
            for idx in grid.get((gx, gy), []):
                cluster_point = np.mean(np.vstack(clusters[idx]["points"]), axis=0)
                dist = float(np.linalg.norm(point - cluster_point))
                if dist <= radius and (best is None or dist < best[0]):
                    best = (dist, idx)
    if best is not None:
        return best[1]
    clusters.append({"points": [], "degrees": []})
    idx = len(clusters) - 1
    grid.setdefault(cell, []).append(idx)
    return idx


def _build_design_strokes(
    edges: list[dict[str, Any]],
    router: SkeletonRouter,
    image_diag: float,
    *,
    extraction_mode: str,
) -> list[dict[str, Any]]:
    """Group raw skeleton edges into longer designer-readable strokes.

    The primary path traces the complete skeleton graph directly, so a continuous red skeleton
    remains continuous even when raw edge tracing introduced small splits. The raw-edge grouping
    below remains as a fallback for unusual graphs.
    """
    router_strokes = _build_router_design_strokes(
        router,
        image_diag,
        extraction_mode=extraction_mode,
    )
    if router_strokes:
        return router_strokes

    items = _design_edge_items(edges)
    if not items:
        return []

    is_pencil = extraction_mode == "pencil_weak_line_art"
    max_gap = max(5.0, float(image_diag) * (0.0075 if is_pencil else 0.005))
    max_angle = float(np.deg2rad(58.0 if is_pencil else 48.0))
    bridge_angle = float(np.deg2rad(76.0 if is_pencil else 66.0))
    min_length = max(3.0, float(image_diag) * 0.002)
    max_edges_per_stroke = 160 if is_pencil else 96
    max_preview_points = 720 if is_pencil else 560

    endpoint_grid = _make_design_endpoint_grid(items, max_gap)
    ordered = sorted(range(len(items)), key=lambda idx: items[idx]["length"], reverse=True)
    globally_used: set[int] = set()
    strokes: list[dict[str, Any]] = []
    for seed_idx in ordered:
        if seed_idx in globally_used:
            continue
        chain: list[tuple[int, bool]] = [(seed_idx, True)]
        local_used = {seed_idx}
        _extend_design_stroke_end(
            chain,
            items,
            endpoint_grid,
            globally_used,
            local_used,
            max_gap=max_gap,
            max_angle=max_angle,
            bridge_angle=bridge_angle,
            max_edges=max_edges_per_stroke,
        )
        chain = [(edge_idx, not forward) for edge_idx, forward in reversed(chain)]
        _extend_design_stroke_end(
            chain,
            items,
            endpoint_grid,
            globally_used,
            local_used,
            max_gap=max_gap,
            max_angle=max_angle,
            bridge_angle=bridge_angle,
            max_edges=max_edges_per_stroke,
        )
        chain = [(edge_idx, not forward) for edge_idx, forward in reversed(chain)]
        globally_used.update(local_used)

        combined = _combine_design_stroke_points(chain, items)
        length = _curve_length(combined)
        if len(combined) < 2 or length < min_length:
            continue
        strokes.append(
            _make_design_stroke(
                len(strokes),
                chain,
                items,
                combined,
                length,
                max_preview_points=max_preview_points,
            )
        )
    strokes.sort(key=lambda item: item["length"], reverse=True)
    for idx, stroke in enumerate(strokes):
        stroke["id"] = f"stroke_{idx:04d}"
    return strokes


def _build_router_design_strokes(
    router: SkeletonRouter,
    image_diag: float,
    *,
    extraction_mode: str,
) -> list[dict[str, Any]]:
    coords = np.asarray(router.coords, dtype=float)
    adjacency = router.adjacency
    if len(coords) == 0 or not adjacency:
        return []

    is_pencil = extraction_mode == "pencil_weak_line_art"
    min_length = max(3.0, float(image_diag) * (0.002 if is_pencil else 0.0035))
    max_preview_points = 760 if is_pencil else 620
    junction_turn = float(np.deg2rad(78.0 if is_pencil else 68.0))
    ambiguous_margin = float(np.deg2rad(10.0))
    max_strokes = 4096

    visited: set[tuple[int, int]] = set()
    degrees = [len(neighbors) for neighbors in adjacency]
    strokes: list[dict[str, Any]] = []

    endpoints = [idx for idx, degree in enumerate(degrees) if degree <= 1]
    for start_idx in sorted(endpoints, key=lambda idx: (coords[idx, 1], coords[idx, 0])):
        for next_idx, _weight in _ordered_router_neighbors(start_idx, adjacency, coords):
            if _router_edge_key(start_idx, next_idx) in visited:
                continue
            path = _trace_router_stroke(
                start_idx,
                next_idx,
                coords,
                adjacency,
                degrees,
                visited,
                junction_turn=junction_turn,
                ambiguous_margin=ambiguous_margin,
            )
            _append_router_stroke(
                strokes,
                path,
                coords,
                min_length=min_length,
                max_preview_points=max_preview_points,
                closed=False,
            )
            if len(strokes) >= max_strokes:
                return _finalize_router_strokes(strokes)

    junctions = [idx for idx, degree in enumerate(degrees) if degree >= 3]
    for start_idx in sorted(junctions, key=lambda idx: (coords[idx, 1], coords[idx, 0])):
        for next_idx, _weight in _ordered_router_neighbors(start_idx, adjacency, coords):
            if _router_edge_key(start_idx, next_idx) in visited:
                continue
            path = _trace_router_stroke(
                start_idx,
                next_idx,
                coords,
                adjacency,
                degrees,
                visited,
                junction_turn=junction_turn,
                ambiguous_margin=ambiguous_margin,
            )
            _append_router_stroke(
                strokes,
                path,
                coords,
                min_length=min_length,
                max_preview_points=max_preview_points,
                closed=False,
            )
            if len(strokes) >= max_strokes:
                return _finalize_router_strokes(strokes)

    for start_idx, neighbors in enumerate(adjacency):
        for next_idx, _weight in neighbors:
            if _router_edge_key(start_idx, next_idx) in visited:
                continue
            path = _trace_router_stroke(
                start_idx,
                next_idx,
                coords,
                adjacency,
                degrees,
                visited,
                junction_turn=junction_turn,
                ambiguous_margin=ambiguous_margin,
            )
            closed = bool(len(path) > 3 and path[0] == path[-1])
            _append_router_stroke(
                strokes,
                path,
                coords,
                min_length=min_length,
                max_preview_points=max_preview_points,
                closed=closed,
            )
            if len(strokes) >= max_strokes:
                return _finalize_router_strokes(strokes)

    return _finalize_router_strokes(strokes)


def _trace_router_stroke(
    start_idx: int,
    next_idx: int,
    coords: np.ndarray,
    adjacency: list[list[tuple[int, float]]],
    degrees: list[int],
    visited: set[tuple[int, int]],
    *,
    junction_turn: float,
    ambiguous_margin: float,
) -> list[int]:
    path = [start_idx]
    previous = start_idx
    current = next_idx
    max_steps = max(16, len(coords) * 2)
    while len(path) < max_steps:
        visited.add(_router_edge_key(previous, current))
        path.append(current)
        candidates = [
            neighbor
            for neighbor, _weight in adjacency[current]
            if neighbor != previous and _router_edge_key(current, neighbor) not in visited
        ]
        if not candidates:
            break
        if current == start_idx and len(path) > 6:
            break
        choice = _choose_router_continuation(
            path,
            current,
            candidates,
            coords,
            degrees,
            junction_turn=junction_turn,
            ambiguous_margin=ambiguous_margin,
        )
        if choice is None:
            break
        previous, current = current, choice
    return path


def _choose_router_continuation(
    path: list[int],
    current: int,
    candidates: list[int],
    coords: np.ndarray,
    degrees: list[int],
    *,
    junction_turn: float,
    ambiguous_margin: float,
) -> int | None:
    if len(candidates) == 1 and degrees[current] <= 2:
        return candidates[0]
    tangent = _router_path_tangent(path, coords)
    scored: list[tuple[float, int]] = []
    current_point = coords[current, :2]
    for candidate in candidates:
        vec = coords[candidate, :2] - current_point
        angle = _angle_between_vectors(tangent, vec)
        scored.append((angle, candidate))
    scored.sort(key=lambda item: item[0])
    best_angle, best_candidate = scored[0]

    if degrees[current] >= 3:
        if best_angle > junction_turn:
            return None
        if len(scored) > 1:
            second_angle = scored[1][0]
            if best_angle > np.deg2rad(30.0) and second_angle - best_angle < ambiguous_margin:
                return None
    return best_candidate


def _append_router_stroke(
    strokes: list[dict[str, Any]],
    path: list[int],
    coords: np.ndarray,
    *,
    min_length: float,
    max_preview_points: int,
    closed: bool,
) -> None:
    if len(path) < 2:
        return
    points = coords[np.asarray(path, dtype=int), :2]
    points = _remove_near_duplicate_points(points, eps=0.25)
    if len(points) < 2:
        return
    length = _curve_length(points)
    if length < min_length:
        return
    smoothed = _smooth_route_points(points, passes=1)
    preview = _as_points3(_downsample_points(smoothed, max_preview_points))
    strokes.append(
        {
            "id": f"stroke_{len(strokes):04d}",
            "label": _semantic_guess(preview),
            "points": _round_points(preview),
            "start_node": _router_coord_node_id(points[0]),
            "end_node": _router_coord_node_id(points[-1]),
            "length": round(float(length), 3),
            "bbox": _bbox(points),
            "source": "skeleton_graph_tracing",
            "source_edge_count": max(1, len(path) - 1),
            "closed": bool(closed),
        }
    )


def _finalize_router_strokes(strokes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strokes.sort(key=lambda item: item["length"], reverse=True)
    for idx, stroke in enumerate(strokes):
        stroke["id"] = f"stroke_{idx:04d}"
    return strokes


def _ordered_router_neighbors(
    node_idx: int,
    adjacency: list[list[tuple[int, float]]],
    coords: np.ndarray,
) -> list[tuple[int, float]]:
    point = coords[node_idx, :2]
    return sorted(adjacency[node_idx], key=lambda item: (coords[item[0], 1] - point[1], coords[item[0], 0] - point[0]))


def _router_path_tangent(path: list[int], coords: np.ndarray) -> np.ndarray:
    if len(path) < 2:
        return np.array([1.0, 0.0])
    current = coords[path[-1], :2]
    back_index = path[max(0, len(path) - 8)]
    vec = current - coords[back_index, :2]
    if np.linalg.norm(vec) <= 1e-9:
        vec = coords[path[-1], :2] - coords[path[-2], :2]
    return _normalize_vec(vec)


def _router_edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _router_coord_node_id(point: np.ndarray) -> str:
    return f"router_node_{int(round(float(point[0])))}_{int(round(float(point[1])))}"


def _design_edge_items(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for edge in edges:
        try:
            points = np.asarray(edge.get("points") or [], dtype=float)
        except Exception:
            continue
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
            continue
        points = points[:, :2]
        items.append(
            {
                "id": str(edge.get("id") or f"edge_{len(items):04d}"),
                "label": str(edge.get("label") or "detail_line"),
                "points": points,
                "length": float(edge.get("length") or _curve_length(points)),
                "start_node": str(edge.get("start_node") or ""),
                "end_node": str(edge.get("end_node") or ""),
                "fragment_promoted": bool(edge.get("fragment_promoted")),
            }
        )
    return items


def _make_design_endpoint_grid(
    items: list[dict[str, Any]],
    radius: float,
) -> dict[str, Any]:
    cell_size = max(float(radius) / 3.0, 3.0)
    grid: dict[tuple[int, int], list[tuple[int, bool, np.ndarray]]] = {}
    for edge_idx, item in enumerate(items):
        for forward, point in ((True, item["points"][0]), (False, item["points"][-1])):
            cell = _grid_cell(point, cell_size)
            grid.setdefault(cell, []).append((edge_idx, forward, point.astype(float, copy=False)))
    return {"cell_size": cell_size, "grid": grid}


def _extend_design_stroke_end(
    chain: list[tuple[int, bool]],
    items: list[dict[str, Any]],
    endpoint_grid: dict[str, Any],
    globally_used: set[int],
    local_used: set[int],
    *,
    max_gap: float,
    max_angle: float,
    bridge_angle: float,
    max_edges: int,
) -> None:
    while len(chain) < max_edges:
        edge_idx, forward = chain[-1]
        current_points = _oriented_design_points(items[edge_idx]["points"], forward)
        end_point = current_points[-1]
        current_tangent = _design_end_tangent(current_points)
        best = _best_design_continuation(
            items,
            endpoint_grid,
            end_point,
            current_tangent,
            globally_used,
            local_used,
            max_gap=max_gap,
            max_angle=max_angle,
            bridge_angle=bridge_angle,
        )
        if best is None:
            return
        _score, next_idx, next_forward = best
        local_used.add(next_idx)
        chain.append((next_idx, next_forward))


def _best_design_continuation(
    items: list[dict[str, Any]],
    endpoint_grid: dict[str, Any],
    end_point: np.ndarray,
    current_tangent: np.ndarray,
    globally_used: set[int],
    local_used: set[int],
    *,
    max_gap: float,
    max_angle: float,
    bridge_angle: float,
) -> tuple[float, int, bool] | None:
    best: tuple[float, int, bool] | None = None
    cell_size = float(endpoint_grid["cell_size"])
    grid = endpoint_grid["grid"]
    search_radius = int(np.ceil(max_gap / max(cell_size, 1e-6)))
    center = _grid_cell(end_point, cell_size)
    max_gap2 = max_gap * max_gap
    nearby: list[tuple[float, int, bool, np.ndarray]] = []
    for gy in range(center[1] - search_radius, center[1] + search_radius + 1):
        for gx in range(center[0] - search_radius, center[0] + search_radius + 1):
            for candidate_idx, candidate_forward, candidate_start in grid.get((gx, gy), []):
                if candidate_idx in globally_used or candidate_idx in local_used:
                    continue
                gap_vec = candidate_start - end_point
                gap2 = float(np.dot(gap_vec, gap_vec))
                if gap2 > max_gap2:
                    continue
                nearby.append((gap2, candidate_idx, candidate_forward, candidate_start))
    if len(nearby) > 40:
        nearby = sorted(nearby, key=lambda item: item[0])[:40]
    for gap2, candidate_idx, candidate_forward, candidate_start in nearby:
        candidate_points = _oriented_design_points(
            items[candidate_idx]["points"], candidate_forward
        )
        if len(candidate_points) < 2:
            continue
        gap_vec = candidate_start - end_point
        gap = float(gap2**0.5)
        candidate_tangent = _design_start_tangent(candidate_points)
        tangent_angle = _angle_between_vectors(current_tangent, candidate_tangent)
        if tangent_angle > max_angle:
            continue
        bridge_penalty = 0.0
        if gap > 1.25:
            bridge = gap_vec / max(gap, 1e-9)
            bridge_in = _angle_between_vectors(current_tangent, bridge)
            bridge_out = _angle_between_vectors(bridge, candidate_tangent)
            if bridge_in > bridge_angle or bridge_out > bridge_angle:
                continue
            bridge_penalty = 0.35 * (bridge_in + bridge_out)
        score = 1.8 * tangent_angle + bridge_penalty + 0.75 * gap / max(max_gap, 1.0)
        if items[candidate_idx].get("label") == "detail_line_fragment":
            score += 0.02
        if best is None or score < best[0]:
            best = (score, candidate_idx, candidate_forward)
    return best


def _make_design_stroke(
    index: int,
    chain: list[tuple[int, bool]],
    items: list[dict[str, Any]],
    points: np.ndarray,
    length: float,
    *,
    max_preview_points: int,
) -> dict[str, Any]:
    first_idx, first_forward = chain[0]
    last_idx, last_forward = chain[-1]
    labels = [items[edge_idx]["label"] for edge_idx, _forward in chain]
    label = _majority_text(labels)
    if label == "detail_line_fragment":
        label = _semantic_guess(points)
    preview = _as_points3(_downsample_points(points, max_preview_points))
    source_ids = [items[edge_idx]["id"] for edge_idx, _forward in chain]
    return {
        "id": f"stroke_{index:04d}",
        "label": label,
        "points": _round_points(preview),
        "start_node": _oriented_design_node(items[first_idx], first_forward, start=True),
        "end_node": _oriented_design_node(items[last_idx], last_forward, start=False),
        "length": round(float(length), 3),
        "bbox": _bbox(points),
        "source": "design_stroke_grouping",
        "source_edge_count": len(source_ids),
        "source_edge_ids": source_ids[:128],
        "source_edge_ids_truncated": len(source_ids) > 128,
    }


def _combine_design_stroke_points(
    chain: list[tuple[int, bool]],
    items: list[dict[str, Any]],
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for edge_idx, forward in chain:
        pts = _oriented_design_points(items[edge_idx]["points"], forward)
        pts = _remove_near_duplicate_points(pts, eps=0.35)
        if len(pts) < 2:
            continue
        parts.append(pts)
    if not parts:
        return np.zeros((0, 2), dtype=float)
    return _remove_near_duplicate_points(np.vstack(parts), eps=0.35)


def _oriented_design_points(points: np.ndarray, forward: bool) -> np.ndarray:
    return points if forward else points[::-1]


def _oriented_design_node(item: dict[str, Any], forward: bool, *, start: bool) -> str:
    if start:
        value = item.get("start_node") if forward else item.get("end_node")
    else:
        value = item.get("end_node") if forward else item.get("start_node")
    return str(value or "")


def _design_start_tangent(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.array([1.0, 0.0])
    k = min(10, len(points) - 1)
    return _normalize_vec(points[k, :2] - points[0, :2])


def _design_end_tangent(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.array([1.0, 0.0])
    k = min(10, len(points) - 1)
    return _normalize_vec(points[-1, :2] - points[-1 - k, :2])


def _angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    a_n = _normalize_vec(a)
    b_n = _normalize_vec(b)
    return float(np.arccos(float(np.clip(np.dot(a_n, b_n), -1.0, 1.0))))


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        return np.array([1.0, 0.0])
    return vec / norm


def _grid_cell(point: np.ndarray, cell_size: float) -> tuple[int, int]:
    return (int(np.floor(float(point[0]) / cell_size)), int(np.floor(float(point[1]) / cell_size)))


def _majority_text(values: list[str]) -> str:
    if not values:
        return "detail_line"
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def _cluster_endpoints(
    raw_edges: list[dict[str, Any]],
    radius: float,
) -> list[tuple[str, str]]:
    clusters: list[dict[str, Any]] = []
    grid: dict[tuple[int, int], list[int]] = {}
    cell_size = max(float(radius), 1.0)
    assignments: list[list[str]] = []
    for edge_idx, edge in enumerate(raw_edges):
        edge_assignments = []
        for side, point in (("start", edge["points"][0]), ("end", edge["points"][-1])):
            cluster_id = _assign_endpoint_cluster(clusters, grid, point[:2], radius, cell_size)
            clusters[cluster_id]["members"].append((edge_idx, side))
            edge_assignments.append(f"node_{cluster_id:04d}")
        assignments.append(edge_assignments)
    return [(item[0], item[1]) for item in assignments]


def _assign_endpoint_cluster(
    clusters: list[dict[str, Any]],
    grid: dict[tuple[int, int], list[int]],
    point: np.ndarray,
    radius: float,
    cell_size: float,
) -> int:
    cell = (int(np.floor(float(point[0]) / cell_size)), int(np.floor(float(point[1]) / cell_size)))
    best: tuple[float, int] | None = None
    seen: set[int] = set()
    for gy in range(cell[1] - 1, cell[1] + 2):
        for gx in range(cell[0] - 1, cell[0] + 2):
            for idx in grid.get((gx, gy), []):
                if idx in seen:
                    continue
                seen.add(idx)
                cluster = clusters[idx]
                dist = float(np.linalg.norm(point - cluster["point"]))
                if dist <= radius and (best is None or dist < best[0]):
                    best = (dist, idx)
    if best is None:
        clusters.append({"point": point.astype(float).copy(), "members": []})
        idx = len(clusters) - 1
        grid.setdefault(cell, []).append(idx)
        return idx
    idx = best[1]
    cluster = clusters[idx]
    count = len(cluster["members"])
    cluster["point"] = (cluster["point"] * count + point) / max(count + 1, 1)
    new_cell = (
        int(np.floor(float(cluster["point"][0]) / cell_size)),
        int(np.floor(float(cluster["point"][1]) / cell_size)),
    )
    if idx not in grid.setdefault(new_cell, []):
        grid[new_cell].append(idx)
    return idx


def _assign_endpoint_cluster_linear(
    clusters: list[dict[str, Any]],
    point: np.ndarray,
    radius: float,
) -> int:
    best: tuple[float, int] | None = None
    for idx, cluster in enumerate(clusters):
        dist = float(np.linalg.norm(point - cluster["point"]))
        if dist <= radius and (best is None or dist < best[0]):
            best = (dist, idx)
    if best is None:
        clusters.append({"point": point.astype(float).copy(), "members": []})
        return len(clusters) - 1
    idx = best[1]
    cluster = clusters[idx]
    count = len(cluster["members"])
    cluster["point"] = (cluster["point"] * count + point) / max(count + 1, 1)
    return idx


def _make_nodes(
    raw_edges: list[dict[str, Any]],
    node_ids: list[tuple[str, str]],
) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    for idx, edge in enumerate(raw_edges):
        edge_id = f"edge_{idx:04d}"
        for side_index, node_id in enumerate(node_ids[idx]):
            point = edge["points"][0 if side_index == 0 else -1, :2]
            node = nodes.setdefault(node_id, {"points": [], "edges": set()})
            node["points"].append(point)
            node["edges"].add(edge_id)
    for node in nodes.values():
        node["point"] = np.mean(np.vstack(node["points"]), axis=0)
        node["edges"] = set(node["edges"])
    return nodes


def _downsample_points(points: np.ndarray, max_count: int) -> np.ndarray:
    if len(points) <= max_count:
        return points
    idx = np.linspace(0, len(points) - 1, max_count).round().astype(int)
    return points[idx]


def _round_points(points: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 3), round(float(y), 3)] for x, y in points[:, :2]]


def _remove_near_duplicate_points(points: np.ndarray, eps: float = 0.5) -> np.ndarray:
    if len(points) <= 1:
        return points
    kept = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point[:2] - kept[-1][:2]) > eps:
            kept.append(point)
    return np.asarray(kept, dtype=float)


def _smooth_route_points(points: np.ndarray, passes: int = 2) -> np.ndarray:
    if len(points) < 4:
        return points.astype(float, copy=True)
    out = points[:, :2].astype(float, copy=True)
    for _ in range(passes):
        smoothed = out.copy()
        smoothed[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out = smoothed
    return out


def _is_distinct_path(path: list[int], existing: list[list[int]], max_overlap: float = 0.82) -> bool:
    if not existing:
        return True
    current = set(path)
    if not current:
        return False
    for other in existing:
        shared = len(current.intersection(other))
        denom = max(min(len(current), len(other)), 1)
        if shared / denom > max_overlap:
            return False
    return True


def _edge_coverage_metrics(
    skeleton: np.ndarray,
    raw_edges: list[dict[str, Any]],
    coverage_fragments: list[list[list[float]]],
) -> dict[str, float | int]:
    cv2 = _require_cv2()
    covered = np.zeros(skeleton.shape, np.uint8)
    for edge in raw_edges:
        pts = np.round(edge["points"][:, :2]).astype(np.int32)
        if len(pts) == 1:
            x, y = pts[0]
            if 0 <= y < covered.shape[0] and 0 <= x < covered.shape[1]:
                covered[y, x] = 255
        elif len(pts) >= 2:
            cv2.polylines(covered, [pts], False, 255, thickness=3, lineType=cv2.LINE_AA)
    for fragment in coverage_fragments:
        pts = np.round(np.asarray(fragment, dtype=float)).astype(np.int32)
        if len(pts) == 1:
            x, y = pts[0]
            if 0 <= y < covered.shape[0] and 0 <= x < covered.shape[1]:
                covered[y, x] = 255
        elif len(pts) >= 2:
            cv2.polylines(covered, [pts], False, 255, thickness=3, lineType=cv2.LINE_AA)
    total = int(np.sum(skeleton > 0))
    missed = int(np.sum((skeleton > 0) & (covered == 0)))
    ratio = 1.0 if total == 0 else 1.0 - missed / max(total, 1)
    return {
        "skeleton_pixels": total,
        "missed_skeleton_pixels": missed,
        "coverage_ratio": round(float(ratio), 6),
    }


if __name__ == "__main__":
    import sys

    print(
        "graph.py is a library: running this file does not write images or JSON.\n"
        "To see skeleton/edges output:\n"
        "  - Run graph_pipeline_walkthrough.py (set IDE_IMAGE / IDE_DEBUG_OUT -> PNG + graph_summary.json)\n"
        "  - Or: autoalias skeleton-review / review-image in browser\n",
        file=sys.stdout,
    )
