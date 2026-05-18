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
    if extraction_mode == "white_on_black_sketch":
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
    if extraction_mode == "white_on_black_sketch":
        skeleton = _prune_skeleton_artifacts(
            skeleton,
            max_spur_length=18.0,
            min_component_pixels=4,
            min_component_extent=4.0,
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
    for chain in chains:
        points = _as_points3(chain)
        length = _curve_length(points)
        if length < min_edge_length or len(points) < 3:
            coverage_fragments.append(_round_points(_downsample_points(points, 12)))
            continue
        bbox = _bbox(points)
        if max(bbox["width"], bbox["height"]) < 3:
            coverage_fragments.append(_round_points(_downsample_points(points, 12)))
            continue
        raw_edges.append(
            {
                "points": points,
                "length": float(length),
                "bbox": bbox,
                "label": _semantic_guess(points),
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

    return {
        "version": 1,
        "image": str(path),
        "image_name": path.name,
        "extraction_mode": extraction_mode,
        "parallel_collapse": parallel_collapse,
        "image_size": {"width": int(w), "height": int(h)},
        "edge_count": len(edges),
        "node_count": len(node_list),
        "coverage": coverage,
        "coverage_fragments": coverage_fragments,
        "edges": edges,
        "nodes": node_list,
    }, router


def graph_snapshot_for_training(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": graph.get("version", 1),
        "image": graph.get("image"),
        "image_name": graph.get("image_name"),
        "extraction_mode": graph.get("extraction_mode"),
        "parallel_collapse": graph.get("parallel_collapse"),
        "image_size": graph.get("image_size"),
        "edge_count": graph.get("edge_count", 0),
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
        "nodes": graph.get("nodes", []),
    }


def _resolve_extraction_mode(image: np.ndarray, requested: str) -> str:
    requested = (requested or "auto").strip().lower()
    aliases = {
        "dark_sketch": "white_on_black_sketch",
        "white_on_black": "white_on_black_sketch",
        "bright_on_dark": "white_on_black_sketch",
        "line_art": "black_on_white_line_art",
        "black_on_white": "black_on_white_line_art",
        "canny": "canny_edges",
        "edges": "canny_edges",
    }
    requested = aliases.get(requested, requested)
    if requested in {"white_on_black_sketch", "black_on_white_line_art", "canny_edges"}:
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


def _cluster_endpoints(
    raw_edges: list[dict[str, Any]],
    radius: float,
) -> list[tuple[str, str]]:
    clusters: list[dict[str, Any]] = []
    assignments: list[list[str]] = []
    for edge_idx, edge in enumerate(raw_edges):
        edge_assignments = []
        for side, point in (("start", edge["points"][0]), ("end", edge["points"][-1])):
            cluster_id = _assign_endpoint_cluster(clusters, point[:2], radius)
            clusters[cluster_id]["members"].append((edge_idx, side))
            edge_assignments.append(f"node_{cluster_id:04d}")
        assignments.append(edge_assignments)
    return [(item[0], item[1]) for item in assignments]


def _assign_endpoint_cluster(
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
