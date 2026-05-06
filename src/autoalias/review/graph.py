from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autoalias.vision.extractor import (
    _curve_length,
    _is_line_art,
    _require_cv2,
    _semantic_guess,
    _skeletonize_zhang_suen,
    _trace_skeleton_chains,
)


@dataclass(slots=True)
class ReviewGraphOptions:
    min_edge_length: float = 3.0
    endpoint_cluster_radius: float = 6.0
    max_points_per_edge: int = 320


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

    def nearest_index(self, point: tuple[float, float]) -> tuple[int, float]:
        p = np.asarray(point, dtype=float)
        d2 = np.sum((self.coords - p) ** 2, axis=1)
        idx = int(np.argmin(d2))
        return idx, float(d2[idx] ** 0.5)

    def _shortest_path(self, start_idx: int, end_idx: int) -> list[int]:
        if start_idx == end_idx:
            return [start_idx]
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
                new_dist = dist + weight
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
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    if _is_line_art(image):
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        ink = cv2.Canny(gray, 55, 150)
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, small, iterations=1)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, small, iterations=1)

    skeleton = _skeletonize_zhang_suen(ink > 0)
    router = SkeletonRouter.from_skeleton(skeleton)
    chains = _trace_skeleton_chains(skeleton)
    raw_edges: list[dict[str, Any]] = []
    coverage_fragments: list[list[list[float]]] = []
    for chain in chains:
        points = _as_points3(chain)
        length = _curve_length(points)
        if length < options.min_edge_length or len(points) < 3:
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
