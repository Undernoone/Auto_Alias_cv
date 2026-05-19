from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class _GraphEdge:
    id: str
    label: str
    points: np.ndarray
    start_node: str
    end_node: str
    length: float


def suggest_geometry_segments(
    graph: dict[str, Any],
    *,
    max_curves: int = 32,
    min_length: float | None = None,
    max_turn_deg: float = 28.0,
    max_junction_turn_deg: float = 18.0,
    max_chain_edges: int = 8,
    max_gap: float = 0.0,
    max_gap_turn_deg: float | None = None,
) -> list[dict[str, Any]]:
    """Suggest editable design curves directly from the skeleton graph.

    The goal is deliberately conservative: keep obvious smooth strokes, join only when
    the next branch is a tangent continuation, and leave ambiguous junctions for the
    human editor.
    """
    edges = _read_edges(graph)
    if not edges:
        return []
    image_size = graph.get("image_size", {}) or {}
    diag = float(
        math.hypot(
            float(image_size.get("width", 0.0) or 0.0),
            float(image_size.get("height", 0.0) or 0.0),
        )
    )
    min_length = float(min_length if min_length is not None else max(24.0, diag * 0.032))
    max_curves = max(1, min(int(max_curves), 512))
    max_turn = math.radians(max(4.0, min(float(max_turn_deg), 70.0)))
    junction_turn = math.radians(max(3.0, min(float(max_junction_turn_deg), 45.0)))
    gap_turn_deg = max_gap_turn_deg if max_gap_turn_deg is not None else max_turn_deg
    gap_turn = math.radians(max(4.0, min(float(gap_turn_deg), 75.0)))
    max_gap = max(0.0, float(max_gap))

    incident = _incident_edges(edges)
    sorted_edges = sorted(edges.values(), key=lambda e: e.length, reverse=True)
    used: set[str] = set()
    suggestions: list[dict[str, Any]] = []

    for edge in sorted_edges:
        if edge.id in used or edge.length < min_length:
            continue
        chain = _grow_chain(
            edge,
            edges,
            incident,
            used,
            min_edge_length=min_length * 0.55,
            max_turn=max_turn,
            junction_turn=junction_turn,
            max_chain_edges=max_chain_edges,
            max_gap=max_gap,
            gap_turn=gap_turn,
        )
        combined = _combine_chain_points(chain)
        length = _curve_length(combined)
        if length < min_length:
            continue
        chain_ids = [item[0].id for item in chain]
        used.update(chain_ids)
        suggestions.append(_make_design_curve(len(suggestions), chain, combined, length))
        if len(suggestions) >= max_curves:
            break

    return suggestions


def _read_edges(graph: dict[str, Any]) -> dict[str, _GraphEdge]:
    out: dict[str, _GraphEdge] = {}
    for item in graph.get("edges", []) or []:
        try:
            points = np.asarray(item.get("points") or [], dtype=float)
        except Exception:
            continue
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
            continue
        points = points[:, :2]
        edge_id = str(item.get("id") or f"edge_{len(out):04d}")
        out[edge_id] = _GraphEdge(
            id=edge_id,
            label=str(item.get("label") or "detail_line"),
            points=points,
            start_node=str(item.get("start_node") or ""),
            end_node=str(item.get("end_node") or ""),
            length=float(item.get("length") or _curve_length(points)),
        )
    return out


def _incident_edges(edges: dict[str, _GraphEdge]) -> dict[str, list[str]]:
    incident: dict[str, list[str]] = {}
    for edge in edges.values():
        if edge.start_node:
            incident.setdefault(edge.start_node, []).append(edge.id)
        if edge.end_node:
            incident.setdefault(edge.end_node, []).append(edge.id)
    return incident


def _grow_chain(
    seed: _GraphEdge,
    edges: dict[str, _GraphEdge],
    incident: dict[str, list[str]],
    globally_used: set[str],
    *,
    min_edge_length: float,
    max_turn: float,
    junction_turn: float,
    max_chain_edges: int,
    max_gap: float,
    gap_turn: float,
) -> list[tuple[_GraphEdge, bool]]:
    chain: list[tuple[_GraphEdge, bool]] = [(seed, True)]
    local_used = {seed.id}
    _extend_chain_end(
        chain,
        edges,
        incident,
        globally_used,
        local_used,
        min_edge_length=min_edge_length,
        max_turn=max_turn,
        junction_turn=junction_turn,
        max_chain_edges=max_chain_edges,
        max_gap=max_gap,
        gap_turn=gap_turn,
    )
    chain = [(edge, not forward) for edge, forward in reversed(chain)]
    _extend_chain_end(
        chain,
        edges,
        incident,
        globally_used,
        local_used,
        min_edge_length=min_edge_length,
        max_turn=max_turn,
        junction_turn=junction_turn,
        max_chain_edges=max_chain_edges,
        max_gap=max_gap,
        gap_turn=gap_turn,
    )
    return [(edge, not forward) for edge, forward in reversed(chain)]


def _extend_chain_end(
    chain: list[tuple[_GraphEdge, bool]],
    edges: dict[str, _GraphEdge],
    incident: dict[str, list[str]],
    globally_used: set[str],
    local_used: set[str],
    *,
    min_edge_length: float,
    max_turn: float,
    junction_turn: float,
    max_chain_edges: int,
    max_gap: float,
    gap_turn: float,
) -> None:
    while len(chain) < max_chain_edges:
        edge, forward = chain[-1]
        node = edge.end_node if forward else edge.start_node
        if not node:
            return
        current_tangent = _end_tangent(_oriented_points(edge, forward))
        node_degree = len(incident.get(node, []))
        turn_limit = junction_turn if node_degree >= 3 else max_turn
        end_point = _oriented_points(edge, forward)[-1, :2]
        best = _best_connected_continuation(
            edges,
            incident.get(node, []),
            globally_used,
            local_used,
            current_tangent,
            end_point,
            min_edge_length=min_edge_length,
            turn_limit=turn_limit,
            max_gap=max_gap,
            gap_turn=gap_turn,
        )
        if best is None:
            return
        _score, next_edge, next_forward = best
        local_used.add(next_edge.id)
        chain.append((next_edge, next_forward))


def _best_connected_continuation(
    edges: dict[str, _GraphEdge],
    incident_ids: list[str],
    globally_used: set[str],
    local_used: set[str],
    current_tangent: np.ndarray,
    end_point: np.ndarray,
    *,
    min_edge_length: float,
    turn_limit: float,
    max_gap: float,
    gap_turn: float,
) -> tuple[float, _GraphEdge, bool] | None:
    best: tuple[float, _GraphEdge, bool] | None = None
    for candidate_id in incident_ids:
        if candidate_id in globally_used or candidate_id in local_used:
            continue
        candidate = edges.get(candidate_id)
        if candidate is None or candidate.length < min_edge_length:
            continue
        # Infer the orientation from the endpoint nearest to the current chain end. This also
        # works when endpoint clustering moved the node center a few pixels away.
        start_gap = float(np.linalg.norm(candidate.points[0, :2] - end_point))
        end_gap = float(np.linalg.norm(candidate.points[-1, :2] - end_point))
        candidate_forward = start_gap <= end_gap
        candidate_points = _oriented_points(candidate, candidate_forward)
        angle = _angle_between(current_tangent, _start_tangent(candidate_points))
        if angle > turn_limit:
            continue
        score = angle + 0.002 * max(0.0, min_edge_length - candidate.length)
        if best is None or score < best[0]:
            best = (score, candidate, candidate_forward)

    if max_gap <= 0:
        return best

    for candidate in edges.values():
        if candidate.id in globally_used or candidate.id in local_used:
            continue
        if candidate.length < min_edge_length:
            continue
        for candidate_forward in (True, False):
            candidate_points = _oriented_points(candidate, candidate_forward)
            candidate_start = candidate_points[0, :2]
            gap = float(np.linalg.norm(candidate_start - end_point))
            if gap <= 1e-6:
                continue
            if gap > max_gap:
                continue
            candidate_tangent = _start_tangent(candidate_points)
            angle = _angle_between(current_tangent, candidate_tangent)
            if angle > gap_turn:
                continue
            gap_vec = (candidate_start - end_point) / gap
            if _angle_between(current_tangent, gap_vec) > gap_turn:
                continue
            if _angle_between(gap_vec, candidate_tangent) > gap_turn:
                continue
            score = angle + 0.65 * (gap / max(max_gap, 1.0))
            if best is None or score < best[0]:
                best = (score, candidate, candidate_forward)
    return best


def _make_design_curve(
    index: int,
    chain: list[tuple[_GraphEdge, bool]],
    points: np.ndarray,
    length: float,
) -> dict[str, Any]:
    chain_ids = [edge.id for edge, _forward in chain]
    labels = [edge.label for edge, _forward in chain]
    semantic = _majority_label(labels)
    route_points = _round_points(points)
    parts = _chain_parts(chain)
    manual_points = _chain_boundary_points(parts)
    route_segments = _chain_route_segments(parts)
    return {
        "id": f"geo_curve_{int(time.time() * 1000):x}_{index:03d}",
        "type": "manual_design_curve",
        "semantic": semantic,
        "edge_ids": chain_ids,
        "manual_points": manual_points,
        "cut_points": manual_points,
        "closed": False,
        "routed_points": route_points,
        "route_segments": route_segments,
        "branch_choices": [0 for _ in route_segments],
        "route_ok": True,
        "source": "geometry_auto_segment",
        "confidence": round(float(min(0.96, 0.45 + length / 900.0)), 3),
        "reason": "geometry stroke tracing: long smooth skeleton path with editable boundary points",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _oriented_points(edge: _GraphEdge, forward: bool) -> np.ndarray:
    return edge.points if forward else edge.points[::-1]


def _combine_chain_points(chain: list[tuple[_GraphEdge, bool]]) -> np.ndarray:
    parts = [part["points"] for part in _chain_parts(chain)]
    if not parts:
        return np.zeros((0, 2), dtype=float)
    combined = np.vstack([part for part in parts if len(part)])
    return _remove_near_duplicate_points(combined)


def _chain_parts(chain: list[tuple[_GraphEdge, bool]]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for edge, forward in chain:
        pts = _oriented_points(edge, forward)
        pts = _remove_near_duplicate_points(pts)
        if len(pts) < 2:
            continue
        parts.append(
            {
                "edge": edge,
                "forward": forward,
                "points": pts,
                "length": _curve_length(pts),
            }
        )
    return parts


def _chain_boundary_points(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not parts:
        return []
    boundaries = [parts[0]["points"][0]]
    before_edges = [""]
    after_edges = [parts[0]["edge"].id]
    for idx, part in enumerate(parts):
        boundaries.append(part["points"][-1])
        before_edges.append(part["edge"].id)
        after_edges.append(parts[idx + 1]["edge"].id if idx + 1 < len(parts) else "")

    out: list[dict[str, Any]] = []
    for order, point in enumerate(boundaries):
        out.append(
            {
                "x": round(float(point[0]), 3),
                "y": round(float(point[1]), 3),
                "order": order,
                "snap_source": "geometry_auto_segment",
                "auto_boundary": True,
                "before_edge_id": before_edges[order],
                "after_edge_id": after_edges[order],
            }
        )
    return out


def _chain_route_segments(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments = []
    for index, part in enumerate(parts):
        edge = part["edge"]
        segment_points = part["points"]
        if index > 0:
            boundary = parts[index - 1]["points"][-1]
            if np.linalg.norm(boundary - segment_points[0]) > 0.5:
                segment_points = np.vstack([boundary, segment_points])
        points = _round_points(segment_points)
        segments.append(
            {
                "ok": True,
                "points": points,
                "segment_index": index,
                "selected_candidate": 0,
                "length": round(float(_curve_length(segment_points)), 3),
                "source": "geometry_auto_segment",
                "edge_id": edge.id,
                "forward": bool(part["forward"]),
            }
        )
    return segments


def _remove_near_duplicate_points(points: np.ndarray, eps: float = 0.5) -> np.ndarray:
    if len(points) <= 1:
        return points
    kept = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(p - kept[-1]) > eps:
            kept.append(p)
    return np.asarray(kept, dtype=float)


def _majority_label(labels: list[str]) -> str:
    if not labels:
        return "detail_line"
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    label = max(counts.items(), key=lambda item: item[1])[0]
    if label.startswith("candidate_"):
        return label.removeprefix("candidate_")
    return label


def _round_points(points: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 3), round(float(y), 3)] for x, y in points[:, :2]]


def _start_tangent(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.array([1.0, 0.0])
    k = min(8, len(points) - 1)
    vec = points[k, :2] - points[0, :2]
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-9)


def _end_tangent(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.array([1.0, 0.0])
    k = min(8, len(points) - 1)
    vec = points[-1, :2] - points[-1 - k, :2]
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-9)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(math.acos(dot))


def _curve_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)))
