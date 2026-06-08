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


@dataclass(slots=True)
class _SpanCut:
    s: float
    point: np.ndarray
    kind: str
    source_edge: str = ""


@dataclass(slots=True)
class _EndpointRecord:
    edge_id: str
    point: np.ndarray
    s: float
    at_start: bool


@dataclass(slots=True)
class _EndpointCluster:
    id: int
    point: np.ndarray
    records: list[_EndpointRecord]


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
    """Suggest curves by splitting strokes only at real geometric intersections.

    This is the active production path. It deliberately does not extend/merge strokes by
    tangent continuity. A span boundary is created only at stroke endpoints and at places
    where another stroke crosses, touches, or branches into the current stroke.
    """
    _ = (max_turn_deg, max_junction_turn_deg, max_chain_edges, max_gap, max_gap_turn_deg)
    return _suggest_intersection_only_segments(
        graph,
        max_curves=max_curves,
        min_length=min_length,
    )


# Legacy chaining logic kept for reference. It is no longer called by
# `suggest_geometry_segments` because it could merge long car outlines into one curve.
def _suggest_geometry_segments_legacy(
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


def _suggest_intersection_only_segments(
    graph: dict[str, Any],
    *,
    max_curves: int,
    min_length: float | None,
) -> list[dict[str, Any]]:
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
    min_length_value = float(min_length if min_length is not None else max(18.0, diag * 0.018))
    max_curves = max(1, min(int(max_curves), 512))
    intersection_radius = max(3.0, diag * 0.0035)

    cuts_by_edge = _intersection_cuts_by_edge(graph, edges, intersection_radius)
    suggestions: list[dict[str, Any]] = []
    sorted_edges = sorted(edges.values(), key=lambda item: item.length, reverse=True)
    accepted_paths: list[np.ndarray] = []
    suggestion_seed = int(time.time() * 1000)
    scan_limit = max(max_curves * 4, max_curves + 80)
    scan_limit = min(max(scan_limit, max_curves), 2048)
    for edge in sorted_edges:
        if edge.length < min_length_value:
            continue
        cuts = _sorted_edge_cuts(edge, cuts_by_edge.get(edge.id, []))
        if len(cuts) < 2:
            continue
        route_segments = _route_segments_from_cuts(
            edge,
            cuts,
            min_span_length=max(4.0, min_length_value * 0.12),
            min_span_chord=max(3.0, diag * 0.0015),
        )
        if not route_segments:
            continue
        for segment in route_segments:
            points = _segment_points_array(segment)
            if len(points) < 2:
                continue
            if _route_segment_duplicates_existing(
                points,
                accepted_paths,
                distance=max(2.6, diag * 0.0016),
            ):
                continue
            manual_points = _manual_points_from_segment(edge, segment)
            if len(manual_points) < 2:
                continue
            single_segment = {**segment, "segment_index": 0, "segment_count": 1}
            is_closed = bool(segment.get("closed"))
            accepted_paths.append(points)
            suggestions.append(
                {
                    "id": f"geo_curve_{suggestion_seed:x}_{len(suggestions):03d}",
                    "type": "manual_design_curve",
                    "semantic": edge.label,
                    "edge_ids": [edge.id],
                    "manual_points": manual_points,
                    "cut_points": manual_points,
                    "closed": is_closed,
                    "routed_points": _round_points(points),
                    "route_segments": [single_segment],
                    "branch_choices": [0],
                    "route_ok": True,
                    "source": "geometry_auto_segment_atomic",
                    "preserve_route_segments": True,
                    "span_split_policy": "atomic_junction_or_corner_interval",
                    "confidence": round(float(min(0.96, 0.42 + float(segment.get("length", 0.0) or 0.0) / 900.0)), 3),
                    "reason": (
                        "atomic geometry split: one editable curve per interval between "
                        "neighboring junction/corner boundaries"
                    ),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
            )
            if len(suggestions) >= scan_limit:
                break
        if len(suggestions) >= scan_limit:
            break
    ordered = _order_suggestions_for_review(suggestions, diag)
    return _limit_ordered_suggestions(ordered, max_curves)


def _intersection_cuts_by_edge(
    graph: dict[str, Any],
    edges: dict[str, _GraphEdge],
    radius: float,
) -> dict[str, list[_SpanCut]]:
    cuts: dict[str, list[_SpanCut]] = {edge_id: [] for edge_id in edges}
    for edge in edges.values():
        cuts[edge.id].append(_SpanCut(0.0, edge.points[0].copy(), "endpoint"))
        cuts[edge.id].append(_SpanCut(edge.length, edge.points[-1].copy(), "endpoint"))

    segment_grid, cell_size = _build_segment_grid(edges, radius)
    endpoint_clusters = _endpoint_clusters(edges, radius)
    _add_shared_endpoint_cuts(cuts, endpoint_clusters)
    _add_endpoint_cluster_touch_cuts(cuts, edges, endpoint_clusters, segment_grid, cell_size, radius)
    # Temporarily disabled by request: validate pure junction-based segmentation first.
    # Corner/blend cuts can be re-enabled after the intersection-only behavior is stable.
    # _add_corner_blend_cuts(cuts, edges, radius)

    # Deliberately not using raw pixel degree>=3 nodes or arbitrary segment intersections
    # here. Those create many false junctions on sketch noise, double outlines, and nearby
    # parallel pencil strokes. The production rule is now: split at shared junctions, at
    # endpoint-to-stroke T/Y junctions, and at localized high-turn blend regions.
    return cuts


def _endpoint_clusters(
    edges: dict[str, _GraphEdge],
    radius: float,
) -> list[_EndpointCluster]:
    records: list[_EndpointRecord] = []
    for edge in edges.values():
        records.append(_EndpointRecord(edge.id, edge.points[0].copy(), 0.0, True))
        records.append(_EndpointRecord(edge.id, edge.points[-1].copy(), edge.length, False))
    if not records:
        return []

    cluster_radius = max(radius * 1.45, 4.0)
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    grid: dict[tuple[int, int], list[int]] = {}
    for index, record in enumerate(records):
        cell = _grid_cell(record.point, cluster_radius)
        for gy in range(cell[1] - 1, cell[1] + 2):
            for gx in range(cell[0] - 1, cell[0] + 2):
                for other_index in grid.get((gx, gy), []):
                    if np.linalg.norm(record.point - records[other_index].point) <= cluster_radius:
                        union(index, other_index)
        grid.setdefault(cell, []).append(index)

    grouped: dict[int, list[_EndpointRecord]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(find(index), []).append(record)

    clusters: list[_EndpointCluster] = []
    for cluster_id, items in enumerate(grouped.values()):
        points = np.vstack([item.point for item in items])
        clusters.append(_EndpointCluster(cluster_id, np.mean(points, axis=0), items))
    return clusters


def _add_shared_endpoint_cuts(
    cuts: dict[str, list[_SpanCut]],
    clusters: list[_EndpointCluster],
) -> None:
    for cluster in clusters:
        edge_ids = {record.edge_id for record in cluster.records}
        if len(edge_ids) < 2:
            continue
        kind = "shared_junction_endpoint" if len(edge_ids) >= 3 else "shared_endpoint"
        source = f"endpoint_cluster_{cluster.id}"
        for record in cluster.records:
            cuts[record.edge_id].append(_SpanCut(record.s, cluster.point.copy(), kind, source))


def _add_endpoint_cluster_touch_cuts(
    cuts: dict[str, list[_SpanCut]],
    edges: dict[str, _GraphEdge],
    clusters: list[_EndpointCluster],
    segment_grid: dict[tuple[int, int], list[dict[str, Any]]],
    cell_size: float,
    radius: float,
) -> None:
    """Split through-strokes when an endpoint junction lands on their middle.

    The graph topology is pixel based, so a real T/Y design junction can be missed when
    branch endpoints are one or two pixels away from the through stroke. This pass detects
    that case from endpoint clusters while rejecting near-parallel overlaps.
    """
    touch_radius = max(radius * 1.85, 5.0)
    endpoint_margin = max(radius * 1.25, 4.0)
    min_cross_angle = math.radians(14.0)
    for cluster in clusters:
        cluster_edge_ids = {record.edge_id for record in cluster.records}
        for other_id, projected, other_s, distance in _nearby_polyline_projections(
            cluster.point,
            segment_grid,
            cell_size,
            touch_radius,
        ):
            if other_id in cluster_edge_ids:
                continue
            other_edge = edges.get(other_id)
            if other_edge is None:
                continue
            if distance > touch_radius:
                continue
            if other_s <= endpoint_margin or other_edge.length - other_s <= endpoint_margin:
                continue
            other_tangent = _polyline_tangent_at_s(other_edge.points, other_s)
            angle_ok = False
            for record in cluster.records:
                edge = edges.get(record.edge_id)
                if edge is None:
                    continue
                endpoint_tangent = _edge_tangent_at_endpoint(edge, record.at_start)
                if _undirected_angle_between(endpoint_tangent, other_tangent) >= min_cross_angle:
                    angle_ok = True
                    break
            if not angle_ok:
                continue
            shared_point = projected.copy()
            source = f"endpoint_cluster_{cluster.id}"
            for record in cluster.records:
                cuts[record.edge_id].append(
                    _SpanCut(record.s, shared_point.copy(), "endpoint_touch_junction", other_id)
                )
            cuts[other_id].append(_SpanCut(other_s, shared_point, "endpoint_touch_projection", source))


def _add_corner_blend_cuts(
    cuts: dict[str, list[_SpanCut]],
    edges: dict[str, _GraphEdge],
    radius: float,
) -> None:
    for edge in edges.values():
        if edge.length < max(42.0, radius * 12.0) or len(edge.points) < 8:
            continue
        existing = cuts.get(edge.id, [])
        regions = _detect_corner_blend_regions(edge, radius, existing)
        for region_index, (start_s, end_s) in enumerate(regions):
            start_point = _interpolate_polyline(edge.points, _cumulative_lengths(edge.points), start_s)
            end_point = _interpolate_polyline(edge.points, _cumulative_lengths(edge.points), end_s)
            source = f"corner_blend_{region_index}"
            cuts[edge.id].append(_SpanCut(start_s, start_point, "corner_blend_start", source))
            cuts[edge.id].append(_SpanCut(end_s, end_point, "corner_blend_end", source))


def _detect_corner_blend_regions(
    edge: _GraphEdge,
    radius: float,
    existing_cuts: list[_SpanCut],
) -> list[tuple[float, float]]:
    length = float(edge.length)
    if length <= 1e-6:
        return []
    sample_count = int(max(72, min(260, length / 2.2)))
    cumulative = _cumulative_lengths(edge.points)
    s_values = np.linspace(0.0, length, sample_count)
    sampled = np.vstack([_interpolate_polyline(edge.points, cumulative, s) for s in s_values])
    sampled = _smooth_xy(sampled, window=9 if sample_count >= 80 else 7)

    vec = np.diff(sampled[:, :2], axis=0)
    seg_len = np.linalg.norm(vec, axis=1)
    if np.count_nonzero(seg_len > 1e-6) < 8:
        return []
    theta = np.unwrap(np.arctan2(vec[:, 1], vec[:, 0]))
    dtheta = np.diff(theta)
    local_ds = (s_values[2:] - s_values[:-2]) * 0.5
    if len(local_ds) != len(dtheta):
        local_ds = np.full(len(dtheta), length / max(len(dtheta), 1), dtype=float)
    turn_s = s_values[1:-1]
    signed_density = dtheta / np.maximum(local_ds, 1e-6)
    signed_density = _smooth_1d(signed_density, window=9 if len(signed_density) >= 48 else 7)

    min_region_len = max(12.0, radius * 4.0, length * 0.030)
    max_region_len = max(42.0, min(length * 0.22, max(90.0, radius * 32.0)))
    endpoint_margin = max(10.0, radius * 4.0, length * 0.035)
    existing_s = [float(cut.s) for cut in existing_cuts]

    candidates: list[tuple[float, float, float]] = []
    for sign in (1, -1):
        signal = np.clip(float(sign) * signed_density, 0.0, None)
        opposite = np.clip(-float(sign) * signed_density, 0.0, None)
        finite = signal[np.isfinite(signal)]
        if len(finite) < 8 or float(np.max(finite)) <= 1e-9:
            continue
        positive = finite[finite > float(np.max(finite)) * 0.025]
        if len(positive) < 5:
            continue
        peak_floor = max(float(np.percentile(positive, 78)), float(np.median(positive)) * 1.55)
        peak_floor = max(peak_floor, 1.0 / max(length * 2.7, 1.0))
        peaks = _corner_peak_indices(signal, peak_floor)
        for peak_idx in peaks:
            scored_items: list[tuple[float, float, float]] = []
            designer_scored = _score_designer_corner_peak(
                signal,
                opposite,
                local_ds,
                turn_s,
                peak_idx,
                length=length,
                radius=radius,
                min_region_len=min_region_len,
                max_region_len=max_region_len,
            )
            if designer_scored is not None:
                scored_items.append(designer_scored)
            support_scored = _score_corner_peak(
                signal,
                opposite,
                local_ds,
                turn_s,
                peak_idx,
                length=length,
                radius=radius,
                min_region_len=min_region_len,
                max_region_len=max_region_len,
            )
            if support_scored is not None:
                scored_items.append(support_scored)
            if not scored_items:
                continue
            scored_items.sort(key=lambda item: item[0])
            score, start_s, end_s = scored_items[0]
            if start_s < endpoint_margin and length - end_s < endpoint_margin:
                continue
            if start_s < endpoint_margin:
                start_s = 0.0
            if length - end_s < endpoint_margin:
                end_s = length
            if end_s - start_s < min_region_len:
                continue
            if _near_existing_cut(start_s, existing_s, endpoint_margin * 0.50) and _near_existing_cut(
                end_s,
                existing_s,
                endpoint_margin * 0.50,
            ):
                continue
            candidates.append((score, start_s, end_s))

    candidates.sort(key=lambda item: item[0])
    out: list[tuple[float, float]] = []
    for _score, start_s, end_s in candidates:
        if _overlaps_existing_region(start_s, end_s, out, min_gap=max(8.0, radius * 3.0)):
            continue
        out.append((start_s, end_s))
        if len(out) >= 3:
            break
    return out


def _corner_peak_indices(signal: np.ndarray, peak_floor: float) -> list[int]:
    if len(signal) < 5:
        return []
    peaks: list[int] = []
    for idx in range(1, len(signal) - 1):
        value = float(signal[idx])
        if value < peak_floor:
            continue
        if value >= float(signal[idx - 1]) and value > float(signal[idx + 1]):
            peaks.append(idx)
    if not peaks:
        best = int(np.nanargmax(signal))
        if float(signal[best]) >= peak_floor:
            peaks.append(best)
    peaks.sort(key=lambda item: float(signal[item]), reverse=True)
    return peaks[:8]


def _score_designer_corner_peak(
    signal: np.ndarray,
    opposite: np.ndarray,
    ds: np.ndarray,
    turn_s: np.ndarray,
    peak_idx: int,
    *,
    length: float,
    radius: float,
    min_region_len: float,
    max_region_len: float,
) -> tuple[float, float, float] | None:
    """Find Alias-style support points around one localized curvature transition.

    A designer does not cut exactly at the maximum-curvature sample. They mark the
    beginning and end of the blend where curvature starts rising from the support span and
    where it returns to the next support span. This routine scores those two boundaries.
    """
    peak = float(signal[peak_idx])
    if peak <= 1e-9:
        return None

    total_positive_turn = _turn_integral(signal, ds)
    min_turn = max(math.radians(10.0), min(math.radians(24.0), total_positive_turn * 0.11))
    max_turn = min(math.radians(122.0), max(math.radians(34.0), total_positive_turn * 0.72))
    best: tuple[float, float, float] | None = None
    for drop_ratio in (0.18, 0.22, 0.26, 0.32, 0.40, 0.48):
        threshold = peak * drop_ratio
        coarse_left, coarse_right = _corner_boundaries_for_peak(signal, peak_idx, threshold)
        left_candidates = _designer_boundary_candidates(signal, coarse_left, peak_idx, side=-1)
        right_candidates = _designer_boundary_candidates(signal, coarse_right, peak_idx, side=1)
        for left in left_candidates:
            for right in right_candidates:
                if right <= left + 2:
                    continue
                start_s = float(turn_s[max(0, left)])
                end_s = float(turn_s[min(len(turn_s) - 1, right)])
                region_len = end_s - start_s
                if region_len < min_region_len or region_len > max_region_len:
                    continue
                if left == 0 and right >= len(signal) - 1:
                    continue

                region_signal = signal[left : right + 1]
                region_opposite = opposite[left : right + 1]
                region_ds = ds[left : right + 1]
                region_turn = _turn_integral(region_signal, region_ds)
                if region_turn < min_turn or region_turn > max_turn:
                    continue
                opposite_turn = _turn_integral(region_opposite, region_ds)
                if opposite_turn > region_turn * 0.13:
                    continue

                peak_shape = _single_peak_violation(region_signal)
                if peak_shape > 0.36:
                    continue
                support_score = _designer_support_score(
                    signal,
                    opposite,
                    ds,
                    turn_s,
                    left,
                    right,
                    peak,
                    radius=radius,
                    length=length,
                )
                if support_score is None:
                    continue

                symmetry = _corner_symmetry_score(signal, ds, turn_s, left, right, peak_idx)
                boundary_drop = (float(signal[left]) + float(signal[right])) / max(2.0 * peak, 1e-9)
                region_ratio = region_len / max(length, 1e-9)
                length_penalty = max(0.0, region_ratio - 0.24) * 115.0
                length_penalty += max(0.0, 0.045 - region_ratio) * 80.0
                edge_bias = 0.0
                if start_s <= max(6.0, radius * 2.0):
                    edge_bias += 10.0
                if length - end_s <= max(6.0, radius * 2.0):
                    edge_bias += 10.0
                score = (
                    support_score
                    + symmetry
                    + peak_shape * 70.0
                    + boundary_drop * 26.0
                    + length_penalty
                    + edge_bias
                    + abs(region_turn - math.radians(48.0)) * 2.5
                )
                if best is None or score < best[0]:
                    best = (score, start_s, end_s)
    return best


def _designer_boundary_candidates(
    signal: np.ndarray,
    coarse: int,
    peak_idx: int,
    *,
    side: int,
) -> list[int]:
    if len(signal) == 0:
        return []
    peak_idx = max(0, min(int(peak_idx), len(signal) - 1))
    coarse = max(0, min(int(coarse), len(signal) - 1))
    radius = max(2, min(12, int(round(len(signal) * 0.045))))
    candidates = {coarse}
    if side < 0:
        lo = max(0, coarse - radius)
        hi = min(peak_idx - 1, coarse + radius)
    else:
        lo = max(peak_idx + 1, coarse - radius)
        hi = min(len(signal) - 1, coarse + radius)
    if hi >= lo:
        window = signal[lo : hi + 1]
        local_min = lo + int(np.nanargmin(window))
        candidates.add(local_min)
        for idx in range(lo + 1, hi):
            if float(signal[idx]) <= float(signal[idx - 1]) and float(signal[idx]) <= float(signal[idx + 1]):
                candidates.add(idx)
    return sorted(candidates)


def _designer_support_score(
    signal: np.ndarray,
    opposite: np.ndarray,
    ds: np.ndarray,
    turn_s: np.ndarray,
    left: int,
    right: int,
    peak: float,
    *,
    radius: float,
    length: float,
) -> float | None:
    start_s = float(turn_s[left])
    end_s = float(turn_s[right])
    support_len = max(10.0, radius * 3.5, min(length * 0.075, max((end_s - start_s) * 0.62, 14.0)))
    guard = max(2.0, radius * 0.9)
    left_mask = (turn_s >= start_s - support_len) & (turn_s <= start_s - guard)
    right_mask = (turn_s >= end_s + guard) & (turn_s <= end_s + support_len)
    left_score = _designer_one_side_support_score(signal, opposite, ds, left_mask, peak)
    right_score = _designer_one_side_support_score(signal, opposite, ds, right_mask, peak)

    missing = 0
    if left_score is None:
        missing += 1
        left_score = 8.0
    if right_score is None:
        missing += 1
        right_score = 8.0
    if missing >= 2:
        return None

    boundary_level = max(float(signal[left]), float(signal[right])) / max(peak, 1e-9)
    if boundary_level > 0.72:
        return None
    return float(left_score + right_score + max(0.0, boundary_level - 0.36) * 22.0)


def _designer_one_side_support_score(
    signal: np.ndarray,
    opposite: np.ndarray,
    ds: np.ndarray,
    mask: np.ndarray,
    peak: float,
) -> float | None:
    count = min(len(mask), len(signal), len(ds))
    if count <= 1:
        return None
    mask = mask[:count]
    if not np.any(mask):
        return None
    values = signal[:count][mask]
    opposite_values = opposite[:count][mask]
    local_ds = ds[:count][mask]
    if len(values) < 2 or float(np.sum(local_ds)) <= 1e-6:
        return None
    mean_density = _turn_integral(values, local_ds) / max(float(np.sum(local_ds)), 1e-6)
    p85_density = float(np.percentile(values, 85)) if len(values) else 0.0
    opposite_density = _turn_integral(opposite_values, local_ds) / max(float(np.sum(local_ds)), 1e-6)
    if mean_density > peak * 0.58 or p85_density > peak * 0.78:
        return None
    if opposite_density > peak * 0.13:
        return None
    return float((mean_density / max(peak, 1e-9)) * 16.0 + (p85_density / max(peak, 1e-9)) * 10.0)


def _score_corner_peak(
    signal: np.ndarray,
    opposite: np.ndarray,
    ds: np.ndarray,
    turn_s: np.ndarray,
    peak_idx: int,
    *,
    length: float,
    radius: float,
    min_region_len: float,
    max_region_len: float,
) -> tuple[float, float, float] | None:
    peak = float(signal[peak_idx])
    if peak <= 1e-9:
        return None
    best: tuple[float, float, float] | None = None
    for drop_ratio in (0.16, 0.20, 0.24, 0.28):
        left, right = _corner_boundaries_for_peak(signal, peak_idx, peak * drop_ratio)
        start_s = float(turn_s[max(0, left)])
        end_s = float(turn_s[min(len(turn_s) - 1, right)])
        region_len = end_s - start_s
        if region_len < min_region_len or region_len > max_region_len:
            continue

        region_signal = signal[left : right + 1]
        region_ds = ds[left : right + 1]
        region_opposite = opposite[left : right + 1]
        region_turn = _turn_integral(region_signal, region_ds)
        if region_turn < math.radians(15.0):
            continue
        if _single_peak_violation(region_signal) > 0.42:
            continue
        if _turn_integral(region_opposite, region_ds) > region_turn * 0.16:
            continue

        support = _corner_support_score(
            signal,
            opposite,
            ds,
            turn_s,
            start_s,
            end_s,
            peak,
            radius=radius,
            length=length,
        )
        if support is None:
            continue
        peak_s = float(turn_s[peak_idx])
        symmetry = _corner_symmetry_score(signal, ds, turn_s, left, right, peak_idx)
        score = support + symmetry + _single_peak_violation(region_signal) * 55.0
        if best is None or score < best[0]:
            best = (score, start_s, end_s)
    return best


def _corner_boundaries_for_peak(signal: np.ndarray, peak_idx: int, threshold: float) -> tuple[int, int]:
    left = int(peak_idx)
    right = int(peak_idx)
    while left > 0 and float(signal[left - 1]) >= threshold:
        left -= 1
    while right < len(signal) - 1 and float(signal[right + 1]) >= threshold:
        right += 1
    return left, right


def _corner_support_score(
    signal: np.ndarray,
    opposite: np.ndarray,
    ds: np.ndarray,
    turn_s: np.ndarray,
    start_s: float,
    end_s: float,
    peak: float,
    *,
    radius: float,
    length: float,
) -> float | None:
    guard = max(radius * 1.4, 3.0)
    support_len = max(14.0, radius * 5.0, min(length * 0.10, max((end_s - start_s) * 0.85, 18.0)))
    left_mask = (turn_s >= start_s - support_len) & (turn_s <= start_s - guard)
    right_mask = (turn_s >= end_s + guard) & (turn_s <= end_s + support_len)
    if _mask_length(ds, left_mask) < support_len * 0.34 or _mask_length(ds, right_mask) < support_len * 0.34:
        return None

    left_score = _low_curvature_support_score(signal, opposite, ds, left_mask, peak)
    right_score = _low_curvature_support_score(signal, opposite, ds, right_mask, peak)
    if left_score is None or right_score is None:
        return None
    return float(left_score + right_score)


def _low_curvature_support_score(
    signal: np.ndarray,
    opposite: np.ndarray,
    ds: np.ndarray,
    mask: np.ndarray,
    peak: float,
) -> float | None:
    support_signal = signal[mask]
    support_opposite = opposite[mask]
    support_ds = ds[mask]
    if len(support_signal) < 2:
        return None
    support_len = max(float(np.sum(support_ds)), 1e-6)
    mean_density = _turn_integral(support_signal, support_ds) / support_len
    opposite_density = _turn_integral(support_opposite, support_ds) / support_len
    max_density = float(np.max(support_signal)) if len(support_signal) else 0.0
    if mean_density > peak * 0.34 or max_density > peak * 0.68:
        return None
    if opposite_density > peak * 0.18:
        return None
    smoothness = float(np.std(support_signal) / max(peak, 1e-9))
    return float((mean_density / max(peak, 1e-9)) * 18.0 + (max_density / max(peak, 1e-9)) * 10.0 + smoothness * 8.0)


def _corner_symmetry_score(
    signal: np.ndarray,
    ds: np.ndarray,
    turn_s: np.ndarray,
    left: int,
    right: int,
    peak_idx: int,
) -> float:
    peak_s = float(turn_s[peak_idx])
    start_s = float(turn_s[left])
    end_s = float(turn_s[right])
    left_len = max(peak_s - start_s, 1e-6)
    right_len = max(end_s - peak_s, 1e-6)
    len_ratio = max(left_len, right_len) / max(min(left_len, right_len), 1e-6)
    left_turn = _turn_integral(signal[left : peak_idx + 1], ds[left : peak_idx + 1])
    right_turn = _turn_integral(signal[peak_idx : right + 1], ds[peak_idx : right + 1])
    turn_ratio = max(left_turn, right_turn) / max(min(left_turn, right_turn), math.radians(0.6))
    return float(max(0.0, len_ratio - 1.65) * 28.0 + max(0.0, turn_ratio - 1.85) * 24.0)


def _single_peak_violation(values: np.ndarray) -> float:
    if len(values) < 5:
        return 0.0
    smooth = _smooth_1d(values, window=5)
    peak_idx = int(np.nanargmax(smooth))
    left = np.diff(smooth[: peak_idx + 1])
    right = np.diff(smooth[peak_idx:])
    scale = max(float(np.nanmax(smooth)), 1e-9)
    wrong_left = np.clip(-left, 0.0, None)
    wrong_right = np.clip(right, 0.0, None)
    penalty = 0.0
    if len(wrong_left):
        penalty += float(np.mean((wrong_left / scale) ** 2))
    if len(wrong_right):
        penalty += float(np.mean((wrong_right / scale) ** 2))
    return penalty


def _turn_integral(values: np.ndarray, ds: np.ndarray) -> float:
    if len(values) == 0 or len(ds) == 0:
        return 0.0
    count = min(len(values), len(ds))
    return float(np.sum(np.asarray(values[:count], dtype=float) * np.asarray(ds[:count], dtype=float)))


def _mask_length(ds: np.ndarray, mask: np.ndarray) -> float:
    if len(ds) == 0 or len(mask) == 0:
        return 0.0
    count = min(len(ds), len(mask))
    return float(np.sum(ds[:count][mask[:count]]))


def _smooth_xy(points: np.ndarray, window: int) -> np.ndarray:
    if window < 3 or len(points) < window:
        return np.asarray(points, dtype=float)
    if window % 2 == 0:
        window += 1
    pts = np.asarray(points, dtype=float)
    pad = window // 2
    padded = np.pad(pts, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    out = np.vstack([np.convolve(padded[:, idx], kernel, mode="valid") for idx in range(pts.shape[1])]).T
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window < 3 or len(values) < window:
        return np.asarray(values, dtype=float)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(np.asarray(values, dtype=float), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _close_boolean_gaps(values: np.ndarray, *, max_gap: int) -> np.ndarray:
    out = np.asarray(values, dtype=bool).copy()
    if len(out) == 0:
        return out
    idx = 0
    while idx < len(out):
        if out[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(out) and not out[idx]:
            idx += 1
        end = idx
        if start > 0 and end < len(out) and end - start <= max_gap:
            out[start:end] = True
    return out


def _boolean_components(values: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    idx = 0
    while idx < len(values):
        if not bool(values[idx]):
            idx += 1
            continue
        start = idx
        while idx + 1 < len(values) and bool(values[idx + 1]):
            idx += 1
        out.append((start, idx))
        idx += 1
    return out


def _near_existing_cut(s: float, existing: list[float], tolerance: float) -> bool:
    return any(abs(float(s) - other) <= tolerance for other in existing)


def _overlaps_existing_region(
    start_s: float,
    end_s: float,
    regions: list[tuple[float, float]],
    *,
    min_gap: float,
) -> bool:
    for a, b in regions:
        if start_s <= b + min_gap and end_s >= a - min_gap:
            return True
    return False


def _read_junction_points(graph: dict[str, Any]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for item in graph.get("nodes", []) or []:
        try:
            degree = int(item.get("degree", 0) or 0)
        except Exception:
            degree = 0
        if degree < 3:
            continue
        try:
            out.append(np.asarray([float(item["x"]), float(item["y"])], dtype=float))
        except Exception:
            continue
    if out:
        return out
    for item in graph.get("junction_points", []) or []:
        try:
            out.append(np.asarray([float(item["x"]), float(item["y"])], dtype=float))
        except Exception:
            continue
    return out


def _project_point_to_polyline(
    points: np.ndarray,
    point: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    if len(points) == 0:
        return np.zeros(2, dtype=float), 0.0, float("inf")
    if len(points) == 1:
        distance = float(np.linalg.norm(points[0, :2] - point))
        return points[0, :2].copy(), 0.0, distance
    cumulative = _cumulative_lengths(points)
    best: tuple[float, np.ndarray, float] | None = None
    for index in range(len(points) - 1):
        projected, t, distance = _project_point_to_segment(point, points[index, :2], points[index + 1, :2])
        s = float(cumulative[index] + t * max(float(cumulative[index + 1] - cumulative[index]), 1e-9))
        if best is None or distance < best[0]:
            best = (distance, projected, s)
    if best is None:
        return points[0, :2].copy(), 0.0, float("inf")
    return best[1], best[2], best[0]


def _nearby_polyline_projections(
    point: np.ndarray,
    segment_grid: dict[tuple[int, int], list[dict[str, Any]]],
    cell_size: float,
    radius: float,
) -> list[tuple[str, np.ndarray, float, float]]:
    center = _grid_cell(point, cell_size)
    search = int(math.ceil(radius / max(cell_size, 1e-6))) + 1
    best_by_edge: dict[str, tuple[float, np.ndarray, float]] = {}
    for gy in range(center[1] - search, center[1] + search + 1):
        for gx in range(center[0] - search, center[0] + search + 1):
            for seg in segment_grid.get((gx, gy), []):
                projected, t, distance = _project_point_to_segment(point, seg["a"], seg["b"])
                if distance > radius:
                    continue
                edge_id = str(seg["edge_id"])
                s = float(seg["s0"] + t * max(float(seg["length"]), 1e-9))
                previous = best_by_edge.get(edge_id)
                if previous is None or distance < previous[0]:
                    best_by_edge[edge_id] = (distance, projected, s)
    return [
        (edge_id, projected, s, distance)
        for edge_id, (distance, projected, s) in best_by_edge.items()
    ]


def _build_segment_grid(
    edges: dict[str, _GraphEdge],
    radius: float,
) -> tuple[dict[tuple[int, int], list[dict[str, Any]]], float]:
    cell_size = max(radius * 2.0, 6.0)
    grid: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for edge in edges.values():
        cumulative = _cumulative_lengths(edge.points)
        for index in range(len(edge.points) - 1):
            a = edge.points[index, :2]
            b = edge.points[index + 1, :2]
            length = float(np.linalg.norm(b - a))
            if length <= 1e-9:
                continue
            seg = {
                "edge_id": edge.id,
                "index": index,
                "a": a,
                "b": b,
                "length": length,
                "s0": float(cumulative[index]),
            }
            x0 = min(float(a[0]), float(b[0])) - radius
            x1 = max(float(a[0]), float(b[0])) + radius
            y0 = min(float(a[1]), float(b[1])) - radius
            y1 = max(float(a[1]), float(b[1])) + radius
            c0 = _grid_cell(np.array([x0, y0]), cell_size)
            c1 = _grid_cell(np.array([x1, y1]), cell_size)
            for gy in range(c0[1], c1[1] + 1):
                for gx in range(c0[0], c1[0] + 1):
                    grid.setdefault((gx, gy), []).append(seg)
    return grid, cell_size


def _segment_pair_key(a: dict[str, Any], b: dict[str, Any]) -> tuple[str, int, str, int]:
    left = (str(a["edge_id"]), int(a["index"]))
    right = (str(b["edge_id"]), int(b["index"]))
    if right < left:
        left, right = right, left
    return (left[0], left[1], right[0], right[1])


def _segment_intersection_point(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray | None:
    r = b - a
    s = d - c
    denom = float(r[0] * s[1] - r[1] * s[0])
    if abs(denom) < 1e-9:
        return None
    q = c - a
    t = float((q[0] * s[1] - q[1] * s[0]) / denom)
    u = float((q[0] * r[1] - q[1] * r[0]) / denom)
    if -1e-6 <= t <= 1.0 + 1e-6 and -1e-6 <= u <= 1.0 + 1e-6:
        return a + np.clip(t, 0.0, 1.0) * r
    return None


def _project_point_to_segment(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-9:
        projected = a
        return projected, 0.0, float(np.linalg.norm(point - projected))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    projected = a + t * ab
    return projected, t, float(np.linalg.norm(point - projected))


def _sorted_edge_cuts(edge: _GraphEdge, cuts: list[_SpanCut]) -> list[_SpanCut]:
    tolerance = 2.0
    prepared: list[_SpanCut] = []
    for cut in cuts:
        s = max(0.0, min(float(cut.s), edge.length))
        point = cut.point.astype(float, copy=False)
        prepared.append(_SpanCut(s, point.copy(), cut.kind, cut.source_edge))
    hard_s = [cut.s for cut in prepared if _is_hard_span_cut(cut.kind)]
    corner_guard = max(8.0, min(42.0, edge.length * 0.035))
    filtered = [
        cut
        for cut in prepared
        if not (_is_corner_span_cut(cut.kind) and any(abs(cut.s - hard) <= corner_guard for hard in hard_s))
    ]
    out: list[_SpanCut] = []
    for cut in sorted(filtered, key=lambda item: item.s):
        s = cut.s
        point = cut.point.astype(float, copy=False)
        if out and abs(s - out[-1].s) <= tolerance:
            if cut.kind != "endpoint":
                merged_kind = out[-1].kind
                if cut.kind not in merged_kind.split("+"):
                    merged_kind = f"{merged_kind}+{cut.kind}"
                out[-1] = _SpanCut(out[-1].s, point.copy(), merged_kind, cut.source_edge or out[-1].source_edge)
            continue
        out.append(_SpanCut(s, point, cut.kind, cut.source_edge))
    if not out or out[0].s > tolerance:
        out.insert(0, _SpanCut(0.0, edge.points[0].copy(), "endpoint"))
    if edge.length - out[-1].s > tolerance:
        out.append(_SpanCut(edge.length, edge.points[-1].copy(), "endpoint"))
    return out


def _is_hard_span_cut(kind: str) -> bool:
    parts = set(str(kind or "").split("+"))
    return bool(
        parts
        & {
            "endpoint",
            "shared_endpoint",
            "shared_junction_endpoint",
            "endpoint_touch_junction",
            "endpoint_touch_projection",
        }
    )


def _is_corner_span_cut(kind: str) -> bool:
    parts = set(str(kind or "").split("+"))
    return bool(parts & {"corner_blend_start", "corner_blend_end"})


def _route_segments_from_cuts(
    edge: _GraphEdge,
    cuts: list[_SpanCut],
    *,
    min_span_length: float,
    min_span_chord: float,
) -> list[dict[str, Any]]:
    cumulative = _cumulative_lengths(edge.points)
    segments: list[dict[str, Any]] = []
    kept_boundaries = _compact_span_cuts(
        edge,
        cuts,
        cumulative,
        min_span_length=min_span_length,
        min_span_chord=min_span_chord,
    )
    if len(kept_boundaries) < 2:
        return []
    segment_count = len(kept_boundaries) - 1
    for index, (start, end) in enumerate(zip(kept_boundaries, kept_boundaries[1:])):
        points = _slice_polyline(edge.points, cumulative, start.s, end.s)
        if len(points) < 2:
            continue
        points = points.copy()
        points[0, :2] = start.point[:2]
        points[-1, :2] = end.point[:2]
        length = _curve_length(points)
        chord = float(np.linalg.norm(points[-1, :2] - points[0, :2]))
        closed_span = chord <= max(2.0, min_span_chord * 0.75) and length >= max(12.0, min_span_length * 3.0)
        if length < max(1.0, min_span_length * 0.5):
            continue
        if chord < min_span_chord and not closed_span:
            continue
        segments.append(
            {
                "ok": True,
                "points": _round_points(points),
                "start_point": _round_point(start.point),
                "end_point": _round_point(end.point),
                "segment_index": index,
                "selected_candidate": 0,
                "length": round(float(length), 3),
                "source": "geometry_auto_segment_junction_corner",
                "edge_id": edge.id,
                "start_kind": start.kind,
                "end_kind": end.kind,
                "start_s": round(float(start.s), 3),
                "end_s": round(float(end.s), 3),
                "segment_count": segment_count,
                "closed": bool(closed_span),
            }
        )
    return segments


def _compact_span_cuts(
    edge: _GraphEdge,
    cuts: list[_SpanCut],
    cumulative: np.ndarray,
    *,
    min_span_length: float,
    min_span_chord: float,
) -> list[_SpanCut]:
    if len(cuts) <= 2:
        return cuts
    sorted_cuts = sorted(cuts, key=lambda item: item.s)
    compacted: list[_SpanCut] = [sorted_cuts[0]]
    for index, cut in enumerate(sorted_cuts[1:], start=1):
        is_last = index == len(sorted_cuts) - 1
        previous = compacted[-1]
        if not is_last and _span_between_cuts_is_too_short(
            edge,
            cumulative,
            previous,
            cut,
            min_span_length=min_span_length,
            min_span_chord=min_span_chord,
        ):
            continue
        compacted.append(cut)

    changed = True
    while changed and len(compacted) > 2:
        changed = False
        for index in range(1, len(compacted) - 1):
            left_short = _span_between_cuts_is_too_short(
                edge,
                cumulative,
                compacted[index - 1],
                compacted[index],
                min_span_length=min_span_length,
                min_span_chord=min_span_chord,
            )
            right_short = _span_between_cuts_is_too_short(
                edge,
                cumulative,
                compacted[index],
                compacted[index + 1],
                min_span_length=min_span_length,
                min_span_chord=min_span_chord,
            )
            if left_short or right_short:
                compacted.pop(index)
                changed = True
                break
    return compacted


def _span_between_cuts_is_too_short(
    edge: _GraphEdge,
    cumulative: np.ndarray,
    start: _SpanCut,
    end: _SpanCut,
    *,
    min_span_length: float,
    min_span_chord: float,
) -> bool:
    if end.s - start.s < min_span_length:
        return True
    a = _interpolate_polyline(edge.points, cumulative, start.s)
    b = _interpolate_polyline(edge.points, cumulative, end.s)
    return float(np.linalg.norm(b - a)) < min_span_chord


def _manual_points_from_cuts(
    edge: _GraphEdge,
    route_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not route_segments:
        return []
    boundaries: list[tuple[float, str, np.ndarray | None]] = []
    first = route_segments[0]
    boundaries.append(
        (
            float(first.get("start_s", 0.0)),
            str(first.get("start_kind", "endpoint")),
            _optional_point(first.get("start_point")),
        )
    )
    for segment in route_segments:
        boundaries.append(
            (
                float(segment.get("end_s", edge.length)),
                str(segment.get("end_kind", "endpoint")),
                _optional_point(segment.get("end_point")),
            )
        )
    cumulative = _cumulative_lengths(edge.points)
    out: list[dict[str, Any]] = []
    for order, (s, kind, boundary_point) in enumerate(boundaries):
        point = boundary_point if boundary_point is not None else _interpolate_polyline(edge.points, cumulative, s)
        out.append(
            {
                "x": round(float(point[0]), 3),
                "y": round(float(point[1]), 3),
                "order": order,
                "snap_source": "geometry_auto_segment_junction_corner",
                "auto_boundary": True,
                "intersection_boundary": kind != "endpoint",
                "boundary_kind": kind,
                "edge_id": edge.id,
            }
        )
    return out


def _manual_points_from_segment(edge: _GraphEdge, segment: dict[str, Any]) -> list[dict[str, Any]]:
    start_s = float(segment.get("start_s", 0.0) or 0.0)
    end_s = float(segment.get("end_s", edge.length) or edge.length)
    start_kind = str(segment.get("start_kind", "endpoint") or "endpoint")
    end_kind = str(segment.get("end_kind", "endpoint") or "endpoint")
    start_point = _optional_point(segment.get("start_point"))
    end_point = _optional_point(segment.get("end_point"))
    cumulative = _cumulative_lengths(edge.points)
    if start_point is None:
        start_point = _interpolate_polyline(edge.points, cumulative, start_s)
    if end_point is None:
        end_point = _interpolate_polyline(edge.points, cumulative, end_s)
    out: list[dict[str, Any]] = []
    for order, (point, kind) in enumerate(((start_point, start_kind), (end_point, end_kind))):
        out.append(
            {
                "x": round(float(point[0]), 3),
                "y": round(float(point[1]), 3),
                "order": order,
                "snap_source": "geometry_auto_segment_atomic",
                "auto_boundary": True,
                "intersection_boundary": kind != "endpoint",
                "boundary_kind": kind,
                "edge_id": edge.id,
            }
        )
    return out


def _order_suggestions_for_review(suggestions: list[dict[str, Any]], diag: float) -> list[dict[str, Any]]:
    """Make auto-generated curve lists feel spatially readable.

    The segmentation core emits candidates by source edge length, which is good for
    filtering but confusing in the editor list. This pass keeps every curve intact and
    only reorders the list: connected pieces stay near each other, and disconnected
    groups are arranged roughly top-to-bottom, left-to-right.
    """
    if len(suggestions) <= 1:
        return suggestions

    infos: list[dict[str, Any]] = []
    for index, curve in enumerate(suggestions):
        points = _curve_points_for_order(curve)
        if len(points) < 2:
            continue
        bbox_min = np.min(points[:, :2], axis=0)
        bbox_max = np.max(points[:, :2], axis=0)
        infos.append(
            {
                "index": index,
                "curve": curve,
                "points": points,
                "start": points[0, :2].copy(),
                "end": points[-1, :2].copy(),
                "center": np.mean(points[:, :2], axis=0),
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "length": _curve_length(points),
            }
        )

    if not infos:
        return suggestions

    endpoint_radius = max(6.0, float(diag) * 0.006)
    adjacency = _curve_adjacency_for_order(infos, endpoint_radius)
    components = _curve_components_for_order(adjacency)

    def component_key(component: list[int]) -> tuple[float, float, float, float]:
        mins = np.vstack([infos[i]["bbox_min"] for i in component])
        maxs = np.vstack([infos[i]["bbox_max"] for i in component])
        centers = np.vstack([infos[i]["center"] for i in component])
        bbox_min = np.min(mins, axis=0)
        bbox_max = np.max(maxs, axis=0)
        center = np.mean(centers, axis=0)
        # Primary reading order is top-to-bottom. The center terms keep very tall
        # components, such as a door loop, from being sorted only by one stray endpoint.
        return (
            round(float(bbox_min[1]) / max(endpoint_radius, 1.0)),
            round(float(bbox_min[0]) / max(endpoint_radius, 1.0)),
            float(center[1]),
            float(center[0]),
        )

    ordered: list[dict[str, Any]] = []
    component_order = 0
    for component in sorted(components, key=component_key):
        local_sequence = _order_component_curves(infos, adjacency, component, endpoint_radius)
        for local_order, info_index in enumerate(local_sequence):
            curve = infos[info_index]["curve"]
            curve["auto_order"] = len(ordered) + 1
            curve["auto_component"] = component_order + 1
            curve["auto_component_order"] = local_order + 1
            ordered.append(curve)
        component_order += 1

    # Preserve any malformed items at the end instead of dropping them.
    ordered_ids = {id(curve) for curve in ordered}
    for curve in suggestions:
        if id(curve) not in ordered_ids:
            ordered.append(curve)
    return ordered


def _limit_ordered_suggestions(suggestions: list[dict[str, Any]], max_curves: int) -> list[dict[str, Any]]:
    """Trim after spatial ordering so short but important strokes are not starved.

    The old flow stopped while scanning curves from longest to shortest. That made the
    saved blue curves over-represent outer silhouettes, wheels, and long rocker lines,
    while dropping shorter window/frame/detail strokes that were already present in the
    design-stroke layer. We now trim only after the review order has been built.
    """
    limit = max(1, min(int(max_curves), len(suggestions)))
    kept = suggestions[:limit]
    for index, curve in enumerate(kept):
        curve["auto_order"] = index + 1
    return kept


def _curve_points_for_order(curve: dict[str, Any]) -> np.ndarray:
    candidates = [curve.get("routed_points")]
    for segment in curve.get("route_segments") or []:
        candidates.append(segment.get("points"))
    for value in candidates:
        try:
            points = np.asarray(value or [], dtype=float)
        except Exception:
            continue
        if points.ndim == 2 and points.shape[0] >= 2 and points.shape[1] >= 2:
            return points[:, :2].astype(float, copy=False)
    manual = curve.get("manual_points") or curve.get("cut_points") or []
    parsed: list[list[float]] = []
    for item in manual:
        point = _optional_point(item)
        if point is not None:
            parsed.append([float(point[0]), float(point[1])])
    if len(parsed) >= 2:
        return np.asarray(parsed, dtype=float)
    return np.zeros((0, 2), dtype=float)


def _curve_adjacency_for_order(
    infos: list[dict[str, Any]],
    endpoint_radius: float,
) -> dict[int, list[int]]:
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(infos))}
    for left in range(len(infos)):
        left_endpoints = (infos[left]["start"], infos[left]["end"])
        for right in range(left + 1, len(infos)):
            right_endpoints = (infos[right]["start"], infos[right]["end"])
            connected = any(
                float(np.linalg.norm(np.asarray(a) - np.asarray(b))) <= endpoint_radius
                for a in left_endpoints
                for b in right_endpoints
            )
            if connected:
                adjacency[left].add(right)
                adjacency[right].add(left)
    return {key: sorted(value) for key, value in adjacency.items()}


def _curve_components_for_order(adjacency: dict[int, list[int]]) -> list[list[int]]:
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency.get(current, []):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    return components


def _order_component_curves(
    infos: list[dict[str, Any]],
    adjacency: dict[int, list[int]],
    component: list[int],
    endpoint_radius: float,
) -> list[int]:
    if len(component) <= 1:
        return component
    remaining = set(component)
    ordered: list[int] = []

    def info_key(index: int) -> tuple[float, float, float]:
        info = infos[index]
        center = info["center"]
        bbox_min = info["bbox_min"]
        return (float(bbox_min[1]), float(bbox_min[0]), -float(info["length"]))

    degree_one = [index for index in component if len([n for n in adjacency.get(index, []) if n in remaining]) <= 1]
    current = min(degree_one or component, key=info_key)

    while remaining:
        if current not in remaining:
            current = min(remaining, key=info_key)
        ordered.append(current)
        remaining.remove(current)
        if not remaining:
            break

        connected_candidates = [n for n in adjacency.get(current, []) if n in remaining]
        if connected_candidates:
            current = min(
                connected_candidates,
                key=lambda idx: (
                    _curve_endpoint_distance(infos[current], infos[idx]),
                    _curve_center_distance(infos[current], infos[idx]),
                    info_key(idx),
                ),
            )
            continue

        current = min(
            remaining,
            key=lambda idx: (
                0 if _curve_endpoint_distance(infos[ordered[-1]], infos[idx]) <= endpoint_radius * 1.65 else 1,
                _curve_center_distance(infos[ordered[-1]], infos[idx]),
                info_key(idx),
            ),
        )
    return ordered


def _curve_endpoint_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_points = (left["start"], left["end"])
    right_points = (right["start"], right["end"])
    return min(float(np.linalg.norm(np.asarray(a) - np.asarray(b))) for a in left_points for b in right_points)


def _curve_center_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return float(np.linalg.norm(np.asarray(left["center"]) - np.asarray(right["center"])))


def _segment_points_array(segment: dict[str, Any]) -> np.ndarray:
    try:
        points = np.asarray(segment.get("points") or [], dtype=float)
    except Exception:
        return np.zeros((0, 2), dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return np.zeros((0, 2), dtype=float)
    return points[:, :2].astype(float, copy=False)


def _route_segment_duplicates_existing(
    points: np.ndarray,
    existing: list[np.ndarray],
    *,
    distance: float,
) -> bool:
    """Reject repeated auto segments without deleting nearby parallel design lines."""
    if len(points) < 2:
        return True
    length = _curve_length(points)
    if length <= 1e-6:
        return True
    sample = _sample_polyline_evenly(points, 32)
    for other in existing:
        if len(other) < 2:
            continue
        other_length = _curve_length(other)
        if other_length <= 1e-6:
            continue
        other_sample = _sample_polyline_evenly(other, 40)
        one_way = _mean_nearest_distance(sample, other_sample)
        if one_way > distance:
            continue
        length_ratio = max(length, other_length) / max(min(length, other_length), 1e-6)
        strict_distance = max(0.8, distance * 0.45)
        subset_distance = max(0.7, distance * 0.35)
        if length_ratio <= 1.18:
            reverse_one_way = _mean_nearest_distance(other_sample, sample)
            if one_way <= strict_distance and reverse_one_way <= strict_distance:
                return True
        elif length <= other_length and one_way <= subset_distance:
            return True
    return False


def _sample_polyline_evenly(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) <= 1:
        return points[:, :2].copy()
    cumulative = _cumulative_lengths(points)
    length = float(cumulative[-1])
    if length <= 1e-6:
        return points[:1, :2].copy()
    sample_count = max(2, min(int(count), max(2, int(length / 2.0))))
    return np.vstack([_interpolate_polyline(points, cumulative, s) for s in np.linspace(0.0, length, sample_count)])


def _mean_nearest_distance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    diff = a[:, None, :2] - b[None, :, :2]
    dist = np.linalg.norm(diff, axis=2)
    return float(np.mean(np.min(dist, axis=1)))


def _optional_point(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    if arr.ndim != 1 or arr.shape[0] < 2:
        return None
    return arr[:2].copy()


def _cumulative_lengths(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=float)
    if len(points) == 1:
        return np.zeros(1, dtype=float)
    distances = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(distances)])


def _interpolate_polyline(points: np.ndarray, cumulative: np.ndarray, s: float) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(2, dtype=float)
    if len(points) == 1 or s <= 0:
        return points[0, :2].copy()
    if s >= cumulative[-1]:
        return points[-1, :2].copy()
    index = int(np.searchsorted(cumulative, s, side="right") - 1)
    index = max(0, min(index, len(points) - 2))
    denom = max(float(cumulative[index + 1] - cumulative[index]), 1e-9)
    t = float((s - cumulative[index]) / denom)
    return points[index, :2] * (1.0 - t) + points[index + 1, :2] * t


def _slice_polyline(
    points: np.ndarray,
    cumulative: np.ndarray,
    s0: float,
    s1: float,
) -> np.ndarray:
    if s1 < s0:
        s0, s1 = s1, s0
    start = _interpolate_polyline(points, cumulative, s0)
    end = _interpolate_polyline(points, cumulative, s1)
    internal = points[(cumulative > s0 + 1e-6) & (cumulative < s1 - 1e-6), :2]
    if len(internal):
        return _remove_near_duplicate_points(np.vstack([start, internal, end]))
    return _remove_near_duplicate_points(np.vstack([start, end]))


def _grid_cell(point: np.ndarray, cell_size: float) -> tuple[int, int]:
    return (int(math.floor(float(point[0]) / cell_size)), int(math.floor(float(point[1]) / cell_size)))


def _read_edges(graph: dict[str, Any]) -> dict[str, _GraphEdge]:
    out: dict[str, _GraphEdge] = {}
    source_items = graph.get("design_strokes") or graph.get("edges", []) or []
    for item in source_items:
        try:
            points = np.asarray(item.get("points") or [], dtype=float)
        except Exception:
            continue
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
            continue
        points = points[:, :2]
        edge_id = str(item.get("id") or f"stroke_{len(out):04d}")
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


def _round_point(point: np.ndarray) -> list[float]:
    arr = np.asarray(point, dtype=float)
    return [round(float(arr[0]), 3), round(float(arr[1]), 3)]


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


def _edge_tangent_at_endpoint(edge: _GraphEdge, at_start: bool) -> np.ndarray:
    return _start_tangent(edge.points) if at_start else _end_tangent(edge.points)


def _polyline_tangent_at_s(points: np.ndarray, s: float) -> np.ndarray:
    if len(points) < 2:
        return np.array([1.0, 0.0])
    cumulative = _cumulative_lengths(points)
    if len(cumulative) < 2 or cumulative[-1] <= 1e-9:
        return _start_tangent(points)
    index = int(np.searchsorted(cumulative, s, side="right") - 1)
    index = max(0, min(index, len(points) - 2))
    left = max(0, index - 3)
    right = min(len(points) - 1, index + 4)
    vec = points[right, :2] - points[left, :2]
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        return _start_tangent(points)
    return vec / norm


def _segment_tangent(segment: dict[str, Any]) -> np.ndarray:
    vec = np.asarray(segment["b"], dtype=float)[:2] - np.asarray(segment["a"], dtype=float)[:2]
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        return np.array([1.0, 0.0])
    return vec / norm


def _segment_crossing_angle(a: dict[str, Any], b: dict[str, Any]) -> float:
    return _undirected_angle_between(_segment_tangent(a), _segment_tangent(b))


def _undirected_angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= 1e-9 or b_norm <= 1e-9:
        return 0.0
    angle = _angle_between(a / a_norm, b / b_norm)
    return min(angle, abs(math.pi - angle))


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(math.acos(dot))


def _curve_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)))
