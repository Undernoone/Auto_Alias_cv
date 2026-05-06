from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autoalias.geometry.bezier import evaluate_bezier
from autoalias.geometry.polyline import resample_polyline, smooth_polyline
from autoalias.models import CurveCandidate, NURBSCurve


@dataclass(slots=True)
class LCornerDecomposition:
    leg_a: np.ndarray
    blend_target: np.ndarray
    leg_b: np.ndarray
    blend_curve: NURBSCurve
    corner_u: float
    group_id: str


def decompose_l_corner_candidate(
    candidate: CurveCandidate,
    min_total_turn_deg: float = 45.0,
    max_total_turn_deg: float = 105.0,
) -> LCornerDecomposition | None:
    """Detect a designer L-corner and split it into leg/blend/leg.

    The blend is a single-span degree-5 Bezier with collinear first/last three CVs, giving a
    G2-like visual entry/exit while keeping the curvature peak near the center of the transition.
    """
    pts = candidate.points
    if len(pts) < 34:
        return None
    pts = resample_polyline(pts, min(180, max(80, len(pts))))
    pts = smooth_polyline(pts, window=5)
    s = _arclength(pts)
    total_len = s[-1]
    if total_len < 45.0:
        return None

    tangents = _window_tangents(pts, window=max(4, len(pts) // 36))
    angles = np.unwrap(np.arctan2(tangents[:, 1], tangents[:, 0]))
    dtheta = np.diff(angles)
    abs_turn = np.abs(dtheta)
    total_signed = float(abs(angles[-1] - angles[0]))
    total_abs = float(np.sum(abs_turn))
    if total_signed < np.deg2rad(min_total_turn_deg) or total_signed > np.deg2rad(max_total_turn_deg):
        return None
    if total_abs > total_signed * 1.7:
        return None  # likely S/wavy, not a clean L corner

    cumulative = np.concatenate([[0.0], np.cumsum(abs_turn)])
    if cumulative[-1] <= 1e-9:
        return None
    i10 = int(np.searchsorted(cumulative, cumulative[-1] * 0.12))
    i50 = int(np.searchsorted(cumulative, cumulative[-1] * 0.50))
    i90 = int(np.searchsorted(cumulative, cumulative[-1] * 0.88))
    turn_span_ratio = float((s[min(i90, len(s) - 1)] - s[max(i10, 0)]) / max(total_len, 1e-9))
    if turn_span_ratio > 0.86:
        return None  # broad arc, not a leg/blend/leg corner

    # Require actual leg length on both sides.
    if s[i50] < total_len * 0.10 or (total_len - s[i50]) < total_len * 0.10:
        return None

    left_width = max(10.0, s[i50] - s[max(0, i10)])
    right_width = max(10.0, s[min(len(s) - 1, i90)] - s[i50])
    half_width = max(left_width, right_width)
    half_width = min(half_width, total_len * 0.22, s[i50] * 0.72, (total_len - s[i50]) * 0.72)
    if half_width < 8.0:
        return None

    s0 = s[i50] - half_width
    s1 = s[i50] + half_width
    i0 = int(np.searchsorted(s, s0))
    i1 = int(np.searchsorted(s, s1))
    if i1 - i0 < 8 or i0 < 4 or i1 > len(pts) - 5:
        return None

    p0 = _interp_by_s(pts, s, s0)
    p5 = _interp_by_s(pts, s, s1)
    leg_a_raw = pts[: i0 + 1]
    leg_b_raw = pts[i1:]
    if not (_is_straight_enough_leg(leg_a_raw) and _is_straight_enough_leg(leg_b_raw)):
        return None
    in_dir = _robust_direction(pts[max(0, i0 - 18) : i0 + 1])
    out_dir = _robust_direction(pts[i1 : min(len(pts), i1 + 19)])
    if np.dot(in_dir, p0[:2] - pts[0, :2]) < 0:
        in_dir = -in_dir
    if np.dot(out_dir, pts[-1, :2] - p5[:2]) < 0:
        out_dir = -out_dir

    group_id = f"lcorner_{abs(hash(candidate.source + candidate.label + str(len(candidate.points)))) % 1000000}"

    leg_a = np.vstack([pts[:i0], p0.reshape(1, 3)])
    blend_target = pts[i0 : i1 + 1]
    if np.linalg.norm(blend_target[0, :2] - p0[:2]) > 1e-6:
        blend_target = np.vstack([p0, blend_target])
    if np.linalg.norm(blend_target[-1, :2] - p5[:2]) > 1e-6:
        blend_target = np.vstack([blend_target, p5])
    leg_b = np.vstack([p5.reshape(1, 3), pts[i1 + 1 :]])
    blend_curve = _make_fair_quintic_blend(candidate.label, p0, p5, in_dir, out_dir, blend_target)
    if blend_curve is None:
        return None
    blend_curve.metadata.update(
        {
            "l_corner_group": group_id,
            "l_corner_role": "blend",
            "preserve_segment": True,
            "corner_u": float(s[i50] / max(total_len, 1e-9)),
            "blend_half_length_px": float(half_width),
        }
    )
    return LCornerDecomposition(
        leg_a=leg_a,
        blend_target=blend_target,
        leg_b=leg_b,
        blend_curve=blend_curve,
        corner_u=float(s[i50] / max(total_len, 1e-9)),
        group_id=group_id,
    )


def _make_fair_quintic_blend(
    label: str,
    p0: np.ndarray,
    p5: np.ndarray,
    in_dir: np.ndarray,
    out_dir: np.ndarray,
    target: np.ndarray | None = None,
) -> NURBSCurve | None:
    chord = float(np.linalg.norm(p5[:2] - p0[:2]))
    if chord < 8.0:
        return None
    chord_axis = (p5[:2] - p0[:2]) / chord
    if float(np.dot(in_dir, chord_axis)) < 0:
        in_dir = -in_dir
    if float(np.dot(out_dir, chord_axis)) < 0:
        out_dir = -out_dir
    cvs, h = _search_blend_handles(p0, p5, in_dir, out_dir, target)
    if cvs is None:
        return None
    curve = NURBSCurve.single_span(label=label, degree=5, cvs=cvs, source="l_corner_fair_blend")
    curve.metadata["blend_handle_length_px"] = float(h)
    return curve


def _search_blend_handles(
    p0: np.ndarray,
    p5: np.ndarray,
    in_dir: np.ndarray,
    out_dir: np.ndarray,
    target: np.ndarray | None,
) -> tuple[np.ndarray | None, float]:
    chord = float(np.linalg.norm(p5[:2] - p0[:2]))
    if chord < 1e-9:
        return None, 0.0
    handle_factors = (0.14, 0.18, 0.22, 0.28, 0.34, 0.42)
    mid_factors = (1.45, 1.65, 1.86, 2.08)
    best: tuple[float, np.ndarray, float] | None = None
    for h0_factor in handle_factors:
        for h1_factor in handle_factors:
            # Class-A CV rhythm: both sides should feel related, not randomly stretched.
            symmetry_penalty = abs(h0_factor - h1_factor) * 0.6
            for mid_factor in mid_factors:
                h0 = chord * h0_factor
                h1 = chord * h1_factor
                cvs = np.zeros((6, 3), dtype=float)
                cvs[0] = p0
                cvs[5] = p5
                cvs[1] = p0
                cvs[2] = p0
                cvs[4] = p5
                cvs[3] = p5
                cvs[1, :2] = p0[:2] + in_dir * h0
                cvs[2, :2] = p0[:2] + in_dir * h0 * mid_factor
                cvs[4, :2] = p5[:2] - out_dir * h1
                cvs[3, :2] = p5[:2] - out_dir * h1 * mid_factor
                if _has_loop_or_bad_projection(cvs):
                    continue
                score = _blend_shape_score(cvs, target) + symmetry_penalty
                if best is None or score < best[0]:
                    best = (score, cvs, 0.5 * (h0 + h1))
    if best is None:
        return None, 0.0
    return best[1], best[2]


def _blend_shape_score(cvs: np.ndarray, target: np.ndarray | None) -> float:
    samples = evaluate_bezier(cvs, np.linspace(0.0, 1.0, 96))
    score = 0.0
    if target is not None and len(target) >= 4:
        probe = target[:: max(1, len(target) // 80), :2]
        diff = probe[:, None, :] - samples[None, :, :2]
        score += float(np.mean(np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1)))) / 12.0
    d2 = np.diff(cvs[:, :2], n=2, axis=0)
    d3 = np.diff(cvs[:, :2], n=3, axis=0)
    chord = max(float(np.linalg.norm(cvs[-1, :2] - cvs[0, :2])), 1e-9)
    score += 0.02 * float(np.mean(np.linalg.norm(d2, axis=1))) / chord
    score += 0.04 * float(np.mean(np.linalg.norm(d3, axis=1))) / chord
    seg = np.linalg.norm(np.diff(cvs[:, :2], axis=0), axis=1)
    positive = seg[seg > 1e-9]
    if len(positive):
        score += 0.04 * float(np.max(positive) / max(np.min(positive), 1e-9))
    return score


def _has_loop_or_bad_projection(cvs: np.ndarray) -> bool:
    pts = evaluate_bezier(cvs, np.linspace(0, 1, 80))
    chord = pts[-1, :2] - pts[0, :2]
    norm = np.linalg.norm(chord)
    if norm <= 1e-9:
        return True
    axis = chord / norm
    proj = pts[:, :2] @ axis
    return bool(np.sum(np.diff(proj) < -norm * 0.04) > 2)


def _arclength(points: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def _interp_by_s(points: np.ndarray, s: np.ndarray, target: float) -> np.ndarray:
    out = np.empty(3, dtype=float)
    for j in range(3):
        out[j] = np.interp(target, s, points[:, j])
    return out


def _window_tangents(points: np.ndarray, window: int) -> np.ndarray:
    tangents = np.empty((len(points), 2), dtype=float)
    for i in range(len(points)):
        a = max(0, i - window)
        b = min(len(points) - 1, i + window)
        v = points[b, :2] - points[a, :2]
        n = np.linalg.norm(v)
        tangents[i] = v / max(n, 1e-9)
    return tangents


def _robust_direction(points: np.ndarray) -> np.ndarray:
    p = points[:, :2]
    if len(p) < 2:
        return np.array([1.0, 0.0])
    centered = p - np.mean(p, axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        d = vh[0]
    except Exception:
        d = p[-1] - p[0]
    if np.dot(d, p[-1] - p[0]) < 0:
        d = -d
    n = np.linalg.norm(d)
    return d / max(n, 1e-9)


def _is_straight_enough_leg(points: np.ndarray) -> bool:
    if len(points) < 8:
        return False
    p = points[:, :2]
    chord = p[-1] - p[0]
    length = np.linalg.norm(chord)
    if length < 18.0:
        return False
    axis = chord / max(length, 1e-9)
    normal = np.array([-axis[1], axis[0]])
    dev = np.abs((p - p[0]) @ normal)
    # Permit hand-drawn line wobble, reject broad arcs masquerading as L legs.
    return float(np.percentile(dev, 90)) <= max(5.0, length * 0.075)
