from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from autoalias.geometry.bezier import evaluate_bezier, signed_curvature_2d


@dataclass(slots=True)
class CurveProfileThresholds:
    straight_sinuosity: float = 1.015
    simple_sinuosity: float = 1.045
    arc_sinuosity: float = 1.16
    s_sign_change_min: int = 1
    corner_turn_deg: float = 34.0


@dataclass(slots=True)
class CurveProfileResult:
    kind: str
    sinuosity: float
    total_turn_deg: float
    max_abs_curvature: float
    curvature_sign_changes: int
    is_straight: bool
    is_s_curve: bool
    is_corner_like: bool


def classify_curve_profile(
    points: np.ndarray,
    thresholds: CurveProfileThresholds | None = None,
) -> CurveProfileResult:
    thresholds = thresholds or CurveProfileThresholds()
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or len(pts) < 2:
        return CurveProfileResult("unknown", 1.0, 0.0, 0.0, 0, False, False, False)
    pts2 = pts[:, :2]
    seg = np.diff(pts2, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    arc_len = float(np.sum(lengths))
    chord = float(np.linalg.norm(pts2[-1] - pts2[0]))
    sinuosity = arc_len / max(chord, 1e-9)
    unit = seg / np.maximum(lengths[:, None], 1e-9)
    if len(unit) >= 2:
        dots = np.sum(unit[:-1] * unit[1:], axis=1)
        turns = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
        total_turn = float(np.sum(turns))
    else:
        total_turn = 0.0
    k = _polyline_signed_curvature(pts2)
    sign_changes = _stable_sign_changes(k)
    max_abs_k = float(np.nanmax(np.abs(k))) if len(k) else 0.0
    is_straight = bool(sinuosity <= thresholds.straight_sinuosity and total_turn < thresholds.corner_turn_deg * 0.35)
    is_s_curve = bool(sign_changes >= thresholds.s_sign_change_min)
    is_corner_like = bool(total_turn >= thresholds.corner_turn_deg and not is_s_curve)
    if is_straight:
        kind = "straight"
    elif is_s_curve:
        kind = "s_curve"
    elif is_corner_like:
        kind = "corner_like"
    elif sinuosity <= thresholds.arc_sinuosity:
        kind = "smooth_arc"
    else:
        kind = "free_curve"
    return CurveProfileResult(kind, sinuosity, total_turn, max_abs_k, sign_changes, is_straight, is_s_curve, is_corner_like)


def classify_nurbs_profile(
    curve: Any,
    thresholds: CurveProfileThresholds | None = None,
    sample_count: int = 160,
) -> CurveProfileResult:
    cvs = np.asarray(getattr(curve, "cvs", curve), dtype=float)
    if cvs.ndim != 2 or len(cvs) < 2:
        return classify_curve_profile(cvs, thresholds)
    u = np.linspace(0.0, 1.0, max(sample_count, 16))
    weights = getattr(curve, "weights", None)
    pts = evaluate_bezier(cvs, u, weights)
    result = classify_curve_profile(pts, thresholds)
    try:
        k = signed_curvature_2d(cvs, np.linspace(0.02, 0.98, max(sample_count, 16)))
        result.max_abs_curvature = float(np.nanmax(np.abs(k))) if len(k) else result.max_abs_curvature
        result.curvature_sign_changes = _stable_sign_changes(k)
        result.is_s_curve = bool(result.curvature_sign_changes >= (thresholds or CurveProfileThresholds()).s_sign_change_min)
    except Exception:
        pass
    return result


def _polyline_signed_curvature(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.zeros(0, dtype=float)
    a = points[:-2]
    b = points[1:-1]
    c = points[2:]
    ab = b - a
    bc = c - b
    ac = c - a
    denom = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ac, axis=1)
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    return np.divide(2.0 * cross, np.maximum(denom, 1e-9))


def _stable_sign_changes(values: np.ndarray) -> int:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 4:
        return 0
    eps = max(float(np.nanmax(np.abs(v))) * 0.06, 1e-9)
    signs = np.sign(np.where(np.abs(v) < eps, 0.0, v))
    nonzero = signs[signs != 0]
    if len(nonzero) < 2:
        return 0
    return int(np.sum(nonzero[1:] * nonzero[:-1] < 0.0))
