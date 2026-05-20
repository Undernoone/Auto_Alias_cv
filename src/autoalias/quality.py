from __future__ import annotations

import numpy as np

from autoalias.geometry.bezier import evaluate_bezier, signed_curvature_2d
from autoalias.geometry.polyline import point_to_point_distances
from autoalias.models import NURBSCurve, QualityReport


class ClassAValidator:
    def __init__(
        self,
        samples: int = 240,
        max_chamfer_px: float = 6.0,
        max_cv_spacing_ratio: float = 6.0,
        max_curvature_peaks: int = 4,
        max_inflections: int = 1,
        max_cv_spacing_rhythm: float = 12.0,
        max_cv_curve_distance_rhythm: float = 14.0,
    ):
        self.samples = samples
        self.max_chamfer_px = max_chamfer_px
        self.max_cv_spacing_ratio = max_cv_spacing_ratio
        self.max_curvature_peaks = max_curvature_peaks
        self.max_inflections = max_inflections
        self.max_cv_spacing_rhythm = max_cv_spacing_rhythm
        self.max_cv_curve_distance_rhythm = max_cv_curve_distance_rhythm

    def validate(self, curve: NURBSCurve, target_points: np.ndarray | None = None) -> QualityReport:
        warnings: list[str] = []
        metrics: dict[str, float | int | bool | list[float] | str] = {
            "degree": curve.degree,
            "span": curve.span_count,
            "single_span": curve.is_single_span,
            "cv_count": len(curve.cvs),
            "knot_count": len(curve.knots),
        }

        if curve.degree not in (3, 4, 5, 6, 7):
            warnings.append("degree outside Alias Class-A target range")
        if not curve.is_single_span:
            warnings.append("curve is not a single-span Bezier/NURBS")

        cv_metrics = self._cv_metrics(curve)
        metrics.update(cv_metrics)
        if cv_metrics["cv_spacing_ratio"] > self.max_cv_spacing_ratio:
            warnings.append("CV spacing ratio is too high")
        if cv_metrics["cv_spacing_rhythm_penalty"] > self.max_cv_spacing_rhythm:
            warnings.append("CV spacing rhythm is not monotone/constant")
        if cv_metrics["cv_curve_distance_rhythm_penalty"] > self.max_cv_curve_distance_rhythm:
            warnings.append("CV-to-curve distance rhythm is not gradual")
        if cv_metrics["control_polygon_turnback"]:
            warnings.append("control polygon has turnback/self-crossing risk")

        curv_metrics = self._curvature_metrics(curve)
        metrics.update(curv_metrics)
        if curv_metrics["curvature_peak_count"] > self.max_curvature_peaks:
            warnings.append("curvature comb has too many peaks")
        if curv_metrics["curvature_oscillation"] > 0.65:
            warnings.append("curvature oscillation is high")
        if curv_metrics["inflection_count"] > self.max_inflections:
            warnings.append("curve has more than one inflection")

        if target_points is not None and len(target_points) >= 2:
            err = self._geometric_error(curve, target_points)
            metrics.update(err)
            if err["chamfer_mean"] > self.max_chamfer_px:
                warnings.append("mean Chamfer error is above target")

        passed = len(warnings) == 0
        return QualityReport(label=curve.label, passed=passed, metrics=metrics, warnings=warnings)

    def _geometric_error(self, curve: NURBSCurve, target_points: np.ndarray) -> dict[str, float]:
        u = np.linspace(0.0, 1.0, self.samples)
        samples = evaluate_bezier(curve.cvs, u, curve.weights)
        dist = point_to_point_distances(samples, target_points)
        forward = np.min(dist, axis=1)
        backward = np.min(dist, axis=0)
        endpoint_error = 0.5 * (
            np.linalg.norm(samples[0, :2] - target_points[0, :2])
            + np.linalg.norm(samples[-1, :2] - target_points[-1, :2])
        )
        return {
            "chamfer_mean": float(np.mean(forward) + np.mean(backward)) / 2.0,
            "hausdorff": float(max(np.max(forward), np.max(backward))),
            "rms_projection_error": float(np.sqrt(np.mean(forward**2))),
            "endpoint_error": float(endpoint_error),
        }

    def _cv_metrics(self, curve: NURBSCurve) -> dict[str, float | bool]:
        seg = np.linalg.norm(np.diff(curve.cvs[:, :2], axis=0), axis=1)
        positive = seg[seg > 1e-9]
        if len(positive) == 0:
            ratio = float("inf")
        else:
            ratio = float(np.max(positive) / max(np.min(positive), 1e-9))
        spacing_rhythm = _sequence_rhythm_metrics(positive, allow_unimodal=False)
        cv_distance_rhythm = _cv_curve_distance_rhythm(curve)
        return {
            "cv_spacing_ratio": ratio,
            "cv_min_spacing": float(np.min(positive)) if len(positive) else 0.0,
            "cv_max_spacing": float(np.max(positive)) if len(positive) else 0.0,
            "cv_spacing_rhythm_penalty": float(spacing_rhythm["penalty"]),
            "cv_spacing_rhythm_shape": str(spacing_rhythm["shape"]),
            "cv_curve_distance_rhythm_penalty": float(cv_distance_rhythm["penalty"]),
            "cv_curve_distance_rhythm_shape": str(cv_distance_rhythm["shape"]),
            "cv_curve_distance_min": float(cv_distance_rhythm["min"]),
            "cv_curve_distance_max": float(cv_distance_rhythm["max"]),
            "control_polygon_turnback": bool(_has_turnback(curve.cvs[:, :2])),
        }

    def _curvature_metrics(self, curve: NURBSCurve) -> dict[str, float | int | list[float]]:
        u = np.linspace(0.01, 0.99, self.samples)
        k = signed_curvature_2d(curve.cvs, u)
        dk = np.gradient(k)
        d2k = np.gradient(dk)
        peaks = _peak_count(np.abs(k))
        inflections = _inflection_locations(u, k)
        denom = np.mean(np.abs(k)) + 1e-9
        return {
            "max_abs_curvature": float(np.max(np.abs(k))),
            "curvature_smoothness": float(np.mean(dk**2)),
            "curvature_jerk": float(np.mean(d2k**2)),
            "curvature_peak_count": int(peaks),
            "curvature_oscillation": float(np.std(dk) / denom),
            "inflection_count": len(inflections),
            "inflection_u": [float(x) for x in inflections],
        }


def _peak_count(values: np.ndarray) -> int:
    v = np.asarray(values, dtype=float)
    if len(v) < 5:
        return 0
    threshold = np.max(v) * 0.08
    count = 0
    for i in range(1, len(v) - 1):
        if v[i] > threshold and v[i] > v[i - 1] and v[i] >= v[i + 1]:
            count += 1
    return count


def _inflection_locations(u: np.ndarray, k: np.ndarray) -> list[float]:
    eps = max(np.max(np.abs(k)) * 0.04, 1e-10)
    signs = np.sign(np.where(np.abs(k) < eps, 0.0, k))
    out: list[float] = []
    last_idx = None
    last_sign = 0.0
    for i, sign in enumerate(signs):
        if sign == 0:
            continue
        if last_sign != 0 and sign != last_sign and last_idx is not None:
            out.append(float(0.5 * (u[last_idx] + u[i])))
        last_sign = sign
        last_idx = i
    return out


def _cv_curve_distance_rhythm(curve: NURBSCurve) -> dict[str, float | str]:
    cvs = np.asarray(curve.cvs[:, :2], dtype=float)
    if len(cvs) <= 3:
        return {"penalty": 0.0, "shape": "short", "min": 0.0, "max": 0.0}
    u = np.linspace(0.0, 1.0, 220)
    samples = evaluate_bezier(curve.cvs, u, curve.weights)[:, :2]
    values: list[float] = []
    for cv in cvs[1:-1]:
        distances = np.linalg.norm(samples - cv, axis=1)
        values.append(float(np.min(distances)))
    if not values:
        return {"penalty": 0.0, "shape": "short", "min": 0.0, "max": 0.0}
    distances_arr = np.asarray(values, dtype=float)
    rhythm = _sequence_rhythm_metrics(distances_arr, allow_unimodal=True)
    return {
        "penalty": float(rhythm["penalty"]),
        "shape": str(rhythm["shape"]),
        "min": float(np.min(distances_arr)),
        "max": float(np.max(distances_arr)),
    }


def _sequence_rhythm_metrics(values: np.ndarray, *, allow_unimodal: bool) -> dict[str, float | str]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return {"penalty": 0.0, "shape": "short"}
    scale = max(float(np.mean(np.abs(v))), float(np.ptp(v)), 1e-9)
    delta = np.diff(v)
    const_penalty = float(np.mean((delta / scale) ** 2) * 100.0)
    increasing_penalty = _monotone_violation(delta, scale, increasing=True)
    decreasing_penalty = _monotone_violation(delta, scale, increasing=False)
    candidates: list[tuple[float, str]] = [
        (const_penalty, "constant"),
        (increasing_penalty, "increasing"),
        (decreasing_penalty, "decreasing"),
    ]
    if allow_unimodal:
        candidates.append((_single_lobe_penalty(v, scale, peak=True), "single_peak"))
        candidates.append((_single_lobe_penalty(v, scale, peak=False), "single_valley"))
    penalty, shape = min(candidates, key=lambda item: item[0])
    return {"penalty": float(penalty), "shape": shape}


def _monotone_violation(delta: np.ndarray, scale: float, *, increasing: bool) -> float:
    if len(delta) == 0:
        return 0.0
    eps = scale * 0.035
    if increasing:
        bad = np.clip(-(delta + eps), 0.0, None)
    else:
        bad = np.clip(delta - eps, 0.0, None)
    return float(np.mean((bad / max(scale, 1e-9)) ** 2) * 100.0)


def _single_lobe_penalty(values: np.ndarray, scale: float, *, peak: bool) -> float:
    if len(values) < 4:
        return 0.0
    best = float("inf")
    for split in range(1, len(values) - 1):
        left = np.diff(values[: split + 1])
        right = np.diff(values[split:])
        if peak:
            penalty = _monotone_violation(left, scale, increasing=True) + _monotone_violation(
                right,
                scale,
                increasing=False,
            )
        else:
            penalty = _monotone_violation(left, scale, increasing=False) + _monotone_violation(
                right,
                scale,
                increasing=True,
            )
        best = min(best, penalty)
    return float(best)


def _has_turnback(points: np.ndarray) -> bool:
    if len(points) < 4:
        return False
    main = points[-1] - points[0]
    norm = np.linalg.norm(main)
    if norm <= 1e-9:
        return True
    axis = main / norm
    projection = points @ axis
    decreases = np.sum(np.diff(projection) < -0.05 * norm)
    if decreases > 0:
        return True
    for i in range(len(points) - 3):
        for j in range(i + 2, len(points) - 1):
            if _segments_intersect(points[i], points[i + 1], points[j], points[j + 1]):
                return True
    return False


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)

