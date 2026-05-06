from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autoalias.geometry.bezier import bernstein_basis, evaluate_bezier, signed_curvature_2d
from autoalias.geometry.polyline import (
    chord_length_parameter,
    estimate_polyline_curvature,
    remove_duplicate_points,
    resample_polyline,
    smooth_polyline,
)
from autoalias.models import CurveCandidate, NURBSCurve


@dataclass(slots=True)
class FittingOptions:
    degree: int | str = "auto"
    sample_count: int = 160
    fair_lambda: float = 0.02
    jerk_lambda: float = 0.004
    endpoint_tangent_lambda: float = 0.02
    max_reweight_iters: int = 3
    prefer_degrees: tuple[int, ...] = (5, 7, 3)


class SingleSpanFitter:
    """Fair single-span Bezier/NURBS fitter for Alias-style control curves."""

    def __init__(self, options: FittingOptions | None = None):
        self.options = options or FittingOptions()

    def fit_candidate(self, candidate: CurveCandidate) -> NURBSCurve:
        pts = self._prepare_points(candidate.points)
        degree = self._select_degree(pts, candidate.label)
        cvs = self._fit_fixed_degree(pts, degree)
        cvs = self._orientation_safe(cvs, pts)
        curve = NURBSCurve.single_span(
            label=candidate.label,
            degree=degree,
            cvs=cvs,
            confidence=candidate.confidence,
            source=candidate.source,
            metadata={"candidate_points": len(candidate.points)},
        )
        return curve

    def fit_candidate_adaptive(
        self,
        candidate: CurveCandidate,
        max_error: float = 3.0,
        max_depth: int = 5,
        min_points: int = 18,
    ) -> list[NURBSCurve]:
        """Fit a visual line as connected single-span curves when one span is not accurate enough."""
        return [curve for curve, _seg in self.fit_candidate_adaptive_pairs(candidate, max_error, max_depth, min_points)]

    def fit_candidate_adaptive_pairs(
        self,
        candidate: CurveCandidate,
        max_error: float = 3.0,
        max_depth: int = 5,
        min_points: int = 18,
    ) -> list[tuple[NURBSCurve, np.ndarray]]:
        """Adaptive fit and return each curve with the target segment used for validation."""
        pts = self._prepare_points(candidate.points)
        segments = self._adaptive_point_segments(pts, max_error, max_depth, min_points)
        curves: list[tuple[NURBSCurve, np.ndarray]] = []
        group_id = f"{candidate.label}_{abs(hash(candidate.source + str(len(candidate.points)))) % 1000000}"
        for idx, seg in enumerate(segments):
            if len(seg) < 4:
                continue
            sub = CurveCandidate(
                label=candidate.label,
                points=seg,
                confidence=candidate.confidence,
                source=candidate.source,
                metadata={**candidate.metadata, "curve_group": group_id, "segment_index": idx},
            )
            curve = self.fit_candidate(sub)
            curve.metadata["curve_group"] = group_id
            curve.metadata["segment_index"] = idx
            curve.metadata["segment_count"] = len(segments)
            curves.append((curve, seg))
        return curves

    def _adaptive_point_segments(
        self,
        points: np.ndarray,
        max_error: float,
        max_depth: int,
        min_points: int,
    ) -> list[np.ndarray]:
        if max_depth <= 0 or len(points) < min_points * 2:
            return [points]
        degree = self._select_degree(points, "adaptive")
        cvs = self._fit_fixed_degree(points, degree)
        err = _target_to_curve_errors(points, cvs)
        max_i = int(np.argmax(err))
        if float(err[max_i]) <= max_error:
            return [points]
        split = max(min_points, min(len(points) - min_points, max_i))
        # Avoid pathological splits near endpoints; midpoint is safer for smooth long curves.
        if split <= min_points or split >= len(points) - min_points:
            split = len(points) // 2
        left = points[: split + 1]
        right = points[split:]
        return self._adaptive_point_segments(left, max_error, max_depth - 1, min_points) + self._adaptive_point_segments(
            right, max_error, max_depth - 1, min_points
        )

    def _prepare_points(self, points: np.ndarray) -> np.ndarray:
        pts = remove_duplicate_points(points, eps=0.5)
        if len(pts) < 4:
            raise ValueError("not enough distinct points to fit")
        pts = resample_polyline(pts, max(self.options.sample_count, 32))
        pts = smooth_polyline(pts, window=5)
        return pts

    def _select_degree(self, points: np.ndarray, label: str) -> int:
        if isinstance(self.options.degree, int):
            degree = int(self.options.degree)
            if degree not in (3, 4, 5, 6, 7):
                raise ValueError("degree must be 3, 4, 5, 6 or 7")
            return degree

        label_l = label.lower()
        curvature = estimate_polyline_curvature(points)
        sign_changes = _count_stable_sign_changes(curvature)
        curve_len = np.sum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))
        chord = np.linalg.norm(points[-1, :2] - points[0, :2])
        sinuosity = curve_len / max(chord, 1e-6)

        if "wheel" in label_l or "arch" in label_l or "lamp" in label_l:
            return 7
        if sign_changes >= 1:
            return 7
        if sinuosity > 1.12 or np.nanmax(np.abs(curvature)) > 0.02:
            return 7
        if sinuosity < 1.015:
            return 3
        return 5

    def _fit_fixed_degree(self, points: np.ndarray, degree: int) -> np.ndarray:
        u = chord_length_parameter(points)
        basis = bernstein_basis(degree, u)
        p0 = points[0]
        p1 = points[-1]
        fixed = basis[:, [0]] * p0 + basis[:, [-1]] * p1
        rhs = points - fixed
        a = basis[:, 1:-1]

        reg_a, reg_rhs = self._regularization_system(degree, p0, p1)
        if len(reg_a):
            a_aug = np.vstack([a, reg_a])
            rhs_aug = np.vstack([rhs, reg_rhs])
        else:
            a_aug = a
            rhs_aug = rhs

        interior, *_ = np.linalg.lstsq(a_aug, rhs_aug, rcond=None)
        cvs = np.vstack([p0, interior, p1])

        for _ in range(self.options.max_reweight_iters):
            cvs = self._fair_reweighted_refit(points, cvs)
        return cvs

    def _regularization_system(
        self, degree: int, p0: np.ndarray, p1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        interior_count = degree - 1
        if interior_count <= 0:
            return np.zeros((0, interior_count)), np.zeros((0, 3))

        rows = []
        rhs = []
        all_count = degree + 1

        def add_diff(order: int, lam: float) -> None:
            if lam <= 0 or all_count <= order:
                return
            scale = np.sqrt(lam)
            coeff = np.array([(-1) ** (order - k) * _binom(order, k) for k in range(order + 1)])
            for start in range(all_count - order):
                row = np.zeros(interior_count)
                known = np.zeros(3)
                for off, c in enumerate(coeff):
                    idx = start + off
                    value = scale * c
                    if idx == 0:
                        known += value * p0
                    elif idx == all_count - 1:
                        known += value * p1
                    else:
                        row[idx - 1] += value
                rows.append(row)
                rhs.append(-known)

        add_diff(2, self.options.fair_lambda)
        add_diff(3, self.options.jerk_lambda)

        if not rows:
            return np.zeros((0, interior_count)), np.zeros((0, 3))
        return np.vstack(rows), np.vstack(rhs)

    def _fair_reweighted_refit(self, points: np.ndarray, cvs: np.ndarray) -> np.ndarray:
        degree = len(cvs) - 1
        u = chord_length_parameter(points)
        basis = bernstein_basis(degree, u)
        fitted = evaluate_bezier(cvs, u)
        err = np.linalg.norm(fitted[:, :2] - points[:, :2], axis=1)
        scale = np.median(err) + 1e-6
        weights = 1.0 / np.maximum(1.0, err / (3.0 * scale))
        weights = np.sqrt(weights)[:, None]

        p0 = points[0]
        p1 = points[-1]
        fixed = basis[:, [0]] * p0 + basis[:, [-1]] * p1
        rhs = (points - fixed) * weights
        a = basis[:, 1:-1] * weights
        reg_a, reg_rhs = self._regularization_system(degree, p0, p1)
        a_aug = np.vstack([a, reg_a])
        rhs_aug = np.vstack([rhs, reg_rhs])
        interior, *_ = np.linalg.lstsq(a_aug, rhs_aug, rcond=None)
        return np.vstack([p0, interior, p1])

    def _orientation_safe(self, cvs: np.ndarray, points: np.ndarray) -> np.ndarray:
        u = np.linspace(0.0, 1.0, len(points))
        forward = np.mean(np.linalg.norm(evaluate_bezier(cvs, u)[:, :2] - points[:, :2], axis=1))
        reverse = np.mean(
            np.linalg.norm(evaluate_bezier(cvs[::-1], u)[:, :2] - points[:, :2], axis=1)
        )
        return cvs[::-1] if reverse < forward else cvs

    def classify_shape(self, curve: NURBSCurve) -> dict[str, object]:
        u = np.linspace(0.02, 0.98, 160)
        k = signed_curvature_2d(curve.cvs, u)
        sign_changes = _count_stable_sign_changes(k)
        max_k = float(np.nanmax(np.abs(k))) if len(k) else 0.0
        return {
            "is_s_curve": sign_changes >= 1,
            "is_l_curve": max_k > 0.03 and sign_changes == 0,
            "curvature_sign_changes": sign_changes,
            "max_abs_curvature": max_k,
        }


def _count_stable_sign_changes(values: np.ndarray, eps_ratio: float = 0.08) -> int:
    v = np.asarray(values, dtype=float)
    if len(v) < 5:
        return 0
    eps = max(np.nanmax(np.abs(v)) * eps_ratio, 1e-9)
    signs = np.sign(np.where(np.abs(v) < eps, 0.0, v))
    nonzero = signs[signs != 0]
    if len(nonzero) < 2:
        return 0
    return int(np.sum(nonzero[1:] * nonzero[:-1] < 0))


def _binom(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    out = 1
    for i in range(1, k + 1):
        out = out * (n - i + 1) // i
    return out


def _target_to_curve_errors(points: np.ndarray, cvs: np.ndarray) -> np.ndarray:
    samples = evaluate_bezier(cvs, np.linspace(0.0, 1.0, max(120, len(points) * 2)))
    diff = points[:, None, :2] - samples[None, :, :2]
    return np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
