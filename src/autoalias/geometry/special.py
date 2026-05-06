from __future__ import annotations

import numpy as np

from autoalias.geometry.bezier import evaluate_derivative, signed_curvature_2d
from autoalias.models import NURBSCurve


def annotate_special_shape(curve: NURBSCurve) -> None:
    """Attach S-curve and L-blend diagnostics without changing the single-span contract."""
    u = np.linspace(0.02, 0.98, 200)
    k = signed_curvature_2d(curve.cvs, u)
    inflections = _inflection_locations(u, k)
    curve.metadata["inflection_u"] = inflections
    curve.metadata["is_s_curve"] = len(inflections) > 0

    peak_i = int(np.argmax(np.abs(k))) if len(k) else 0
    peak_k = float(k[peak_i]) if len(k) else 0.0
    effective_radius = float(1.0 / max(abs(peak_k), 1e-9))
    curve.metadata["max_abs_curvature"] = abs(peak_k)
    curve.metadata["effective_radius_px"] = effective_radius

    if _looks_like_l_curve(k, inflections):
        center = evaluate_derivative(curve.cvs, np.array([u[peak_i]]), order=0)[0]
        curve.metadata["is_l_curve"] = True
        curve.metadata["blend_center"] = center.tolist()
        curve.metadata["blend_u"] = float(u[peak_i])
        curve.metadata["blend_effective_radius_px"] = effective_radius
        curve.metadata["blend_continuity_estimate"] = _blend_continuity(curve, u[peak_i])
    else:
        curve.metadata["is_l_curve"] = False


def _inflection_locations(u: np.ndarray, k: np.ndarray) -> list[float]:
    if len(k) == 0:
        return []
    eps = max(float(np.max(np.abs(k))) * 0.04, 1e-10)
    signs = np.sign(np.where(np.abs(k) < eps, 0.0, k))
    out: list[float] = []
    last_idx: int | None = None
    last_sign = 0.0
    for i, sign in enumerate(signs):
        if sign == 0:
            continue
        if last_idx is not None and last_sign != 0 and sign != last_sign:
            out.append(float((u[last_idx] + u[i]) * 0.5))
        last_idx = i
        last_sign = sign
    return out


def _looks_like_l_curve(k: np.ndarray, inflections: list[float]) -> bool:
    if len(k) < 20 or inflections:
        return False
    abs_k = np.abs(k)
    peak = float(np.max(abs_k))
    if peak <= 1e-9:
        return False
    high = abs_k > peak * 0.55
    concentration = float(np.sum(high)) / len(high)
    return 0.04 <= concentration <= 0.35


def _blend_continuity(curve: NURBSCurve, u_peak: float) -> dict[str, float | str]:
    us = np.clip(np.array([u_peak - 0.08, u_peak + 0.08]), 0.02, 0.98)
    d1 = evaluate_derivative(curve.cvs[:, :2], us, order=1)
    t = d1 / np.maximum(np.linalg.norm(d1, axis=1, keepdims=True), 1e-9)
    tangent_angle = float(np.degrees(np.arccos(np.clip(np.dot(t[0], t[1]), -1.0, 1.0))))
    k = signed_curvature_2d(curve.cvs, us)
    curvature_jump = float(abs(k[1] - k[0]))
    if curvature_jump < 0.002:
        grade = "G2/G3-like visual blend"
    elif curvature_jump < 0.01:
        grade = "G2-like visual blend"
    else:
        grade = "fair but curvature needs review"
    return {
        "tangent_angle_deg_across_blend": tangent_angle,
        "curvature_jump_across_blend": curvature_jump,
        "grade": grade,
    }

