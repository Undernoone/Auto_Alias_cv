from __future__ import annotations

import numpy as np

from autoalias.geometry.bezier import evaluate_bezier
from autoalias.geometry.fitting import FittingOptions, SingleSpanFitter
from autoalias.models import CurveCandidate, NURBSCurve


def build_alias_design_curves(
    curves: list[NURBSCurve],
    max_gap: float = 18.0,
    max_angle_deg: float = 26.0,
    max_iterations: int = 90,
) -> list[NURBSCurve]:
    """Create the production Alias curve set.

    Coverage repair fragments are useful diagnostically, but they make Alias unusable when mixed
    into the main file. This function removes tiny repair-only fragments and merges G1-compatible
    adjacent segments into longer single-span design curves.
    """
    primary = [c for c in curves if not _is_repair_only(c)]
    repair = [c for c in curves if _is_repair_only(c)]

    merged = _merge_g1_segments(primary, max_gap, np.deg2rad(max_angle_deg), max_iterations)
    merged = _absorb_repair_fragments(merged, repair, max_gap=12.0, max_angle=np.deg2rad(34.0))
    merged = _snap_compatible_endpoints(merged, snap_gap=20.0, max_axis_angle=np.deg2rad(52.0))
    return sorted(merged, key=lambda c: _curve_length(c), reverse=True)


def _merge_g1_segments(
    curves: list[NURBSCurve],
    max_gap: float,
    max_angle: float,
    max_iterations: int,
) -> list[NURBSCurve]:
    items = list(curves)
    for _ in range(max_iterations):
        best: tuple[float, int, int, NURBSCurve] | None = None
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                candidate = _try_merge_curves(items[i], items[j], max_gap, max_angle)
                if candidate is None:
                    continue
                gap, merged = candidate
                score = gap - 0.001 * (_curve_length(items[i]) + _curve_length(items[j]))
                if best is None or score < best[0]:
                    best = (score, i, j, merged)
        if best is None:
            break
        _, i, j, merged = best
        items = [c for k, c in enumerate(items) if k not in (i, j)]
        items.append(merged)
    return items


def _absorb_repair_fragments(
    primary: list[NURBSCurve],
    repair: list[NURBSCurve],
    max_gap: float,
    max_angle: float,
) -> list[NURBSCurve]:
    items = list(primary)
    for frag in sorted(repair, key=_curve_length, reverse=True):
        if frag.metadata.get("preserve_segment"):
            items.append(frag)
            continue
        best: tuple[float, int, NURBSCurve] | None = None
        for i, curve in enumerate(items):
            candidate = _try_merge_curves(curve, frag, max_gap, max_angle)
            if candidate is None:
                continue
            gap, merged = candidate
            if best is None or gap < best[0]:
                best = (gap, i, merged)
        if best is not None:
            _, i, merged = best
            items[i] = merged
        elif _curve_length(frag) > 28.0:
            # Keep only meaningful standalone repairs in the production file.
            items.append(frag)
    return items


def _try_merge_curves(
    a: NURBSCurve,
    b: NURBSCurve,
    max_gap: float,
    max_angle: float,
) -> tuple[float, NURBSCurve] | None:
    if a.metadata.get("preserve_segment") or b.metadata.get("preserve_segment"):
        return None
    combos = [
        (a, b, False, False, _endpoint(a, 1), _endpoint(b, 0), _tangent(a, 1), _tangent(b, 0)),
        (b, a, False, False, _endpoint(b, 1), _endpoint(a, 0), _tangent(b, 1), _tangent(a, 0)),
        (a, b, False, True, _endpoint(a, 1), _endpoint(b, 1), _tangent(a, 1), -_tangent(b, 1)),
        (a, b, True, False, _endpoint(a, 0), _endpoint(b, 0), -_tangent(a, 0), _tangent(b, 0)),
    ]
    best: tuple[float, NURBSCurve] | None = None
    for first, second, rev_first, rev_second, p0, p1, t0, t1 in combos:
        gap = float(np.linalg.norm(p0[:2] - p1[:2]))
        if gap > max_gap:
            continue
        angle = _angle(t0, t1)
        if angle > max_angle:
            continue
        # Do not merge across a real L corner. Use a wider angle only for very tiny gaps.
        if gap > 3.0 and angle > max_angle * 0.75:
            continue
        points = _sample_curve(first, reverse=rev_first)
        other = _sample_curve(second, reverse=rev_second)
        if np.linalg.norm(points[-1, :2] - other[0, :2]) < 1.0:
            joined = np.vstack([points, other[1:]])
        else:
            joined = np.vstack([points, other])
        merged = _fit_joined_curve(first, second, joined)
        if merged is None:
            continue
        if best is None or gap < best[0]:
            best = (gap, merged)
    return best


def _fit_joined_curve(a: NURBSCurve, b: NURBSCurve, points: np.ndarray) -> NURBSCurve | None:
    fitter = SingleSpanFitter(FittingOptions(degree="auto", sample_count=max(120, len(points))))
    label = a.label if _curve_length(a) >= _curve_length(b) else b.label
    try:
        curve = fitter.fit_candidate(
            CurveCandidate(
                label=label,
                points=points,
                confidence=max(a.confidence, b.confidence),
                source="alias_post_merge",
                metadata={
                    "merged_from": [a.label, b.label],
                    "source_a": a.source,
                    "source_b": b.source,
                },
            )
        )
    except Exception:
        return None
    curve.metadata["post_merged"] = True
    curve.metadata["merged_sources"] = [a.source, b.source]
    return curve


def _sample_curve(curve: NURBSCurve, reverse: bool = False, n: int | None = None) -> np.ndarray:
    n = n or max(16, min(80, int(_curve_length(curve) / 8.0)))
    pts = evaluate_bezier(curve.cvs, np.linspace(0.0, 1.0, n), curve.weights)
    return pts[::-1] if reverse else pts


def _endpoint(curve: NURBSCurve, side: int) -> np.ndarray:
    return curve.cvs[0] if side == 0 else curve.cvs[-1]


def _tangent(curve: NURBSCurve, side: int) -> np.ndarray:
    if side == 0:
        v = curve.cvs[1, :2] - curve.cvs[0, :2]
    else:
        v = curve.cvs[-1, :2] - curve.cvs[-2, :2]
    n = np.linalg.norm(v)
    return v / max(n, 1e-9)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def _curve_length(curve: NURBSCurve) -> float:
    pts = evaluate_bezier(curve.cvs, np.linspace(0.0, 1.0, 80), curve.weights)
    return float(np.sum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)))


def _snap_compatible_endpoints(
    curves: list[NURBSCurve],
    snap_gap: float,
    max_axis_angle: float,
    iterations: int = 8,
) -> list[NURBSCurve]:
    items = [_copy_curve(c) for c in curves]
    for _ in range(iterations):
        pairs: list[tuple[float, int, int, int, int]] = []
        for i, a in enumerate(items):
            for j in range(i + 1, len(items)):
                b = items[j]
                for side_a in (0, 1):
                    pa = _endpoint(a, side_a)
                    axis_a = _endpoint_axis(a, side_a)
                    for side_b in (0, 1):
                        pb = _endpoint(b, side_b)
                        gap = float(np.linalg.norm(pa[:2] - pb[:2]))
                        if gap <= 1e-6 or gap > snap_gap:
                            continue
                        axis_b = _endpoint_axis(b, side_b)
                        if _endpoint_pair_is_compatible(pa, pb, axis_a, axis_b, gap, max_axis_angle):
                            pairs.append((gap, i, side_a, j, side_b))
        if not pairs:
            break
        used: set[tuple[int, int]] = set()
        changed = False
        for gap, i, side_i, j, side_j in sorted(pairs):
            key_i = (i, side_i)
            key_j = (j, side_j)
            if key_i in used or key_j in used:
                continue
            pi = _endpoint(items[i], side_i)
            pj = _endpoint(items[j], side_j)
            if float(np.linalg.norm(pi[:2] - pj[:2])) > snap_gap:
                continue
            target = 0.5 * (pi + pj)
            _move_endpoint_preserve_tangent(items[i], side_i, target)
            _move_endpoint_preserve_tangent(items[j], side_j, target)
            used.add(key_i)
            used.add(key_j)
            changed = True
        if not changed:
            break
    return items


def _endpoint_pair_is_compatible(
    pa: np.ndarray,
    pb: np.ndarray,
    axis_a: np.ndarray,
    axis_b: np.ndarray,
    gap: float,
    max_axis_angle: float,
) -> bool:
    if gap <= 5.0:
        return True
    axis_dot = abs(float(np.dot(axis_a, axis_b)))
    if axis_dot < float(np.cos(max_axis_angle)):
        return False
    v = pb[:2] - pa[:2]
    v_norm = np.linalg.norm(v)
    if v_norm <= 1e-9:
        return True
    v = v / v_norm
    # The gap should lie roughly along the shared tangent axis; otherwise nearby parallel lines
    # would get incorrectly snapped together.
    return abs(float(np.dot(axis_a, v))) >= np.cos(np.deg2rad(58.0))


def _endpoint_axis(curve: NURBSCurve, side: int) -> np.ndarray:
    t = _tangent(curve, side)
    n = np.linalg.norm(t)
    return t / max(n, 1e-9)


def _move_endpoint_preserve_tangent(curve: NURBSCurve, side: int, target: np.ndarray) -> None:
    if side == 0:
        delta = target - curve.cvs[0]
        curve.cvs[0] += delta
        if len(curve.cvs) > 1:
            curve.cvs[1] += delta
    else:
        delta = target - curve.cvs[-1]
        curve.cvs[-1] += delta
        if len(curve.cvs) > 1:
            curve.cvs[-2] += delta
    curve.metadata["endpoint_snapped"] = True


def _copy_curve(curve: NURBSCurve) -> NURBSCurve:
    return NURBSCurve(
        label=curve.label,
        degree=curve.degree,
        cvs=curve.cvs.copy(),
        weights=curve.weights.copy(),
        knots=curve.knots.copy(),
        u_min=curve.u_min,
        u_max=curve.u_max,
        confidence=curve.confidence,
        source=curve.source,
        metadata=dict(curve.metadata),
    )


def _is_repair_only(curve: NURBSCurve) -> bool:
    source = curve.source
    if "uncovered_component" in source:
        return True
    if "uncovered_pass" in source and _curve_length(curve) < 32.0:
        return True
    return False
