from __future__ import annotations

import numpy as np

from autoalias.geometry.bezier import evaluate_bezier
from autoalias.geometry.fitting import FittingOptions, SingleSpanFitter
from autoalias.geometry.postprocess import build_alias_design_curves
from autoalias.models import CurveCandidate, NURBSCurve


def beautify_alias_curves(curves: list[NURBSCurve]) -> list[NURBSCurve]:
    """Produce an Alias-facing fair version of the curve set.

    This pass intentionally allows mild deviation from the source line art. The goal is clean CVs,
    smooth curvature combs, connected endpoints, and fair L-corner blends.
    """
    if not curves:
        return []
    out = [_copy_curve(c) for c in curves]
    out = _beautify_individual_curves(out)
    out = _beautify_l_corner_groups(out)
    out = _snap_and_average_near_endpoints(out)
    out = build_alias_design_curves(out)
    out = _beautify_individual_curves(out, second_pass=True)
    out = _beautify_l_corner_groups(out)
    out = _insert_fair_corner_bridges(out)
    out = _align_blend_neighbor_handles(out)
    out = _snap_and_average_near_endpoints(out)
    return out


def _beautify_individual_curves(
    curves: list[NURBSCurve],
    second_pass: bool = False,
) -> list[NURBSCurve]:
    out: list[NURBSCurve] = []
    for curve in curves:
        if curve.degree < 3:
            out.append(curve)
            continue
        if curve.metadata.get("l_corner_role") == "blend":
            out.append(curve)
            continue
        out.append(_fair_refit_curve(curve, second_pass=second_pass))
    return out


def _fair_refit_curve(curve: NURBSCurve, second_pass: bool = False) -> NURBSCurve:
    samples = evaluate_bezier(curve.cvs, np.linspace(0.0, 1.0, 180), curve.weights)
    degree = _beauty_degree(curve)
    options = FittingOptions(
        degree=degree,
        sample_count=160,
        fair_lambda=0.18 if not second_pass else 0.28,
        jerk_lambda=0.05 if not second_pass else 0.10,
        max_reweight_iters=2,
    )
    fitter = SingleSpanFitter(options)
    try:
        fair = fitter.fit_candidate(
            CurveCandidate(
                label=curve.label,
                points=samples,
                confidence=curve.confidence,
                source=f"{curve.source}+beauty_refit",
                metadata=dict(curve.metadata),
            )
        )
    except Exception:
        fair = _copy_curve(curve)
    fair.metadata.update(curve.metadata)
    fair.metadata["beautified"] = True
    fair.metadata["beauty_source"] = curve.source
    return _blend_back_if_too_far(curve, fair, max_mean_deviation=10.0 if second_pass else 14.0)


def _beauty_degree(curve: NURBSCurve) -> int:
    if curve.metadata.get("l_corner_role") == "leg_a" or curve.metadata.get("l_corner_role") == "leg_b":
        return 3
    if curve.degree <= 3:
        return 3
    if curve.degree in (4, 5):
        return 5
    return 7


def _blend_back_if_too_far(original: NURBSCurve, fair: NURBSCurve, max_mean_deviation: float) -> NURBSCurve:
    u = np.linspace(0.0, 1.0, 160)
    a = evaluate_bezier(original.cvs, u, original.weights)
    b = evaluate_bezier(fair.cvs, u, fair.weights)
    dev = float(np.mean(np.linalg.norm(a[:, :2] - b[:, :2], axis=1)))
    if dev <= max_mean_deviation:
        return fair
    # Interpolate CVs if degrees match; otherwise keep the fair curve only partially by refitting
    # the blended sample points.
    alpha = max(0.25, min(0.80, max_mean_deviation / max(dev, 1e-9)))
    if fair.degree == original.degree and len(fair.cvs) == len(original.cvs):
        cvs = (1.0 - alpha) * original.cvs + alpha * fair.cvs
        result = NURBSCurve.single_span(
            original.label,
            original.degree,
            cvs,
            weights=original.weights.copy(),
            confidence=original.confidence,
            source=f"{original.source}+beauty_blend",
            metadata={**original.metadata, "beautified": True, "beauty_alpha": float(alpha)},
        )
        return result
    samples = (1.0 - alpha) * a + alpha * b
    try:
        return SingleSpanFitter(FittingOptions(degree=fair.degree, fair_lambda=0.22, jerk_lambda=0.06)).fit_candidate(
            CurveCandidate(original.label, samples, original.confidence, f"{original.source}+beauty_blend", dict(original.metadata))
        )
    except Exception:
        return original


def _beautify_l_corner_groups(curves: list[NURBSCurve]) -> list[NURBSCurve]:
    groups: dict[str, dict[str, NURBSCurve]] = {}
    passthrough: list[NURBSCurve] = []
    for curve in curves:
        group = curve.metadata.get("l_corner_group")
        role = curve.metadata.get("l_corner_role")
        if group and role in {"leg_a", "blend", "leg_b"}:
            groups.setdefault(str(group), {})[str(role)] = curve
        else:
            passthrough.append(curve)

    rebuilt: list[NURBSCurve] = []
    for group_id, parts in groups.items():
        if not {"leg_a", "blend", "leg_b"} <= set(parts):
            rebuilt.extend(parts.values())
            continue
        leg_a = _make_leg_fair(parts["leg_a"], end_side=1)
        leg_b = _make_leg_fair(parts["leg_b"], end_side=0)
        blend = _make_corner_blend_from_legs(parts["blend"], leg_a, leg_b)
        if blend is None:
            rebuilt.extend([leg_a, parts["blend"], leg_b])
            continue
        _set_endpoint(leg_a, 1, blend.cvs[0])
        _set_endpoint(leg_b, 0, blend.cvs[-1])
        _set_endpoint_handle_direction(leg_a, 1, -_unit(blend.cvs[1, :2] - blend.cvs[0, :2]))
        _set_endpoint_handle_direction(leg_b, 0, _unit(blend.cvs[-1, :2] - blend.cvs[-2, :2]))
        for role, curve in (("leg_a", leg_a), ("blend", blend), ("leg_b", leg_b)):
            curve.metadata["l_corner_group"] = group_id
            curve.metadata["l_corner_role"] = role
            curve.metadata["preserve_segment"] = True
            curve.metadata["beautified"] = True
        rebuilt.extend([leg_a, blend, leg_b])
    return passthrough + rebuilt


def _make_leg_fair(curve: NURBSCurve, end_side: int) -> NURBSCurve:
    # Legs should read like clean Alias construction strokes. Degree 3 is enough and avoids
    # accidental waviness on nearly straight portions.
    samples = evaluate_bezier(curve.cvs, np.linspace(0.0, 1.0, 90), curve.weights)
    try:
        fair = SingleSpanFitter(FittingOptions(degree=3, fair_lambda=0.35, jerk_lambda=0.12)).fit_candidate(
            CurveCandidate(curve.label, samples, curve.confidence, f"{curve.source}+beauty_leg", dict(curve.metadata))
        )
    except Exception:
        return curve
    fair.metadata.update(curve.metadata)
    fair.metadata["beautified"] = True
    return fair


def _make_corner_blend_from_legs(
    old_blend: NURBSCurve,
    leg_a: NURBSCurve,
    leg_b: NURBSCurve,
) -> NURBSCurve | None:
    p0 = leg_a.cvs[-1].copy()
    p5 = leg_b.cvs[0].copy()
    in_dir = _unit(leg_a.cvs[-1, :2] - leg_a.cvs[-2, :2])
    out_dir = _unit(leg_b.cvs[1, :2] - leg_b.cvs[0, :2])
    chord = float(np.linalg.norm(p5[:2] - p0[:2]))
    if chord < 6.0:
        return None
    chord_axis = _unit(p5[:2] - p0[:2])
    if float(np.dot(in_dir, chord_axis)) < 0:
        in_dir = -in_dir
    if float(np.dot(out_dir, chord_axis)) < 0:
        out_dir = -out_dir
    target = evaluate_bezier(old_blend.cvs, np.linspace(0.0, 1.0, 90), old_blend.weights)
    curve = _make_quintic_blend(
        old_blend.label,
        p0,
        p5,
        in_dir,
        out_dir,
        source=f"{old_blend.source}+beauty_blend",
        target=target,
    )
    if curve is None:
        return None
    curve.label = old_blend.label
    curve.confidence = old_blend.confidence
    curve.source = f"{old_blend.source}+beauty_blend"
    curve.metadata.update(dict(old_blend.metadata))
    curve.metadata["beauty_blend_symmetric_handles"] = True
    return curve


def _insert_fair_corner_bridges(curves: list[NURBSCurve]) -> list[NURBSCurve]:
    """Insert fair blend curves across broken L-corner gaps.

    The image extractor can split a single styling line into two entities right before and after
    the bend. G1 merging correctly refuses those because their tangent angle is large, so this pass
    treats them as a designer corner: keep both legs and add one clean quintic blend between them.
    """
    out = [_copy_curve(c) for c in curves]
    endpoints = []
    for idx, curve in enumerate(out):
        if curve.metadata.get("l_corner_role") == "blend":
            continue
        if curve.metadata.get("l_corner_group"):
            continue
        if _curve_length(curve) < 32.0:
            continue
        for side in (0, 1):
            endpoints.append(
                {
                    "idx": idx,
                    "side": side,
                    "point": _get_endpoint(curve, side).copy(),
                    "leave": _endpoint_leave_dir(curve, side),
                    "enter": _endpoint_enter_dir(curve, side),
                    "label": curve.label,
                }
            )

    proposals: list[tuple[float, int, int, int, int, NURBSCurve]] = []
    for a_i in range(len(endpoints)):
        a = endpoints[a_i]
        for b_i in range(a_i + 1, len(endpoints)):
            b = endpoints[b_i]
            if a["idx"] == b["idx"]:
                continue
            for first, second in ((a, b), (b, a)):
                bridge = _try_make_corner_bridge(first, second, out)
                if bridge is None:
                    continue
                gap = float(np.linalg.norm(first["point"][:2] - second["point"][:2]))
                angle = _angle(first["leave"], second["enter"])
                label_penalty = 8.0 if first["label"] != second["label"] else 0.0
                score = gap + label_penalty + abs(angle - np.deg2rad(82.0)) * 3.0
                proposals.append(
                    (
                        score,
                        int(first["idx"]),
                        int(first["side"]),
                        int(second["idx"]),
                        int(second["side"]),
                        bridge,
                    )
                )

    used: set[tuple[int, int]] = set()
    bridges: list[NURBSCurve] = []
    for _score, ia, sa, ib, sb, bridge in sorted(proposals, key=lambda item: item[0]):
        if (ia, sa) in used or (ib, sb) in used:
            continue
        if len(bridges) >= max(8, len(out) // 8):
            break
        rebuilt = _make_trimmed_corner_bridge(out[ia], sa, out[ib], sb, bridge)
        if rebuilt is None:
            continue
        out[ia], out[ib], bridge = rebuilt
        used.add((ia, sa))
        used.add((ib, sb))
        _mark_bridge_leg(out[ia], sa, bridge.metadata["beauty_bridge_group"])
        _mark_bridge_leg(out[ib], sb, bridge.metadata["beauty_bridge_group"])
        bridges.append(bridge)
    return out + bridges


def _make_trimmed_corner_bridge(
    leg_a: NURBSCurve,
    side_a: int,
    leg_b: NURBSCurve,
    side_b: int,
    provisional: NURBSCurve,
) -> tuple[NURBSCurve, NURBSCurve, NURBSCurve] | None:
    """Build a real designer fillet by trimming both legs before inserting the blend."""
    len_a = _curve_length(leg_a)
    len_b = _curve_length(leg_b)
    if min(len_a, len_b) < 54.0:
        return None
    gap = float(provisional.metadata.get("bridge_gap_px", 0.0))
    angle_deg = float(provisional.metadata.get("bridge_angle_deg", 90.0))
    angle_scale = np.sin(np.deg2rad(max(30.0, min(130.0, angle_deg))) * 0.5)
    desired = max(18.0, gap * 1.75, 32.0 * angle_scale)
    trim = min(48.0, len_a * 0.24, len_b * 0.24, desired)
    if trim < 12.0:
        return None

    trimmed_a = _trim_curve_endpoint(leg_a, side_a, trim)
    trimmed_b = _trim_curve_endpoint(leg_b, side_b, trim)
    if trimmed_a is None or trimmed_b is None:
        return None

    p0 = _get_endpoint(trimmed_a, side_a).copy()
    p5 = _get_endpoint(trimmed_b, side_b).copy()
    in_dir = _endpoint_leave_dir(trimmed_a, side_a)
    out_dir = _endpoint_enter_dir(trimmed_b, side_b)
    bridge = _make_quintic_blend(
        label=provisional.label,
        p0=p0,
        p5=p5,
        in_dir=in_dir,
        out_dir=out_dir,
        source="beauty_trimmed_corner_bridge",
    )
    if bridge is None:
        return None
    bridge.metadata.update(provisional.metadata)
    bridge.metadata.update(
        {
            "beauty_auto_corner_bridge": True,
            "beauty_trimmed_corner_bridge": True,
            "trim_length_px": float(trim),
            "bridge_gap_px": float(gap),
            "bridge_angle_deg": float(angle_deg),
            "preserve_segment": True,
        }
    )
    _set_endpoint_handle_direction(trimmed_a, side_a, -_unit(bridge.cvs[1, :2] - bridge.cvs[0, :2]))
    _set_endpoint_handle_direction(trimmed_b, side_b, _unit(bridge.cvs[-1, :2] - bridge.cvs[-2, :2]))
    return trimmed_a, trimmed_b, bridge


def _trim_curve_endpoint(curve: NURBSCurve, side: int, distance: float) -> NURBSCurve | None:
    samples = evaluate_bezier(curve.cvs, np.linspace(0.0, 1.0, 260), curve.weights)
    length = float(np.sum(np.linalg.norm(np.diff(samples[:, :2], axis=0), axis=1)))
    if distance <= 1.0 or distance >= length * 0.42:
        return None
    if side == 1:
        kept = _trim_samples_from_end(samples, distance)
    else:
        kept = _trim_samples_from_start(samples, distance)
    if kept is None or len(kept) < 8:
        return None
    degree = 3 if _is_nearly_straight_samples(kept) else min(max(curve.degree, 5), 7)
    try:
        trimmed = SingleSpanFitter(
            FittingOptions(
                degree=degree,
                sample_count=max(80, min(180, len(kept))),
                fair_lambda=0.18,
                jerk_lambda=0.08,
                max_reweight_iters=2,
            )
        ).fit_candidate(
            CurveCandidate(
                curve.label,
                kept,
                curve.confidence,
                f"{curve.source}+corner_trim",
                dict(curve.metadata),
            )
        )
    except Exception:
        return None
    trimmed.metadata.update(curve.metadata)
    trimmed.metadata["beauty_corner_trimmed"] = True
    trimmed.metadata["corner_trim_length_px"] = float(distance)
    return trimmed


def _trim_samples_from_end(samples: np.ndarray, distance: float) -> np.ndarray | None:
    rev = samples[::-1]
    cut = _point_at_arclength(rev, distance)
    if cut is None:
        return None
    idx, point = cut
    keep_start = len(samples) - idx
    kept = samples[:keep_start]
    if len(kept) == 0 or np.linalg.norm(kept[-1, :2] - point[:2]) > 1e-6:
        kept = np.vstack([kept, point])
    return kept


def _trim_samples_from_start(samples: np.ndarray, distance: float) -> np.ndarray | None:
    cut = _point_at_arclength(samples, distance)
    if cut is None:
        return None
    idx, point = cut
    kept = samples[idx:]
    if len(kept) == 0 or np.linalg.norm(kept[0, :2] - point[:2]) > 1e-6:
        kept = np.vstack([point, kept])
    else:
        kept[0] = point
    return kept


def _point_at_arclength(samples: np.ndarray, distance: float) -> tuple[int, np.ndarray] | None:
    if len(samples) < 2:
        return None
    seg = np.linalg.norm(np.diff(samples[:, :2], axis=0), axis=1)
    acc = 0.0
    for i, length in enumerate(seg):
        next_acc = acc + float(length)
        if next_acc >= distance:
            t = (distance - acc) / max(float(length), 1e-9)
            point = (1.0 - t) * samples[i] + t * samples[i + 1]
            return i + 1, point
        acc = next_acc
    return None


def _is_nearly_straight_samples(samples: np.ndarray) -> bool:
    if len(samples) < 8:
        return True
    chord = samples[-1, :2] - samples[0, :2]
    length = np.linalg.norm(chord)
    if length <= 1e-9:
        return False
    normal = np.array([-chord[1], chord[0]]) / length
    dev = np.abs((samples[:, :2] - samples[0, :2]) @ normal)
    return float(np.percentile(dev, 92)) < max(3.5, length * 0.028)


def _try_make_corner_bridge(
    first: dict[str, object],
    second: dict[str, object],
    curves: list[NURBSCurve],
) -> NURBSCurve | None:
    p0 = np.asarray(first["point"], dtype=float)
    p5 = np.asarray(second["point"], dtype=float)
    first_is_candidate = str(first["label"]).startswith("candidate_")
    second_is_candidate = str(second["label"]).startswith("candidate_")
    if not (first_is_candidate or second_is_candidate):
        return None
    gap = float(np.linalg.norm(p5[:2] - p0[:2]))
    if gap < 5.0 or gap > 34.0:
        return None

    c0 = curves[int(first["idx"])]
    c1 = curves[int(second["idx"])]
    if gap > min(_curve_length(c0), _curve_length(c1)) * 0.42:
        return None

    in_dir = np.asarray(first["leave"], dtype=float)
    out_dir = np.asarray(second["enter"], dtype=float)
    angle = _angle(in_dir, out_dir)
    if angle < np.deg2rad(34.0) or angle > np.deg2rad(132.0):
        return None

    chord = _unit(p5[:2] - p0[:2])
    if float(np.dot(in_dir, chord)) < 0.16 or float(np.dot(out_dir, chord)) < 0.16:
        return None

    # Avoid bridging two separate parallel styling bands. If labels disagree, require a very
    # compact gap and a strong corner angle.
    if first["label"] != second["label"] and (gap > 18.0 or angle < np.deg2rad(55.0)):
        return None

    bridge = _make_quintic_blend(
        label=str(first["label"]),
        p0=p0,
        p5=p5,
        in_dir=in_dir,
        out_dir=out_dir,
        source="beauty_auto_corner_bridge",
    )
    if bridge is None:
        return None
    group_id = f"beauty_bridge_{abs(hash((int(first['idx']), int(first['side']), int(second['idx']), int(second['side'])))) % 1000000}"
    bridge.metadata.update(
        {
            "beauty_auto_corner_bridge": True,
            "beauty_bridge_group": group_id,
            "preserve_segment": True,
            "bridge_gap_px": float(gap),
            "bridge_angle_deg": float(np.rad2deg(angle)),
            "bridge_leg_a": int(first["idx"]),
            "bridge_leg_b": int(second["idx"]),
        }
    )
    return bridge


def _make_quintic_blend(
    label: str,
    p0: np.ndarray,
    p5: np.ndarray,
    in_dir: np.ndarray,
    out_dir: np.ndarray,
    source: str,
    target: np.ndarray | None = None,
) -> NURBSCurve | None:
    chord = float(np.linalg.norm(p5[:2] - p0[:2]))
    if chord < 5.0:
        return None
    in_dir = _unit(in_dir)
    out_dir = _unit(out_dir)
    searched = _search_quintic_blend_handles(p0, p5, in_dir, out_dir, target)
    if searched is None:
        return None
    cvs, h = searched
    return NURBSCurve.single_span(
        label=label,
        degree=5,
        cvs=cvs,
        source=source,
        metadata={
            "beautified": True,
            "beauty_blend_symmetric_handles": True,
            "blend_handle_length_px": float(h),
        },
    )


def _search_quintic_blend_handles(
    p0: np.ndarray,
    p5: np.ndarray,
    in_dir: np.ndarray,
    out_dir: np.ndarray,
    target: np.ndarray | None,
) -> tuple[np.ndarray, float] | None:
    chord = float(np.linalg.norm(p5[:2] - p0[:2]))
    if chord <= 1e-9:
        return None
    handle_factors = (0.14, 0.18, 0.22, 0.28, 0.34, 0.42)
    mid_factors = (1.45, 1.65, 1.86, 2.08)
    best: tuple[float, np.ndarray, float] | None = None
    for h0_factor in handle_factors:
        for h1_factor in handle_factors:
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
                if _blend_has_bad_projection(cvs):
                    continue
                score = _blend_target_score(cvs, target) + symmetry_penalty
                if best is None or score < best[0]:
                    best = (score, cvs.copy(), 0.5 * (h0 + h1))
    if best is None:
        return None
    return best[1], best[2]


def _blend_target_score(cvs: np.ndarray, target: np.ndarray | None) -> float:
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


def _blend_has_bad_projection(cvs: np.ndarray) -> bool:
    pts = evaluate_bezier(cvs, np.linspace(0.0, 1.0, 80))
    chord = pts[-1, :2] - pts[0, :2]
    length = np.linalg.norm(chord)
    if length <= 1e-9:
        return True
    axis = chord / length
    proj = pts[:, :2] @ axis
    return bool(np.any(np.diff(proj) < -length * 0.025))


def _mark_bridge_leg(curve: NURBSCurve, side: int, group_id: str) -> None:
    curve.metadata.setdefault("beauty_bridge_groups", [])
    groups = curve.metadata["beauty_bridge_groups"]
    if isinstance(groups, list):
        groups.append({"group": group_id, "side": int(side)})
    curve.metadata["beauty_bridge_leg"] = True


def _align_blend_neighbor_handles(curves: list[NURBSCurve]) -> list[NURBSCurve]:
    out = [_copy_curve(c) for c in curves]
    for blend in [c for c in out if c.metadata.get("l_corner_role") == "blend" or c.metadata.get("beauty_auto_corner_bridge")]:
        start = blend.cvs[0]
        end = blend.cvs[-1]
        start_t = _unit(blend.cvs[1, :2] - blend.cvs[0, :2])
        end_t = _unit(blend.cvs[-1, :2] - blend.cvs[-2, :2])
        for curve in out:
            if curve is blend:
                continue
            for side in (0, 1):
                endpoint = _get_endpoint(curve, side)
                if float(np.linalg.norm(endpoint[:2] - start[:2])) < 1.5:
                    _set_endpoint_handle_direction(curve, side, -start_t)
                elif float(np.linalg.norm(endpoint[:2] - end[:2])) < 1.5:
                    _set_endpoint_handle_direction(curve, side, end_t)
    return out


def _snap_and_average_near_endpoints(curves: list[NURBSCurve]) -> list[NURBSCurve]:
    out = [_copy_curve(c) for c in curves]
    for _ in range(5):
        changed = False
        endpoints = []
        for i, curve in enumerate(out):
            endpoints.append((i, 0, curve.cvs[0].copy(), _endpoint_dir(curve, 0)))
            endpoints.append((i, 1, curve.cvs[-1].copy(), _endpoint_dir(curve, 1)))
        used: set[tuple[int, int]] = set()
        pairs: list[tuple[float, int, int, int, int]] = []
        for a in range(len(endpoints)):
            ia, sa, pa, ta = endpoints[a]
            for b in range(a + 1, len(endpoints)):
                ib, sb, pb, tb = endpoints[b]
                if ia == ib:
                    continue
                gap = float(np.linalg.norm(pa[:2] - pb[:2]))
                if gap > 14.0 or gap < 1e-6:
                    continue
                if abs(float(np.dot(ta, tb))) < np.cos(np.deg2rad(58.0)):
                    continue
                pairs.append((gap, ia, sa, ib, sb))
        for gap, ia, sa, ib, sb in sorted(pairs):
            if (ia, sa) in used or (ib, sb) in used:
                continue
            target = 0.5 * (_get_endpoint(out[ia], sa) + _get_endpoint(out[ib], sb))
            _set_endpoint(out[ia], sa, target)
            _set_endpoint(out[ib], sb, target)
            out[ia].metadata["beauty_endpoint_snapped"] = True
            out[ib].metadata["beauty_endpoint_snapped"] = True
            used.add((ia, sa))
            used.add((ib, sb))
            changed = True
        if not changed:
            break
    return out


def _get_endpoint(curve: NURBSCurve, side: int) -> np.ndarray:
    return curve.cvs[0] if side == 0 else curve.cvs[-1]


def _set_endpoint(curve: NURBSCurve, side: int, target: np.ndarray) -> None:
    if side == 0:
        delta = target - curve.cvs[0]
        curve.cvs[0] += delta
        curve.cvs[1] += delta
    else:
        delta = target - curve.cvs[-1]
        curve.cvs[-1] += delta
        curve.cvs[-2] += delta


def _endpoint_dir(curve: NURBSCurve, side: int) -> np.ndarray:
    if side == 0:
        return _unit(curve.cvs[1, :2] - curve.cvs[0, :2])
    return _unit(curve.cvs[-1, :2] - curve.cvs[-2, :2])


def _endpoint_leave_dir(curve: NURBSCurve, side: int) -> np.ndarray:
    if side == 0:
        return _unit(curve.cvs[0, :2] - curve.cvs[1, :2])
    return _unit(curve.cvs[-1, :2] - curve.cvs[-2, :2])


def _endpoint_enter_dir(curve: NURBSCurve, side: int) -> np.ndarray:
    if side == 0:
        return _unit(curve.cvs[1, :2] - curve.cvs[0, :2])
    return _unit(curve.cvs[-2, :2] - curve.cvs[-1, :2])


def _set_endpoint_handle_direction(curve: NURBSCurve, side: int, interior_dir: np.ndarray) -> None:
    direction = _unit(np.asarray(interior_dir, dtype=float))
    if len(curve.cvs) < 2:
        return
    if side == 0:
        length = float(np.linalg.norm(curve.cvs[1, :2] - curve.cvs[0, :2]))
        curve.cvs[1, :2] = curve.cvs[0, :2] + direction * max(length, 1e-6)
    else:
        length = float(np.linalg.norm(curve.cvs[-2, :2] - curve.cvs[-1, :2]))
        curve.cvs[-2, :2] = curve.cvs[-1, :2] + direction * max(length, 1e-6)
    curve.metadata["beauty_handle_aligned"] = True


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arccos(np.clip(float(np.dot(_unit(a), _unit(b))), -1.0, 1.0)))


def _curve_length(curve: NURBSCurve) -> float:
    pts = evaluate_bezier(curve.cvs, np.linspace(0.0, 1.0, 80), curve.weights)
    return float(np.sum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)))


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(n, 1e-9)


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
