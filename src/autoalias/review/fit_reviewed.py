from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autoalias.exporters import write_iges, write_json_bundle, write_svg_preview
from autoalias.geometry.bezier import bernstein_basis, evaluate_bezier, signed_curvature_2d
from autoalias.geometry.fitting import FittingOptions, SingleSpanFitter, _binom
from autoalias.geometry.polyline import chord_length_parameter, resample_polyline, smooth_polyline
from autoalias.geometry.polyline import remove_duplicate_points
from autoalias.models import CurveCandidate, NURBSCurve, QualityReport
from autoalias.quality import ClassAValidator


@dataclass(slots=True)
class ReviewedFitResult:
    out: Path
    curves: list[NURBSCurve]
    reports: list[QualityReport]
    skipped_count: int


@dataclass(slots=True)
class _G2Constraint:
    point: np.ndarray
    d1: np.ndarray
    d2: np.ndarray


G2_SPLIT_MAX_SHIFT_PX = 18.0


def fit_reviewed_annotations(
    annotation_paths: list[str | Path],
    out: str | Path,
    *,
    degree: int | str = "auto",
    min_points: int = 8,
) -> ReviewedFitResult:
    """Fit Alias-ready curves strictly from manually saved design-curve annotations.

    The interactive review JSON is intentionally verbose because it stores the full
    routed skeleton path. This exporter consumes that dense path as a target, but the
    resulting Alias JSON/IGES contains compact single-span NURBS curves only.
    """
    resolved_paths = _expand_annotation_paths(annotation_paths)
    out_path = Path(out).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    validator = ClassAValidator()
    curves: list[NURBSCurve] = []
    reports: list[QualityReport] = []
    candidates: list[CurveCandidate] = []
    skipped = 0
    background_image: str | Path | None = None

    for annotation_path in resolved_paths:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_path = data.get("graph", {}).get("image")
        if background_image is None and image_path:
            background_image = image_path

        for index, design_curve in enumerate(data.get("design_curves", []), start=1):
            raw_segments = _curve_segments(design_curve)
            if not raw_segments:
                skipped += 1
                continue
            segments: list[dict[str, Any]] = []
            for segment in raw_segments:
                points = segment["points"]
                if len(points) < 2:
                    skipped += 1
                    continue
                try:
                    points = remove_duplicate_points(points, eps=0.5)
                    points = _ensure_minimum_fit_points(points)
                    if len(points) < min_points and segment["segment_count"] <= 1:
                        skipped += 1
                        continue
                    if len(points) < 4:
                        skipped += 1
                        continue
                    segments.append({**segment, "points": points})
                except Exception:
                    skipped += 1
            if not segments:
                continue

            fitted = _fit_design_curve_chain(
                design_curve,
                annotation_path,
                index,
                segments,
                degree,
                validator,
            )
            for curve, candidate, report in fitted:
                curves.append(curve)
                candidates.append(candidate)
                reports.append(report)

    write_json_bundle(out_path / "reviewed_curves.json", curves, reports)
    if curves:
        write_iges(out_path / "reviewed_curves.igs", curves)
        write_svg_preview(
            out_path / "reviewed_preview.svg",
            curves,
            candidates=candidates,
            background_image=background_image,
            show_labels=True,
            show_comb=True,
            show_cvs=True,
            show_candidates=True,
        )
        write_svg_preview(
            out_path / "reviewed_clean_preview.svg",
            curves,
            candidates=None,
            background_image=background_image,
            show_labels=False,
            show_comb=False,
            show_cvs=False,
            show_candidates=False,
        )
    return ReviewedFitResult(out=out_path, curves=curves, reports=reports, skipped_count=skipped)


def _expand_annotation_paths(paths: list[str | Path]) -> list[Path]:
    expanded: list[Path] = []
    for path_like in paths:
        text = str(path_like)
        if any(ch in text for ch in "*?["):
            expanded.extend(Path(match).resolve() for match in sorted(glob.glob(text)))
        else:
            expanded.append(Path(text).resolve())
    seen: set[str] = set()
    out: list[Path] = []
    for path in expanded:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            out.append(path)
    if not out:
        raise FileNotFoundError("no reviewed annotation JSON files matched the input path(s)")
    return out


def _fit_lowest_degree(
    candidate: CurveCandidate,
    target_points: np.ndarray,
    degree: int | str,
    validator: ClassAValidator,
) -> NURBSCurve:
    if isinstance(degree, int):
        return SingleSpanFitter(FittingOptions(degree=degree)).fit_candidate(candidate)
    best_curve: NURBSCurve | None = None
    best_score = float("inf")
    for candidate_degree in (3, 4, 5, 6, 7):
        curve = SingleSpanFitter(FittingOptions(degree=candidate_degree)).fit_candidate(candidate)
        report = validator.validate(curve, target_points)
        chamfer = float(report.metrics.get("chamfer_mean", 999.0))
        side = _cv_side_consistency_penalty(curve, target_points)
        warnings = len(report.warnings)
        score = warnings * 1000.0 + chamfer + side * 2.6 + candidate_degree * 0.01
        if report.passed and side < 34.0:
            return curve
        if score < best_score:
            best_score = score
            best_curve = curve
    if best_curve is None:
        raise ValueError("failed to fit any degree")
    return best_curve


def _fit_chain_lowest_degree(
    candidate: CurveCandidate,
    target_points: np.ndarray,
    degree: int | str,
    validator: ClassAValidator,
    *,
    segment_index: int,
    segment_count: int,
    closed: bool,
) -> NURBSCurve:
    if isinstance(degree, int):
        return SingleSpanFitter(FittingOptions(degree=degree)).fit_candidate(candidate)
    start_constrained = closed or segment_index > 1
    end_constrained = closed or segment_index < segment_count
    simplicity = _target_curve_simplicity(target_points)
    # Two-sided G2 needs independent start/end derivative CVs. Degree 3/4 can
    # look clean in isolation, but the two ends fight over the same control
    # points, so middle/closed segments must start at degree 5.
    if start_constrained and end_constrained:
        degrees = (5, 6, 7)
    else:
        degrees = (3, 4, 5, 6, 7)
    best_curve: NURBSCurve | None = None
    best_score = float("inf")
    for candidate_degree in degrees:
        curve = SingleSpanFitter(FittingOptions(degree=candidate_degree)).fit_candidate(candidate)
        report = validator.validate(curve, target_points)
        chamfer = float(report.metrics.get("chamfer_mean", 999.0))
        spacing = float(report.metrics.get("cv_spacing_ratio", 999.0))
        oscillation = float(report.metrics.get("curvature_oscillation", 999.0))
        dent = _cv_dent_penalty(curve.cvs)
        side = _cv_side_consistency_penalty(curve, target_points)
        corridor = _cv_target_corridor_penalty(curve, target_points)
        layout = _cv_layout_penalty(curve.cvs)
        warnings = len(report.warnings)
        if bool(simplicity["simple"]):
            score = (
                warnings * 1600.0
                + max(0.0, chamfer - 5.2) * 55.0
                + max(0.0, spacing - 4.8) * 70.0
                + max(0.0, oscillation - 0.58) * 80.0
                + dent * 0.24
                + side * 2.2
                + corridor * 2.4
                + layout * 0.14
                + candidate_degree * 42.0
            )
            if report.passed and dent < 18.0 and side < 32.0 and corridor < 34.0 and spacing < 5.6:
                curve.metadata["degree_selected_for_simple_curve"] = True
                curve.metadata["simplicity_sinuosity"] = simplicity["sinuosity"]
                curve.metadata["simplicity_max_angle_deg"] = simplicity["max_angle_deg"]
                return curve
        else:
            score = (
                warnings * 1400.0
                + max(0.0, chamfer - 2.8) * 180.0
                + max(0.0, spacing - 3.8) * 85.0
                + max(0.0, oscillation - 0.42) * 180.0
                + dent * 0.28
                + side * 2.7
                + corridor * 2.8
                + layout * 0.18
                + candidate_degree * 0.05
            )
        if (
            report.passed
            and dent < 14.0
            and side < 30.0
            and corridor < 32.0
            and chamfer < 3.2
            and spacing < 4.4
            and oscillation < 0.5
        ):
            return curve
        if score < best_score:
            best_score = score
            best_curve = curve
    if best_curve is None:
        raise ValueError("failed to fit any chain degree")
    return best_curve


def _fit_design_curve_chain(
    design_curve: dict[str, Any],
    annotation_path: Path,
    design_index: int,
    segments: list[dict[str, Any]],
    degree: int | str,
    validator: ClassAValidator,
    *,
    allow_auto_merge: bool = False,
) -> list[tuple[NURBSCurve, CurveCandidate, QualityReport]]:
    """Fit one reviewed curve.

    A design curve with several manual/AI split points is a continuity chain. Its
    exported IGES entities are still one-span Bezier/NURBS curves, but adjacent
    entities share the same endpoint, tangent and curvature vector at every split.
    """
    if len(segments) <= 1:
        out: list[tuple[NURBSCurve, CurveCandidate, QualityReport]] = []
        for segment_index, segment in enumerate(segments, start=1):
            points = segment["points"]
            label = _curve_label(
                design_curve,
                annotation_path,
                design_index,
                segment_index,
                segment["segment_count"],
            )
            candidate = _make_candidate(label, points, annotation_path, design_curve)
            curve = _fit_lowest_degree(candidate, points, degree, validator)
            curve.source = "manual_review_fit"
            curve.metadata = _curve_metadata(
                design_curve,
                annotation_path,
                segment,
                points,
                segment_index,
                fit_policy="split_boundaries_then_single_span_degree_3_to_7",
            )
            out.append((curve, candidate, validator.validate(curve, points)))
        return out

    closed = bool(design_curve.get("closed", False))
    original_segments = [{**seg, "points": np.asarray(seg["points"], dtype=float).copy()} for seg in segments]
    original_segment_count = len(original_segments)
    if allow_auto_merge:
        segments, merge_diagnostics = _merge_smooth_chain_segments(
            segments,
            closed=closed,
            degree=degree,
            validator=validator,
        )
    else:
        merge_diagnostics = {}
    segments, split_diagnostics = _auto_adjust_g2_split_points(
        segments,
        closed=closed,
        max_shift_px=G2_SPLIT_MAX_SHIFT_PX,
    )
    fitted_curves: list[tuple[NURBSCurve, CurveCandidate, np.ndarray, dict[str, Any], int]] = []
    out = []
    for segment_index, segment in enumerate(segments, start=1):
        points = segment["points"]
        label = _curve_label(
            design_curve,
            annotation_path,
            design_index,
            segment_index,
            segment["segment_count"],
        )
        candidate = _make_candidate(label, points, annotation_path, design_curve)
        curve = _fit_chain_lowest_degree(
            candidate,
            points,
            degree,
            validator,
            segment_index=segment_index,
            segment_count=segment["segment_count"],
            closed=closed,
        )
        curve.source = "manual_review_g2_fit"
        curve.metadata = _curve_metadata(
            design_curve,
            annotation_path,
            segment,
            points,
            segment_index,
            fit_policy="manual_split_boundaries_with_lowest_degree_g2_fairing",
        )
        curve.metadata["g2_chain"] = True
        curve.metadata["g2_method"] = "limited_split_adjustment_plus_local_fairness_search"
        curve.metadata["chain_original_segment_count"] = original_segment_count
        curve.metadata["chain_merged_segment_count"] = len(segments)
        curve.metadata["chain_merge"] = merge_diagnostics.get(segment_index - 1, {})
        curve.metadata["g2_split_adjustment"] = split_diagnostics.get(segment_index - 1, {})
        fitted_curves.append((curve, candidate, points, segment, segment_index))

    _apply_g2_endpoint_fairing([item[0] for item in fitted_curves], [item[2] for item in fitted_curves], closed=closed)
    if not isinstance(degree, int):
        _simplify_simple_chain_degrees(fitted_curves, validator, closed=closed)
    _promote_failed_chain_degrees(fitted_curves, validator, closed=closed)
    _repair_cv_side_flips(fitted_curves, validator, closed=closed)
    _apply_g2_endpoint_fairing([item[0] for item in fitted_curves], [item[2] for item in fitted_curves], closed=closed)
    if not isinstance(degree, int):
        _promote_bad_layout_degrees(fitted_curves, validator, closed=closed)
        _apply_g2_endpoint_fairing([item[0] for item in fitted_curves], [item[2] for item in fitted_curves], closed=closed)
    _repair_cv_side_flips(fitted_curves, validator, closed=closed)
    _apply_g2_endpoint_fairing([item[0] for item in fitted_curves], [item[2] for item in fitted_curves], closed=closed)
    if not isinstance(degree, int):
        _promote_bad_layout_degrees(fitted_curves, validator, closed=closed)
        _apply_g2_endpoint_fairing([item[0] for item in fitted_curves], [item[2] for item in fitted_curves], closed=closed)
    for curve, candidate, points, segment, segment_index in fitted_curves:
        _stamp_cv_side_diagnostics(curve, points)
        _stamp_cv_target_corridor_diagnostics(curve, points)
        curve.metadata["g2_start_constrained"] = closed or segment_index > 1
        curve.metadata["g2_end_constrained"] = closed or segment_index < len(fitted_curves)
        out.append((curve, candidate, validator.validate(curve, points)))
    _stamp_g2_diagnostics(out, closed=closed)
    if allow_auto_merge and len(segments) < original_segment_count and any(not item[2].passed for item in out):
        return _fit_design_curve_chain(
            design_curve,
            annotation_path,
            design_index,
            original_segments,
            degree,
            validator,
            allow_auto_merge=False,
        )
    if (
        not isinstance(degree, int)
        and any(not item[2].passed for item in out)
        and any(item[0].degree < 7 for item in out)
    ):
        return _fit_design_curve_chain(
            design_curve,
            annotation_path,
            design_index,
            original_segments,
            7,
            validator,
            allow_auto_merge=False,
        )
    return out


def _make_candidate(
    label: str,
    points: np.ndarray,
    annotation_path: Path,
    design_curve: dict[str, Any],
) -> CurveCandidate:
    return CurveCandidate(
        label=label,
        points=points,
        confidence=1.0,
        source="manual_review",
        metadata={
            "annotation_file": str(annotation_path),
            "annotation_id": design_curve.get("id"),
        },
    )


def _curve_metadata(
    design_curve: dict[str, Any],
    annotation_path: Path,
    segment: dict[str, Any],
    points: np.ndarray,
    segment_index: int,
    *,
    fit_policy: str,
) -> dict[str, Any]:
    return {
        "annotation_file": str(annotation_path),
        "annotation_id": design_curve.get("id"),
        "semantic": design_curve.get("semantic") or "manual_design_curve",
        "closed": bool(design_curve.get("closed", False)),
        "manual_point_count": len(design_curve.get("manual_points") or design_curve.get("cut_points") or []),
        "target_point_count": int(len(points)),
        "boundary_segment_index": segment_index - 1,
        "boundary_segment_count": segment["segment_count"],
        "boundary_start_order": segment.get("start_order"),
        "boundary_end_order": segment.get("end_order"),
        "fit_policy": fit_policy,
    }


def _fit_g2_constrained_segment(
    candidate: CurveCandidate,
    target_points: np.ndarray,
    *,
    start_constraint: _G2Constraint | None,
    end_constraint: _G2Constraint | None,
    requested_degree: int | str,
) -> NURBSCurve:
    # A two-sided G2-constrained Bezier needs enough CVs to keep fit freedom.
    # Degree 7 remains single-span, Alias-friendly, and gives two free interior CVs.
    if isinstance(requested_degree, int):
        degree = max(7 if start_constraint is not None and end_constraint is not None else 5, requested_degree)
    else:
        degree = 7
    degree = min(max(int(degree), 3), 7)
    if start_constraint is not None and end_constraint is not None and degree < 5:
        degree = 5

    pts = _prepare_fit_points(target_points)
    cvs = _fit_fixed_degree_with_g2_constraints(
        pts,
        degree,
        start_constraint=start_constraint,
        end_constraint=end_constraint,
    )
    return NURBSCurve.single_span(
        label=candidate.label,
        degree=degree,
        cvs=cvs,
        confidence=candidate.confidence,
        source=candidate.source,
        metadata={"candidate_points": len(candidate.points)},
    )


def _apply_g2_endpoint_fairing(
    curves: list[NURBSCurve],
    target_segments: list[np.ndarray],
    *,
    closed: bool,
    only_join_indices: set[int] | None = None,
) -> None:
    if len(curves) <= 1:
        return
    join_count = len(curves) if closed and len(curves) > 2 else len(curves) - 1
    for join_index in range(join_count):
        if only_join_indices is not None and join_index not in only_join_indices:
            continue
        left = curves[join_index]
        right = curves[(join_index + 1) % len(curves)]
        left_points = target_segments[join_index]
        right_points = target_segments[(join_index + 1) % len(curves)]
        _fair_single_g2_join(left, right, left_points, right_points)


def _adjacent_join_indices(index: int, curve_count: int, *, closed: bool) -> set[int]:
    if curve_count <= 1:
        return set()
    join_count = curve_count if closed and curve_count > 2 else curve_count - 1
    candidates = {index - 1, index}
    if closed:
        candidates = {item % curve_count for item in candidates}
    return {item for item in candidates if 0 <= item < join_count}


def _promote_failed_chain_degrees(
    fitted_curves: list[tuple[NURBSCurve, CurveCandidate, np.ndarray, dict[str, Any], int]],
    validator: ClassAValidator,
    *,
    closed: bool,
) -> None:
    changed = False
    for index, (curve, candidate, points, segment, segment_index) in enumerate(list(fitted_curves)):
        report = validator.validate(curve, points)
        if report.passed or curve.degree >= 7:
            continue
        best_promoted: NURBSCurve | None = None
        best_score = float("inf")
        for candidate_degree in range(curve.degree + 1, 8):
            try:
                promoted = SingleSpanFitter(FittingOptions(degree=candidate_degree)).fit_candidate(candidate)
                promoted.source = curve.source
                promoted.metadata = dict(curve.metadata)
                promoted.metadata["degree_promoted_after_fairing"] = True
                promoted.metadata["degree_before_promotion"] = curve.degree
                promoted.metadata["degree_after_promotion"] = candidate_degree
                new_report = validator.validate(promoted, points)
                new_chamfer = float(new_report.metrics.get("chamfer_mean", 999.0))
                new_spacing = float(new_report.metrics.get("cv_spacing_ratio", 999.0))
                side = _cv_side_consistency_penalty(promoted, points)
                corridor = _cv_target_corridor_penalty(promoted, points)
                score = (
                    len(new_report.warnings) * 1000.0
                    + new_chamfer * 20.0
                    + max(0.0, new_spacing - 4.0) * 50.0
                    + side * 2.4
                    + corridor * 2.5
                )
                if score < best_score:
                    best_score = score
                    best_promoted = promoted
                if new_report.passed and side < 34.0 and corridor < 36.0:
                    break
            except Exception:
                continue
        if best_promoted is not None:
            fitted_curves[index] = (best_promoted, candidate, points, segment, segment_index)
            changed = True
    if changed:
        _apply_g2_endpoint_fairing(
            [item[0] for item in fitted_curves],
            [item[2] for item in fitted_curves],
            closed=closed,
        )


def _repair_cv_side_flips(
    fitted_curves: list[tuple[NURBSCurve, CurveCandidate, np.ndarray, dict[str, Any], int]],
    validator: ClassAValidator,
    *,
    closed: bool,
) -> None:
    for index, (curve, candidate, points, segment, segment_index) in enumerate(list(fitted_curves)):
        current_side = _cv_side_consistency_penalty(curve, points)
        current_corridor = _cv_target_corridor_penalty(curve, points)
        if current_side <= 48.0 and current_corridor <= 54.0:
            continue
        best_curve = curve
        best_score = _side_repair_score(curve, points, validator)
        start_constrained = closed or index > 0
        end_constrained = closed or index < len(fitted_curves) - 1
        min_degree = 5 if start_constrained and end_constrained else 3
        for candidate_degree in range(min_degree, 8):
            try:
                trial = SingleSpanFitter(FittingOptions(degree=candidate_degree)).fit_candidate(candidate)
                trial.source = curve.source
                trial.metadata = dict(curve.metadata)
                trial.metadata["cv_side_repaired"] = True
                trial.metadata["cv_side_repair_from_degree"] = curve.degree
                trial.metadata["cv_side_repair_to_degree"] = candidate_degree
                score = _side_repair_score(trial, points, validator)
                if score < best_score:
                    best_score = score
                    best_curve = trial
            except Exception:
                continue
        repaired_side = _cv_side_consistency_penalty(best_curve, points)
        repaired_corridor = _cv_target_corridor_penalty(best_curve, points)
        if (
            best_curve is not curve
            and repaired_side + repaired_corridor <= current_side + current_corridor - 32.0
        ):
            fitted_curves[index] = (best_curve, candidate, points, segment, segment_index)


def _side_repair_score(curve: NURBSCurve, points: np.ndarray, validator: ClassAValidator) -> float:
    report = validator.validate(curve, points)
    chamfer = float(report.metrics.get("chamfer_mean", 999.0))
    spacing = float(report.metrics.get("cv_spacing_ratio", 999.0))
    oscillation = float(report.metrics.get("curvature_oscillation", 999.0))
    side = _cv_side_consistency_penalty(curve, points)
    corridor = _cv_target_corridor_penalty(curve, points)
    dent = _cv_dent_penalty(curve.cvs)
    return (
        side * 14.0
        + corridor * 14.0
        + len(report.warnings) * 360.0
        + max(0.0, chamfer - 2.6) * 58.0
        + max(0.0, spacing - 4.2) * 45.0
        + max(0.0, oscillation - 0.52) * 120.0
        + dent * 0.8
        + curve.degree * 4.0
    )


def _promote_bad_layout_degrees(
    fitted_curves: list[tuple[NURBSCurve, CurveCandidate, np.ndarray, dict[str, Any], int]],
    validator: ClassAValidator,
    *,
    closed: bool,
) -> None:
    if not fitted_curves:
        return
    target_segments = [item[2] for item in fitted_curves]
    for index, (curve, candidate, points, segment, segment_index) in enumerate(list(fitted_curves)):
        side = _cv_side_consistency_penalty(curve, points)
        corridor = _cv_target_corridor_penalty(curve, points)
        report = validator.validate(curve, points)
        if curve.degree >= 7:
            continue
        if report.passed and side < 120.0 and corridor < 120.0:
            continue

        start_constrained = closed or index > 0
        end_constrained = closed or index < len(fitted_curves) - 1
        min_degree = 5 if start_constrained and end_constrained else 3
        best_curves: list[NURBSCurve] | None = None
        best_passes = bool(report.passed)
        best_score = _chain_local_quality_score(
            [item[0] for item in fitted_curves],
            target_segments,
            validator,
            index,
            closed=closed,
        )
        for target_degree in range(max(curve.degree + 1, min_degree), 8):
            try:
                trial_curves = [_clone_curve(item[0]) for item in fitted_curves]
                trial = SingleSpanFitter(FittingOptions(degree=target_degree)).fit_candidate(candidate)
                trial.source = curve.source
                trial.metadata = dict(curve.metadata)
                trial.metadata["degree_promoted_for_cv_layout"] = True
                trial.metadata["degree_before_layout_promotion"] = curve.degree
                trial.metadata["degree_after_layout_promotion"] = target_degree
                trial_curves[index] = trial
                _apply_g2_endpoint_fairing(
                    trial_curves,
                    target_segments,
                    closed=closed,
                    only_join_indices=_adjacent_join_indices(index, len(trial_curves), closed=closed),
                )
                trial_report = validator.validate(trial_curves[index], points)
                score = _chain_local_quality_score(trial_curves, target_segments, validator, index, closed=closed)
                score += _cv_side_consistency_penalty(trial_curves[index], points) * 2.8
                score += _cv_target_corridor_penalty(trial_curves[index], points) * 2.8
                if (not best_passes and trial_report.passed) or score < best_score:
                    best_score = score
                    best_passes = bool(trial_report.passed)
                    best_curves = trial_curves
            except Exception:
                continue
        if best_curves is not None:
            for update_index, updated in enumerate(best_curves):
                old = fitted_curves[update_index]
                fitted_curves[update_index] = (updated, old[1], old[2], old[3], old[4])


def _simplify_simple_chain_degrees(
    fitted_curves: list[tuple[NURBSCurve, CurveCandidate, np.ndarray, dict[str, Any], int]],
    validator: ClassAValidator,
    *,
    closed: bool,
) -> None:
    if not fitted_curves:
        return
    for index, (curve, candidate, points, segment, segment_index) in enumerate(list(fitted_curves)):
        if curve.degree <= 5:
            continue
        simple = _target_curve_simplicity(points)
        if not simple["simple"] and not simple["smooth_arc"]:
            continue
        start_constrained = closed or index > 0
        end_constrained = closed or index < len(fitted_curves) - 1
        if bool(simple["simple"]):
            target_degrees = (5,) if start_constrained and end_constrained else (3, 4, 5)
        else:
            target_degrees = (5, 6) if start_constrained and end_constrained else (4, 5, 6)
        baseline_score = _chain_local_quality_score(
            [item[0] for item in fitted_curves],
            [item[2] for item in fitted_curves],
            validator,
            index,
            closed=closed,
        )
        for target_degree in target_degrees:
            if target_degree >= curve.degree:
                continue
            trial_curves = [_clone_curve(item[0]) for item in fitted_curves]
            trial = SingleSpanFitter(FittingOptions(degree=target_degree)).fit_candidate(candidate)
            trial.source = curve.source
            trial.metadata = dict(curve.metadata)
            trial.metadata["degree_simplified_from"] = curve.degree
            trial.metadata["degree_simplified_to"] = target_degree
            trial.metadata["simplicity_sinuosity"] = simple["sinuosity"]
            trial.metadata["simplicity_max_angle_deg"] = simple["max_angle_deg"]
            trial.metadata["simplicity_smooth_arc"] = simple["smooth_arc"]
            trial_curves[index] = trial
            target_segments = [item[2] for item in fitted_curves]
            _apply_g2_endpoint_fairing(
                trial_curves,
                target_segments,
                closed=closed,
                only_join_indices=_adjacent_join_indices(index, len(trial_curves), closed=closed),
            )
            report = validator.validate(trial_curves[index], points)
            if not report.passed:
                continue
            if _cv_dent_penalty(trial_curves[index].cvs) > 14.0:
                continue
            if _cv_side_consistency_penalty(trial_curves[index], points) > 30.0:
                continue
            if _cv_target_corridor_penalty(trial_curves[index], points) > 34.0:
                continue
            trial_score = _chain_local_quality_score(
                trial_curves,
                target_segments,
                validator,
                index,
                closed=closed,
            )
            tolerance = 110.0 if bool(simple["smooth_arc"]) else 18.0
            if trial_score <= baseline_score + tolerance:
                for update_index, updated in enumerate(trial_curves):
                    old = fitted_curves[update_index]
                    fitted_curves[update_index] = (updated, old[1], old[2], old[3], old[4])
                break


def _chain_local_quality_score(
    curves: list[NURBSCurve],
    target_segments: list[np.ndarray],
    validator: ClassAValidator,
    index: int,
    *,
    closed: bool,
) -> float:
    if not curves:
        return 1e9
    indices = {index}
    if len(curves) > 1:
        if index > 0:
            indices.add(index - 1)
        elif closed:
            indices.add(len(curves) - 1)
        if index < len(curves) - 1:
            indices.add(index + 1)
        elif closed:
            indices.add(0)
    score = 0.0
    for item_index in indices:
        report = validator.validate(curves[item_index], target_segments[item_index])
        chamfer = float(report.metrics.get("chamfer_mean", 999.0))
        spacing = float(report.metrics.get("cv_spacing_ratio", 999.0))
        oscillation = float(report.metrics.get("curvature_oscillation", 999.0))
        score += len(report.warnings) * 900.0
        score += max(0.0, chamfer - 2.8) * 110.0
        score += max(0.0, spacing - 4.0) * 65.0
        score += max(0.0, oscillation - 0.48) * 120.0
        score += _cv_dent_penalty(curves[item_index].cvs) * 0.35
        score += _cv_side_consistency_penalty(curves[item_index], target_segments[item_index]) * 1.8
        score += _cv_target_corridor_penalty(curves[item_index], target_segments[item_index]) * 1.9
        score += curves[item_index].degree * 0.08
    return score


def _target_curve_simplicity(points: np.ndarray) -> dict[str, float | bool]:
    pts = remove_duplicate_points(np.asarray(points, dtype=float), eps=0.5)
    if len(pts) < 4:
        return {"simple": True, "smooth_arc": True, "sinuosity": 1.0, "max_angle_deg": 0.0, "sag_ratio": 0.0}
    if len(pts) > 12:
        pts = smooth_polyline(resample_polyline(pts, 90), window=7)
    length = _polyline_length(pts)
    chord = float(np.linalg.norm(pts[-1, :2] - pts[0, :2]))
    if chord < 1e-6:
        return {"simple": False, "smooth_arc": False, "sinuosity": 999.0, "max_angle_deg": 180.0, "sag_ratio": 999.0}
    sinuosity = length / chord
    vec = np.diff(pts[:, :2], axis=0)
    dist = np.linalg.norm(vec, axis=1)
    unit_vec = vec / np.maximum(dist[:, None], 1e-9)
    dots = np.sum(unit_vec[:-1] * unit_vec[1:], axis=1)
    angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0))) if len(dots) else np.zeros(0)
    max_angle = float(np.nanmax(angles)) if len(angles) else 0.0
    chord_vec = pts[-1, :2] - pts[0, :2]
    sag = np.abs(np.cross(chord_vec, pts[:, :2] - pts[0, :2]) / chord)
    sag_ratio = float(np.nanmax(sag) / chord) if len(sag) else 0.0
    simple = bool(sinuosity < 1.045 and max_angle < 15.0 and sag_ratio < 0.04)
    curvature = []
    for idx in range(1, len(pts) - 1):
        curvature.append(_signed_three_point_curvature(pts[idx - 1, :2], pts[idx, :2], pts[idx + 1, :2]))
    curv = np.asarray(curvature, dtype=float)
    curv = curv[np.isfinite(curv)]
    if len(curv):
        curv_abs = np.abs(curv)
        denom = float(np.mean(curv_abs) + 1e-9)
        curv_osc = float(np.std(np.diff(curv)) / denom) if len(curv) > 2 else 0.0
        sign_changes = _count_curve_sign_changes(curv)
    else:
        curv_osc = 0.0
        sign_changes = 0
    # Skeletons extracted from drawings often have tiny alternating curvature
    # signs even on a visually clean arc. For degree selection, a small local
    # direction change is a better signal than raw curvature sign noise.
    smooth_arc = bool(sinuosity < 1.16 and max_angle < 5.0 and (sign_changes <= 2 or curv_osc < 1.35))
    return {
        "simple": simple,
        "smooth_arc": smooth_arc,
        "sinuosity": float(sinuosity),
        "max_angle_deg": max_angle,
        "sag_ratio": sag_ratio,
    }


def _count_curve_sign_changes(values: np.ndarray) -> int:
    if len(values) < 5:
        return 0
    eps = max(float(np.nanmax(np.abs(values))) * 0.08, 1e-9)
    signs = np.sign(np.where(np.abs(values) < eps, 0.0, values))
    nonzero = signs[signs != 0]
    if len(nonzero) < 2:
        return 0
    return int(np.sum(nonzero[1:] * nonzero[:-1] < 0.0))


def _merge_smooth_chain_segments(
    segments: list[dict[str, Any]],
    *,
    closed: bool,
    degree: int | str,
    validator: ClassAValidator,
    max_passes: int = 5,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    if len(segments) <= 2:
        return segments, {}
    current = [{**seg, "points": remove_duplicate_points(np.asarray(seg["points"], dtype=float), eps=0.5)} for seg in segments]
    for _ in range(max_passes):
        if len(current) <= 2:
            break
        merged: list[dict[str, Any]] = []
        changed = False
        index = 0
        while index < len(current):
            if index < len(current) - 1 and _can_merge_chain_pair(
                current[index],
                current[index + 1],
                degree=degree,
                validator=validator,
                closed=closed,
            ):
                merged.append(_merged_chain_segment(current[index], current[index + 1]))
                changed = True
                index += 2
            else:
                merged.append(current[index])
                index += 1
        current = _refresh_segment_counts(merged)
        if not changed:
            break
    diagnostics: dict[int, dict[str, Any]] = {}
    for index, segment in enumerate(current):
        source_count = len(segment.get("source_segment_indices", []))
        if source_count > 1:
            diagnostics[index] = {
                "auto_merged": True,
                "source_segment_count": source_count,
                "source_segment_indices": segment.get("source_segment_indices", []),
            }
    return current, diagnostics


def _can_merge_chain_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    degree: int | str,
    validator: ClassAValidator,
    closed: bool,
) -> bool:
    left_points = np.asarray(left["points"], dtype=float)
    right_points = np.asarray(right["points"], dtype=float)
    if len(left_points) < 4 or len(right_points) < 4:
        return False
    angle = _polyline_join_angle(left_points, right_points)
    if angle > 24.0:
        return False
    combined = remove_duplicate_points(_combine_join_points(left_points, right_points), eps=0.5)
    if len(combined) < 8:
        return False
    if _polyline_length(combined) > 980.0:
        return False
    try:
        candidate = CurveCandidate(label="merge_probe", points=combined, source="merge_probe")
        segment_count = int(left.get("segment_count", 2)) + int(right.get("segment_count", 2)) - 1
        curve = _fit_chain_lowest_degree(
            candidate,
            combined,
            degree,
            validator,
            segment_index=2,
            segment_count=max(segment_count, 3),
            closed=closed,
        )
        report = validator.validate(curve, combined)
        chamfer = float(report.metrics.get("chamfer_mean", 999.0))
        spacing = float(report.metrics.get("cv_spacing_ratio", 999.0))
        oscillation = float(report.metrics.get("curvature_oscillation", 999.0))
        if not report.passed:
            return False
        if chamfer > 3.6:
            return False
        if spacing > 4.2:
            return False
        if bool(report.metrics.get("control_polygon_turnback", False)):
            return False
        if oscillation > 0.5:
            return False
        if _cv_dent_penalty(curve.cvs) > 16.0:
            return False
        if _cv_side_consistency_penalty(curve, combined) > 32.0:
            return False
        if _cv_target_corridor_penalty(curve, combined) > 36.0:
            return False
        return True
    except Exception:
        return False


def _merged_chain_segment(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_sources = list(left.get("source_segment_indices") or [left.get("start_order", 0)])
    right_sources = list(right.get("source_segment_indices") or [right.get("start_order", 0)])
    return {
        **left,
        "points": remove_duplicate_points(
            _combine_join_points(np.asarray(left["points"], dtype=float), np.asarray(right["points"], dtype=float)),
            eps=0.5,
        ),
        "end_order": right.get("end_order"),
        "source_segment_indices": left_sources + right_sources,
    }


def _refresh_segment_counts(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    count = len(segments)
    return [{**seg, "segment_count": count} for seg in segments]


def _polyline_join_angle(left: np.ndarray, right: np.ndarray) -> float:
    left_len = _polyline_length(left)
    right_len = _polyline_length(right)
    point = 0.5 * (left[-1, :2] + right[0, :2])
    before = _point_before_end(left, min(max(left_len * 0.12, 6.0), 34.0))[:2]
    after = _point_after_start(right, min(max(right_len * 0.12, 6.0), 34.0))[:2]
    return _angle_between_2d(point - before, after - point)


def _auto_adjust_g2_split_points(
    segments: list[dict[str, Any]],
    *,
    closed: bool,
    max_shift_px: float,
    min_segment_points: int = 8,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Move split boundaries slightly along the routed skeleton to improve G2 fairing.

    The user/AI split point remains the design intent, but the exported boundary may
    slide a small distance along the same skeleton path when the exact picked point
    creates a bad curvature-comb transition. This does not add curve entities.
    """
    if len(segments) <= 1:
        return segments, {}

    adjusted = [{**seg, "points": np.asarray(seg["points"], dtype=float).copy()} for seg in segments]
    diagnostics: dict[int, dict[str, Any]] = {}
    join_count = len(adjusted) if closed and len(adjusted) > 2 else len(adjusted) - 1

    for join_index in range(join_count):
        left_index = join_index
        right_index = (join_index + 1) % len(adjusted)
        left = remove_duplicate_points(adjusted[left_index]["points"], eps=0.5)
        right = remove_duplicate_points(adjusted[right_index]["points"], eps=0.5)
        if len(left) < min_segment_points or len(right) < min_segment_points:
            continue

        combined = _combine_join_points(left, right)
        if len(combined) < min_segment_points * 2:
            continue
        original = len(left) - 1
        best = _best_g2_split_index(
            combined,
            original_index=original,
            max_shift_px=max_shift_px,
            min_left=min_segment_points,
            min_right=min_segment_points,
        )
        if best is None:
            continue
        split_index, score, shift = best
        if split_index == original:
            diagnostics[left_index] = {
                "join_after_segment": left_index,
                "shift_px": 0.0,
                "score": round(float(score), 6),
            }
            continue
        new_left = combined[: split_index + 1]
        new_right = combined[split_index:]
        if len(new_left) < min_segment_points or len(new_right) < min_segment_points:
            continue

        adjusted[left_index]["points"] = new_left
        adjusted[right_index]["points"] = new_right
        payload = {
            "join_after_segment": left_index,
            "shift_px": round(float(shift), 3),
            "max_shift_px": float(max_shift_px),
            "score": round(float(score), 6),
            "auto_adjusted": True,
        }
        diagnostics[left_index] = payload
        diagnostics[right_index] = {
            "join_before_segment": left_index,
            "shift_px": round(float(shift), 3),
            "max_shift_px": float(max_shift_px),
            "score": round(float(score), 6),
            "auto_adjusted": True,
        }
    return adjusted, diagnostics


def _combine_join_points(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0:
        return right
    if len(right) == 0:
        return left
    if np.linalg.norm(left[-1, :2] - right[0, :2]) < 1.5:
        join = 0.5 * (left[-1] + right[0])
        return np.vstack([left[:-1], join, right[1:]])
    return np.vstack([left, right])


def _best_g2_split_index(
    points: np.ndarray,
    *,
    original_index: int,
    max_shift_px: float,
    min_left: int,
    min_right: int,
) -> tuple[int, float, float] | None:
    s = _arc_lengths(points)
    if original_index <= 0 or original_index >= len(points) - 1:
        return None
    original_s = float(s[original_index])
    candidates: list[tuple[int, float, float]] = []
    for idx in range(min_left - 1, len(points) - min_right + 1):
        shift = abs(float(s[idx]) - original_s)
        if shift > max_shift_px:
            continue
        score = _g2_split_score(points, idx, shift=shift, max_shift_px=max_shift_px)
        candidates.append((idx, score, shift))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[1])
    top: dict[int, tuple[int, float, float]] = {}
    for item in candidates[:8]:
        top[item[0]] = item
    original = min(candidates, key=lambda item: abs(item[0] - original_index))
    top[original[0]] = original

    scored: list[tuple[int, float, float]] = []
    for idx, local_score, shift in top.values():
        fit_score = _g2_split_fit_score(
            points,
            idx,
            local_score=local_score,
            shift=shift,
            max_shift_px=max_shift_px,
            min_left=min_left,
            min_right=min_right,
        )
        scored.append((idx, fit_score, shift))
    return min(scored, key=lambda item: item[1])


def _g2_split_score(points: np.ndarray, idx: int, *, shift: float, max_shift_px: float) -> float:
    point = points[idx]
    total_len = max(_polyline_length(points), 1.0)
    look = min(max(total_len * 0.045, 8.0), 36.0)
    before = _point_at_arc(points, max(0.0, _arc_lengths(points)[idx] - look))
    after = _point_at_arc(points, min(total_len, _arc_lengths(points)[idx] + look))
    t_left = _unit(point[:2] - before[:2])
    t_right = _unit(after[:2] - point[:2])
    if np.linalg.norm(t_left) < 1e-9 or np.linalg.norm(t_right) < 1e-9:
        return 1e9
    tangent_angle = _angle_between_2d(t_left, t_right)

    k_left = _local_polyline_curvature(points, idx, side="left")
    k_right = _local_polyline_curvature(points, idx, side="right")
    k_delta = abs(k_left - k_right)
    k_peak = max(abs(k_left), abs(k_right))
    shift_penalty = (shift / max(max_shift_px, 1.0)) ** 2

    # Prefer places where the skeleton itself already has a smooth tangent and
    # similar local curvature. Shift penalty keeps the user's point authoritative.
    return (
        tangent_angle * tangent_angle * 6.0
        + min(k_delta * 25000.0, 80.0)
        + min(k_peak * 1800.0, 35.0)
        + shift_penalty * 18.0
    )


def _g2_split_fit_score(
    points: np.ndarray,
    idx: int,
    *,
    local_score: float,
    shift: float,
    max_shift_px: float,
    min_left: int,
    min_right: int,
) -> float:
    left_points = points[: idx + 1]
    right_points = points[idx:]
    if len(left_points) < min_left or len(right_points) < min_right:
        return 1e9
    try:
        fitter = SingleSpanFitter(
            FittingOptions(
                degree=7,
                sample_count=96,
                fair_lambda=0.035,
                jerk_lambda=0.009,
                max_reweight_iters=1,
            )
        )
        left = fitter.fit_candidate(CurveCandidate(label="split_left_probe", points=left_points))
        right = fitter.fit_candidate(CurveCandidate(label="split_right_probe", points=right_points))
        raw_tangent_angle = _angle_between(_bezier_d1_end(left), _bezier_d1_start(right))
        raw_curv_delta = abs(_bezier_curvature_end(left) - _bezier_curvature_start(right))

        fair_left = _clone_curve(left)
        fair_right = _clone_curve(right)
        _apply_g2_join_variant(
            fair_left,
            fair_right,
            left_points,
            right_points,
            tangent_weight=0.82,
            handle_scale=0.92,
            curvature_scale=0.42,
        )

        fair_error = _bezier_mean_error(fair_left, left_points) + _bezier_mean_error(fair_right, right_points)
        layout = _cv_layout_penalty(fair_left.cvs) + _cv_layout_penalty(fair_right.cvs)
        side = _cv_side_consistency_penalty(fair_left, left_points) + _cv_side_consistency_penalty(fair_right, right_points)
        corridor = _cv_target_corridor_penalty(fair_left, left_points) + _cv_target_corridor_penalty(fair_right, right_points)
        comb = _join_curvature_comb_penalty(fair_left, fair_right)
        flow_shape = _join_curvature_flow_penalty(fair_left, fair_right, left_points, right_points)
        collapse = _join_curvature_collapse_penalty(fair_left, fair_right, left_points, right_points)
        shift_penalty = (shift / max(max_shift_px, 1.0)) ** 2
        return (
            local_score * 0.35
            + raw_tangent_angle * raw_tangent_angle * 0.18
            + min(raw_curv_delta * 28000.0, 70.0)
            + min(fair_error * 1.7, 120.0)
            + min(layout * 4.0, 90.0)
            + min(side * 2.0, 140.0)
            + min(corridor * 2.1, 150.0)
            + min(comb, 90.0)
            + min(flow_shape, 160.0)
            + min(collapse, 220.0)
            + shift_penalty * 16.0
        )
    except Exception:
        return 1e9


def _clone_curve(curve: NURBSCurve) -> NURBSCurve:
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


def _bezier_mean_error(curve: NURBSCurve, points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    pts = remove_duplicate_points(points, eps=0.5)
    u = chord_length_parameter(pts)
    sampled = evaluate_bezier(curve.cvs, u, curve.weights)
    return float(np.mean(np.linalg.norm(sampled[:, :2] - pts[:, :2], axis=1)))


def _cv_layout_penalty(cvs: np.ndarray) -> float:
    if len(cvs) < 4:
        return 0.0
    vec = np.diff(cvs[:, :2], axis=0)
    dist = np.linalg.norm(vec, axis=1)
    valid = dist[dist > 1e-6]
    if len(valid) < 2:
        return 100.0
    spacing_ratio = float(np.max(valid) / max(np.min(valid), 1e-6))
    penalty = max(0.0, spacing_ratio - 4.0) * 2.0
    unit_vec = vec / np.maximum(dist[:, None], 1e-9)
    dots = np.sum(unit_vec[:-1] * unit_vec[1:], axis=1)
    turnbacks = np.clip(-dots - 0.05, 0.0, None)
    penalty += float(np.sum(turnbacks) * 25.0)
    angle_changes = np.arccos(np.clip(dots, -1.0, 1.0))
    if len(angle_changes) >= 2:
        penalty += float(np.mean(np.abs(np.diff(angle_changes))) * 4.0)
    return penalty


def _cv_dent_penalty(cvs: np.ndarray) -> float:
    pts = np.asarray(cvs[:, :2], dtype=float)
    if len(pts) < 4:
        return 0.0
    vec = np.diff(pts, axis=0)
    dist = np.linalg.norm(vec, axis=1)
    if np.count_nonzero(dist > 1e-6) < 3:
        return 80.0
    unit_vec = vec / np.maximum(dist[:, None], 1e-9)
    dots = np.sum(unit_vec[:-1] * unit_vec[1:], axis=1)
    angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    penalty = float(np.sum(np.clip(angles - 34.0, 0.0, None) ** 1.35) * 0.42)
    if len(angles) >= 2:
        penalty += float(np.sum(np.clip(np.abs(np.diff(angles)) - 18.0, 0.0, None)) * 0.9)

    local_ratios = []
    for idx in range(1, len(pts) - 1):
        a = pts[idx - 1]
        b = pts[idx]
        c = pts[idx + 1]
        ac = c - a
        ac_len = float(np.linalg.norm(ac))
        if ac_len < 1e-6:
            penalty += 35.0
            continue
        sag = abs(float(np.cross(ac, b - a))) / ac_len
        local_scale = max(min(float(np.linalg.norm(b - a)), float(np.linalg.norm(c - b))), 1.0)
        local_ratios.append(sag / local_scale)
    if local_ratios:
        ratios = np.asarray(local_ratios, dtype=float)
        penalty += float(np.sum(np.clip(ratios - 0.62, 0.0, None)) * 18.0)
        if len(ratios) >= 2:
            penalty += float(np.sum(np.clip(np.abs(np.diff(ratios)) - 0.36, 0.0, None)) * 10.0)

    signs = []
    chord = pts[-1] - pts[0]
    chord_len = float(np.linalg.norm(chord))
    if chord_len > 1e-6:
        for p in pts[1:-1]:
            side = float(np.cross(chord, p - pts[0]) / chord_len)
            if abs(side) > max(chord_len * 0.012, 0.8):
                signs.append(np.sign(side))
    if len(signs) >= 2:
        penalty += float(np.sum(np.asarray(signs[1:]) * np.asarray(signs[:-1]) < 0.0) * 18.0)
    return penalty


def _cv_side_consistency_penalty(curve: NURBSCurve, target_points: np.ndarray) -> float:
    return float(_cv_side_consistency_metrics(curve, target_points)["penalty"])


def _stamp_cv_side_diagnostics(curve: NURBSCurve, target_points: np.ndarray) -> None:
    metrics = _cv_side_consistency_metrics(curve, target_points)
    curve.metadata["cv_side_penalty"] = round(float(metrics["penalty"]), 4)
    curve.metadata["cv_side_switches"] = int(metrics["side_switches"])
    curve.metadata["cv_side_allowed_switches"] = int(metrics["allowed_switches"])
    curve.metadata["cv_target_is_s_curve"] = bool(metrics["target_is_s_curve"])


def _cv_target_corridor_penalty(curve: NURBSCurve, target_points: np.ndarray) -> float:
    return float(_cv_target_corridor_metrics(curve, target_points)["penalty"])


def _stamp_cv_target_corridor_diagnostics(curve: NURBSCurve, target_points: np.ndarray) -> None:
    metrics = _cv_target_corridor_metrics(curve, target_points)
    curve.metadata["cv_target_corridor_penalty"] = round(float(metrics["penalty"]), 4)
    curve.metadata["cv_target_side_switches"] = int(metrics["target_side_switches"])
    curve.metadata["cv_target_wrong_side_count"] = int(metrics["wrong_side_count"])
    curve.metadata["cv_target_max_normal_offset"] = round(float(metrics["max_normal_offset"]), 4)
    curve.metadata["cv_target_max_tangent_offset"] = round(float(metrics["max_tangent_offset"]), 4)


def _cv_target_corridor_metrics(curve: NURBSCurve, target_points: np.ndarray) -> dict[str, float | int]:
    cvs = np.asarray(curve.cvs[:, :2], dtype=float)
    target = remove_duplicate_points(np.asarray(target_points, dtype=float), eps=0.5)
    length = _polyline_length(target)
    if len(cvs) < 4 or length < 1e-6:
        return {
            "penalty": 0.0,
            "target_side_switches": 0,
            "wrong_side_count": 0,
            "max_normal_offset": 0.0,
            "max_tangent_offset": 0.0,
        }

    signature = _target_curvature_signature(target)
    target_is_s_curve = bool(signature["s_like"])
    dominant_side = _dominant_target_curvature_side(signature)
    chord = float(np.linalg.norm(target[-1, :2] - target[0, :2]))
    if chord > 1e-6:
        sag = float(np.nanmax(np.abs((target[-1, 0] - target[0, 0]) * (target[:, 1] - target[0, 1]) - (target[-1, 1] - target[0, 1]) * (target[:, 0] - target[0, 0])) / chord))
    else:
        sag = 0.0
    normal_limit = max(8.0, min(length * 0.20, max(length * 0.075, sag * 1.7 + 4.0)))
    tangent_limit = max(8.0, min(length * 0.14, length * 0.065 + 8.0))
    side_eps = max(length * 0.005, 0.75)

    signed_values: list[float] = []
    tangent_offsets: list[float] = []
    active_signs: list[int] = []
    wrong_side_count = 0
    penalty = 0.0

    for cv_index in range(1, len(cvs) - 1):
        fraction = cv_index / float(len(cvs) - 1)
        ref_point, tangent = _target_point_tangent_at_fraction(target, fraction)
        if np.linalg.norm(tangent) < 1e-9:
            continue
        delta = cvs[cv_index] - ref_point
        signed = float(tangent[0] * delta[1] - tangent[1] * delta[0])
        along = float(np.dot(delta, tangent))
        signed_values.append(signed)
        tangent_offsets.append(along)

        if abs(signed) > side_eps:
            sign = 1 if signed > 0.0 else -1
            active_signs.append(sign)
            if dominant_side and not target_is_s_curve and sign != dominant_side:
                wrong_side_count += 1
                penalty += 54.0 + min(abs(signed) / max(normal_limit, 1.0), 2.0) * 18.0
        else:
            active_signs.append(0)

        penalty += max(0.0, abs(signed) - normal_limit) / max(normal_limit, 1.0) * 42.0
        penalty += max(0.0, abs(along) - tangent_limit) / max(tangent_limit, 1.0) * 32.0

    nonzero = [sign for sign in active_signs if sign != 0]
    side_switches = int(sum(a * b < 0 for a, b in zip(nonzero[:-1], nonzero[1:]))) if len(nonzero) >= 2 else 0
    allowed_switches = 1 if target_is_s_curve else 0
    penalty += max(0, side_switches - allowed_switches) * 95.0

    if len(signed_values) >= 4:
        signed_arr = np.asarray(signed_values, dtype=float)
        tangent_arr = np.asarray(tangent_offsets, dtype=float)
        normal_second = np.diff(signed_arr, n=2)
        tangent_second = np.diff(tangent_arr, n=2)
        penalty += max(0.0, float(np.mean(np.abs(normal_second))) - normal_limit * 0.34) / max(normal_limit, 1.0) * 38.0
        penalty += max(0.0, float(np.mean(np.abs(tangent_second))) - tangent_limit * 0.32) / max(tangent_limit, 1.0) * 26.0

    penalty += _endpoint_target_boundary_penalty(cvs, target, length)
    max_normal = float(np.nanmax(np.abs(signed_values))) if signed_values else 0.0
    max_tangent = float(np.nanmax(np.abs(tangent_offsets))) if tangent_offsets else 0.0
    return {
        "penalty": min(float(penalty), 360.0),
        "target_side_switches": side_switches,
        "wrong_side_count": wrong_side_count,
        "max_normal_offset": max_normal,
        "max_tangent_offset": max_tangent,
    }


def _dominant_target_curvature_side(signature: dict[str, float | int | bool]) -> int:
    positive = int(signature.get("positive", 0))
    negative = int(signature.get("negative", 0))
    if positive >= max(3, negative * 2):
        return 1
    if negative >= max(3, positive * 2):
        return -1
    return 0


def _target_point_tangent_at_fraction(points: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    length = max(_polyline_length(points), 1e-9)
    distance = float(np.clip(fraction, 0.0, 1.0)) * length
    look = min(max(length * 0.018, 2.0), 16.0)
    point = _point_at_arc(points, distance)[:2]
    before = _point_at_arc(points, max(0.0, distance - look))[:2]
    after = _point_at_arc(points, min(length, distance + look))[:2]
    tangent = _unit(after - before)
    if np.linalg.norm(tangent) < 1e-9:
        tangent = _unit(points[-1, :2] - points[0, :2])
    return point, tangent


def _endpoint_target_boundary_penalty(cvs: np.ndarray, target: np.ndarray, length: float) -> float:
    if len(cvs) < 4:
        return 0.0
    start, start_tangent = _target_point_tangent_at_fraction(target, 0.0)
    end, end_tangent = _target_point_tangent_at_fraction(target, 1.0)
    if np.linalg.norm(start_tangent) < 1e-9 or np.linalg.norm(end_tangent) < 1e-9:
        return 0.0
    tol = max(length * 0.018, 1.4)
    normal_limit = max(length * 0.10, 7.0)
    penalty = 0.0

    start_indices = [idx for idx in (1, 2) if idx < len(cvs) - 1]
    start_proj = [0.0]
    for idx in start_indices:
        delta = cvs[idx] - start
        forward = float(np.dot(delta, start_tangent))
        normal = abs(float(start_tangent[0] * delta[1] - start_tangent[1] * delta[0]))
        start_proj.append(forward)
        penalty += max(0.0, -forward - tol) / tol * 38.0
        penalty += max(0.0, normal - normal_limit) / max(normal_limit, 1.0) * 22.0
    if len(start_proj) >= 3:
        penalty += float(np.sum(np.clip(-(np.diff(start_proj)) - tol * 0.25, 0.0, None)) / tol * 18.0)

    end_indices = [idx for idx in (-2, -3) if abs(idx) <= len(cvs)]
    end_proj = [0.0]
    for idx in end_indices:
        delta = end - cvs[idx]
        backward = float(np.dot(delta, end_tangent))
        normal = abs(float(end_tangent[0] * delta[1] - end_tangent[1] * delta[0]))
        end_proj.append(backward)
        penalty += max(0.0, -backward - tol) / tol * 38.0
        penalty += max(0.0, normal - normal_limit) / max(normal_limit, 1.0) * 22.0
    if len(end_proj) >= 3:
        penalty += float(np.sum(np.clip(-(np.diff(end_proj)) - tol * 0.25, 0.0, None)) / tol * 18.0)
    return penalty


def _cv_side_consistency_metrics(curve: NURBSCurve, target_points: np.ndarray) -> dict[str, float | int | bool]:
    cvs = np.asarray(curve.cvs[:, :2], dtype=float)
    target = remove_duplicate_points(np.asarray(target_points, dtype=float), eps=0.5)
    length = _polyline_length(target)
    if len(cvs) < 4 or length < 1e-6:
        return {
            "penalty": 0.0,
            "side_switches": 0,
            "allowed_switches": 0,
            "active_cv_count": 0,
            "target_is_s_curve": False,
        }

    sample_count = max(180, len(cvs) * 44)
    u_dense = np.linspace(0.0, 1.0, sample_count)
    sampled = evaluate_bezier(curve.cvs, u_dense, curve.weights)[:, :2]
    if not np.all(np.isfinite(sampled)):
        return {
            "penalty": 220.0,
            "side_switches": 99,
            "allowed_switches": 0,
            "active_cv_count": len(cvs) - 2,
            "target_is_s_curve": False,
        }

    cv_u = np.linspace(0.0, 1.0, len(cvs))
    signed_distances: list[float] = []
    active_signs: list[int] = []
    side_eps = max(length * 0.0045, 0.7)
    for cv_index in range(1, len(cvs) - 1):
        sample_index = int(round(float(cv_u[cv_index]) * (sample_count - 1)))
        lo = max(0, sample_index - 2)
        hi = min(sample_count - 1, sample_index + 2)
        tangent = sampled[hi] - sampled[lo]
        tangent_len = float(np.linalg.norm(tangent))
        if tangent_len < 1e-9:
            continue
        delta = cvs[cv_index] - sampled[sample_index]
        signed = float((tangent[0] * delta[1] - tangent[1] * delta[0]) / tangent_len)
        signed_distances.append(signed)
        if abs(signed) > side_eps:
            active_signs.append(1 if signed > 0.0 else -1)
        else:
            active_signs.append(0)

    nonzero = [sign for sign in active_signs if sign != 0]
    if len(nonzero) < 2:
        return {
            "penalty": 0.0,
            "side_switches": 0,
            "allowed_switches": 0,
            "active_cv_count": len(nonzero),
            "target_is_s_curve": False,
        }

    side_switches = int(sum(a * b < 0 for a, b in zip(nonzero[:-1], nonzero[1:])))
    target_signature = _target_curvature_signature(target)
    target_is_s_curve = bool(target_signature["s_like"])
    allowed_switches = 1 if target_is_s_curve else 0
    positive = int(sum(sign > 0 for sign in nonzero))
    negative = int(sum(sign < 0 for sign in nonzero))
    weaker_side_count = min(positive, negative)

    penalty = float(max(0, side_switches - allowed_switches) * 105.0)
    if not target_is_s_curve and weaker_side_count > 0:
        penalty += float(weaker_side_count * 38.0 + min(weaker_side_count, 2) * 24.0)

    signed_arr = np.asarray(signed_distances, dtype=float)
    if len(signed_arr) >= 4:
        amplitude = max(float(np.percentile(np.abs(signed_arr), 75)), side_eps)
        second = np.diff(signed_arr, n=2)
        roughness = float(np.mean(np.abs(second)) / max(amplitude, 1e-6))
        penalty += max(0.0, roughness - 0.85) * 20.0
        kink_limit = max(side_eps * 4.0, amplitude * 1.15)
        penalty += float(np.sum(np.clip(np.abs(second) - kink_limit, 0.0, None)) / max(length * 0.018, 1.0) * 9.0)

    return {
        "penalty": min(penalty, 320.0),
        "side_switches": side_switches,
        "allowed_switches": allowed_switches,
        "active_cv_count": len(nonzero),
        "target_is_s_curve": target_is_s_curve,
    }


def _target_curvature_signature(points: np.ndarray) -> dict[str, float | int | bool]:
    pts = remove_duplicate_points(np.asarray(points, dtype=float), eps=0.5)
    if len(pts) < 7:
        return {"sign_changes": 0, "positive": 0, "negative": 0, "s_like": False}
    if len(pts) > 120:
        pts = smooth_polyline(resample_polyline(pts, 120), window=7)
    curvature = []
    for idx in range(1, len(pts) - 1):
        curvature.append(_signed_three_point_curvature(pts[idx - 1, :2], pts[idx, :2], pts[idx + 1, :2]))
    values = np.asarray(curvature, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 5:
        return {"sign_changes": 0, "positive": 0, "negative": 0, "s_like": False}
    eps = max(float(np.nanmax(np.abs(values))) * 0.10, 1e-7)
    signs = np.sign(np.where(np.abs(values) < eps, 0.0, values))
    nonzero = signs[signs != 0]
    if len(nonzero) < 3:
        return {"sign_changes": 0, "positive": 0, "negative": 0, "s_like": False}
    positive = int(np.sum(nonzero > 0))
    negative = int(np.sum(nonzero < 0))
    sign_changes = int(np.sum(nonzero[1:] * nonzero[:-1] < 0.0))
    smaller_fraction = min(positive, negative) / max(positive + negative, 1)
    s_like = bool(sign_changes >= 1 and smaller_fraction >= 0.18)
    return {
        "sign_changes": sign_changes,
        "positive": positive,
        "negative": negative,
        "s_like": s_like,
    }


def _endpoint_cv_dent_penalty(curve: NURBSCurve, *, side: str) -> float:
    cvs = curve.cvs[-5:] if side == "end" else curve.cvs[:5]
    if len(cvs) < 4:
        return 0.0
    penalty = _cv_dent_penalty(cvs) * 1.35
    pts = cvs[:, :2]
    axis = _unit(pts[-1] - pts[0])
    if np.linalg.norm(axis) > 1e-9:
        proj = pts @ axis
        backwards = np.clip(-(np.diff(proj)), 0.0, None)
        penalty += float(np.sum(backwards) * 4.5)
    vec = np.diff(pts, axis=0)
    dist = np.linalg.norm(vec, axis=1)
    positive = dist[dist > 1e-6]
    if len(positive) >= 2:
        ratio = float(np.max(positive) / max(np.min(positive), 1e-6))
        penalty += max(0.0, ratio - 3.4) * 7.0
    return penalty


def _join_curvature_comb_penalty(left: NURBSCurve, right: NURBSCurve) -> float:
    try:
        u_left = np.linspace(0.58, 1.0, 54)
        u_right = np.linspace(0.0, 0.42, 54)
        k_left = signed_curvature_2d(left.cvs, u_left)
        k_right = signed_curvature_2d(right.cvs, u_right)
        k = np.concatenate([k_left, k_right])
        k = k[np.isfinite(k)]
        if len(k) < 6:
            return 0.0
        dk = np.diff(k)
        d2k = np.diff(k, n=2) if len(k) >= 3 else np.zeros(0, dtype=float)
        join = len(k_left) - 1
        local_jump = abs(float(k[join + 1] - k[join])) if join + 1 < len(k) else 0.0
        max_jump = float(np.max(np.abs(dk))) if len(dk) else 0.0
        wiggle = float(np.mean(np.abs(d2k))) if len(d2k) else 0.0
        peak = float(np.nanmax(np.abs(k)))
        end_peak = max(abs(float(k_left[-1])), abs(float(k_right[0])))
        return min(
            local_jump * 130000.0
            + max_jump * 36000.0
            + wiggle * 62000.0
            + max(0.0, end_peak - peak * 0.82) * 1600.0,
            180.0,
        )
    except Exception:
        return 120.0


def _join_curvature_flow_penalty(
    left: NURBSCurve,
    right: NURBSCurve,
    left_points: np.ndarray,
    right_points: np.ndarray,
) -> float:
    try:
        u_left = np.linspace(0.45, 1.0, 76)
        u_right = np.linspace(0.0, 0.55, 76)
        k_left = signed_curvature_2d(left.cvs, u_left)
        k_right = signed_curvature_2d(right.cvs, u_right)
        k = np.concatenate([k_left, k_right])
        k = k[np.isfinite(k)]
        if len(k) < 24:
            return 0.0

        target = _combine_join_points(np.asarray(left_points, dtype=float), np.asarray(right_points, dtype=float))
        target_is_s = bool(_target_curvature_signature(target).get("s_like", False))
        peak = float(np.nanpercentile(np.abs(k), 92))
        if peak < 1e-8:
            return 0.0

        eps = max(peak * 0.055, 1e-8)
        signs = np.sign(np.where(np.abs(k) < eps, 0.0, k))
        nonzero = signs[signs != 0]
        sign_changes = int(np.sum(nonzero[1:] * nonzero[:-1] < 0.0)) if len(nonzero) >= 2 else 0

        center = len(k_left) - 1
        k_abs = np.abs(k)
        smooth = _smooth_1d(k_abs, window=7)
        join_band = smooth[max(0, center - 2) : min(len(smooth), center + 4)]
        join_value = float(np.mean(join_band)) if len(join_band) else float(smooth[center])

        left_band = smooth[max(0, int(len(k_left) * 0.22)) : max(1, int(len(k_left) * 0.82))]
        right_band = smooth[len(k_left) + int(len(k_right) * 0.18) : len(k_left) + max(1, int(len(k_right) * 0.78))]
        left_ref = float(np.nanmedian(left_band)) if len(left_band) else join_value
        right_ref = float(np.nanmedian(right_band)) if len(right_band) else join_value
        low_ref = min(left_ref, right_ref)
        high_ref = max(left_ref, right_ref)

        penalty = 0.0
        if not target_is_s and sign_changes:
            penalty += max(0, sign_changes - 0) * 150.0

        # A Class-A blend should not pinch at the split when both neighboring
        # sides already carry meaningful curvature in the same direction.
        if low_ref > peak * 0.22 and join_value < low_ref * 0.68:
            penalty += (low_ref * 0.68 - join_value) / max(low_ref, 1e-9) * 360.0
        if high_ref > peak * 0.25 and join_value > high_ref * 1.65:
            penalty += (join_value - high_ref * 1.65) / max(high_ref, 1e-9) * 220.0

        local = smooth[max(0, center - 10) : min(len(smooth), center + 11)]
        if len(local) >= 7:
            local_p75 = float(np.nanpercentile(local, 75))
            local_p20 = float(np.nanpercentile(local, 20))
            if join_value <= local_p20 and local_p75 > join_value * 1.42 and local_p75 > peak * 0.18:
                penalty += (local_p75 - join_value) / max(local_p75, 1e-9) * 250.0

        dk = np.diff(smooth)
        if len(dk) >= 6:
            dk_eps = max(float(np.nanpercentile(np.abs(dk), 70)) * 0.18, peak * 0.01, 1e-10)
            dk_sign = np.sign(np.where(np.abs(dk) < dk_eps, 0.0, dk))
            nz = dk_sign[dk_sign != 0]
            extrema = int(np.sum(nz[1:] * nz[:-1] < 0.0)) if len(nz) >= 2 else 0
            penalty += max(0, extrema - (2 if target_is_s else 1)) * 42.0

        d2 = np.diff(smooth, n=2)
        if len(d2):
            jerk = float(np.nanmean(np.abs(d2)) / max(peak, 1e-9))
            penalty += max(0.0, jerk - 0.018) * 850.0
        return min(float(penalty), 720.0)
    except Exception:
        return 260.0


def _smooth_1d(values: np.ndarray, *, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 3 or window <= 1:
        return arr
    width = min(int(window), len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
    width = max(width, 3)
    kernel = np.ones(width, dtype=float) / float(width)
    pad = width // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _local_polyline_curvature(points: np.ndarray, idx: int, *, side: str) -> float:
    count = len(points)
    if count < 5:
        return 0.0
    if side == "left":
        a = max(0, idx - 5)
        b = max(0, idx - 2)
        c = idx
    else:
        a = idx
        b = min(count - 1, idx + 2)
        c = min(count - 1, idx + 5)
    if len({a, b, c}) < 3:
        return 0.0
    return _signed_three_point_curvature(points[a, :2], points[b, :2], points[c, :2])


def _arc_lengths(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(len(points), dtype=float)
    d = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def _point_at_arc(points: np.ndarray, distance: float) -> np.ndarray:
    s = _arc_lengths(points)
    if len(points) == 0:
        return np.zeros(3, dtype=float)
    if distance <= 0.0:
        return points[0]
    if distance >= s[-1]:
        return points[-1]
    idx = int(np.searchsorted(s, distance, side="right"))
    idx = max(1, min(idx, len(points) - 1))
    denom = max(float(s[idx] - s[idx - 1]), 1e-9)
    t = (distance - float(s[idx - 1])) / denom
    return points[idx - 1] * (1.0 - t) + points[idx] * t


def _angle_between_2d(a: np.ndarray, b: np.ndarray) -> float:
    au = _unit(a)
    bu = _unit(b)
    if np.linalg.norm(au) < 1e-9 or np.linalg.norm(bu) < 1e-9:
        return 180.0
    return float(np.degrees(np.arccos(float(np.clip(np.dot(au, bu), -1.0, 1.0)))))


def _fair_single_g2_join(
    left: NURBSCurve,
    right: NURBSCurve,
    left_points: np.ndarray,
    right_points: np.ndarray,
) -> None:
    if left.degree < 3 or right.degree < 3 or len(left.cvs) < 4 or len(right.cvs) < 4:
        return
    base_left = _clone_curve(left)
    base_right = _clone_curve(right)
    base_tangent_angle = _angle_between(_bezier_d1_end(base_left), _bezier_d1_start(base_right))
    base_curvature_delta = abs(_bezier_curvature_end(base_left) - _bezier_curvature_start(base_right))
    base_is_already_g2 = base_tangent_angle < 0.22 and base_curvature_delta < 8e-5
    best_left: NURBSCurve | None = base_left if base_is_already_g2 else None
    best_right: NURBSCurve | None = base_right if base_is_already_g2 else None
    best_score = _pair_fairness_score(base_left, base_right, left_points, right_points) if base_is_already_g2 else float("inf")
    local_len = max(min(_polyline_length(left_points), _polyline_length(right_points)), 1.0)
    target_curvature = _target_join_curvature(left_points, right_points)
    desired_curvature = _desired_join_curvature(
        _bezier_curvature_end(base_left),
        _bezier_curvature_start(base_right),
        target_curvature,
        left_points,
        right_points,
        local_len,
    )
    if abs(desired_curvature) > _meaningful_join_curvature(local_len):
        curvature_scales = (0.78, 1.0, 1.24)
    else:
        curvature_scales = (0.0, 0.45, 0.82, 1.0)

    for tangent_weight in (0.55, 0.82, 1.0):
        for handle_scale in (0.68, 0.92, 1.18):
            for curvature_scale in curvature_scales:
                trial_left = _clone_curve(base_left)
                trial_right = _clone_curve(base_right)
                if not _apply_g2_join_variant(
                    trial_left,
                    trial_right,
                    left_points,
                    right_points,
                    tangent_weight=tangent_weight,
                    handle_scale=handle_scale,
                    curvature_scale=curvature_scale,
                ):
                    continue
                score = _pair_fairness_score(trial_left, trial_right, left_points, right_points)
                if score < best_score:
                    best_score = score
                    best_left = trial_left
                    best_right = trial_right

    if best_left is None or best_right is None:
        return
    left.cvs = best_left.cvs
    right.cvs = best_right.cvs


def _apply_g2_join_variant(
    left: NURBSCurve,
    right: NURBSCurve,
    left_points: np.ndarray,
    right_points: np.ndarray,
    *,
    tangent_weight: float,
    handle_scale: float,
    curvature_scale: float,
) -> bool:
    point = 0.5 * (left.cvs[-1] + right.cvs[0])

    left_tangent = _unit(_bezier_d1_end(left)[:2])
    right_tangent = _unit(_bezier_d1_start(right)[:2])
    target_tangent = _target_join_tangent(left_points, right_points)
    fitted_tangent = _unit(left_tangent + right_tangent)
    if np.linalg.norm(fitted_tangent) < 1e-9:
        fitted_tangent = left_tangent if np.linalg.norm(left_tangent) > 1e-9 else right_tangent
    if np.linalg.norm(target_tangent) > 1e-9 and np.linalg.norm(fitted_tangent) > 1e-9:
        tangent = _unit(fitted_tangent * (1.0 - tangent_weight) + target_tangent * tangent_weight)
    elif np.linalg.norm(target_tangent) > 1e-9:
        tangent = target_tangent
    else:
        tangent = fitted_tangent
    if np.linalg.norm(tangent) < 1e-9:
        return False

    left_len = _polyline_length(left_points)
    right_len = _polyline_length(right_points)
    local_len = max(min(left_len, right_len), 1.0)
    original_handle = 0.5 * (
        np.linalg.norm(left.cvs[-1, :2] - left.cvs[-2, :2])
        + np.linalg.norm(right.cvs[1, :2] - right.cvs[0, :2])
    )
    base_handle = min(max(original_handle, local_len * 0.045, 2.0), local_len * 0.18, 28.0)
    handle = min(max(base_handle * handle_scale, local_len * 0.026, 1.4), local_len * 0.22, 34.0)

    k_left = _bezier_curvature_end(left)
    k_right = _bezier_curvature_start(right)
    target_curvature = _target_join_curvature(left_points, right_points)
    desired_curvature = _desired_join_curvature(
        k_left,
        k_right,
        target_curvature,
        left_points,
        right_points,
        local_len,
    )
    base_curvature = _blend_join_curvature(k_left, k_right, target_curvature, local_len)
    meaningful = _meaningful_join_curvature(local_len)
    if abs(desired_curvature) > meaningful and abs(base_curvature) < abs(desired_curvature) * 0.42:
        base_curvature = desired_curvature
    curvature = base_curvature * curvature_scale

    speed = handle * min(left.degree, right.degree)
    d1 = np.array([tangent[0] * speed, tangent[1] * speed, 0.0], dtype=float)
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    d2 = np.array(
        [normal[0] * curvature * speed * speed, normal[1] * curvature * speed * speed, 0.0],
        dtype=float,
    )
    max_second_cv_offset = min(max(handle * 0.34, 1.2), local_len * 0.075, 10.0)
    second_cv_offset = d2 / float(min(left.degree, right.degree) * (min(left.degree, right.degree) - 1))
    second_cv_offset_len = float(np.linalg.norm(second_cv_offset[:2]))
    if second_cv_offset_len > max_second_cv_offset > 1e-9:
        d2 *= max_second_cv_offset / second_cv_offset_len

    _apply_end_derivatives(left, point, d1, d2)
    _apply_start_derivatives(right, point, d1, d2)
    if _endpoint_cv_dent_penalty(left, side="end") > 260.0:
        return False
    if _endpoint_cv_dent_penalty(right, side="start") > 260.0:
        return False
    return True


def _pair_fairness_score(
    left: NURBSCurve,
    right: NURBSCurve,
    left_points: np.ndarray,
    right_points: np.ndarray,
) -> float:
    join_gap = float(np.linalg.norm(left.cvs[-1, :2] - right.cvs[0, :2]))
    tangent_angle = _angle_between(_bezier_d1_end(left), _bezier_d1_start(right))
    curvature_delta = abs(_bezier_curvature_end(left) - _bezier_curvature_start(right))
    error = _bezier_mean_error(left, left_points) + _bezier_mean_error(right, right_points)
    layout = _cv_layout_penalty(left.cvs) + _cv_layout_penalty(right.cvs)
    dent = _cv_dent_penalty(left.cvs) + _cv_dent_penalty(right.cvs)
    side = _cv_side_consistency_penalty(left, left_points) + _cv_side_consistency_penalty(right, right_points)
    corridor = _cv_target_corridor_penalty(left, left_points) + _cv_target_corridor_penalty(right, right_points)
    end_dent = _endpoint_cv_dent_penalty(left, side="end") + _endpoint_cv_dent_penalty(right, side="start")
    comb = _join_curvature_comb_penalty(left, right)
    flow_shape = _join_curvature_flow_penalty(left, right, left_points, right_points)
    flow = _curve_end_flow_penalty(left, side="end") + _curve_end_flow_penalty(right, side="start")
    collapse = _join_curvature_collapse_penalty(left, right, left_points, right_points)
    return (
        join_gap * 500.0
        + tangent_angle * tangent_angle * 18.0
        + min(curvature_delta * 90000.0, 140.0)
        + min(error * 1.6, 120.0)
        + min(layout * 5.5, 140.0)
        + min(dent * 8.0, 220.0)
        + min(side * 10.0, 360.0)
        + min(corridor * 9.0, 360.0)
        + min(end_dent * 14.0, 260.0)
        + min(comb, 180.0)
        + min(flow_shape, 520.0)
        + min(flow, 180.0)
        + min(collapse, 520.0)
    )


def _curve_end_flow_penalty(curve: NURBSCurve, *, side: str) -> float:
    try:
        if side == "end":
            u = np.linspace(0.56, 1.0, 56)
        else:
            u = np.linspace(0.0, 0.44, 56)
        k = signed_curvature_2d(curve.cvs, u)
        k = k[np.isfinite(k)]
        if len(k) < 8:
            return 0.0
        dk = np.diff(k)
        d2k = np.diff(k, n=2)
        sign_changes = np.sum(np.signbit(dk[1:]) != np.signbit(dk[:-1])) if len(dk) > 2 else 0
        return float(
            min(np.mean(np.abs(dk)) * 65000.0, 90.0)
            + min(np.mean(np.abs(d2k)) * 42000.0, 90.0)
            + max(0, int(sign_changes) - 1) * 18.0
        )
    except Exception:
        return 120.0


def _join_curvature_collapse_penalty(
    left: NURBSCurve,
    right: NURBSCurve,
    left_points: np.ndarray,
    right_points: np.ndarray,
) -> float:
    left_len = _polyline_length(left_points)
    right_len = _polyline_length(right_points)
    local_len = max(min(left_len, right_len), 1.0)
    k_left = _bezier_curvature_end(left)
    k_right = _bezier_curvature_start(right)
    samples = _join_curvature_samples(left_points, right_points)
    target_curvature = _select_target_join_curvature(samples)
    desired = _desired_join_curvature(k_left, k_right, target_curvature, left_points, right_points, local_len, samples=samples)
    meaningful = _meaningful_join_curvature(local_len)
    if abs(desired) <= meaningful:
        return 0.0

    joined = 0.5 * (k_left + k_right)
    if not np.isfinite(joined):
        return 300.0
    desired_abs = max(abs(desired), meaningful)
    joined_abs = abs(joined)
    penalty = 0.0
    if desired * joined <= 0.0:
        penalty += 260.0
    ratio = joined_abs / desired_abs
    if ratio < 0.46:
        penalty += (0.46 - ratio) * 780.0
    elif ratio > 1.95:
        penalty += min(ratio - 1.95, 2.4) * 120.0
    return min(float(penalty), 620.0)


def _target_join_tangent(left_points: np.ndarray, right_points: np.ndarray) -> np.ndarray:
    left_len = _polyline_length(left_points)
    right_len = _polyline_length(right_points)
    lookback = min(max(left_len * 0.18, 8.0), 48.0)
    lookahead = min(max(right_len * 0.18, 8.0), 48.0)
    point = 0.5 * (left_points[-1, :2] + right_points[0, :2])
    before = _point_before_end(left_points, lookback)[:2]
    after = _point_after_start(right_points, lookahead)[:2]
    tangent = _unit(after - before)
    if np.linalg.norm(tangent) < 1e-9:
        t0 = _unit(point - before)
        t1 = _unit(after - point)
        tangent = _unit(t0 + t1)
    return tangent


def _meaningful_join_curvature(local_len: float) -> float:
    return float(max(8e-6, min(6.5e-5, 0.0045 / max(local_len, 1.0))))


def _join_curvature_samples(left_points: np.ndarray, right_points: np.ndarray) -> np.ndarray:
    left_len = _polyline_length(left_points)
    right_len = _polyline_length(right_points)
    local_len = max(min(left_len, right_len), 1.0)
    point = 0.5 * (left_points[-1, :2] + right_points[0, :2])
    values: list[float] = []
    for factor in (0.08, 0.14, 0.22, 0.34):
        look = min(max(local_len * factor, 6.0), 62.0)
        before = _point_before_end(left_points, look)[:2]
        after = _point_after_start(right_points, look)[:2]
        values.append(_signed_three_point_curvature(before, point, after))
    values.append(_local_polyline_curvature(left_points, len(left_points) - 1, side="left"))
    values.append(_local_polyline_curvature(right_points, 0, side="right"))
    return np.asarray([value for value in values if np.isfinite(value)], dtype=float)


def _target_join_curvature(left_points: np.ndarray, right_points: np.ndarray) -> float:
    left_len = _polyline_length(left_points)
    right_len = _polyline_length(right_points)
    local_len = max(min(left_len, right_len), 1.0)
    curv = _join_curvature_samples(left_points, right_points)
    return _select_target_join_curvature(curv)


def _select_target_join_curvature(curv: np.ndarray) -> float:
    if len(curv) == 0:
        return 0.0
    eps = max(float(np.nanmax(np.abs(curv))) * 0.12, 1e-7)
    nonzero = curv[np.abs(curv) > eps]
    if len(nonzero) == 0:
        return 0.0
    positive = nonzero[nonzero > 0.0]
    negative = nonzero[nonzero < 0.0]
    if len(positive) and len(negative):
        if np.sum(np.abs(positive)) >= np.sum(np.abs(negative)) * 1.45:
            selected = positive
        elif np.sum(np.abs(negative)) >= np.sum(np.abs(positive)) * 1.45:
            selected = negative
        else:
            return 0.0
    else:
        selected = nonzero
    sign = 1.0 if float(np.sum(selected)) >= 0.0 else -1.0
    magnitude = float(np.nanpercentile(np.abs(selected), 62))
    if magnitude < 1e-6:
        return 0.0
    return float(np.clip(sign * magnitude * 0.92, -0.02, 0.02))


def _desired_join_curvature(
    k_left: float,
    k_right: float,
    target_curvature: float,
    left_points: np.ndarray,
    right_points: np.ndarray,
    local_len: float,
    *,
    samples: np.ndarray | None = None,
) -> float:
    meaningful = _meaningful_join_curvature(local_len)
    if samples is None:
        samples = _join_curvature_samples(left_points, right_points)
    candidates: list[float] = []
    for value in (target_curvature, k_left, k_right):
        if np.isfinite(value) and abs(value) > meaningful * 0.72:
            candidates.append(float(value))
    if len(samples):
        sample_eps = max(float(np.nanmax(np.abs(samples))) * 0.16, meaningful * 0.72)
        candidates.extend(float(value) for value in samples if abs(value) > sample_eps)
    if not candidates:
        return 0.0

    arr = np.asarray(candidates, dtype=float)
    pos = arr[arr > 0.0]
    neg = arr[arr < 0.0]
    if len(pos) and len(neg):
        pos_strength = float(np.sum(np.abs(pos)))
        neg_strength = float(np.sum(np.abs(neg)))
        balance = min(pos_strength, neg_strength) / max(pos_strength, neg_strength, 1e-12)
        sample_pos = samples[samples > meaningful * 0.72]
        sample_neg = samples[samples < -meaningful * 0.72]
        sample_is_balanced = bool(len(sample_pos) and len(sample_neg) and balance > 0.42)
        strong_target = abs(target_curvature) > meaningful * 3.0
        if sample_is_balanced and not strong_target:
            return 0.0
        if strong_target:
            selected = pos if target_curvature > 0.0 else neg
            if len(selected) == 0:
                selected = pos if pos_strength >= neg_strength else neg
        else:
            selected = pos if pos_strength >= neg_strength else neg
    else:
        selected = arr

    sign = 1.0 if float(np.sum(selected)) >= 0.0 else -1.0
    magnitude = float(np.nanpercentile(np.abs(selected), 58))
    if magnitude <= meaningful:
        return 0.0
    max_abs = min(0.02, 2.0 / max(local_len, 1.0))
    return float(np.clip(sign * magnitude, -max_abs, max_abs))


def _blend_join_curvature(k_left: float, k_right: float, target_curvature: float, local_len: float) -> float:
    if not np.isfinite(k_left) or not np.isfinite(k_right):
        k_left = 0.0
        k_right = 0.0
    if not np.isfinite(target_curvature):
        target_curvature = 0.0
    if abs(target_curvature) > 2e-5:
        fit_value = 0.5 * (k_left + k_right) if k_left * k_right >= 0.0 else 0.0
        if fit_value and target_curvature * fit_value > 0.0 and abs(target_curvature) < abs(fit_value) * 0.42:
            value = target_curvature * 0.42 + fit_value * 0.58
        else:
            value = target_curvature * 0.72 + fit_value * 0.28
    elif abs(k_left) < 1e-4 or abs(k_right) < 1e-4 or k_left * k_right < 0.0:
        value = 0.0
    else:
        smaller_abs = min(abs(k_left), abs(k_right))
        larger_abs = max(abs(k_left), abs(k_right))
        smaller = k_left if abs(k_left) <= abs(k_right) else k_right
        if smaller_abs < larger_abs * 0.35:
            value = smaller * 0.68 + 0.5 * (k_left + k_right) * 0.32
        else:
            value = 0.5 * (k_left + k_right)
    max_abs = min(0.02, 2.0 / max(local_len, 1.0))
    return float(np.clip(value, -max_abs, max_abs))


def _apply_start_derivatives(curve: NURBSCurve, point: np.ndarray, d1: np.ndarray, d2: np.ndarray) -> None:
    p = curve.degree
    curve.cvs[0] = point
    curve.cvs[1] = point + d1 / p
    curve.cvs[2] = d2 / (p * (p - 1)) + 2.0 * curve.cvs[1] - curve.cvs[0]


def _apply_end_derivatives(curve: NURBSCurve, point: np.ndarray, d1: np.ndarray, d2: np.ndarray) -> None:
    p = curve.degree
    curve.cvs[-1] = point
    curve.cvs[-2] = point - d1 / p
    curve.cvs[-3] = d2 / (p * (p - 1)) - curve.cvs[-1] + 2.0 * curve.cvs[-2]


def _prepare_fit_points(points: np.ndarray) -> np.ndarray:
    pts = remove_duplicate_points(points, eps=0.5)
    if len(pts) < 4:
        raise ValueError("not enough distinct points to fit")
    pts = resample_polyline(pts, 160)
    pts = smooth_polyline(pts, window=5)
    return pts


def _fit_fixed_degree_with_g2_constraints(
    points: np.ndarray,
    degree: int,
    *,
    start_constraint: _G2Constraint | None,
    end_constraint: _G2Constraint | None,
) -> np.ndarray:
    u = chord_length_parameter(points)
    basis = bernstein_basis(degree, u)
    fixed: dict[int, np.ndarray] = {
        0: (start_constraint.point if start_constraint is not None else points[0]).astype(float),
        degree: (end_constraint.point if end_constraint is not None else points[-1]).astype(float),
    }
    if start_constraint is not None:
        p = fixed[0]
        fixed[1] = p + start_constraint.d1 / degree
        fixed[2] = start_constraint.d2 / (degree * (degree - 1)) + 2.0 * fixed[1] - p
    if end_constraint is not None:
        p = fixed[degree]
        fixed[degree - 1] = p - end_constraint.d1 / degree
        fixed[degree - 2] = end_constraint.d2 / (degree * (degree - 1)) - p + 2.0 * fixed[degree - 1]

    free_indices = [idx for idx in range(degree + 1) if idx not in fixed]
    if not free_indices:
        return np.vstack([fixed[idx] for idx in range(degree + 1)])

    fixed_part = np.zeros_like(points, dtype=float)
    for idx, cv in fixed.items():
        fixed_part += basis[:, [idx]] * cv
    a = basis[:, free_indices]
    rhs = points - fixed_part

    reg_a, reg_rhs = _free_cv_regularization(degree, fixed, free_indices)
    if len(reg_a):
        a = np.vstack([a, reg_a])
        rhs = np.vstack([rhs, reg_rhs])

    free, *_ = np.linalg.lstsq(a, rhs, rcond=None)
    cvs = [fixed[idx] if idx in fixed else free[free_indices.index(idx)] for idx in range(degree + 1)]
    return np.vstack(cvs)


def _free_cv_regularization(
    degree: int,
    fixed: dict[int, np.ndarray],
    free_indices: list[int],
    *,
    fair_lambda: float = 0.02,
    jerk_lambda: float = 0.004,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    rhs: list[np.ndarray] = []

    def add_diff(order: int, lam: float) -> None:
        if lam <= 0 or degree + 1 <= order:
            return
        scale = float(lam) ** 0.5
        coeff = np.array([(-1) ** (order - k) * _binom(order, k) for k in range(order + 1)], dtype=float)
        for start in range(degree + 1 - order):
            row = np.zeros(len(free_indices), dtype=float)
            known = np.zeros(3, dtype=float)
            for off, c in enumerate(coeff):
                idx = start + off
                value = scale * c
                if idx in fixed:
                    known += value * fixed[idx]
                else:
                    row[free_indices.index(idx)] += value
            rows.append(row)
            rhs.append(-known)

    add_diff(2, fair_lambda)
    add_diff(3, jerk_lambda)
    if not rows:
        return np.zeros((0, len(free_indices))), np.zeros((0, 3))
    return np.vstack(rows), np.vstack(rhs)


def _g2_join_constraints(
    segments: list[dict[str, Any]],
    *,
    closed: bool,
) -> dict[int, _G2Constraint]:
    count = len(segments)
    join_count = count if closed and count > 2 else count - 1
    constraints: dict[int, _G2Constraint] = {}
    for join_index in range(join_count):
        prev_points = np.asarray(segments[join_index]["points"], dtype=float)
        next_points = np.asarray(segments[(join_index + 1) % count]["points"], dtype=float)
        prev_len = _polyline_length(prev_points)
        next_len = _polyline_length(next_points)
        if prev_len <= 1e-6 or next_len <= 1e-6:
            continue
        point = 0.5 * (prev_points[-1] + next_points[0])
        lookback = min(max(prev_len * 0.22, 8.0), 55.0)
        lookahead = min(max(next_len * 0.22, 8.0), 55.0)
        before = _point_before_end(prev_points, lookback)
        after = _point_after_start(next_points, lookahead)
        t_prev = _unit(point[:2] - before[:2])
        t_next = _unit(after[:2] - point[:2])
        tangent = _unit(t_prev + t_next)
        if np.linalg.norm(tangent) < 1e-8:
            tangent = t_next if np.linalg.norm(t_next) > 1e-8 else t_prev
        handle = min(max(min(prev_len, next_len) * 0.20, 4.0), 45.0)
        d1_xy = tangent * handle * 7.0
        curvature = _signed_three_point_curvature(before[:2], point[:2], after[:2])
        curvature = float(np.clip(curvature * 0.35, -0.012, 0.012))
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        d2_xy = normal * curvature * float(np.dot(d1_xy, d1_xy))
        constraints[join_index] = _G2Constraint(
            point=point.astype(float),
            d1=np.array([d1_xy[0], d1_xy[1], 0.0], dtype=float),
            d2=np.array([d2_xy[0], d2_xy[1], 0.0], dtype=float),
        )
    return constraints


def _stamp_g2_diagnostics(
    fitted: list[tuple[NURBSCurve, CurveCandidate, QualityReport]],
    *,
    closed: bool,
) -> None:
    if len(fitted) <= 1:
        return
    join_count = len(fitted) if closed and len(fitted) > 2 else len(fitted) - 1
    for join_index in range(join_count):
        left = fitted[join_index][0]
        right = fitted[(join_index + 1) % len(fitted)][0]
        gap = float(np.linalg.norm(left.cvs[-1, :2] - right.cvs[0, :2]))
        tan = _angle_between(
            _bezier_d1_end(left),
            _bezier_d1_start(right),
        )
        curv = abs(_bezier_curvature_end(left) - _bezier_curvature_start(right))
        left_curvature = _bezier_curvature_end(left)
        right_curvature = _bezier_curvature_start(right)
        target_curvature = _target_join_curvature(
            fitted[join_index][1].points,
            fitted[(join_index + 1) % len(fitted)][1].points,
        )
        local_len = max(
            min(
                _polyline_length(fitted[join_index][1].points),
                _polyline_length(fitted[(join_index + 1) % len(fitted)][1].points),
            ),
            1.0,
        )
        desired_curvature = _desired_join_curvature(
            left_curvature,
            right_curvature,
            target_curvature,
            fitted[join_index][1].points,
            fitted[(join_index + 1) % len(fitted)][1].points,
            local_len,
        )
        flow = _join_curvature_flow_penalty(
            left,
            right,
            fitted[join_index][1].points,
            fitted[(join_index + 1) % len(fitted)][1].points,
        )
        collapse = _join_curvature_collapse_penalty(
            left,
            right,
            fitted[join_index][1].points,
            fitted[(join_index + 1) % len(fitted)][1].points,
        )
        left.metadata[f"g2_join_{join_index}_gap"] = gap
        left.metadata[f"g2_join_{join_index}_tangent_angle_deg"] = tan
        left.metadata[f"g2_join_{join_index}_curvature_delta"] = curv
        left.metadata[f"g2_join_{join_index}_left_curvature"] = left_curvature
        left.metadata[f"g2_join_{join_index}_right_curvature"] = right_curvature
        left.metadata[f"g2_join_{join_index}_target_curvature"] = target_curvature
        left.metadata[f"g2_join_{join_index}_desired_curvature"] = desired_curvature
        left.metadata[f"g2_join_{join_index}_curvature_flow_penalty"] = flow
        left.metadata[f"g2_join_{join_index}_curvature_collapse_penalty"] = collapse
        right.metadata[f"g2_join_{join_index}_gap"] = gap
        right.metadata[f"g2_join_{join_index}_tangent_angle_deg"] = tan
        right.metadata[f"g2_join_{join_index}_curvature_delta"] = curv
        right.metadata[f"g2_join_{join_index}_left_curvature"] = left_curvature
        right.metadata[f"g2_join_{join_index}_right_curvature"] = right_curvature
        right.metadata[f"g2_join_{join_index}_target_curvature"] = target_curvature
        right.metadata[f"g2_join_{join_index}_desired_curvature"] = desired_curvature
        right.metadata[f"g2_join_{join_index}_curvature_flow_penalty"] = flow
        right.metadata[f"g2_join_{join_index}_curvature_collapse_penalty"] = collapse


def _bezier_d1_start(curve: NURBSCurve) -> np.ndarray:
    return curve.degree * (curve.cvs[1] - curve.cvs[0])


def _bezier_d1_end(curve: NURBSCurve) -> np.ndarray:
    return curve.degree * (curve.cvs[-1] - curve.cvs[-2])


def _bezier_d2_start(curve: NURBSCurve) -> np.ndarray:
    p = curve.degree
    return p * (p - 1) * (curve.cvs[2] - 2 * curve.cvs[1] + curve.cvs[0])


def _bezier_d2_end(curve: NURBSCurve) -> np.ndarray:
    p = curve.degree
    return p * (p - 1) * (curve.cvs[-1] - 2 * curve.cvs[-2] + curve.cvs[-3])


def _bezier_curvature_start(curve: NURBSCurve) -> float:
    return _curvature_from_derivatives(_bezier_d1_start(curve), _bezier_d2_start(curve))


def _bezier_curvature_end(curve: NURBSCurve) -> float:
    return _curvature_from_derivatives(_bezier_d1_end(curve), _bezier_d2_end(curve))


def _curvature_from_derivatives(d1: np.ndarray, d2: np.ndarray) -> float:
    speed = float(np.linalg.norm(d1[:2]))
    if speed < 1e-9:
        return 0.0
    cross = float(d1[0] * d2[1] - d1[1] * d2[0])
    return cross / (speed**3)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    au = _unit(a[:2])
    bu = _unit(b[:2])
    if np.linalg.norm(au) < 1e-9 or np.linalg.norm(bu) < 1e-9:
        return 180.0
    dot = float(np.clip(np.dot(au, bu), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _point_before_end(points: np.ndarray, distance: float) -> np.ndarray:
    return _point_from_polyline_end(points, distance, from_end=True)


def _point_after_start(points: np.ndarray, distance: float) -> np.ndarray:
    return _point_from_polyline_end(points, distance, from_end=False)


def _point_from_polyline_end(points: np.ndarray, distance: float, *, from_end: bool) -> np.ndarray:
    pts = points[::-1] if from_end else points
    remaining = float(distance)
    for idx in range(len(pts) - 1):
        a = pts[idx]
        b = pts[idx + 1]
        seg = float(np.linalg.norm(b[:2] - a[:2]))
        if seg >= remaining and seg > 1e-9:
            t = remaining / seg
            return a * (1.0 - t) + b * t
        remaining -= seg
    return pts[-1]


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)))


def _unit(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros_like(v, dtype=float)
    return v / n


def _signed_three_point_curvature(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    bc = c - b
    ac = c - a
    lab = float(np.linalg.norm(ab))
    lbc = float(np.linalg.norm(bc))
    lac = float(np.linalg.norm(ac))
    denom = lab * lbc * lac
    if denom < 1e-9:
        return 0.0
    cross = float(ab[0] * bc[1] - ab[1] * bc[0])
    return 2.0 * cross / denom


def _curve_segments(curve: dict[str, Any]) -> list[dict[str, Any]]:
    manual = curve.get("manual_points") or curve.get("cut_points") or []
    route_segments = curve.get("route_segments") or []
    expected_count = max(0, len(manual) - 1)
    if curve.get("closed") and len(manual) >= 3:
        expected_count += 1

    out: list[dict[str, Any]] = []
    if expected_count > 0 and len(route_segments) >= expected_count:
        for index, segment in enumerate(route_segments[:expected_count]):
            points = _as_points3(segment.get("points") or [])
            if len(points) < 2:
                points = _manual_pair_points(manual, index, bool(curve.get("closed")))
            out.append(
                {
                    "points": points,
                    "start_order": index,
                    "end_order": 0 if bool(curve.get("closed")) and index == len(manual) - 1 else index + 1,
                    "segment_count": expected_count,
                }
            )
        return out

    if expected_count > 0:
        for index in range(expected_count):
            out.append(
                {
                    "points": _manual_pair_points(manual, index, bool(curve.get("closed"))),
                    "start_order": index,
                    "end_order": 0 if bool(curve.get("closed")) and index == len(manual) - 1 else index + 1,
                    "segment_count": expected_count,
                }
            )
        return out

    points = _curve_points(curve)
    if len(points) >= 2:
        return [{"points": points, "start_order": 0, "end_order": 1, "segment_count": 1}]
    return []


def _curve_points(curve: dict[str, Any]) -> np.ndarray:
    routed = curve.get("routed_points") or []
    if len(routed) >= 4:
        return _as_points3(routed)
    manual = curve.get("manual_points") or curve.get("cut_points") or []
    points = [[p["x"], p["y"]] if isinstance(p, dict) else p for p in manual]
    if curve.get("closed") and len(points) >= 2:
        points.append(points[0])
    return _as_points3(points)


def _manual_pair_points(manual: list[Any], index: int, closed: bool) -> np.ndarray:
    if len(manual) < 2:
        return np.zeros((0, 3), dtype=float)
    start = manual[index]
    end_index = 0 if closed and index == len(manual) - 1 else index + 1
    if end_index >= len(manual):
        return np.zeros((0, 3), dtype=float)
    end = manual[end_index]
    a = _manual_point_xy(start)
    b = _manual_point_xy(end)
    if a is None or b is None:
        return np.zeros((0, 3), dtype=float)
    return _line_points(a, b, count=8)


def _manual_point_xy(point: Any) -> tuple[float, float] | None:
    if isinstance(point, dict) and "x" in point and "y" in point:
        return (float(point["x"]), float(point["y"]))
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return (float(point[0]), float(point[1]))
    return None


def _ensure_minimum_fit_points(points: np.ndarray) -> np.ndarray:
    if len(points) >= 4:
        return points
    if len(points) < 2:
        return points
    return _line_points((float(points[0, 0]), float(points[0, 1])), (float(points[-1, 0]), float(points[-1, 1])), count=8)


def _line_points(a: tuple[float, float], b: tuple[float, float], count: int = 8) -> np.ndarray:
    u = np.linspace(0.0, 1.0, count)
    arr = np.column_stack(
        [
            a[0] * (1.0 - u) + b[0] * u,
            a[1] * (1.0 - u) + b[1] * u,
            np.zeros(count, dtype=float),
        ]
    )
    return arr


def _curve_label(curve: dict[str, Any], path: Path, index: int, segment_index: int, segment_count: int) -> str:
    semantic = str(curve.get("semantic") or "manual_design_curve")
    stem = "".join(ch if ch.isalnum() else "_" for ch in path.stem)
    if segment_count <= 1:
        return f"{semantic}_{stem}_{index:03d}"
    return f"{semantic}_{stem}_{index:03d}_seg{segment_index:03d}"


def _as_points3(points: Any) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        return np.zeros((0, 3), dtype=float)
    if arr.shape[1] == 2:
        arr = np.column_stack([arr, np.zeros(len(arr), dtype=float)])
    return arr
