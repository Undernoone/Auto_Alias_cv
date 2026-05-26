from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover - scipy is optional for portable installs.
    minimize = None

from autoalias.exporters import (
    WireExportResult,
    write_iges,
    write_json_bundle,
    write_svg_preview,
    write_wire_from_iges,
    write_wire_status,
)
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
    wire_result: WireExportResult | None = None


@dataclass(slots=True)
class _G2Constraint:
    point: np.ndarray
    d1: np.ndarray
    d2: np.ndarray


G2_SPLIT_MAX_SHIFT_PX = 18.0
ENABLE_G2_CONSTRAINTS = False
ENABLE_G2_EDITOR_OVERRIDES = False
PRECISION_FIT_MAX_DEGREE = 24
PRECISION_FIT_DEGREE_TIEBREAK_WEIGHT = 0.015


def fit_reviewed_annotations(
    annotation_paths: list[str | Path],
    out: str | Path,
    *,
    degree: int | str = "auto",
    min_points: int = 8,
    max_fit_points: int | None = None,
    diagnostic_preview: bool = True,
    fast_mode: bool = False,
    fit_mode: str = "manual_class_a_g2",
    wire_export: bool = False,
    iges_to_al: str | Path | None = None,
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
    wire_result: WireExportResult | None = None

    for annotation_path in resolved_paths:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_path = data.get("graph", {}).get("image")
        image_size = data.get("graph", {}).get("image_size", {}) or {}
        image_diag = float(
            np.hypot(
                float(image_size.get("width", 0.0) or 0.0),
                float(image_size.get("height", 0.0) or 0.0),
            )
        )
        if background_image is None and image_path:
            background_image = image_path

        for index, design_curve in enumerate(data.get("design_curves", []), start=1):
            raw_segments = _curve_segments(design_curve)
            raw_segments = _compact_auto_export_segments(raw_segments, design_curve, image_diag=image_diag)
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
                    points = _limit_fit_points(points, max_fit_points)
                    if not _has_enough_distinct_fit_points(points):
                        skipped += 1
                        continue
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

            overrides = _curve_alias_overrides(design_curve, expected_count=len(segments))
            if overrides:
                for segment_index, (segment, override) in enumerate(zip(segments, overrides, strict=False), start=1):
                    points = segment["points"]
                    label = _curve_label(
                        design_curve,
                        annotation_path,
                        index,
                        segment_index,
                        segment["segment_count"],
                    )
                    candidate = _make_candidate(label, points, annotation_path, design_curve)
                    cvs = _as_points3(override.get("cvs") or override.get("cv") or [])
                    degree = int(override.get("degree") or max(len(cvs) - 1, 3))
                    if len(cvs) != degree + 1:
                        degree = len(cvs) - 1
                    if degree < 3 or degree > 7 or len(cvs) != degree + 1:
                        skipped += 1
                        continue
                    curve = NURBSCurve.single_span(
                        label=label,
                        degree=degree,
                        cvs=cvs,
                        confidence=1.0,
                        source="manual_g2_constraint_cv_override",
                        metadata=_curve_metadata(
                            design_curve,
                            annotation_path,
                            segment,
                            points,
                            segment_index,
                            fit_policy="manual_dynamic_g2_cv_override",
                        ),
                    )
                    curve.metadata["dynamic_g2_override"] = True
                    report = validator.validate(curve, points)
                    curves.append(curve)
                    candidates.append(candidate)
                    reports.append(report)
                continue

            fitted = _fit_design_curve_chain(
                design_curve,
                annotation_path,
                index,
                segments,
                degree,
                validator,
                fast_mode=fast_mode,
                fit_mode=fit_mode,
            )
            for curve, candidate, report in fitted:
                curves.append(curve)
                candidates.append(candidate)
                reports.append(report)

    write_json_bundle(out_path / "reviewed_curves.json", curves, reports)
    if curves:
        iges_path = out_path / "reviewed_curves.igs"
        write_iges(iges_path, curves)
        if wire_export:
            wire_result = write_wire_from_iges(
                iges_path,
                out_path / "reviewed_curves.wire",
                converter=iges_to_al,
            )
            write_wire_status(out_path / "reviewed_curves.wire_status.json", wire_result)
        if diagnostic_preview:
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
    return ReviewedFitResult(
        out=out_path,
        curves=curves,
        reports=reports,
        skipped_count=skipped,
        wire_result=wire_result,
    )


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
        curve = SingleSpanFitter(FittingOptions(degree=degree)).fit_candidate(candidate)
        return _refine_non_s_single_side_curve(curve, target_points)
    best_curve: NURBSCurve | None = None
    best_score = float("inf")
    for candidate_degree in (3, 4, 5, 6, 7):
        curve = SingleSpanFitter(FittingOptions(degree=candidate_degree)).fit_candidate(candidate)
        curve = _refine_non_s_single_side_curve(curve, target_points)
        report = validator.validate(curve, target_points)
        chamfer = float(report.metrics.get("chamfer_mean", 999.0))
        side = _cv_side_consistency_penalty(curve, target_points)
        forbidden_side = _has_forbidden_cv_side_switch(curve, target_points)
        forbidden_curvature = _has_forbidden_curvature_sign_change(curve, target_points)
        blend_fairness = _curve_blend_fairness_metrics(curve, target_points)
        exceeds_precision = _curve_exceeds_precision_budget(curve, target_points)
        warnings = len(report.warnings)
        score = (
            warnings * 1000.0
            + chamfer
            + side * 2.6
            + (10000.0 if forbidden_side else 0.0)
            + (10000.0 if forbidden_curvature else 0.0)
            + (8000.0 if bool(blend_fairness["forbidden"]) else 0.0)
            + float(blend_fairness["penalty"]) * 40.0
            + _curvature_sign_penalty(curve, target_points)
            + (5000.0 if exceeds_precision else 0.0)
            + candidate_degree * 0.01
        )
        if report.passed and not forbidden_side and not forbidden_curvature and not bool(blend_fairness["forbidden"]) and not exceeds_precision and side < 34.0:
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
        curve = SingleSpanFitter(FittingOptions(degree=degree)).fit_candidate(candidate)
        return _refine_non_s_single_side_curve(curve, target_points)
    start_constrained = closed or segment_index > 1
    end_constrained = closed or segment_index < segment_count
    simplicity = _target_curve_simplicity(target_points)
    # Two-sided G2 needs independent start/end derivative CVs. When G2 is
    # disabled we allow low degree again so fitting quality can be isolated.
    if ENABLE_G2_CONSTRAINTS and start_constrained and end_constrained:
        degrees = (5, 6, 7)
    else:
        degrees = (3, 4, 5, 6, 7)
    best_curve: NURBSCurve | None = None
    best_score = float("inf")
    for candidate_degree in degrees:
        curve = SingleSpanFitter(FittingOptions(degree=candidate_degree)).fit_candidate(candidate)
        curve = _refine_non_s_single_side_curve(curve, target_points)
        report = validator.validate(curve, target_points)
        chamfer = float(report.metrics.get("chamfer_mean", 999.0))
        spacing = float(report.metrics.get("cv_spacing_ratio", 999.0))
        oscillation = float(report.metrics.get("curvature_oscillation", 999.0))
        dent = _cv_dent_penalty(curve.cvs)
        side = _cv_side_consistency_penalty(curve, target_points)
        corridor = _cv_target_corridor_penalty(curve, target_points)
        layout = _cv_layout_penalty(curve.cvs)
        forbidden_side = _has_forbidden_cv_side_switch(curve, target_points)
        forbidden_curvature = _has_forbidden_curvature_sign_change(curve, target_points)
        blend_fairness = _curve_blend_fairness_metrics(curve, target_points)
        exceeds_precision = _curve_exceeds_precision_budget(curve, target_points)
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
                + (10000.0 if forbidden_side else 0.0)
                + (10000.0 if forbidden_curvature else 0.0)
                + (8000.0 if bool(blend_fairness["forbidden"]) else 0.0)
                + float(blend_fairness["penalty"]) * 40.0
                + _curvature_sign_penalty(curve, target_points)
                + (5000.0 if exceeds_precision else 0.0)
                + candidate_degree * 42.0
            )
            if report.passed and not forbidden_side and not forbidden_curvature and not bool(blend_fairness["forbidden"]) and not exceeds_precision and dent < 18.0 and side < 32.0 and corridor < 34.0 and spacing < 5.6:
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
                + (10000.0 if forbidden_side else 0.0)
                + (10000.0 if forbidden_curvature else 0.0)
                + (8000.0 if bool(blend_fairness["forbidden"]) else 0.0)
                + float(blend_fairness["penalty"]) * 40.0
                + _curvature_sign_penalty(curve, target_points)
                + (5000.0 if exceeds_precision else 0.0)
                + candidate_degree * 0.05
            )
        if (
            report.passed
            and not forbidden_side
            and not forbidden_curvature
            and not bool(blend_fairness["forbidden"])
            and not exceeds_precision
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
    fast_mode: bool = False,
    fit_mode: str = "manual_class_a_g2",
) -> list[tuple[NURBSCurve, CurveCandidate, QualityReport]]:
    """Fit one reviewed curve.

    A design curve with several manual/AI split points is a continuity chain. Its
    exported IGES entities are still one-span Bezier/NURBS curves, but adjacent
    entities share the same endpoint, tangent and curvature vector at every split.
    """
    if fast_mode:
        return _fit_design_curve_chain_fast(
            design_curve,
            annotation_path,
            design_index,
            segments,
            degree,
            validator,
        )
    if _is_precision_fit_mode(fit_mode):
        return _fit_design_curve_chain_precision(
            design_curve,
            annotation_path,
            design_index,
            segments,
            degree,
            validator,
        )
    if _should_use_manual_class_a_g2(design_curve, segments, fit_mode):
        return _fit_design_curve_chain_class_a_g2(
            design_curve,
            annotation_path,
            design_index,
            segments,
            degree,
            validator,
        )
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
            fit_notes = _fit_metadata_notes(curve)
            curve.source = "manual_review_fit"
            curve.metadata = _curve_metadata(
                design_curve,
                annotation_path,
                segment,
                points,
                segment_index,
                fit_policy="split_boundaries_then_single_span_degree_3_to_7",
            )
            curve.metadata.update(fit_notes)
            _stamp_cv_side_diagnostics(curve, points)
            _stamp_cv_target_corridor_diagnostics(curve, points)
            _stamp_curvature_sign_diagnostics(curve, points)
            _stamp_blend_fairness_diagnostics(curve, points)
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
    if ENABLE_G2_CONSTRAINTS:
        segments, split_diagnostics = _auto_adjust_g2_split_points(
            segments,
            closed=closed,
            max_shift_px=G2_SPLIT_MAX_SHIFT_PX,
        )
    else:
        split_diagnostics = {}
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
        fit_notes = _fit_metadata_notes(curve)
        curve.source = "manual_review_g2_fit" if ENABLE_G2_CONSTRAINTS else "manual_review_fit"
        curve.metadata = _curve_metadata(
            design_curve,
            annotation_path,
            segment,
            points,
            segment_index,
            fit_policy=(
                "manual_split_boundaries_with_lowest_degree_g2_fairing"
                if ENABLE_G2_CONSTRAINTS
                else "manual_split_boundaries_with_lowest_degree_no_g2"
            ),
        )
        curve.metadata["g2_chain"] = bool(ENABLE_G2_CONSTRAINTS)
        curve.metadata["g2_method"] = (
            "limited_split_adjustment_plus_local_fairness_search"
            if ENABLE_G2_CONSTRAINTS
            else "disabled_for_fit_debugging"
        )
        curve.metadata["chain_original_segment_count"] = original_segment_count
        curve.metadata["chain_merged_segment_count"] = len(segments)
        curve.metadata["chain_merge"] = merge_diagnostics.get(segment_index - 1, {})
        curve.metadata["g2_split_adjustment"] = split_diagnostics.get(segment_index - 1, {})
        curve.metadata.update(fit_notes)
        fitted_curves.append((curve, candidate, points, segment, segment_index))

    if ENABLE_G2_CONSTRAINTS:
        _apply_g2_endpoint_fairing([item[0] for item in fitted_curves], [item[2] for item in fitted_curves], closed=closed)
    if not isinstance(degree, int):
        _simplify_simple_chain_degrees(fitted_curves, validator, closed=closed)
    _promote_failed_chain_degrees(fitted_curves, validator, closed=closed)
    _repair_cv_side_flips(fitted_curves, validator, closed=closed)
    if ENABLE_G2_CONSTRAINTS:
        _apply_g2_endpoint_fairing([item[0] for item in fitted_curves], [item[2] for item in fitted_curves], closed=closed)
    if not isinstance(degree, int):
        _promote_bad_layout_degrees(fitted_curves, validator, closed=closed)
        if ENABLE_G2_CONSTRAINTS:
            _apply_g2_endpoint_fairing([item[0] for item in fitted_curves], [item[2] for item in fitted_curves], closed=closed)
    _repair_cv_side_flips(fitted_curves, validator, closed=closed)
    if ENABLE_G2_CONSTRAINTS:
        _apply_g2_endpoint_fairing([item[0] for item in fitted_curves], [item[2] for item in fitted_curves], closed=closed)
    if not isinstance(degree, int):
        _promote_bad_layout_degrees(fitted_curves, validator, closed=closed)
        if ENABLE_G2_CONSTRAINTS:
            _apply_g2_endpoint_fairing([item[0] for item in fitted_curves], [item[2] for item in fitted_curves], closed=closed)
    if ENABLE_G2_CONSTRAINTS:
        _refit_chain_with_current_g2_constraints(fitted_curves, validator, requested_degree=degree, closed=closed)
        _apply_g2_endpoint_fairing([item[0] for item in fitted_curves], [item[2] for item in fitted_curves], closed=closed)
    _fair_free_interior_cvs_for_chain(fitted_curves, closed=closed)
    for curve, candidate, points, segment, segment_index in fitted_curves:
        _stamp_cv_side_diagnostics(curve, points)
        _stamp_cv_target_corridor_diagnostics(curve, points)
        _stamp_curvature_sign_diagnostics(curve, points)
        _stamp_blend_fairness_diagnostics(curve, points)
        curve.metadata["g2_start_constrained"] = bool(ENABLE_G2_CONSTRAINTS and (closed or segment_index > 1))
        curve.metadata["g2_end_constrained"] = bool(ENABLE_G2_CONSTRAINTS and (closed or segment_index < len(fitted_curves)))
        out.append((curve, candidate, validator.validate(curve, points)))
    if ENABLE_G2_CONSTRAINTS:
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
            fit_mode=fit_mode,
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
            fit_mode=fit_mode,
        )
    return out


def _should_use_manual_class_a_g2(
    design_curve: dict[str, Any],
    segments: list[dict[str, Any]],
    fit_mode: str,
) -> bool:
    if str(fit_mode or "").lower() not in {"manual_class_a_g2", "class_a_g2"}:
        return False
    if len(segments) <= 1:
        return False
    source = str(design_curve.get("source") or "").lower()
    if source.startswith("geometry_auto_segment"):
        return False
    return True


def _is_precision_fit_mode(fit_mode: str) -> bool:
    return str(fit_mode or "").strip().lower() in {
        "precision",
        "accuracy",
        "fit_accuracy",
        "precision_no_cv",
        "ignore_cv",
    }


def _fit_design_curve_chain_class_a_g2(
    design_curve: dict[str, Any],
    annotation_path: Path,
    design_index: int,
    segments: list[dict[str, Any]],
    degree: int | str,
    validator: ClassAValidator,
) -> list[tuple[NURBSCurve, CurveCandidate, QualityReport]]:
    """Fit a manual split chain as Alias-style single-span G2 curves.

    The important difference from the old endpoint fairing path is that the G2
    boundary conditions are estimated from the routed target first, then each span is
    solved with those constraints. That preserves G0/G1 before asking for curvature
    continuity and avoids the "pull one CV after fitting" jump-point failure mode.
    """
    closed = bool(design_curve.get("closed", False))
    normalized_segments = [
        {**segment, "points": remove_duplicate_points(np.asarray(segment["points"], dtype=float), eps=0.5)}
        for segment in segments
    ]
    try:
        return _fit_design_curve_chain_global_c2(
            design_curve,
            annotation_path,
            design_index,
            normalized_segments,
            degree,
            validator,
        )
    except Exception:
        # Fall back to the older constrained-span path if the global solve is singular
        # for a degenerate hand route.
        pass

    base_curves: list[NURBSCurve] = []
    for segment_index, segment in enumerate(normalized_segments, start=1):
        points = segment["points"]
        label = _curve_label(
            design_curve,
            annotation_path,
            design_index,
            segment_index,
            segment["segment_count"],
        )
        candidate = _make_candidate(label, points, annotation_path, design_curve)
        try:
            base_curves.append(
                _fit_chain_lowest_degree(
                    candidate,
                    points,
                    degree,
                    validator,
                    segment_index=segment_index,
                    segment_count=segment["segment_count"],
                    closed=closed,
                )
            )
        except Exception:
            base_curves.append(_fit_lowest_degree(candidate, points, degree, validator))
    fitted_curves: list[tuple[NURBSCurve, CurveCandidate, np.ndarray, dict[str, Any], int]] = []
    constraints = _class_a_g2_join_constraints(
        normalized_segments,
        closed=closed,
        base_curves=base_curves,
        curvature_scale=0.62,
        handle_scale=1.0,
    )

    for segment_index, segment in enumerate(normalized_segments, start=1):
        points = segment["points"]
        label = _curve_label(
            design_curve,
            annotation_path,
            design_index,
            segment_index,
            segment["segment_count"],
        )
        candidate = _make_candidate(label, points, annotation_path, design_curve)
        start_constraint = None
        end_constraint = None
        if segment_index > 1 or (closed and len(normalized_segments) > 2):
            start_constraint = constraints.get((segment_index - 2) % len(normalized_segments))
        if segment_index < len(normalized_segments) or (closed and len(normalized_segments) > 2):
            end_constraint = constraints.get((segment_index - 1) % len(normalized_segments))

        curve = _fit_class_a_g2_segment(
            candidate,
            points,
            degree,
            validator,
            start_constraint=start_constraint,
            end_constraint=end_constraint,
        )
        curve.source = "manual_review_class_a_g2_fit"
        curve.metadata = _curve_metadata(
            design_curve,
            annotation_path,
            segment,
            points,
            segment_index,
            fit_policy="manual_class_a_g2_global_constraints_single_span",
        )
        curve.metadata.update(_fit_metadata_notes(curve))
        curve.metadata["class_a_g2"] = True
        curve.metadata["class_a_g2_start_constrained"] = start_constraint is not None
        curve.metadata["class_a_g2_end_constrained"] = end_constraint is not None
        curve.metadata["class_a_g2_method"] = "target_curvature_constrained_least_squares"
        fitted_curves.append((curve, candidate, points, segment, segment_index))

    target_segments = [item[2] for item in fitted_curves]
    curves = [item[0] for item in fitted_curves]
    _fair_free_interior_cvs_for_curves(curves, closed=closed, target_segments=target_segments)
    _repair_class_a_cv_layout(curves, target_segments, closed=closed)

    out: list[tuple[NURBSCurve, CurveCandidate, QualityReport]] = []
    for curve, candidate, points, _segment, _segment_index in fitted_curves:
        _stamp_cv_side_diagnostics(curve, points)
        _stamp_cv_target_corridor_diagnostics(curve, points)
        _stamp_curvature_sign_diagnostics(curve, points)
        _stamp_blend_fairness_diagnostics(curve, points)
        out.append((curve, candidate, validator.validate(curve, points)))
    _stamp_class_a_g2_diagnostics(out, closed=closed)
    return out


def _review_fit_has_visual_fairness_failure(
    fitted: list[tuple[NURBSCurve, CurveCandidate, QualityReport]],
) -> bool:
    for curve, candidate, report in fitted:
        points = np.asarray(candidate.points, dtype=float)
        if not bool(report.passed):
            return True
        if _has_forbidden_curvature_sign_change(curve, points):
            return True
        if _has_forbidden_cv_side_switch(curve, points):
            return True
        if bool(_curve_blend_fairness_metrics(curve, points)["forbidden"]):
            return True
        if _curve_exceeds_precision_budget(curve, points):
            return True
    return False


def _review_fit_visual_fairness_score(
    fitted: list[tuple[NURBSCurve, CurveCandidate, QualityReport]],
) -> float:
    score = 0.0
    for curve, candidate, report in fitted:
        points = np.asarray(candidate.points, dtype=float)
        score += 20000.0 if not bool(report.passed) else 0.0
        score += 22000.0 if _has_forbidden_curvature_sign_change(curve, points) else 0.0
        score += 18000.0 if _has_forbidden_cv_side_switch(curve, points) else 0.0
        blend = _curve_blend_fairness_metrics(curve, points)
        score += 16000.0 if bool(blend["forbidden"]) else 0.0
        score += float(blend["penalty"]) * 120.0
        score += 14000.0 if _curve_exceeds_precision_budget(curve, points) else 0.0
        score += _bezier_mean_error(curve, points) * 30.0
        score += _cv_side_consistency_penalty(curve, points) * 12.0
        score += _cv_target_corridor_penalty(curve, points) * 12.0
        score += _curvature_sign_penalty(curve, points) * 3.0
        score += _cv_layout_penalty(curve.cvs) * 8.0
        score += _cv_dent_penalty(curve.cvs) * 3.0
    return float(score)


def _fit_design_curve_chain_global_c2(
    design_curve: dict[str, Any],
    annotation_path: Path,
    design_index: int,
    segments: list[dict[str, Any]],
    degree: int | str,
    validator: ClassAValidator,
) -> list[tuple[NURBSCurve, CurveCandidate, QualityReport]]:
    closed = bool(design_curve.get("closed", False))
    segment_count = len(segments)
    if segment_count <= 1:
        raise ValueError("global C2 chain requires at least two segments")
    solve_degree = _select_global_c2_solve_degree(segments, degree, closed=closed)
    prior_cvs = _global_c2_cv_priors(segments, degree=solve_degree)
    best: list[tuple[NURBSCurve, CurveCandidate, QualityReport]] | None = None
    best_score = float("inf")
    best_prior_weight = 0.0
    for prior_weight in (0.0, 0.18, 0.38, 0.72, 1.15, 1.75, 2.6):
        curves = _solve_global_c2_bezier_chain(
            segments,
            degree=solve_degree,
            closed=closed,
            cv_priors=prior_cvs if prior_weight > 0.0 else None,
            cv_prior_weight=prior_weight,
            visual_optimize=False,
        )
        fitted = _package_global_c2_fit(
            design_curve,
            annotation_path,
            design_index,
            segments,
            curves,
            solve_degree,
            validator,
            prior_weight=prior_weight,
        )
        score = _review_fit_visual_fairness_score(fitted)
        if score < best_score:
            best = fitted
            best_score = score
            best_prior_weight = prior_weight
        if not _review_fit_has_visual_fairness_failure(fitted):
            best = fitted
            best_prior_weight = prior_weight
            break
    if best is not None and _review_fit_has_visual_fairness_failure(best):
        try:
            curves = _solve_global_c2_bezier_chain(
                segments,
                degree=solve_degree,
                closed=closed,
                cv_priors=prior_cvs if best_prior_weight > 0.0 else None,
                cv_prior_weight=best_prior_weight,
                visual_optimize=True,
                hard_join_curvature=False,
            )
            optimized = _package_global_c2_fit(
                design_curve,
                annotation_path,
                design_index,
                segments,
                curves,
                solve_degree,
                validator,
                prior_weight=best_prior_weight,
            )
            optimized_score = _review_fit_visual_fairness_score(optimized)
            if optimized_score <= best_score:
                best = optimized
                best_score = optimized_score
        except Exception:
            pass
    if best is not None and _review_fit_has_visual_fairness_failure(best):
        try:
            curves = _solve_global_c2_bezier_chain(
                segments,
                degree=solve_degree,
                closed=closed,
                cv_priors=prior_cvs if best_prior_weight > 0.0 else None,
                cv_prior_weight=best_prior_weight,
                visual_optimize=True,
                hard_join_curvature=True,
            )
            hard_fit = _package_global_c2_fit(
                design_curve,
                annotation_path,
                design_index,
                segments,
                curves,
                solve_degree,
                validator,
                prior_weight=best_prior_weight,
            )
            hard_score = _review_fit_visual_fairness_score(hard_fit)
            if hard_score <= best_score or not _review_fit_has_visual_fairness_failure(hard_fit):
                best = hard_fit
                best_score = hard_score
        except Exception:
            pass
    if (
        best is not None
        and not isinstance(degree, int)
        and solve_degree < 7
        and _review_fit_has_visual_fairness_failure(best)
    ):
        # Degree 5 is the preferred Class-A minimum for simple G2 chains, but it
        # has no free CV after both endpoint C2 constraints are satisfied. If it
        # visibly loses the hand-routed path or creates a poor comb, retry the
        # same chain as degree 7 instead of exporting an underfit curve.
        return _fit_design_curve_chain_global_c2(
            design_curve,
            annotation_path,
            design_index,
            segments,
            7,
            validator,
        )
    if best is None:
        if not isinstance(degree, int) and solve_degree < 7:
            return _fit_design_curve_chain_global_c2(
                design_curve,
                annotation_path,
                design_index,
                segments,
                7,
                validator,
            )
        raise ValueError("global C2 solve failed")
    for curve, _candidate, _report in best:
        curve.metadata["class_a_g2_cv_prior_weight"] = float(best_prior_weight)
        curve.metadata["class_a_g2_visual_fairness_score"] = round(float(best_score), 4)
        curve.metadata["class_a_g2_auto_low_degree"] = bool(not isinstance(degree, int) and solve_degree == 5)
    _stamp_class_a_g2_diagnostics(best, closed=closed)
    return best


def _select_global_c2_solve_degree(
    segments: list[dict[str, Any]],
    requested_degree: int | str,
    *,
    closed: bool,
) -> int:
    if isinstance(requested_degree, int):
        return min(max(int(requested_degree), 5), 7)
    return 5 if _global_c2_chain_prefers_degree5(segments, closed=closed) else 7


def _global_c2_chain_prefers_degree5(segments: list[dict[str, Any]], *, closed: bool) -> bool:
    """Return True when a hand-split G2 chain should use the Class-A minimum degree.

    Degree 5 is the lowest practical single-span Bezier degree for a segment that
    may be C2-constrained at both ends. It is ideal for near-straight styling
    lines and broad one-direction arcs, but too restrictive for S-curves, tight
    blends, loops, and noisy multi-feature chains.
    """
    if not segments:
        return False
    if closed and len(segments) > 2:
        return False

    simple_or_arc = 0
    total_length = 0.0
    worst_angle = 0.0
    worst_sinuosity = 1.0
    worst_sag = 0.0
    for segment in segments:
        pts = remove_duplicate_points(np.asarray(segment.get("points"), dtype=float), eps=0.5)
        if len(pts) < 4:
            continue
        total_length += _polyline_length(pts)
        if _target_has_macro_s_shape(pts):
            return False
        simplicity = _target_curve_simplicity(pts)
        max_angle = float(simplicity.get("max_angle_deg", 180.0) or 180.0)
        sinuosity = float(simplicity.get("sinuosity", 999.0) or 999.0)
        sag_ratio = float(simplicity.get("sag_ratio", 999.0) or 999.0)
        worst_angle = max(worst_angle, max_angle)
        worst_sinuosity = max(worst_sinuosity, sinuosity)
        worst_sag = max(worst_sag, sag_ratio)

        if bool(simplicity.get("simple")):
            simple_or_arc += 1
            continue
        if bool(simplicity.get("smooth_arc")) and sinuosity < 1.14 and max_angle < 7.5 and sag_ratio < 0.12:
            simple_or_arc += 1
            continue
        return False

    if simple_or_arc == 0 or total_length < 10.0:
        return False
    # Long open roof/belt lines can have several gentle spans; keep them degree 5
    # as long as each span is visually calm. Tight or very wavy chains stay at 7.
    return bool(worst_angle < 11.0 and worst_sinuosity < 1.18 and worst_sag < 0.16)


def _package_global_c2_fit(
    design_curve: dict[str, Any],
    annotation_path: Path,
    design_index: int,
    segments: list[dict[str, Any]],
    curves: list[NURBSCurve],
    solve_degree: int,
    validator: ClassAValidator,
    *,
    prior_weight: float,
) -> list[tuple[NURBSCurve, CurveCandidate, QualityReport]]:
    fitted: list[tuple[NURBSCurve, CurveCandidate, QualityReport]] = []
    for segment_index, (curve, segment) in enumerate(zip(curves, segments, strict=False), start=1):
        points = segment["points"]
        label = _curve_label(
            design_curve,
            annotation_path,
            design_index,
            segment_index,
            segment["segment_count"],
        )
        candidate = _make_candidate(label, points, annotation_path, design_curve)
        curve.label = label
        curve.confidence = candidate.confidence
        curve.source = "manual_review_class_a_global_c2_fit"
        curve.metadata = _curve_metadata(
            design_curve,
            annotation_path,
            segment,
            points,
            segment_index,
            fit_policy="manual_class_a_global_c2_least_squares",
        )
        curve.metadata.update(_fit_metadata_notes(curve))
        curve.metadata["class_a_g2"] = True
        curve.metadata["class_a_g2_method"] = "global_constrained_least_squares_c0_c1_c2_with_cv_prior"
        curve.metadata["class_a_g2_global_degree"] = solve_degree
        curve.metadata["class_a_g2_cv_prior_weight"] = float(prior_weight)
        _stamp_cv_side_diagnostics(curve, points)
        _stamp_cv_target_corridor_diagnostics(curve, points)
        _stamp_curvature_sign_diagnostics(curve, points)
        _stamp_blend_fairness_diagnostics(curve, points)
        fitted.append((curve, candidate, validator.validate(curve, points)))
    return fitted


def _global_c2_cv_priors(segments: list[dict[str, Any]], *, degree: int) -> list[np.ndarray]:
    priors: list[np.ndarray] = []
    for index, segment in enumerate(segments):
        points = np.asarray(segment["points"], dtype=float)
        candidate = CurveCandidate(
            label=f"global_c2_prior_{index + 1:03d}",
            points=points,
            confidence=1.0,
            source="manual_review_prior",
        )
        try:
            curve = SingleSpanFitter(FittingOptions(degree=degree)).fit_candidate(candidate)
            curve = _refine_non_s_single_side_curve(curve, points)
            priors.append(np.asarray(curve.cvs, dtype=float))
        except Exception:
            priors.append(np.zeros((degree + 1, 3), dtype=float))
    return priors


def _solve_global_c2_bezier_chain(
    segments: list[dict[str, Any]],
    *,
    degree: int,
    closed: bool,
    cv_priors: list[np.ndarray] | None = None,
    cv_prior_weight: float = 0.0,
    visual_optimize: bool = False,
    hard_join_curvature: bool = False,
) -> list[NURBSCurve]:
    count = len(segments)
    cv_count = degree + 1
    var_count = count * cv_count
    rows: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    constraints: list[np.ndarray] = []
    constraint_rhs: list[np.ndarray] = []

    def var(seg_index: int, cv_index: int) -> int:
        return seg_index * cv_count + cv_index

    def add_row(coeffs: dict[int, float], value: np.ndarray, weight: float = 1.0) -> None:
        row = np.zeros(var_count, dtype=float)
        for index, coeff in coeffs.items():
            row[index] += float(coeff) * weight
        rows.append(row)
        rhs.append(np.asarray(value, dtype=float) * weight)

    def add_constraint(coeffs: dict[int, float], value: np.ndarray | None = None) -> None:
        row = np.zeros(var_count, dtype=float)
        for index, coeff in coeffs.items():
            row[index] += float(coeff)
        constraints.append(row)
        if value is None:
            constraint_rhs.append(np.zeros(3, dtype=float))
        else:
            constraint_rhs.append(np.asarray(value, dtype=float))

    prepared_points: list[np.ndarray] = []
    lengths: list[float] = []
    for seg_index, segment in enumerate(segments):
        pts = remove_duplicate_points(np.asarray(segment["points"], dtype=float), eps=0.5)
        if len(pts) < 4:
            raise ValueError("not enough distinct points for global C2 solve")
        sample_count = max(36, min(140, int(_polyline_length(pts) / 2.0)))
        pts = resample_polyline(pts, sample_count)
        pts = smooth_polyline(pts, window=5)
        prepared_points.append(pts)
        lengths.append(max(_polyline_length(pts), 1.0))
        u = chord_length_parameter(pts)
        basis = bernstein_basis(degree, u)
        for row_basis, point in zip(basis, pts, strict=False):
            add_row({var(seg_index, i): float(row_basis[i]) for i in range(cv_count)}, point, weight=1.0)

    if cv_priors is not None and cv_prior_weight > 0.0:
        for seg_index, cvs in enumerate(cv_priors[:count]):
            if len(cvs) != cv_count:
                continue
            for cv_index in range(1, degree):
                edge_factor = 1.45 if cv_index in (1, 2, degree - 2, degree - 1) else 1.0
                add_row(
                    {var(seg_index, cv_index): 1.0},
                    np.asarray(cvs[cv_index], dtype=float),
                    weight=float(cv_prior_weight) * edge_factor,
                )

    join_points = _global_chain_join_points(prepared_points, closed=closed)
    if closed:
        for seg_index in range(count):
            add_constraint({var(seg_index, 0): 1.0}, join_points[seg_index])
            add_constraint({var(seg_index, degree): 1.0}, join_points[(seg_index + 1) % count])
    else:
        for seg_index in range(count):
            add_constraint({var(seg_index, 0): 1.0}, join_points[seg_index])
            add_constraint({var(seg_index, degree): 1.0}, join_points[seg_index + 1])

    join_count = count if closed and count > 2 else count - 1
    for join_index in range(join_count):
        left = join_index
        right = (join_index + 1) % count
        left_len = lengths[left]
        right_len = lengths[right]
        p = float(degree)
        add_constraint(
            {
                var(left, degree): p / left_len,
                var(left, degree - 1): -p / left_len,
                var(right, 1): -p / right_len,
                var(right, 0): p / right_len,
            }
        )
        pp = float(degree * (degree - 1))
        add_constraint(
            {
                var(left, degree): pp / (left_len * left_len),
                var(left, degree - 1): -2.0 * pp / (left_len * left_len),
                var(left, degree - 2): pp / (left_len * left_len),
                var(right, 2): -pp / (right_len * right_len),
                var(right, 1): 2.0 * pp / (right_len * right_len),
                var(right, 0): -pp / (right_len * right_len),
            }
        )
        if hard_join_curvature:
            target_d2 = _global_join_curvature_prior_target(prepared_points, lengths, left, right)
            if target_d2 is not None:
                scale = pp / max(left_len * left_len, 1e-9)
                add_constraint(
                    {
                        var(left, degree): scale,
                        var(left, degree - 1): -2.0 * scale,
                        var(left, degree - 2): scale,
                    },
                    target_d2,
                )
        else:
            _add_global_join_curvature_prior_row(
                add_row,
                var,
                prepared_points,
                lengths,
                left,
                right,
                degree,
                weight=30.0,
            )

    _add_global_fairness_rows(rows, rhs, var, count, degree, var_count)

    a = np.vstack(rows)
    b = np.vstack(rhs)
    c = np.vstack(constraints)
    d = np.vstack(constraint_rhs)
    ata = a.T @ a
    atb = a.T @ b
    # Tiny diagonal damping protects nearly straight chains without visually moving CVs.
    ata += np.eye(var_count, dtype=float) * 1e-8
    zeros = np.zeros((c.shape[0], c.shape[0]), dtype=float)
    kkt = np.block([[ata, c.T], [c, zeros]])
    right = np.vstack([atb, d])
    solution, *_ = np.linalg.lstsq(kkt, right, rcond=None)
    cvs_flat = solution[:var_count]
    if visual_optimize:
        cvs_flat = _optimize_global_c2_visual_nullspace(
            cvs_flat,
            c,
            d,
            prepared_points,
            degree=degree,
            cv_priors=cv_priors,
        )

    curves: list[NURBSCurve] = []
    for seg_index in range(count):
        cvs = np.vstack([cvs_flat[var(seg_index, i)] for i in range(cv_count)])
        curves.append(
            NURBSCurve.single_span(
                label=f"class_a_segment_{seg_index + 1:03d}",
                degree=degree,
                cvs=cvs,
                confidence=1.0,
                source="manual_review_class_a_global_c2_fit",
                metadata={},
            )
        )
    return curves


def _add_global_join_curvature_prior_row(
    add_row,
    var,
    prepared_points: list[np.ndarray],
    lengths: list[float],
    left: int,
    right: int,
    degree: int,
    *,
    weight: float,
) -> None:
    target_d2 = _global_join_curvature_prior_target(prepared_points, lengths, left, right)
    if target_d2 is None:
        return
    pp = float(degree * (degree - 1))
    scale = pp / max(lengths[left] * lengths[left], 1e-9)
    add_row(
        {
            var(left, degree): scale,
            var(left, degree - 1): -2.0 * scale,
            var(left, degree - 2): scale,
        },
        target_d2,
        weight=weight,
    )


def _global_join_curvature_prior_target(
    prepared_points: list[np.ndarray],
    lengths: list[float],
    left: int,
    right: int,
) -> np.ndarray | None:
    left_points = prepared_points[left]
    right_points = prepared_points[right]
    local_len = max(min(lengths[left], lengths[right]), 1.0)
    tangent = _target_join_tangent(left_points, right_points)
    if np.linalg.norm(tangent) < 1e-9:
        return None
    normal = np.array([-tangent[1], tangent[0], 0.0], dtype=float)
    desired_sign = _global_join_desired_curvature_sign(left_points, right_points)
    if desired_sign == 0:
        return None
    samples = _join_curvature_samples(left_points, right_points)
    finite = np.asarray([abs(value) for value in samples if np.isfinite(value) and abs(value) > 1e-8], dtype=float)
    if len(finite):
        magnitude = float(np.nanpercentile(finite, 52)) * 0.42
    else:
        magnitude = 0.0
    magnitude = float(np.clip(magnitude, 0.00008, min(0.0042, 0.42 / local_len)))
    return normal * float(desired_sign) * magnitude


def _global_join_desired_curvature_sign(left_points: np.ndarray, right_points: np.ndarray) -> int:
    left_sign = _dominant_target_polyline_curvature_sign(left_points)
    right_sign = _dominant_target_polyline_curvature_sign(right_points)
    if left_sign and right_sign:
        if left_sign == right_sign:
            return int(left_sign)
        left_strength = _target_curvature_sign_strength(left_points, left_sign)
        right_strength = _target_curvature_sign_strength(right_points, right_sign)
        return int(left_sign if left_strength >= right_strength else right_sign)
    if left_sign:
        return int(left_sign)
    if right_sign:
        return int(right_sign)
    target = _target_join_curvature(left_points, right_points)
    if target > 1e-8:
        return 1
    if target < -1e-8:
        return -1
    return 0


def _target_curvature_sign_strength(points: np.ndarray, sign: int) -> float:
    pts = remove_duplicate_points(np.asarray(points, dtype=float), eps=0.5)
    if len(pts) < 5 or sign == 0:
        return 0.0
    pts = smooth_polyline(resample_polyline(pts, min(max(len(pts), 50), 120)), window=7)
    values = []
    for idx in range(1, len(pts) - 1):
        value = _signed_three_point_curvature(pts[idx - 1, :2], pts[idx, :2], pts[idx + 1, :2])
        if sign * value > 0.0:
            values.append(abs(value))
    return float(np.sum(values))


def _global_chain_join_points(points: list[np.ndarray], *, closed: bool) -> list[np.ndarray]:
    count = len(points)
    if closed:
        out: list[np.ndarray] = []
        for index in range(count):
            prev_end = points[(index - 1) % count][-1]
            this_start = points[index][0]
            out.append(0.5 * (prev_end + this_start))
        return out
    out = [points[0][0]]
    for index in range(count - 1):
        out.append(0.5 * (points[index][-1] + points[index + 1][0]))
    out.append(points[-1][-1])
    return out


def _add_global_fairness_rows(
    rows: list[np.ndarray],
    rhs: list[np.ndarray],
    var,
    segment_count: int,
    degree: int,
    var_count: int,
) -> None:
    cv_count = degree + 1

    def add_diff(order: int, lam: float) -> None:
        if lam <= 0 or cv_count <= order:
            return
        scale = float(lam) ** 0.5
        coeff = np.array([(-1) ** (order - k) * _binom(order, k) for k in range(order + 1)], dtype=float)
        for seg_index in range(segment_count):
            for start in range(cv_count - order):
                row = np.zeros(var_count, dtype=float)
                for off, c in enumerate(coeff):
                    row[var(seg_index, start + off)] += scale * float(c)
                rows.append(row)
                rhs.append(np.zeros(3, dtype=float))

    add_diff(2, 0.012)
    add_diff(3, 0.010)


def _fit_class_a_g2_segment(
    candidate: CurveCandidate,
    points: np.ndarray,
    degree: int | str,
    validator: ClassAValidator,
    *,
    start_constraint: _G2Constraint | None,
    end_constraint: _G2Constraint | None,
) -> NURBSCurve:
    degree_candidates = _class_a_degree_candidates(points, degree, start_constraint, end_constraint)
    best_curve: NURBSCurve | None = None
    best_score = float("inf")
    for candidate_degree in degree_candidates:
        try:
            fit_points = _prepare_fit_points(points)
            cvs = _fit_fixed_degree_with_g2_constraints(
                fit_points,
                candidate_degree,
                start_constraint=start_constraint,
                end_constraint=end_constraint,
            )
            curve = NURBSCurve.single_span(
                label=candidate.label,
                degree=candidate_degree,
                cvs=cvs,
                confidence=candidate.confidence,
                source="manual_review_class_a_g2_fit",
                metadata={"candidate_points": len(candidate.points)},
            )
            _fair_free_interior_cvs_guarded(curve, points)
            report = validator.validate(curve, points)
            score = _class_a_segment_score(curve, points, report)
            if score < best_score:
                best_score = score
                best_curve = curve
            if _class_a_segment_is_good_enough(curve, points, report):
                break
        except Exception:
            continue
    if best_curve is None:
        return _fit_lowest_degree(candidate, points, degree, validator)
    return best_curve


def _class_a_degree_candidates(
    points: np.ndarray,
    requested_degree: int | str,
    start_constraint: _G2Constraint | None,
    end_constraint: _G2Constraint | None,
) -> list[int]:
    if isinstance(requested_degree, int):
        degree = min(max(int(requested_degree), 3), 7)
        if start_constraint is not None and end_constraint is not None:
            degree = max(degree, 5)
        elif start_constraint is not None or end_constraint is not None:
            degree = max(degree, 4)
        return [degree]

    both = start_constraint is not None and end_constraint is not None
    one = (start_constraint is not None) != (end_constraint is not None)
    simplicity = _target_curve_simplicity(points)
    if both:
        # Two-sided G2 fixes P0/P1/P2 and Pn/Pn-1/Pn-2. Degree 5 has no free
        # interior CVs and degree 6 only has one, which makes the fit drift far
        # from the hand-routed skeleton. Degree 7 is still single-span but leaves
        # two free interior CVs for Alias-style shape control.
        return [7]
    if one:
        return [5, 6, 7, 4] if bool(simplicity["smooth_arc"]) else [6, 7, 5, 4]
    return [3, 4, 5, 6, 7]


def _class_a_segment_score(curve: NURBSCurve, points: np.ndarray, report: QualityReport) -> float:
    mean_error = float(report.metrics.get("chamfer_mean", 999.0))
    max_error = _bezier_max_error(curve, points)
    layout = _cv_layout_penalty(curve.cvs)
    dent = _cv_dent_penalty(curve.cvs)
    side = _cv_side_consistency_penalty(curve, points)
    corridor = _cv_target_corridor_penalty(curve, points)
    sign = 450.0 if _has_forbidden_curvature_sign_change(curve, points) else 0.0
    forbidden_side = 450.0 if _has_forbidden_cv_side_switch(curve, points) else 0.0
    blend = _curve_blend_fairness_metrics(curve, points)
    blend_penalty = 500.0 if bool(blend["forbidden"]) else float(blend.get("score", 0.0))
    warnings = len(report.warnings) * 60.0
    return float(
        warnings
        + mean_error * 18.0
        + max(0.0, max_error - _max_fit_budget(points)) * 9.0
        + layout * 4.0
        + dent * 8.0
        + side * 11.0
        + corridor * 7.0
        + sign
        + forbidden_side
        + blend_penalty
        + curve.degree * 0.08
    )


def _class_a_segment_is_good_enough(curve: NURBSCurve, points: np.ndarray, report: QualityReport) -> bool:
    if _has_forbidden_cv_side_switch(curve, points):
        return False
    if _has_forbidden_curvature_sign_change(curve, points):
        return False
    if bool(_curve_blend_fairness_metrics(curve, points)["forbidden"]):
        return False
    if _curve_exceeds_precision_budget(curve, points):
        return False
    mean_error = float(report.metrics.get("chamfer_mean", 999.0))
    return bool(mean_error <= _mean_fit_budget(points) * 0.82)


def _fit_design_curve_chain_fast(
    design_curve: dict[str, Any],
    annotation_path: Path,
    design_index: int,
    segments: list[dict[str, Any]],
    degree: int | str,
    validator: ClassAValidator,
) -> list[tuple[NURBSCurve, CurveCandidate, QualityReport]]:
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
        curve, report = _fit_fast_layout_curve(candidate, points, degree, validator)
        curve.source = "manual_review_fast_fit"
        curve.metadata = _curve_metadata(
            design_curve,
            annotation_path,
            segment,
            points,
            segment_index,
            fit_policy="fast_single_pass_single_span_export",
        )
        curve.metadata["fast_mode"] = True
        curve.metadata["fast_degree"] = curve.degree
        out.append((curve, candidate, report))
    return out


def _fit_design_curve_chain_precision(
    design_curve: dict[str, Any],
    annotation_path: Path,
    design_index: int,
    segments: list[dict[str, Any]],
    degree: int | str,
    validator: ClassAValidator,
) -> list[tuple[NURBSCurve, CurveCandidate, QualityReport]]:
    """Fit reviewed spans with image accuracy as the primary objective.

    This mode intentionally ignores Class-A CV aesthetics and G2 coupling. It is
    for logos, icons, decals and dense local details where the user wants the
    Alias curve to trace the routed skeleton as closely as possible.
    """
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
        curve, report = _fit_precision_curve(candidate, points, degree)
        precision_notes = dict(curve.metadata)
        curve.source = "manual_review_precision_fit"
        curve.metadata = _curve_metadata(
            design_curve,
            annotation_path,
            segment,
            points,
            segment_index,
            fit_policy="precision_fit_ignore_cv_aesthetic",
        )
        curve.metadata.update(_fit_metadata_notes(curve))
        curve.metadata["precision_fit"] = True
        curve.metadata["precision_fit_ignores_cv_aesthetic"] = True
        if "precision_fit_score" in precision_notes:
            curve.metadata["precision_fit_score"] = precision_notes["precision_fit_score"]
        if "precision_fit_sample_count" in precision_notes:
            curve.metadata["precision_fit_sample_count"] = precision_notes["precision_fit_sample_count"]
        for key, value in precision_notes.items():
            if str(key).startswith("precision_"):
                curve.metadata[key] = value
        curve.metadata["precision_fit_mean_error"] = round(_bezier_mean_error(curve, points), 4)
        curve.metadata["precision_fit_max_error"] = round(_bezier_max_error(curve, points), 4)
        out.append((curve, candidate, _precision_quality_report(curve, points)))
    return out


def _fit_precision_curve(
    candidate: CurveCandidate,
    points: np.ndarray,
    degree: int | str,
) -> tuple[NURBSCurve, QualityReport]:
    if isinstance(degree, int):
        return _fit_best_precision_regularization(candidate, points, int(degree))

    trials: list[tuple[int, float, float, bool, NURBSCurve, QualityReport]] = []
    for candidate_degree in _precision_degree_candidates(points):
        curve, report, score = _fit_scored_precision_degree(candidate, points, candidate_degree)
        mean_error = _bezier_mean_error(curve, points)
        max_error = _bezier_max_error(curve, points)
        fit_score = _precision_fit_score(mean_error, max_error)
        stability = _precision_cv_stability_metrics(curve, points)
        editable = _precision_cv_is_editable(stability, points)
        trials.append((candidate_degree, fit_score, score, editable, curve, report))
    if not trials:
        raise ValueError("failed to fit precision curve")
    selected = _select_precision_trial(trials, points)
    selected_degree, fit_score, total_score, editable, curve, report = selected
    curve.metadata["precision_fit_score"] = round(float(total_score), 4)
    curve.metadata["precision_fit_selection_score"] = round(float(fit_score), 4)
    curve.metadata["precision_fit_selected_lowest_close_degree"] = True
    curve.metadata["precision_fit_cv_editable"] = bool(editable)
    curve.metadata["precision_fit_selected_degree"] = int(selected_degree)
    return curve, report


def _fit_best_precision_regularization(
    candidate: CurveCandidate,
    points: np.ndarray,
    degree: int,
) -> tuple[NURBSCurve, QualityReport]:
    curve, report, score = _fit_scored_precision_degree(candidate, points, degree)
    curve.metadata["precision_fit_score"] = round(float(score), 4)
    return curve, report


def _fit_scored_precision_degree(
    candidate: CurveCandidate,
    points: np.ndarray,
    degree: int,
) -> tuple[NURBSCurve, QualityReport, float]:
    best: tuple[float, NURBSCurve, QualityReport] | None = None
    for regularization in _precision_regularization_candidates(degree):
        curve = _fit_precision_fixed_degree(candidate, points, degree, regularization_strength=regularization)
        report = _precision_quality_report(curve, points)
        mean_error = _bezier_mean_error(curve, points)
        max_error = _bezier_max_error(curve, points)
        stability = _precision_cv_stability_metrics(curve, points)
        score = (
            _precision_fit_score(mean_error, max_error)
            + float(stability["penalty"])
            + degree * PRECISION_FIT_DEGREE_TIEBREAK_WEIGHT
        )
        if best is None or score < best[0]:
            best = (score, curve, report)
    if best is None:
        raise ValueError("failed to fit precision curve")
    return best[1], best[2], best[0]


def _precision_fit_score(mean_error: float, max_error: float) -> float:
    return float(mean_error) + float(max_error) * 0.35


def _select_precision_trial(
    trials: list[tuple[int, float, float, bool, NURBSCurve, QualityReport]],
    points: np.ndarray,
) -> tuple[int, float, float, bool, NURBSCurve, QualityReport]:
    """Choose the lowest degree that is visually as accurate as the best fit.

    Precision mode does not have a fixed degree. It first rejects CV polygons
    that leave the target corridor, then finds the best remaining image fit.
    Among curves within a small perceptual error band of that best fit, it
    chooses the lowest degree. This keeps the curve close to the source without
    producing high-degree flying control polygons.
    """
    editable_trials = [trial for trial in trials if trial[3]]
    pool = editable_trials or trials
    best_fit = min(trial[1] for trial in pool)
    tolerance = _precision_fit_equivalence_tolerance(points, best_fit)
    close = [trial for trial in pool if trial[1] <= best_fit + tolerance]
    if close:
        return min(
            close,
            key=lambda trial: (
                trial[0],
                _precision_cv_selection_penalty(trial[4], points),
                trial[1],
                trial[2],
            ),
        )
    return min(
        pool,
        key=lambda trial: (
            trial[1],
            _precision_cv_selection_penalty(trial[4], points),
            trial[0],
        ),
    )


def _precision_fit_equivalence_tolerance(points: np.ndarray, best_fit_score: float) -> float:
    length = max(_polyline_length(np.asarray(points, dtype=float)), 1.0)
    return max(0.42, min(6.5, best_fit_score * 0.18 + length * 0.0045))


def _precision_cv_selection_penalty(curve: NURBSCurve, points: np.ndarray) -> float:
    stability = _precision_cv_stability_metrics(curve, points)
    return float(stability["penalty"]) + max(0.0, float(stability["precision_cv_polyline_ratio"]) - 1.8) * 18.0


def _precision_regularization_candidates(degree: int) -> tuple[float, ...]:
    if degree <= 7:
        return (0.22, 0.42, 0.75, 1.1)
    if degree <= 13:
        return (0.18, 0.34, 0.58, 0.92, 1.35)
    return (0.14, 0.26, 0.46, 0.74, 1.08, 1.55)


def _fit_precision_fixed_degree(
    candidate: CurveCandidate,
    points: np.ndarray,
    degree: int,
    *,
    regularization_strength: float = 1.0,
) -> NURBSCurve:
    degree = min(max(int(degree), 3), PRECISION_FIT_MAX_DEGREE)
    sample_count = max(260, min(1200, int(len(points) * 1.6)))
    fit_points = remove_duplicate_points(np.asarray(points, dtype=float), eps=0.35)
    fit_points = resample_polyline(fit_points, max(sample_count, degree + 8))
    u = chord_length_parameter(fit_points)
    basis = bernstein_basis(degree, u)
    p0 = fit_points[0]
    p1 = fit_points[-1]
    fixed = basis[:, [0]] * p0 + basis[:, [-1]] * p1
    rhs = fit_points - fixed
    a = basis[:, 1:-1]
    if a.shape[1] > 0:
        a_aug, rhs_aug = _precision_augmented_system(
            a,
            rhs,
            fit_points,
            degree,
            p0,
            p1,
            regularization_strength=regularization_strength,
        )
        interior, *_ = np.linalg.lstsq(a_aug, rhs_aug, rcond=None)
        cvs = np.vstack([p0, interior, p1])
    else:
        cvs = np.vstack([p0, p1])
    cvs = _clamp_precision_cvs_to_target_corridor(cvs, fit_points)
    curve = NURBSCurve.single_span(
        label=candidate.label,
        degree=degree,
        cvs=cvs,
        confidence=candidate.confidence,
        source="manual_review_precision_fit",
        metadata={"candidate_points": len(candidate.points)},
    )
    curve.metadata["precision_fit_sample_count"] = sample_count
    curve.metadata["precision_fit_max_degree"] = PRECISION_FIT_MAX_DEGREE
    curve.metadata["precision_fit_regularization"] = round(float(regularization_strength), 4)
    curve.metadata.update(_precision_cv_stability_metrics(curve, points))
    return curve


def _precision_degree_candidates(points: np.ndarray) -> tuple[int, ...]:
    distinct = len(remove_duplicate_points(np.asarray(points, dtype=float), eps=0.5))
    max_degree = min(PRECISION_FIT_MAX_DEGREE, max(7, distinct - 1))
    base = [3, 5, 7, 9, 11, 12, 13, 16, 20, 24]
    degrees = [degree for degree in base if degree <= max_degree]
    if max_degree not in degrees:
        degrees.append(max_degree)
    return tuple(dict.fromkeys(degrees))


def _precision_augmented_system(
    fit_matrix: np.ndarray,
    fit_rhs: np.ndarray,
    fit_points: np.ndarray,
    degree: int,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    regularization_strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Regularize high-degree precision fits so CVs stay editable in Alias.

    A pure Bernstein least-squares solve is numerically legal but badly
    conditioned at high degree. It may trace the pixels while sending control
    vertices far away from the curve. The extra rows keep the unknown interior
    CVs close to a chord-length prior and preserve the prior's second
    difference rhythm.
    """
    interior_count = max(degree - 1, 0)
    if interior_count <= 0:
        return fit_matrix, fit_rhs

    length = max(_polyline_length(fit_points), 1.0)
    prior = resample_polyline(fit_points, degree + 1)
    prior = _smooth_precision_cv_prior(prior, fit_points)

    rows: list[np.ndarray] = [fit_matrix]
    rhs_rows: list[np.ndarray] = [fit_rhs]

    sample_count = max(len(fit_points), 1)
    strength = max(0.05, float(regularization_strength))
    prior_weight = max(0.35, min(80.0, (sample_count / max(interior_count, 1)) * (0.28 + degree * 0.038) * strength))
    rows.append(np.eye(interior_count, dtype=float) * np.sqrt(prior_weight))
    rhs_rows.append(prior[1:-1] * np.sqrt(prior_weight))

    if interior_count >= 2:
        diff_rows, diff_rhs = _precision_difference_prior_rows(prior, p0, p1, order=1)
        if len(diff_rows):
            rhythm_weight = max(0.8, min(18.0, prior_weight * 0.28))
            rows.append(diff_rows * np.sqrt(rhythm_weight))
            rhs_rows.append(diff_rhs * np.sqrt(rhythm_weight))

    if interior_count >= 3:
        diff2_rows, diff2_rhs = _precision_difference_prior_rows(prior, p0, p1, order=2)
        if len(diff2_rows):
            smooth_weight = max(0.35, min(12.0, prior_weight * 0.16))
            rows.append(diff2_rows * np.sqrt(smooth_weight))
            rhs_rows.append(diff2_rhs * np.sqrt(smooth_weight))

    ridge = max(1e-5, min(0.018, 0.0004 + degree * degree * 0.000018))
    rows.append(np.eye(interior_count, dtype=float) * np.sqrt(ridge))
    rhs_rows.append(prior[1:-1] * np.sqrt(ridge))

    augmented_a = np.vstack(rows)
    augmented_rhs = np.vstack(rhs_rows)
    if not np.all(np.isfinite(augmented_a)) or not np.all(np.isfinite(augmented_rhs)):
        return fit_matrix, fit_rhs
    # Length is intentionally referenced so future tuning can remain
    # length-scaled without changing the public behavior.
    _ = length
    return augmented_a, augmented_rhs


def _smooth_precision_cv_prior(prior: np.ndarray, fit_points: np.ndarray) -> np.ndarray:
    if len(prior) < 5:
        return prior
    smoothed = prior.copy()
    # Keep endpoints exact; only calm the interior reference polygon.
    for idx in range(1, len(prior) - 1):
        smoothed[idx] = prior[idx] * 0.5 + (prior[idx - 1] + prior[idx + 1]) * 0.25
    smoothed[0] = fit_points[0]
    smoothed[-1] = fit_points[-1]
    return smoothed


def _precision_difference_prior_rows(
    prior: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    degree = len(prior) - 1
    interior_count = max(degree - 1, 0)
    if interior_count <= 0:
        return np.zeros((0, 0), dtype=float), np.zeros((0, 3), dtype=float)

    full_rows: list[np.ndarray] = []
    if order == 1:
        for idx in range(degree):
            row = np.zeros(degree + 1, dtype=float)
            row[idx] = -1.0
            row[idx + 1] = 1.0
            full_rows.append(row)
    elif order == 2:
        for idx in range(1, degree):
            row = np.zeros(degree + 1, dtype=float)
            row[idx - 1] = 1.0
            row[idx] = -2.0
            row[idx + 1] = 1.0
            full_rows.append(row)
    else:
        return np.zeros((0, interior_count), dtype=float), np.zeros((0, 3), dtype=float)

    full = np.vstack(full_rows)
    target = full @ prior
    boundary = full[:, [0]] * p0 + full[:, [-1]] * p1
    return full[:, 1:-1], target - boundary


def _clamp_precision_cvs_to_target_corridor(cvs: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    target = remove_duplicate_points(np.asarray(target_points, dtype=float), eps=0.5)
    if len(target) < 4 or len(cvs) < 4:
        return cvs
    length = max(_polyline_length(target), 1.0)
    # This is a guardrail, not a projector onto the curve. Bezier CVs may sit
    # away from the visible curve, but they must not leave the design corridor.
    normal_limit = max(10.0, min(90.0, length * 0.18))
    tangent_limit = max(12.0, min(110.0, length * 0.22))
    out = np.asarray(cvs, dtype=float).copy()
    for cv_index in range(1, len(out) - 1):
        fraction = cv_index / float(len(out) - 1)
        ref, tangent = _target_point_tangent_at_fraction(target, fraction)
        if np.linalg.norm(tangent) < 1e-9:
            continue
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        delta = out[cv_index, :2] - ref
        along = float(np.clip(np.dot(delta, tangent), -tangent_limit, tangent_limit))
        side = float(np.clip(np.dot(delta, normal), -normal_limit, normal_limit))
        out[cv_index, :2] = ref + tangent * along + normal * side
    return out


def _precision_cv_stability_metrics(curve: NURBSCurve, target_points: np.ndarray) -> dict[str, float | int | bool]:
    target = remove_duplicate_points(np.asarray(target_points, dtype=float), eps=0.5)
    cvs = np.asarray(curve.cvs, dtype=float)
    length = max(_polyline_length(target), 1.0)
    if len(target) < 4 or len(cvs) < 4:
        return {
            "precision_cv_stability_penalty": 0.0,
            "precision_cv_max_reference_distance": 0.0,
            "precision_cv_polyline_ratio": 1.0,
            "precision_cv_corridor_penalty": 0.0,
            "penalty": 0.0,
        }

    ref_distances: list[float] = []
    tangent_offsets: list[float] = []
    normal_offsets: list[float] = []
    for cv_index in range(1, len(cvs) - 1):
        fraction = cv_index / float(len(cvs) - 1)
        ref, tangent = _target_point_tangent_at_fraction(target, fraction)
        if np.linalg.norm(tangent) < 1e-9:
            continue
        delta = cvs[cv_index, :2] - ref
        ref_distances.append(float(np.linalg.norm(delta)))
        tangent_offsets.append(abs(float(np.dot(delta, tangent))))
        normal_offsets.append(abs(float(tangent[0] * delta[1] - tangent[1] * delta[0])))

    cv_length = _polyline_length(cvs)
    ratio = cv_length / length
    max_ref = max(ref_distances) if ref_distances else 0.0
    max_tangent = max(tangent_offsets) if tangent_offsets else 0.0
    max_normal = max(normal_offsets) if normal_offsets else 0.0
    corridor = _cv_target_corridor_metrics(curve, target)
    corridor_penalty = float(corridor["penalty"])
    distance_limit = max(14.0, min(120.0, length * 0.24))
    tangent_limit = max(16.0, min(130.0, length * 0.26))
    normal_limit = max(11.0, min(95.0, length * 0.20))

    penalty = 0.0
    penalty += max(0.0, ratio - 2.9) * 30.0
    penalty += max(0.0, max_ref - distance_limit) / max(distance_limit, 1.0) * 115.0
    penalty += max(0.0, max_tangent - tangent_limit) / max(tangent_limit, 1.0) * 52.0
    penalty += max(0.0, max_normal - normal_limit) / max(normal_limit, 1.0) * 78.0
    penalty += corridor_penalty * 0.42
    penalty += _cv_layout_penalty(cvs) * 0.22
    penalty += _cv_dent_penalty(cvs) * 0.34
    if int(corridor["target_side_switches"]) > 0:
        penalty += 90.0 * int(corridor["target_side_switches"])
    if int(corridor["wrong_side_count"]) > 0:
        penalty += 70.0 * int(corridor["wrong_side_count"])

    return {
        "precision_cv_stability_penalty": round(float(penalty), 4),
        "precision_cv_max_reference_distance": round(float(max_ref), 4),
        "precision_cv_max_tangent_offset": round(float(max_tangent), 4),
        "precision_cv_max_normal_offset": round(float(max_normal), 4),
        "precision_cv_polyline_ratio": round(float(ratio), 4),
        "precision_cv_corridor_penalty": round(float(corridor_penalty), 4),
        "precision_cv_side_switches": int(corridor["target_side_switches"]),
        "precision_cv_wrong_side_count": int(corridor["wrong_side_count"]),
        "penalty": float(penalty),
    }


def _precision_cv_is_editable(metrics: dict[str, float | int | bool], target_points: np.ndarray) -> bool:
    length = max(_polyline_length(np.asarray(target_points, dtype=float)), 1.0)
    max_reference = float(metrics.get("precision_cv_max_reference_distance", 0.0) or 0.0)
    max_normal = float(metrics.get("precision_cv_max_normal_offset", 0.0) or 0.0)
    ratio = float(metrics.get("precision_cv_polyline_ratio", 1.0) or 1.0)
    penalty = float(metrics.get("precision_cv_stability_penalty", 0.0) or 0.0)
    side_switches = int(metrics.get("precision_cv_side_switches", 0) or 0)
    wrong_side = int(metrics.get("precision_cv_wrong_side_count", 0) or 0)

    reference_limit = max(18.0, min(160.0, length * 0.30))
    normal_limit = max(12.0, min(110.0, length * 0.22))
    if max_reference > reference_limit:
        return False
    if max_normal > normal_limit:
        return False
    if ratio > 3.6:
        return False
    if penalty > 170.0:
        return False
    if wrong_side > 0:
        return False
    if side_switches > 0 and not _target_has_macro_s_shape(np.asarray(target_points, dtype=float)):
        return False
    return True


def _precision_quality_report(curve: NURBSCurve, target_points: np.ndarray) -> QualityReport:
    mean_error = _bezier_mean_error(curve, target_points)
    max_error = _bezier_max_error(curve, target_points)
    length = max(_polyline_length(target_points), 1.0)
    mean_budget = max(2.4, min(9.0, length * 0.012))
    max_budget = max(7.0, min(28.0, length * 0.04))
    stability = _precision_cv_stability_metrics(curve, target_points)
    warnings: list[str] = []
    if mean_error > mean_budget:
        warnings.append("precision mean fit error is above target")
    if max_error > max_budget:
        warnings.append("precision max fit error is above target")
    if float(stability["precision_cv_stability_penalty"]) > 120.0:
        warnings.append("precision CV polygon leaves the target corridor")
    if int(stability["precision_cv_side_switches"]) > 0:
        warnings.append("precision CV polygon changes side on a non-S target")
    metrics: dict[str, float | int | bool | list[float] | str] = {
        "degree": curve.degree,
        "span": curve.span_count,
        "single_span": curve.is_single_span,
        "cv_count": len(curve.cvs),
        "knot_count": len(curve.knots),
        "precision_fit": True,
        "precision_mean_error": float(mean_error),
        "precision_max_error": float(max_error),
        "precision_mean_budget": float(mean_budget),
        "precision_max_budget": float(max_budget),
        "degree_limit_ignored": bool(curve.degree > 7),
    }
    metrics.update(stability)
    return QualityReport(label=curve.label, passed=len(warnings) == 0, metrics=metrics, warnings=warnings)


def _fit_fast_layout_curve(
    candidate: CurveCandidate,
    points: np.ndarray,
    degree: int | str,
    validator: ClassAValidator,
) -> tuple[NURBSCurve, QualityReport]:
    if isinstance(degree, int):
        curve = SingleSpanFitter(FittingOptions(degree=int(degree))).fit_candidate(candidate)
        return curve, validator.validate(curve, points)

    base_degree = _fast_degree_for_points(points)
    simplicity = _target_curve_simplicity(points)
    if bool(simplicity.get("simple")):
        degrees = tuple(dict.fromkeys([base_degree, min(base_degree + 1, 7), 5]))
    elif base_degree >= 7:
        degrees = (5, 6, 7)
    else:
        degrees = tuple(dict.fromkeys([3, base_degree, min(base_degree + 1, 7), 7]))

    best: tuple[float, NURBSCurve, QualityReport] | None = None
    for candidate_degree in degrees:
        curve = SingleSpanFitter(FittingOptions(degree=candidate_degree)).fit_candidate(candidate)
        report = validator.validate(curve, points)
        score = _fast_layout_score(curve, report)
        if best is None or score < best[0]:
            best = (score, curve, report)
        if report.passed and candidate_degree <= base_degree:
            break
    if best is None:
        raise ValueError("failed to fit fast layout curve")
    best[1].metadata["fast_layout_score"] = round(float(best[0]), 4)
    return best[1], best[2]


def _fast_layout_score(curve: NURBSCurve, report: QualityReport) -> float:
    metrics = report.metrics
    inflections = int(metrics.get("inflection_count", 0) or 0)
    spacing_rhythm = float(metrics.get("cv_spacing_rhythm_penalty", 0.0) or 0.0)
    distance_rhythm = float(metrics.get("cv_curve_distance_rhythm_penalty", 0.0) or 0.0)
    spacing = float(metrics.get("cv_spacing_ratio", 1.0) or 1.0)
    chamfer = float(metrics.get("chamfer_mean", 0.0) or 0.0)
    oscillation = float(metrics.get("curvature_oscillation", 0.0) or 0.0)
    turnback = bool(metrics.get("control_polygon_turnback", False))
    return float(
        len(report.warnings) * 1800.0
        + max(0, inflections - 1) * 9000.0
        + spacing_rhythm * 130.0
        + distance_rhythm * 110.0
        + max(0.0, spacing - 5.0) * 220.0
        + chamfer * 24.0
        + max(0.0, oscillation - 0.55) * 800.0
        + (12000.0 if turnback else 0.0)
        + curve.degree * 28.0
    )


def _fast_degree_for_points(points: np.ndarray) -> int:
    simplicity = _target_curve_simplicity(points)
    if bool(simplicity.get("simple")):
        max_angle = float(simplicity.get("max_angle_deg", 0.0) or 0.0)
        return 3 if max_angle < 9.0 else 4
    sinuosity = float(simplicity.get("sinuosity", 1.0) or 1.0)
    max_angle = float(simplicity.get("max_angle_deg", 0.0) or 0.0)
    if sinuosity > 1.22 or max_angle > 95.0:
        return 7
    return 5


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


def _fit_metadata_notes(curve: NURBSCurve) -> dict[str, Any]:
    prefixes = (
        "non_s_",
        "degree_selected_",
        "simplicity_",
    )
    return {key: value for key, value in curve.metadata.items() if any(key.startswith(prefix) for prefix in prefixes)}


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
    if not ENABLE_G2_CONSTRAINTS:
        return
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
                promoted = _refine_non_s_single_side_curve(promoted, points)
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
                hard_failure = (
                    _has_forbidden_cv_side_switch(promoted, points)
                    or _has_forbidden_curvature_sign_change(promoted, points)
                    or bool(_curve_blend_fairness_metrics(promoted, points)["forbidden"])
                    or _curve_exceeds_precision_budget(promoted, points)
                )
                if hard_failure:
                    continue
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


def _refit_chain_with_current_g2_constraints(
    fitted_curves: list[tuple[NURBSCurve, CurveCandidate, np.ndarray, dict[str, Any], int]],
    validator: ClassAValidator,
    *,
    requested_degree: int | str,
    closed: bool,
) -> None:
    if not ENABLE_G2_CONSTRAINTS:
        return
    if len(fitted_curves) <= 1:
        return
    curves = [item[0] for item in fitted_curves]
    target_segments = [item[2] for item in fitted_curves]
    base_score = _chain_total_quality_score(curves, target_segments, validator, closed=closed)
    base_jump = _chain_total_endpoint_jump(curves, target_segments, closed=closed)
    if base_jump < 180.0:
        return

    constraints = _current_chain_g2_constraints(curves, closed=closed)
    trial_curves: list[NURBSCurve] = []
    for index, (curve, candidate, points, segment, segment_index) in enumerate(fitted_curves):
        start_constraint = None
        end_constraint = None
        if index > 0 or (closed and len(fitted_curves) > 2):
            start_constraint = constraints.get((index - 1) % len(fitted_curves))
        if index < len(fitted_curves) - 1 or (closed and len(fitted_curves) > 2):
            end_constraint = constraints.get(index)
        if start_constraint is None and end_constraint is None:
            trial_curves.append(_clone_curve(curve))
            continue
        if isinstance(requested_degree, int):
            refit_degree: int | str = max(int(requested_degree), curve.degree)
        elif start_constraint is not None and end_constraint is not None:
            refit_degree = 7
        else:
            refit_degree = max(curve.degree, 5)
        refit = _fit_g2_constrained_segment(
            candidate,
            points,
            start_constraint=start_constraint,
            end_constraint=end_constraint,
            requested_degree=refit_degree,
        )
        refit.source = curve.source
        refit.metadata = dict(curve.metadata)
        refit.metadata["g2_constraint_refit"] = True
        refit.metadata["g2_constraint_refit_from_degree"] = curve.degree
        refit.metadata["g2_constraint_refit_to_degree"] = refit.degree
        trial_curves.append(refit)

    _fair_free_interior_cvs_for_curves(trial_curves, target_segments=target_segments, closed=closed)
    if not _chain_passes_g0_g1(trial_curves, target_segments, closed=closed):
        return
    trial_score = _chain_total_quality_score(trial_curves, target_segments, validator, closed=closed)
    trial_jump = _chain_total_endpoint_jump(trial_curves, target_segments, closed=closed)
    if trial_jump < base_jump * 0.60 and trial_score < base_score * 0.96:
        for index, updated in enumerate(trial_curves):
            old = fitted_curves[index]
            fitted_curves[index] = (updated, old[1], old[2], old[3], old[4])


def _current_chain_g2_constraints(curves: list[NURBSCurve], *, closed: bool) -> dict[int, _G2Constraint]:
    constraints: dict[int, _G2Constraint] = {}
    if len(curves) <= 1:
        return constraints
    join_count = len(curves) if closed and len(curves) > 2 else len(curves) - 1
    for join_index in range(join_count):
        left = curves[join_index]
        right = curves[(join_index + 1) % len(curves)]
        point = 0.5 * (left.cvs[-1] + right.cvs[0])
        d1 = 0.5 * (_bezier_d1_end(left) + _bezier_d1_start(right))
        if np.linalg.norm(d1[:2]) < 1e-8:
            d1 = _bezier_d1_end(left) if np.linalg.norm(_bezier_d1_end(left)[:2]) > 1e-8 else _bezier_d1_start(right)
        d2 = 0.5 * (_bezier_d2_end(left) + _bezier_d2_start(right))
        constraints[join_index] = _G2Constraint(point=point.copy(), d1=d1.copy(), d2=d2.copy())
    return constraints


def _chain_passes_g0_g1(curves: list[NURBSCurve], target_segments: list[np.ndarray], *, closed: bool) -> bool:
    if len(curves) <= 1:
        return True
    join_count = len(curves) if closed and len(curves) > 2 else len(curves) - 1
    for join_index in range(join_count):
        local_len = max(
            min(
                _polyline_length(target_segments[join_index]),
                _polyline_length(target_segments[(join_index + 1) % len(curves)]),
            ),
            1.0,
        )
        if not _join_passes_g0_g1(curves[join_index], curves[(join_index + 1) % len(curves)], local_len):
            return False
    return True


def _chain_total_endpoint_jump(curves: list[NURBSCurve], target_segments: list[np.ndarray], *, closed: bool) -> float:
    return float(sum(_curve_endpoint_jump_score(curves, target_segments, index, closed=closed) for index in range(len(curves))))


def _chain_total_quality_score(
    curves: list[NURBSCurve],
    target_segments: list[np.ndarray],
    validator: ClassAValidator,
    *,
    closed: bool,
) -> float:
    score = 0.0
    for index in range(len(curves)):
        score += _chain_local_quality_score(curves, target_segments, validator, index, closed=closed)
    return float(score)


def _repair_cv_side_flips(
    fitted_curves: list[tuple[NURBSCurve, CurveCandidate, np.ndarray, dict[str, Any], int]],
    validator: ClassAValidator,
    *,
    closed: bool,
) -> None:
    for index, (curve, candidate, points, segment, segment_index) in enumerate(list(fitted_curves)):
        current_side = _cv_side_consistency_penalty(curve, points)
        current_corridor = _cv_target_corridor_penalty(curve, points)
        current_forbidden = _has_forbidden_cv_side_switch(curve, points)
        current_curv_forbidden = _has_forbidden_curvature_sign_change(curve, points)
        current_blend_forbidden = bool(_curve_blend_fairness_metrics(curve, points)["forbidden"])
        if not current_forbidden and not current_curv_forbidden and not current_blend_forbidden and current_side <= 32.0 and current_corridor <= 36.0:
            continue
        best_curve = curve
        best_score = _side_repair_score(curve, points, validator)
        start_constrained = closed or index > 0
        end_constrained = closed or index < len(fitted_curves) - 1
        min_degree = 5 if start_constrained and end_constrained else 3
        for candidate_degree in range(min_degree, 8):
            try:
                trial = SingleSpanFitter(FittingOptions(degree=candidate_degree)).fit_candidate(candidate)
                trial = _refine_non_s_single_side_curve(trial, points)
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
        repaired_forbidden = _has_forbidden_cv_side_switch(best_curve, points)
        repaired_curv_forbidden = _has_forbidden_curvature_sign_change(best_curve, points)
        repaired_blend_forbidden = bool(_curve_blend_fairness_metrics(best_curve, points)["forbidden"])
        if (
            best_curve is not curve
            and not repaired_forbidden
            and not repaired_curv_forbidden
            and not repaired_blend_forbidden
            and repaired_side + repaired_corridor <= current_side + current_corridor - (18.0 if current_forbidden else 32.0)
            and not _curve_exceeds_precision_budget(best_curve, points)
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
    forbidden_side = _has_forbidden_cv_side_switch(curve, points)
    forbidden_curvature = _has_forbidden_curvature_sign_change(curve, points)
    blend_fairness = _curve_blend_fairness_metrics(curve, points)
    exceeds_precision = _curve_exceeds_precision_budget(curve, points)
    return (
        side * 14.0
        + corridor * 14.0
        + len(report.warnings) * 360.0
        + max(0.0, chamfer - 2.6) * 58.0
        + max(0.0, spacing - 4.2) * 45.0
        + max(0.0, oscillation - 0.52) * 120.0
        + dent * 0.8
        + (10000.0 if forbidden_side else 0.0)
        + (10000.0 if forbidden_curvature else 0.0)
        + (8000.0 if bool(blend_fairness["forbidden"]) else 0.0)
        + float(blend_fairness["penalty"]) * 36.0
        + _curvature_sign_penalty(curve, points)
        + (5000.0 if exceeds_precision else 0.0)
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
        forbidden_side = _has_forbidden_cv_side_switch(curve, points)
        forbidden_curvature = _has_forbidden_curvature_sign_change(curve, points)
        blend_forbidden = bool(_curve_blend_fairness_metrics(curve, points)["forbidden"])
        jump = _curve_endpoint_jump_score(
            [item[0] for item in fitted_curves],
            target_segments,
            index,
            closed=closed,
        )
        report = validator.validate(curve, points)
        if curve.degree >= 7:
            continue
        if report.passed and not forbidden_side and not forbidden_curvature and not blend_forbidden and side < 120.0 and corridor < 120.0 and jump < 130.0:
            continue

        start_constrained = closed or index > 0
        end_constrained = closed or index < len(fitted_curves) - 1
        min_degree = 5 if start_constrained and end_constrained else 3
        best_curves: list[NURBSCurve] | None = None
        best_passes = bool(report.passed)
        best_jump = jump
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
                trial = _refine_non_s_single_side_curve(trial, points)
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
                _fair_free_interior_cvs_for_curves(trial_curves, target_segments=target_segments, closed=closed)
                trial_report = validator.validate(trial_curves[index], points)
                trial_has_hard_failure = (
                    _has_forbidden_cv_side_switch(trial_curves[index], points)
                    or _has_forbidden_curvature_sign_change(trial_curves[index], points)
                    or bool(_curve_blend_fairness_metrics(trial_curves[index], points)["forbidden"])
                    or _curve_exceeds_precision_budget(trial_curves[index], points)
                )
                if trial_has_hard_failure:
                    continue
                score = _chain_local_quality_score(trial_curves, target_segments, validator, index, closed=closed)
                score += _cv_side_consistency_penalty(trial_curves[index], points) * 2.8
                score += _cv_target_corridor_penalty(trial_curves[index], points) * 2.8
                if _has_forbidden_cv_side_switch(trial_curves[index], points):
                    score += 10000.0
                if _has_forbidden_curvature_sign_change(trial_curves[index], points):
                    score += 10000.0
                blend_fairness = _curve_blend_fairness_metrics(trial_curves[index], points)
                if bool(blend_fairness["forbidden"]):
                    score += 8000.0
                score += float(blend_fairness["penalty"]) * 36.0
                score += _curvature_sign_penalty(trial_curves[index], points)
                if _curve_exceeds_precision_budget(trial_curves[index], points):
                    score += 5000.0
                trial_jump = _curve_endpoint_jump_score(trial_curves, target_segments, index, closed=closed)
                score += trial_jump * 6.0
                jump_improved = jump > 180.0 and trial_jump < best_jump * 0.88
                if (not best_passes and trial_report.passed) or score < best_score or jump_improved:
                    best_score = score
                    best_jump = trial_jump
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
        baseline_jump = _curve_endpoint_jump_score(
            [item[0] for item in fitted_curves],
            [item[2] for item in fitted_curves],
            index,
            closed=closed,
        )
        for target_degree in target_degrees:
            if target_degree >= curve.degree:
                continue
            trial_curves = [_clone_curve(item[0]) for item in fitted_curves]
            trial = SingleSpanFitter(FittingOptions(degree=target_degree)).fit_candidate(candidate)
            trial = _refine_non_s_single_side_curve(trial, points)
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
            _fair_free_interior_cvs_for_curves(trial_curves, target_segments=target_segments, closed=closed)
            report = validator.validate(trial_curves[index], points)
            if not report.passed:
                continue
            if _has_forbidden_cv_side_switch(trial_curves[index], points):
                continue
            if _has_forbidden_curvature_sign_change(trial_curves[index], points):
                continue
            if bool(_curve_blend_fairness_metrics(trial_curves[index], points)["forbidden"]):
                continue
            if _curve_exceeds_precision_budget(trial_curves[index], points):
                continue
            if _cv_dent_penalty(trial_curves[index].cvs) > 14.0:
                continue
            if _cv_side_consistency_penalty(trial_curves[index], points) > 30.0:
                continue
            if _cv_target_corridor_penalty(trial_curves[index], points) > 34.0:
                continue
            trial_jump = _curve_endpoint_jump_score(trial_curves, target_segments, index, closed=closed)
            if trial_jump > max(130.0, baseline_jump * 1.08):
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
        if _has_forbidden_cv_side_switch(curves[item_index], target_segments[item_index]):
            score += 10000.0
        if _has_forbidden_curvature_sign_change(curves[item_index], target_segments[item_index]):
            score += 10000.0
        blend_fairness = _curve_blend_fairness_metrics(curves[item_index], target_segments[item_index])
        if bool(blend_fairness["forbidden"]):
            score += 8000.0
        score += float(blend_fairness["penalty"]) * 28.0
        score += _curvature_sign_penalty(curves[item_index], target_segments[item_index])
        if _curve_exceeds_precision_budget(curves[item_index], target_segments[item_index]):
            score += 5000.0
        score += _curve_endpoint_jump_score(curves, target_segments, item_index, closed=closed) * 3.2
        score += curves[item_index].degree * 0.08
    return score


def _curve_endpoint_jump_score(
    curves: list[NURBSCurve],
    target_segments: list[np.ndarray],
    index: int,
    *,
    closed: bool,
) -> float:
    if not curves or index < 0 or index >= len(curves):
        return 0.0
    score = 0.0
    curve = curves[index]
    if index > 0 or (closed and len(curves) > 2):
        prev_index = (index - 1) % len(curves)
        local_len = max(min(_polyline_length(target_segments[prev_index]), _polyline_length(target_segments[index])), 1.0)
        score += _endpoint_cv_jump_penalty(curve, side="start", local_len=local_len)
    if index < len(curves) - 1 or (closed and len(curves) > 2):
        next_index = (index + 1) % len(curves)
        local_len = max(min(_polyline_length(target_segments[index]), _polyline_length(target_segments[next_index])), 1.0)
        score += _endpoint_cv_jump_penalty(curve, side="end", local_len=local_len)
    return float(score)


def _fair_free_interior_cvs_for_chain(
    fitted_curves: list[tuple[NURBSCurve, CurveCandidate, np.ndarray, dict[str, Any], int]],
    *,
    closed: bool,
) -> None:
    curves = [item[0] for item in fitted_curves]
    target_segments = [item[2] for item in fitted_curves]
    _fair_free_interior_cvs_for_curves(curves, target_segments=target_segments, closed=closed)


def _smooth_endpoint_bridges_for_chain(
    fitted_curves: list[tuple[NURBSCurve, CurveCandidate, np.ndarray, dict[str, Any], int]],
    *,
    closed: bool,
) -> None:
    curves = [item[0] for item in fitted_curves]
    target_segments = [item[2] for item in fitted_curves]
    _smooth_endpoint_bridges_for_curves(curves, target_segments, closed=closed)


def _smooth_endpoint_bridges_for_curves(
    curves: list[NURBSCurve],
    target_segments: list[np.ndarray],
    *,
    closed: bool,
) -> None:
    if len(curves) <= 1:
        return
    join_count = len(curves) if closed and len(curves) > 2 else len(curves) - 1
    for join_index in range(join_count):
        local_len = max(
            min(
                _polyline_length(target_segments[join_index]),
                _polyline_length(target_segments[(join_index + 1) % len(curves)]),
            ),
            1.0,
        )
        _smooth_endpoint_bridge_cv(curves[join_index], side="end", local_len=local_len)
        _smooth_endpoint_bridge_cv(curves[(join_index + 1) % len(curves)], side="start", local_len=local_len)


def _fair_free_interior_cvs_for_curves(
    curves: list[NURBSCurve],
    *,
    closed: bool,
    target_segments: list[np.ndarray] | None = None,
) -> None:
    if not curves:
        return
    for index, curve in enumerate(curves):
        start_constrained = closed or index > 0
        end_constrained = closed or index < len(curves) - 1
        if start_constrained and end_constrained:
            if target_segments is None:
                continue
            _fair_free_interior_cvs_guarded(curve, target_segments[index])


def _fair_free_interior_cvs_guarded(curve: NURBSCurve, points: np.ndarray) -> None:
    base = _clone_curve(curve)
    base_mean = _bezier_mean_error(base, points)
    base_max = _bezier_max_error(base, points)
    base_layout = _cv_layout_penalty(base.cvs)
    base_dent = _cv_dent_penalty(base.cvs)
    base_side = _cv_side_consistency_penalty(base, points)
    base_corridor = _cv_target_corridor_penalty(base, points)
    best: NURBSCurve | None = None
    best_score = base_layout * 0.7 + base_dent + base_side * 2.0 + base_corridor * 2.0 + base_mean * 8.0
    for strength in (0.16, 0.28, 0.40):
        trial = _clone_curve(base)
        _fair_free_interior_cvs(trial, strength=strength)
        mean = _bezier_mean_error(trial, points)
        max_error = _bezier_max_error(trial, points)
        if mean > min(_mean_fit_budget(points), base_mean + max(0.45, base_mean * 0.12)):
            continue
        if max_error > min(_max_fit_budget(points), base_max + max(1.2, base_max * 0.12)):
            continue
        if _has_forbidden_cv_side_switch(trial, points):
            continue
        if _has_forbidden_curvature_sign_change(trial, points):
            continue
        if bool(_curve_blend_fairness_metrics(trial, points)["forbidden"]):
            continue
        side = _cv_side_consistency_penalty(trial, points)
        corridor = _cv_target_corridor_penalty(trial, points)
        if side > base_side + 5.0 or corridor > base_corridor + 5.0:
            continue
        layout = _cv_layout_penalty(trial.cvs)
        dent = _cv_dent_penalty(trial.cvs)
        score = layout * 0.7 + dent + side * 2.0 + corridor * 2.0 + mean * 8.0
        if score < best_score - 3.0:
            best_score = score
            best = trial
    if best is not None:
        curve.cvs = best.cvs


def _fair_free_interior_cvs(curve: NURBSCurve, *, strength: float = 0.28) -> None:
    cvs = curve.cvs
    count = len(cvs)
    if count <= 6:
        return
    first_free = 3
    last_free = count - 4
    if last_free < first_free:
        return
    start_anchor = cvs[2].copy()
    end_anchor = cvs[-3].copy()
    denom = float((count - 3) - 2)
    for idx in range(first_free, last_free + 1):
        t = float(idx - 2) / max(denom, 1.0)
        target = start_anchor * (1.0 - t) + end_anchor * t
        cvs[idx] = cvs[idx] * (1.0 - strength) + target * strength


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


def _bezier_max_error(curve: NURBSCurve, points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    pts = remove_duplicate_points(points, eps=0.5)
    u = chord_length_parameter(pts)
    sampled = evaluate_bezier(curve.cvs, u, curve.weights)
    return float(np.max(np.linalg.norm(sampled[:, :2] - pts[:, :2], axis=1)))


def _mean_fit_budget(points: np.ndarray) -> float:
    length = max(_polyline_length(points), 1.0)
    return max(3.2, min(8.0, length * 0.012))


def _max_fit_budget(points: np.ndarray) -> float:
    length = max(_polyline_length(points), 1.0)
    return max(8.0, min(22.0, length * 0.035))


def _curve_exceeds_precision_budget(curve: NURBSCurve, points: np.ndarray) -> bool:
    return _bezier_mean_error(curve, points) > _mean_fit_budget(points) or _bezier_max_error(curve, points) > _max_fit_budget(points)


def _curve_curvature_sign_metrics(curve: NURBSCurve, target_points: np.ndarray, sample_count: int = 120) -> dict[str, float | int | bool]:
    u = np.linspace(0.0, 1.0, max(sample_count, 24))
    d1_cvs = curve.degree * np.diff(curve.cvs, axis=0)
    d2_cvs = curve.degree * (curve.degree - 1) * np.diff(curve.cvs, n=2, axis=0)
    d1 = evaluate_bezier(d1_cvs, u)
    d2 = evaluate_bezier(d2_cvs, u)
    speed = np.linalg.norm(d1[:, :2], axis=1)
    cross_vals = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    curvature = np.divide(cross_vals, np.maximum(speed, 1e-9) ** 3)
    curvature = curvature[np.isfinite(curvature)]
    if len(curvature) == 0:
        return {
            "forbidden": False,
            "sign_flips": 0,
            "positive": 0,
            "negative": 0,
            "k_min": 0.0,
            "k_max": 0.0,
            "target_is_s_curve": False,
        }
    target_is_s_curve = _target_has_macro_s_shape(target_points)
    max_abs = float(np.nanmax(np.abs(curvature)))
    length = max(_polyline_length(target_points), 1.0)
    # Ignore numerical fuzz on nearly straight spans, but do not ignore visible
    # curvature reversals. The length-scaled floor avoids treating pure lines as S.
    eps = max(max_abs * 0.035, 1.0 / (length * length) * 0.35, 1e-8)
    signs = np.sign(np.where(np.abs(curvature) < eps, 0.0, curvature))
    nonzero = signs[signs != 0]
    positive = int(np.sum(nonzero > 0))
    negative = int(np.sum(nonzero < 0))
    sign_flips = int(np.sum(nonzero[1:] * nonzero[:-1] < 0.0)) if len(nonzero) > 1 else 0
    forbidden = bool((not target_is_s_curve) and positive > 0 and negative > 0)
    return {
        "forbidden": forbidden,
        "sign_flips": sign_flips,
        "positive": positive,
        "negative": negative,
        "k_min": float(np.nanmin(curvature)),
        "k_max": float(np.nanmax(curvature)),
        "target_is_s_curve": bool(target_is_s_curve),
    }


def _has_forbidden_curvature_sign_change(curve: NURBSCurve, target_points: np.ndarray) -> bool:
    return bool(_curve_curvature_sign_metrics(curve, target_points)["forbidden"])


def _stamp_curvature_sign_diagnostics(curve: NURBSCurve, target_points: np.ndarray) -> None:
    metrics = _curve_curvature_sign_metrics(curve, target_points)
    curve.metadata["curvature_sign_forbidden"] = bool(metrics["forbidden"])
    curve.metadata["curvature_sign_flips"] = int(metrics["sign_flips"])
    curve.metadata["curvature_positive_samples"] = int(metrics["positive"])
    curve.metadata["curvature_negative_samples"] = int(metrics["negative"])
    curve.metadata["curvature_k_min"] = round(float(metrics["k_min"]), 9)
    curve.metadata["curvature_k_max"] = round(float(metrics["k_max"]), 9)
    curve.metadata["target_macro_s_curve"] = bool(metrics["target_is_s_curve"])


def _stamp_blend_fairness_diagnostics(curve: NURBSCurve, target_points: np.ndarray) -> None:
    metrics = _curve_blend_fairness_metrics(curve, target_points)
    curve.metadata["blend_fairness_forbidden"] = bool(metrics["forbidden"])
    curve.metadata["blend_fairness_penalty"] = round(float(metrics["penalty"]), 4)
    curve.metadata["blend_strict_side_leak"] = bool(metrics["strict_side_leak"])
    curve.metadata["blend_lobe_extrema"] = int(metrics["lobe_extrema"])
    curve.metadata["blend_lobe_penalty"] = round(float(metrics["lobe_penalty"]), 4)
    curve.metadata["blend_max_cv_turn_deg"] = round(float(metrics["max_cv_turn_deg"]), 4)
    curve.metadata["blend_strict_wrong_curvature"] = round(float(metrics["strict_wrong_curvature"]), 9)
    curve.metadata["blend_wrong_cv_turn_count"] = int(metrics["wrong_cv_turn_count"])
    curve.metadata["blend_max_wrong_cv_turn"] = round(float(metrics["max_wrong_cv_turn"]), 6)


def _curvature_sign_penalty(curve: NURBSCurve, target_points: np.ndarray) -> float:
    metrics = _curve_curvature_sign_metrics(curve, target_points)
    if bool(metrics["target_is_s_curve"]):
        return 0.0
    positive = int(metrics["positive"])
    negative = int(metrics["negative"])
    flips = int(metrics["sign_flips"])
    weaker = min(positive, negative)
    return float(weaker * 180.0 + flips * 420.0)


def _refine_non_s_single_side_curve(curve: NURBSCurve, target_points: np.ndarray) -> NURBSCurve:
    """Nudge a fitted Bezier away from false inflections on non-S manual segments.

    The manual split path is allowed to be noisy, but a normal automotive
    segment should not make Alias' curvature comb jump to the other side.  This
    local optimizer keeps the same single-span degree and fixed endpoints while
    penalizing opposite signed curvature and CVs crossing the outside side.
    """
    if minimize is None or curve.degree < 3 or _target_has_macro_s_shape(target_points):
        return curve
    target = remove_duplicate_points(np.asarray(target_points, dtype=float), eps=0.5)
    if len(target) < 6 or _polyline_length(target) < 6.0:
        return curve

    side_forbidden = _has_forbidden_cv_side_switch(curve, target)
    curv_forbidden = _has_forbidden_curvature_sign_change(curve, target)
    blend_metrics = _curve_blend_fairness_metrics(curve, target)
    blend_forbidden = bool(blend_metrics["forbidden"])
    if not side_forbidden and not curv_forbidden and not blend_forbidden:
        return curve

    desired_curvature_sign = _dominant_curve_curvature_sign(curve, target)
    if desired_curvature_sign == 0:
        desired_curvature_sign = _dominant_target_polyline_curvature_sign(target)
    if desired_curvature_sign == 0:
        desired_curvature_sign = 1

    desired_cv_side = _dominant_cv_target_side(curve, target)
    if desired_cv_side == 0:
        desired_cv_side = -desired_curvature_sign

    work_count = int(min(180, max(80, len(target))))
    work_target = smooth_polyline(resample_polyline(target, work_count), window=5)
    u_fit = chord_length_parameter(work_target)
    basis = bernstein_basis(curve.degree, u_fit)
    target_xy = work_target[:, :2]

    u_curv = np.linspace(0.015, 0.985, 96)
    d1_basis = bernstein_basis(curve.degree - 1, u_curv)
    d2_basis = bernstein_basis(curve.degree - 2, u_curv) if curve.degree >= 2 else np.zeros((len(u_curv), 1))

    base = np.asarray(curve.cvs, dtype=float).copy()
    base_xy = base[:, :2].copy()
    p0 = base[0].copy()
    p1 = base[-1].copy()
    length = max(_polyline_length(target), 1.0)
    fit_scale = max(_mean_fit_budget(target), 1.0)
    drift_limit = max(length * 0.12, 12.0)
    side_eps = 0.12
    refs: list[np.ndarray] = []
    tangents: list[np.ndarray] = []
    for cv_index in range(1, curve.degree):
        ref, tangent = _target_point_tangent_at_fraction(work_target, cv_index / float(curve.degree))
        refs.append(ref)
        tangents.append(tangent)
    refs_arr = np.asarray(refs, dtype=float)
    tangents_arr = np.asarray(tangents, dtype=float)

    base_k = _signed_curvature_samples_from_cvs(base_xy, d1_basis, d2_basis, curve.degree)
    finite_base_k = base_k[np.isfinite(base_k)]
    k_scale = (
        max(float(np.nanpercentile(np.abs(finite_base_k), 80)), 1.0 / max(length * 2.2, 1.0), 1e-5)
        if len(finite_base_k)
        else max(1.0 / max(length * 2.2, 1.0), 1e-5)
    )

    def build_cvs(x: np.ndarray) -> np.ndarray:
        cvs = base.copy()
        cvs[1:-1, :2] = x.reshape(curve.degree - 1, 2)
        cvs[0] = p0
        cvs[-1] = p1
        return cvs

    def objective_factory(curvature_weight: float):
        def objective(x: np.ndarray) -> float:
            cvs = build_cvs(x)
            xy = cvs[:, :2]
            sampled = basis @ xy
            data = float(np.mean(np.sum((sampled - target_xy) ** 2, axis=1)) / (fit_scale**2))
            d2 = np.diff(xy, n=2, axis=0)
            d3 = np.diff(xy, n=3, axis=0) if len(xy) >= 4 else np.zeros((0, 2))
            fair = float(np.mean(np.sum(d2**2, axis=1)) / (length**2)) if len(d2) else 0.0
            jerk = float(np.mean(np.sum(d3**2, axis=1)) / (length**2)) if len(d3) else 0.0
            drift = float(np.mean(np.sum((xy[1:-1] - base_xy[1:-1]) ** 2, axis=1)) / (drift_limit**2))

            k = _signed_curvature_samples_from_cvs(xy, d1_basis, d2_basis, curve.degree)
            wrong_k = np.clip(-desired_curvature_sign * k, 0.0, None)
            curv_sign = float(np.mean((wrong_k / k_scale) ** 2))
            fair_k = desired_curvature_sign * k
            strict_wrong_k = np.clip(-(fair_k) - k_scale * 0.006, 0.0, None)
            comb_leak = float(np.mean((strict_wrong_k / k_scale) ** 2))
            lobe = _curvature_lobe_penalty_from_values(fair_k, scale=k_scale)

            side_penalty = 0.0
            if desired_cv_side != 0 and len(refs_arr):
                signed = tangents_arr[:, 0] * (xy[1:-1, 1] - refs_arr[:, 1]) - tangents_arr[:, 1] * (
                    xy[1:-1, 0] - refs_arr[:, 0]
                )
                wrong_side = np.clip(-(desired_cv_side * signed) - side_eps, 0.0, None)
                side_penalty = float(np.mean((wrong_side / max(side_eps * 4.0, 1.0)) ** 2))

            edge = np.diff(xy, axis=0)
            edge_len = np.linalg.norm(edge, axis=1)
            unit = edge / np.maximum(edge_len[:, None], 1e-9)
            turn = 0.0
            if len(unit) >= 2:
                dots = np.sum(unit[:-1] * unit[1:], axis=1)
                turn = float(np.mean(np.clip(-dots - 0.02, 0.0, None) ** 2))
                angles = np.arccos(np.clip(dots, -1.0, 1.0))
                cv_turn = float(np.mean(np.clip(angles - np.deg2rad(24.0), 0.0, None) ** 2))
                cross_turn = edge[:-1, 0] * edge[1:, 1] - edge[:-1, 1] * edge[1:, 0]
                cross_norm = cross_turn / np.maximum(edge_len[:-1] * edge_len[1:], 1e-9)
                bad_turn_side = np.clip(-(desired_curvature_sign * cross_norm) - 0.003, 0.0, None)
                cv_turn_side = float(np.mean(bad_turn_side**2))
            else:
                cv_turn = 0.0
                cv_turn_side = 0.0

            return (
                data * 34.0
                + fair * 16.0
                + jerk * 5.5
                + drift * 2.2
                + curv_sign * curvature_weight
                + comb_leak * curvature_weight * 2.6
                + lobe * 240.0
                + side_penalty * 110.0
                + turn * 180.0
                + cv_turn * 260.0
                + cv_turn_side * 620.0
            )

        return objective

    base_score = _single_side_acceptance_score(curve, target)
    best_curve = curve
    best_score = base_score
    x0 = base[1:-1, :2].reshape(-1)
    max_shift = max(length * 0.28, 18.0)
    bounds = [(float(value - max_shift), float(value + max_shift)) for value in x0]
    for curvature_weight in (140.0, 420.0, 1100.0):
        try:
            result = minimize(
                objective_factory(curvature_weight),
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 140, "ftol": 1e-7, "maxls": 20},
            )
        except Exception:
            continue
        if not hasattr(result, "x"):
            continue
        trial = _clone_curve(curve)
        trial.cvs = build_cvs(np.asarray(result.x, dtype=float))
        _repair_cv_turn_side(trial, desired_curvature_sign, target)
        if _bezier_mean_error(trial, target) > min(_mean_fit_budget(target), _bezier_mean_error(curve, target) + 2.4):
            continue
        if _bezier_max_error(trial, target) > min(_max_fit_budget(target), _bezier_max_error(curve, target) + 7.5):
            continue
        trial_blend = _curve_blend_fairness_metrics(trial, target)
        if bool(trial_blend["strict_side_leak"]) and not bool(blend_metrics["strict_side_leak"]):
            continue
        score = _single_side_acceptance_score(trial, target)
        if score < best_score - 18.0:
            best_score = score
            best_curve = trial

    if best_curve is not curve:
        best_curve.metadata = dict(curve.metadata)
        best_curve.metadata["non_s_single_side_refined"] = True
        best_curve.metadata["non_s_single_side_from_score"] = round(float(base_score), 4)
        best_curve.metadata["non_s_single_side_to_score"] = round(float(best_score), 4)
        best_curve.metadata["non_s_desired_curvature_sign"] = int(desired_curvature_sign)
        best_curve.metadata["non_s_desired_cv_side"] = int(desired_cv_side)
    return best_curve


def _single_side_acceptance_score(curve: NURBSCurve, target_points: np.ndarray) -> float:
    blend = _curve_blend_fairness_metrics(curve, target_points)
    return float(
        _bezier_mean_error(curve, target_points) * 24.0
        + _bezier_max_error(curve, target_points) * 2.4
        + _cv_side_consistency_penalty(curve, target_points) * 14.0
        + _cv_target_corridor_penalty(curve, target_points) * 12.0
        + _curvature_sign_penalty(curve, target_points) * 3.2
        + float(blend["penalty"]) * 24.0
        + _cv_layout_penalty(curve.cvs) * 3.0
        + _cv_dent_penalty(curve.cvs) * 1.4
        + (25000.0 if _has_forbidden_curvature_sign_change(curve, target_points) else 0.0)
        + (18000.0 if _has_forbidden_cv_side_switch(curve, target_points) else 0.0)
        + (14000.0 if bool(blend["forbidden"]) else 0.0)
    )


def _curve_blend_fairness_metrics(curve: NURBSCurve, target_points: np.ndarray) -> dict[str, float | int | bool]:
    if _target_has_macro_s_shape(target_points):
        return {
            "forbidden": False,
            "penalty": 0.0,
            "strict_side_leak": False,
            "lobe_extrema": 0,
            "lobe_penalty": 0.0,
            "max_cv_turn_deg": 0.0,
            "strict_wrong_curvature": 0.0,
            "wrong_cv_turn_count": 0,
            "max_wrong_cv_turn": 0.0,
        }
    u = np.linspace(0.01, 0.99, 140)
    d1_cvs = curve.degree * np.diff(curve.cvs[:, :2], axis=0)
    d2_cvs = curve.degree * (curve.degree - 1) * np.diff(curve.cvs[:, :2], n=2, axis=0)
    d1 = evaluate_bezier(d1_cvs, u)
    d2 = evaluate_bezier(d2_cvs, u)
    speed = np.linalg.norm(d1, axis=1)
    k = np.divide(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0], np.maximum(speed, 1e-9) ** 3)
    k = k[np.isfinite(k)]
    if len(k) < 16:
        return {
            "forbidden": False,
            "penalty": 0.0,
            "strict_side_leak": False,
            "lobe_extrema": 0,
            "lobe_penalty": 0.0,
            "max_cv_turn_deg": 0.0,
            "strict_wrong_curvature": 0.0,
            "wrong_cv_turn_count": 0,
            "max_wrong_cv_turn": 0.0,
        }
    desired = _dominant_curve_curvature_sign(curve, target_points)
    if desired == 0:
        desired = _dominant_target_polyline_curvature_sign(target_points)
    if desired == 0:
        desired = 1
    fair_k = desired * k
    length = max(_polyline_length(target_points), 1.0)
    k_scale = max(float(np.nanpercentile(np.abs(k), 88)), 1.0 / max(length * 2.2, 1.0), 1e-6)
    strict_wrong = float(max(0.0, -float(np.nanmin(fair_k))))
    strict_side_leak = bool(strict_wrong > max(k_scale * 0.006, 1.0 / max(length * length * 18.0, 1.0)))
    lobe_penalty, lobe_extrema = _curvature_lobe_penalty_from_values(fair_k, scale=k_scale, return_extrema=True)
    max_cv_turn, mean_bad_turn = _cv_turn_metrics(curve.cvs)
    wrong_cv_turn_count, max_wrong_cv_turn = _cv_turn_side_metrics(curve.cvs, desired)
    penalty = (
        lobe_penalty * 1.8
        + max(0.0, max_cv_turn - 34.0) * 0.18
        + mean_bad_turn * 5.0
        + (strict_wrong / k_scale) * 38.0
        + wrong_cv_turn_count * 18.0
        + max_wrong_cv_turn * 42.0
    )
    forbidden = bool(
        strict_side_leak
        or lobe_extrema > 2
        or max_cv_turn > 48.0
        or lobe_penalty > 1.75
        or wrong_cv_turn_count > 0
    )
    return {
        "forbidden": forbidden,
        "penalty": float(min(penalty, 260.0)),
        "strict_side_leak": strict_side_leak,
        "lobe_extrema": int(lobe_extrema),
        "lobe_penalty": float(lobe_penalty),
        "max_cv_turn_deg": float(max_cv_turn),
        "strict_wrong_curvature": float(strict_wrong),
        "wrong_cv_turn_count": int(wrong_cv_turn_count),
        "max_wrong_cv_turn": float(max_wrong_cv_turn),
    }


def _curvature_lobe_penalty_from_values(
    signed_k: np.ndarray,
    *,
    scale: float,
    return_extrema: bool = False,
) -> float | tuple[float, int]:
    values = np.asarray(signed_k, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 12:
        return (0.0, 0) if return_extrema else 0.0
    kk = np.maximum(values, 0.0)
    window = 9 if len(kk) >= 40 else 5
    kernel = np.ones(window, dtype=float) / float(window)
    smooth = np.convolve(kk, kernel, mode="same")
    dk = np.diff(smooth)
    eps = max(float(np.nanmax(np.abs(dk))) * 0.08, scale * 0.004, 1e-10)
    signs = np.sign(np.where(np.abs(dk) < eps, 0.0, dk))
    nonzero = signs[signs != 0]
    extrema = int(np.sum(nonzero[1:] * nonzero[:-1] < 0.0)) if len(nonzero) > 1 else 0

    peak = int(np.nanargmax(smooth))
    before = np.diff(smooth[: peak + 1])
    after = np.diff(smooth[peak:])
    wrong_before = np.clip(-before, 0.0, None)
    wrong_after = np.clip(after, 0.0, None)
    monotone = (
        float(np.mean((wrong_before / max(scale, 1e-9)) ** 2)) if len(wrong_before) else 0.0
    ) + (
        float(np.mean((wrong_after / max(scale, 1e-9)) ** 2)) if len(wrong_after) else 0.0
    )
    rough = np.diff(smooth, n=2)
    roughness = float(np.mean(np.abs(rough)) / max(scale, 1e-9)) if len(rough) else 0.0
    penalty = float(monotone * 34.0 + roughness * 4.5 + max(0, extrema - 1) * 0.45)
    return (penalty, extrema) if return_extrema else penalty


def _cv_turn_metrics(cvs: np.ndarray) -> tuple[float, float]:
    pts = np.asarray(cvs[:, :2], dtype=float)
    if len(pts) < 4:
        return 0.0, 0.0
    edge = np.diff(pts, axis=0)
    length = np.linalg.norm(edge, axis=1)
    if np.count_nonzero(length > 1e-6) < 3:
        return 180.0, 180.0
    unit = edge / np.maximum(length[:, None], 1e-9)
    dots = np.sum(unit[:-1] * unit[1:], axis=1)
    angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    max_turn = float(np.nanmax(angles)) if len(angles) else 0.0
    bad = np.clip(angles - 24.0, 0.0, None)
    mean_bad = float(np.mean(bad / 24.0)) if len(bad) else 0.0
    return max_turn, mean_bad


def _cv_turn_side_metrics(cvs: np.ndarray, desired_sign: int) -> tuple[int, float]:
    if desired_sign == 0:
        return 0, 0.0
    pts = np.asarray(cvs[:, :2], dtype=float)
    if len(pts) < 4:
        return 0, 0.0
    edge = np.diff(pts, axis=0)
    length = np.linalg.norm(edge, axis=1)
    if np.count_nonzero(length > 1e-6) < 3:
        return 0, 0.0
    cross = edge[:-1, 0] * edge[1:, 1] - edge[:-1, 1] * edge[1:, 0]
    normalized = cross / np.maximum(length[:-1] * length[1:], 1e-9)
    wrong = -(desired_sign * normalized)
    active = wrong > 0.003
    if not np.any(active):
        return 0, 0.0
    return int(np.sum(active)), float(np.nanmax(wrong[active]))


def _repair_cv_turn_side(curve: NURBSCurve, desired_sign: int, target_points: np.ndarray) -> None:
    if desired_sign == 0 or _target_has_macro_s_shape(target_points):
        return
    cvs = np.asarray(curve.cvs, dtype=float).copy()
    if len(cvs) < 5:
        return
    length = max(_polyline_length(target_points), 1.0)
    max_step = max(1.2, min(9.0, length * 0.035))
    margin = 0.010
    for _ in range(5):
        changed = False
        pts = cvs[:, :2]
        edge = np.diff(pts, axis=0)
        lens = np.linalg.norm(edge, axis=1)
        for turn_index in range(len(edge) - 1):
            a = edge[turn_index]
            b = edge[turn_index + 1]
            la = float(lens[turn_index])
            lb = float(lens[turn_index + 1])
            if la < 1e-6 or lb < 1e-6:
                continue
            signed = desired_sign * float(a[0] * b[1] - a[1] * b[0])
            target_cross = margin * la * lb
            if signed >= target_cross:
                continue
            if turn_index >= len(edge) - 2:
                move_index = turn_index
                if move_index <= 0 or move_index >= len(cvs) - 1:
                    continue
                normal = desired_sign * np.array([-b[1], b[0]], dtype=float) / max(lb, 1e-9)
                delta = (target_cross - signed) / max(lb, 1e-9)
            else:
                move_index = turn_index + 2
                if move_index <= 0 or move_index >= len(cvs) - 1:
                    continue
                normal = desired_sign * np.array([-a[1], a[0]], dtype=float) / max(la, 1e-9)
                delta = (target_cross - signed) / max(la, 1e-9)
            delta = float(np.clip(delta, 0.0, max_step))
            cvs[move_index, :2] += normal * delta
            changed = True
        if not changed:
            break
    repaired = _clone_curve(curve)
    repaired.cvs = cvs
    if _bezier_mean_error(repaired, target_points) <= _bezier_mean_error(curve, target_points) + 1.8 and _bezier_max_error(
        repaired, target_points
    ) <= _bezier_max_error(curve, target_points) + 6.0:
        curve.cvs = cvs
        curve.metadata["cv_turn_side_repaired"] = True


def _optimize_global_c2_visual_nullspace(
    cvs_flat: np.ndarray,
    constraints: np.ndarray,
    constraint_rhs: np.ndarray,
    prepared_points: list[np.ndarray],
    *,
    degree: int,
    cv_priors: list[np.ndarray] | None,
) -> np.ndarray:
    if minimize is None:
        return cvs_flat
    try:
        from scipy.linalg import null_space
    except Exception:
        return cvs_flat
    c = np.asarray(constraints, dtype=float)
    if c.size == 0:
        return cvs_flat
    ns = null_space(c)
    if ns.size == 0 or ns.shape[1] == 0:
        return cvs_flat

    base = np.asarray(cvs_flat, dtype=float).copy()
    n_vars = base.shape[0]
    cv_count = degree + 1
    segment_count = len(prepared_points)
    if n_vars != segment_count * cv_count:
        return cvs_flat

    # The KKT solve should already satisfy C0/C1/C2. Project the start point
    # through the nullspace so every optimizer step remains exactly on that
    # same G2 constraint manifold.
    x0 = np.zeros((ns.shape[1], 2), dtype=float)
    priors = cv_priors if cv_priors is not None else [None] * segment_count
    segment_data = _global_visual_objective_data(prepared_points, priors, degree)
    if not segment_data:
        return cvs_flat

    base_score = _global_visual_objective(base[:, :2], ns, x0, segment_data, degree)
    try:
        result = minimize(
            lambda z: _global_visual_objective(base[:, :2], ns, z.reshape(ns.shape[1], 2), segment_data, degree),
            x0.reshape(-1),
            method="L-BFGS-B",
            options={"maxiter": 52, "ftol": 1e-7, "maxls": 16},
        )
    except Exception:
        return cvs_flat
    if not getattr(result, "success", False) and not hasattr(result, "x"):
        return cvs_flat

    z = np.asarray(result.x, dtype=float).reshape(ns.shape[1], 2)
    optimized_xy = base[:, :2] + ns @ z
    trial = base.copy()
    trial[:, :2] = optimized_xy
    trial_score = _global_visual_objective(base[:, :2], ns, z, segment_data, degree)
    if not np.isfinite(trial_score) or trial_score > base_score * 0.985:
        return cvs_flat
    # Guard against numerical leakage from the equality manifold.
    if np.linalg.norm(c @ trial - constraint_rhs) > 1e-5:
        return cvs_flat
    return trial


def _global_visual_objective_data(
    prepared_points: list[np.ndarray],
    cv_priors: list[np.ndarray | None],
    degree: int,
) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    for seg_index, points in enumerate(prepared_points):
        pts = np.asarray(points, dtype=float)
        if len(pts) < 4:
            continue
        length = max(_polyline_length(pts), 1.0)
        u_fit = chord_length_parameter(pts)
        u_curv = np.linspace(0.015, 0.985, 80)
        prior = cv_priors[seg_index] if seg_index < len(cv_priors) else None
        if prior is not None and len(prior) == degree + 1:
            prior_xy = np.asarray(prior[:, :2], dtype=float)
            prior_curve = NURBSCurve.single_span(
                label="visual_prior",
                degree=degree,
                cvs=np.column_stack([prior_xy, np.zeros(len(prior_xy))]),
            )
            desired_sign = _dominant_curve_curvature_sign(prior_curve, pts)
        else:
            prior_xy = None
            desired_sign = 0
        if desired_sign == 0:
            desired_sign = _dominant_target_polyline_curvature_sign(pts)
        if desired_sign == 0:
            desired_sign = 1
        desired_side = 0
        if prior_xy is not None:
            prior_curve = NURBSCurve.single_span(
                label="visual_prior_side",
                degree=degree,
                cvs=np.column_stack([prior_xy, np.zeros(len(prior_xy))]),
            )
            desired_side = _dominant_cv_target_side(prior_curve, pts)
        if desired_side == 0:
            desired_side = -desired_sign
        refs = []
        tangents = []
        for cv_index in range(1, degree):
            ref, tangent = _target_point_tangent_at_fraction(pts, cv_index / float(degree))
            refs.append(ref)
            tangents.append(tangent)
        data.append(
            {
                "segment_index": seg_index,
                "points": pts,
                "length": length,
                "basis": bernstein_basis(degree, u_fit),
                "target_xy": pts[:, :2],
                "d1_basis": bernstein_basis(degree - 1, u_curv),
                "d2_basis": bernstein_basis(degree - 2, u_curv),
                "prior_xy": prior_xy,
                "desired_sign": int(desired_sign),
                "desired_side": int(desired_side),
                "refs": np.asarray(refs, dtype=float),
                "tangents": np.asarray(tangents, dtype=float),
            }
        )
    return data


def _global_visual_objective(
    base_xy: np.ndarray,
    nullspace: np.ndarray,
    z: np.ndarray,
    segment_data: list[dict[str, Any]],
    degree: int,
) -> float:
    xy = base_xy + nullspace @ z
    cv_count = degree + 1
    total = 0.0
    for item in segment_data:
        seg_index = int(item["segment_index"])
        start = seg_index * cv_count
        seg_xy = xy[start : start + cv_count]
        length = float(item["length"])
        sampled = item["basis"] @ seg_xy
        target = item["target_xy"]
        fit_scale = max(_mean_fit_budget(item["points"]), 1.0)
        data = float(np.mean(np.sum((sampled - target) ** 2, axis=1)) / (fit_scale**2))

        d2 = np.diff(seg_xy, n=2, axis=0)
        d3 = np.diff(seg_xy, n=3, axis=0)
        fair = float(np.mean(np.sum(d2**2, axis=1)) / (length**2)) if len(d2) else 0.0
        jerk = float(np.mean(np.sum(d3**2, axis=1)) / (length**2)) if len(d3) else 0.0

        prior_xy = item["prior_xy"]
        prior_term = 0.0
        if prior_xy is not None and len(prior_xy) == len(seg_xy):
            drift_limit = max(length * 0.11, 10.0)
            prior_term = float(np.mean(np.sum((seg_xy[1:-1] - prior_xy[1:-1]) ** 2, axis=1)) / (drift_limit**2))

        k = _signed_curvature_samples_from_cvs(seg_xy, item["d1_basis"], item["d2_basis"], degree)
        k = k[np.isfinite(k)]
        desired_sign = int(item["desired_sign"])
        if len(k):
            k_scale = max(float(np.nanpercentile(np.abs(k), 82)), 1.0 / max(length * 2.6, 1.0), 1e-6)
            fair_k = desired_sign * k
            wrong = np.clip(-fair_k - k_scale * 0.004, 0.0, None)
            sign_penalty = float(np.mean((wrong / k_scale) ** 2))
            lobe_penalty = float(_curvature_lobe_penalty_from_values(fair_k, scale=k_scale))
        else:
            sign_penalty = 0.0
            lobe_penalty = 0.0

        side_penalty = 0.0
        refs = item["refs"]
        tangents = item["tangents"]
        desired_side = int(item["desired_side"])
        if desired_side and len(refs) == degree - 1:
            interior = seg_xy[1:-1]
            signed = tangents[:, 0] * (interior[:, 1] - refs[:, 1]) - tangents[:, 1] * (interior[:, 0] - refs[:, 0])
            side_eps = max(length * 0.004, 0.55)
            wrong_side = np.clip(-(desired_side * signed) - side_eps, 0.0, None)
            side_penalty = float(np.mean((wrong_side / max(side_eps * 3.5, 1.0)) ** 2))

        edge = np.diff(seg_xy, axis=0)
        edge_len = np.linalg.norm(edge, axis=1)
        turn_side = 0.0
        turn_angle = 0.0
        turnback = 0.0
        if len(edge) >= 2 and desired_sign:
            unit = edge / np.maximum(edge_len[:, None], 1e-9)
            dots = np.sum(unit[:-1] * unit[1:], axis=1)
            angles = np.arccos(np.clip(dots, -1.0, 1.0))
            turn_angle = float(np.mean(np.clip(angles - np.deg2rad(28.0), 0.0, None) ** 2))
            turnback = float(np.mean(np.clip(-dots - 0.015, 0.0, None) ** 2))
            cross = edge[:-1, 0] * edge[1:, 1] - edge[:-1, 1] * edge[1:, 0]
            normalized = cross / np.maximum(edge_len[:-1] * edge_len[1:], 1e-9)
            wrong_turn = np.clip(-(desired_sign * normalized) - 0.0025, 0.0, None)
            turn_side = float(np.mean(wrong_turn**2))

        total += (
            data * 42.0
            + fair * 24.0
            + jerk * 7.0
            + prior_term * 12.0
            + sign_penalty * 22000.0
            + lobe_penalty * 420.0
            + side_penalty * 520.0
            + turn_angle * 18000.0
            + turnback * 12000.0
            + turn_side * 14000.0
        )
    return float(total)


def _signed_curvature_samples_from_cvs(
    cvs_xy: np.ndarray,
    d1_basis: np.ndarray,
    d2_basis: np.ndarray,
    degree: int,
) -> np.ndarray:
    d1_cvs = degree * np.diff(cvs_xy, axis=0)
    d2_cvs = degree * (degree - 1) * np.diff(cvs_xy, n=2, axis=0)
    d1 = d1_basis @ d1_cvs
    d2 = d2_basis @ d2_cvs
    speed = np.linalg.norm(d1, axis=1)
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    return np.divide(cross, np.maximum(speed, 1e-9) ** 3)


def _dominant_curve_curvature_sign(curve: NURBSCurve, target_points: np.ndarray) -> int:
    metrics = _curve_curvature_sign_metrics(curve, target_points, sample_count=120)
    positive = int(metrics["positive"])
    negative = int(metrics["negative"])
    if positive >= max(6, negative * 1.25):
        return 1
    if negative >= max(6, positive * 1.25):
        return -1
    return 0


def _dominant_target_polyline_curvature_sign(points: np.ndarray) -> int:
    pts = remove_duplicate_points(np.asarray(points, dtype=float), eps=0.5)
    if len(pts) < 8:
        return 0
    pts = smooth_polyline(resample_polyline(pts, 120), window=9)
    v1 = pts[1:-1, :2] - pts[:-2, :2]
    v2 = pts[2:, :2] - pts[1:-1, :2]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    if len(cross) == 0:
        return 0
    eps = max(float(np.nanpercentile(np.abs(cross), 70)) * 0.08, 1e-6)
    signs = np.sign(np.where(np.abs(cross) < eps, 0.0, cross))
    positive = int(np.sum(signs > 0.0))
    negative = int(np.sum(signs < 0.0))
    if positive >= max(4, negative * 1.25):
        return 1
    if negative >= max(4, positive * 1.25):
        return -1
    return 0


def _dominant_cv_target_side(curve: NURBSCurve, target_points: np.ndarray) -> int:
    target = remove_duplicate_points(np.asarray(target_points, dtype=float), eps=0.5)
    if len(target) < 4:
        return 0
    signs: list[int] = []
    length = max(_polyline_length(target), 1.0)
    eps = max(length * 0.004, 0.65)
    for cv_index in range(1, len(curve.cvs) - 1):
        ref, tangent = _target_point_tangent_at_fraction(target, cv_index / float(len(curve.cvs) - 1))
        signed = float(tangent[0] * (curve.cvs[cv_index, 1] - ref[1]) - tangent[1] * (curve.cvs[cv_index, 0] - ref[0]))
        if abs(signed) > eps:
            signs.append(1 if signed > 0.0 else -1)
    if not signs:
        return _dominant_target_chord_side(target)
    positive = sum(1 for value in signs if value > 0)
    negative = sum(1 for value in signs if value < 0)
    if positive > negative:
        return 1
    if negative > positive:
        return -1
    return 0


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


def _has_forbidden_cv_side_switch(curve: NURBSCurve, target_points: np.ndarray) -> bool:
    metrics = _cv_side_consistency_metrics(curve, target_points)
    if _target_has_macro_s_shape(target_points):
        return False
    if int(metrics["side_switches"]) > int(metrics["allowed_switches"]):
        return True
    corridor = _cv_target_corridor_metrics(curve, target_points)
    return int(corridor["target_side_switches"]) > 0 or int(corridor["wrong_side_count"]) > 0


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
    target_is_s_curve = _target_has_macro_s_shape(target)
    dominant_side = _dominant_target_chord_side(target)
    if dominant_side == 0:
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


def _dominant_target_chord_side(points: np.ndarray) -> int:
    target = remove_duplicate_points(np.asarray(points, dtype=float), eps=0.5)
    if len(target) < 4:
        return 0
    start = target[0, :2]
    end = target[-1, :2]
    chord = end - start
    chord_len = float(np.linalg.norm(chord))
    if chord_len < 1e-9:
        return 0
    signed_distance = (
        chord[0] * (target[:, 1] - start[1]) - chord[1] * (target[:, 0] - start[0])
    ) / chord_len
    eps = max(chord_len * 0.004, 0.55)
    positive = int(np.sum(signed_distance > eps))
    negative = int(np.sum(signed_distance < -eps))
    if positive >= max(4, negative * 2):
        return 1
    if negative >= max(4, positive * 2):
        return -1
    median = float(np.nanmedian(signed_distance))
    if abs(median) > eps * 1.6:
        return 1 if median > 0.0 else -1
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
    target_is_s_curve = _target_has_macro_s_shape(target)
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


def _target_has_macro_s_shape(points: np.ndarray) -> bool:
    pts = remove_duplicate_points(np.asarray(points, dtype=float), eps=0.5)
    if len(pts) < 8:
        return False
    length = _polyline_length(pts)
    chord = pts[-1, :2] - pts[0, :2]
    chord_len = float(np.linalg.norm(chord))
    if length < 1e-6 or chord_len < 1e-6:
        return False
    if len(pts) > 96:
        pts = smooth_polyline(resample_polyline(pts, 96), window=9)
    signed = ((pts[-1, 0] - pts[0, 0]) * (pts[:, 1] - pts[0, 1]) - (pts[-1, 1] - pts[0, 1]) * (pts[:, 0] - pts[0, 0])) / chord_len
    tol = max(length * 0.012, 1.25)
    pos = signed[signed > tol]
    neg = signed[signed < -tol]
    if len(pos) < max(3, len(signed) * 0.08) or len(neg) < max(3, len(signed) * 0.08):
        return False
    pos_amp = float(np.nanpercentile(pos, 80)) if len(pos) else 0.0
    neg_amp = float(abs(np.nanpercentile(neg, 20))) if len(neg) else 0.0
    return bool(min(pos_amp, neg_amp) > max(length * 0.025, 3.0))


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
        target_is_s = _target_has_macro_s_shape(target)
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
    local_len = max(min(_polyline_length(left_points), _polyline_length(right_points)), 1.0)
    base_g01 = _join_g0_g1_metrics(base_left, base_right, local_len)
    base_tangent_angle = _angle_between(_bezier_d1_end(base_left), _bezier_d1_start(base_right))
    base_curvature_delta = abs(_bezier_curvature_end(base_left) - _bezier_curvature_start(base_right))
    base_is_already_g2 = (
        bool(base_g01["g0_ok"])
        and bool(base_g01["g1_ok"])
        and base_tangent_angle < 0.22
        and base_curvature_delta < 8e-5
    )
    best_left: NURBSCurve | None = base_left if base_is_already_g2 else None
    best_right: NURBSCurve | None = base_right if base_is_already_g2 else None
    best_score = _pair_fairness_score(base_left, base_right, left_points, right_points) if base_is_already_g2 else float("inf")
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
                if _pair_precision_regression_too_high(base_left, base_right, trial_left, trial_right, left_points, right_points):
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


def _pair_precision_regression_too_high(
    base_left: NURBSCurve,
    base_right: NURBSCurve,
    trial_left: NURBSCurve,
    trial_right: NURBSCurve,
    left_points: np.ndarray,
    right_points: np.ndarray,
) -> bool:
    base_mean = _bezier_mean_error(base_left, left_points) + _bezier_mean_error(base_right, right_points)
    trial_mean = _bezier_mean_error(trial_left, left_points) + _bezier_mean_error(trial_right, right_points)
    base_max = max(_bezier_max_error(base_left, left_points), _bezier_max_error(base_right, right_points))
    trial_max = max(_bezier_max_error(trial_left, left_points), _bezier_max_error(trial_right, right_points))
    mean_budget = _mean_fit_budget(left_points) + _mean_fit_budget(right_points)
    max_budget = max(_max_fit_budget(left_points), _max_fit_budget(right_points))
    if trial_mean > min(mean_budget, base_mean + max(0.9, base_mean * 0.18)):
        return True
    if trial_max > min(max_budget, base_max + max(1.8, base_max * 0.18)):
        return True
    base_side = _cv_side_consistency_penalty(base_left, left_points) + _cv_side_consistency_penalty(base_right, right_points)
    trial_side = _cv_side_consistency_penalty(trial_left, left_points) + _cv_side_consistency_penalty(trial_right, right_points)
    base_corridor = _cv_target_corridor_penalty(base_left, left_points) + _cv_target_corridor_penalty(base_right, right_points)
    trial_corridor = _cv_target_corridor_penalty(trial_left, left_points) + _cv_target_corridor_penalty(trial_right, right_points)
    if trial_side > base_side + 12.0 or trial_corridor > base_corridor + 12.0:
        return True
    return False


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
    _smooth_endpoint_bridge_cv(left, side="end", local_len=local_len)
    _smooth_endpoint_bridge_cv(right, side="start", local_len=local_len)
    if not _join_passes_g0_g1(left, right, local_len):
        return False
    if _endpoint_cv_dent_penalty(left, side="end") > 260.0:
        return False
    if _endpoint_cv_dent_penalty(right, side="start") > 260.0:
        return False
    if _has_forbidden_cv_side_switch(left, left_points) or _has_forbidden_cv_side_switch(right, right_points):
        return False
    if _curve_exceeds_precision_budget(left, left_points) or _curve_exceeds_precision_budget(right, right_points):
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
    local_len = max(min(_polyline_length(left_points), _polyline_length(right_points)), 1.0)
    hierarchy = _join_g0_g1_penalty(left, right, local_len)
    endpoint_jump = (
        _endpoint_cv_jump_penalty(left, side="end", local_len=local_len)
        + _endpoint_cv_jump_penalty(right, side="start", local_len=local_len)
    )
    return (
        min(hierarchy, 5000.0)
        + join_gap * 500.0
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
        + min(endpoint_jump * 8.0, 520.0)
    )


def _join_g0_g1_metrics(left: NURBSCurve, right: NURBSCurve, local_len: float) -> dict[str, float | bool]:
    gap = float(np.linalg.norm(left.cvs[-1, :2] - right.cvs[0, :2]))
    left_d1 = _bezier_d1_end(left)
    right_d1 = _bezier_d1_start(right)
    left_speed = float(np.linalg.norm(left_d1[:2]))
    right_speed = float(np.linalg.norm(right_d1[:2]))
    tangent_angle = _angle_between(left_d1, right_d1)
    gap_tol = max(float(local_len) * 1e-4, 0.025)
    tangent_tol = 0.22
    speed_ok = left_speed > 1e-7 and right_speed > 1e-7
    return {
        "gap": gap,
        "gap_tol": gap_tol,
        "tangent_angle_deg": tangent_angle,
        "tangent_tol_deg": tangent_tol,
        "left_speed": left_speed,
        "right_speed": right_speed,
        "g0_ok": bool(gap <= gap_tol),
        "g1_ok": bool(gap <= gap_tol and speed_ok and tangent_angle <= tangent_tol),
    }


def _join_passes_g0_g1(left: NURBSCurve, right: NURBSCurve, local_len: float) -> bool:
    metrics = _join_g0_g1_metrics(left, right, local_len)
    return bool(metrics["g0_ok"]) and bool(metrics["g1_ok"])


def _join_g0_g1_penalty(left: NURBSCurve, right: NURBSCurve, local_len: float) -> float:
    metrics = _join_g0_g1_metrics(left, right, local_len)
    gap = float(metrics["gap"])
    gap_tol = float(metrics["gap_tol"])
    tangent = float(metrics["tangent_angle_deg"])
    tangent_tol = float(metrics["tangent_tol_deg"])
    penalty = 0.0
    if gap > gap_tol:
        penalty += ((gap - gap_tol) / max(gap_tol, 1e-6)) ** 2 * 2800.0
    if tangent > tangent_tol:
        penalty += ((tangent - tangent_tol) / max(tangent_tol, 1e-6)) ** 2 * 1800.0
    if float(metrics["left_speed"]) <= 1e-7 or float(metrics["right_speed"]) <= 1e-7:
        penalty += 3200.0
    return float(penalty)


def _endpoint_cv_jump_penalty(curve: NURBSCurve, *, side: str, local_len: float) -> float:
    cvs = np.asarray(curve.cvs[:, :2], dtype=float)
    if len(cvs) < 4:
        return 0.0
    if side == "start":
        origin = cvs[0]
        tangent = _unit(_bezier_d1_start(curve)[:2])
        local = cvs[: min(len(cvs), 5)]
        rel = local - origin
    else:
        origin = cvs[-1]
        tangent = _unit(_bezier_d1_end(curve)[:2])
        local = cvs[max(0, len(cvs) - 5) :][::-1]
        rel = origin - local
    if np.linalg.norm(tangent) < 1e-9 or len(local) < 4:
        return 240.0

    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    proj = rel @ tangent
    normal_offsets = (local - origin) @ normal
    if side == "end":
        normal_offsets = -normal_offsets

    tol = max(float(local_len) * 0.006, 0.55)
    penalty = 0.0
    handle_proj = proj[1:]
    if len(handle_proj):
        penalty += float(np.sum(np.clip(-handle_proj, 0.0, None)) / tol * 80.0)
    if len(handle_proj) >= 2:
        increments = np.diff(handle_proj)
        penalty += float(np.sum(np.clip(-increments - tol * 0.20, 0.0, None)) / tol * 70.0)

    active = [value for value in normal_offsets[1:] if abs(float(value)) > tol]
    if len(active) >= 2:
        signs = np.sign(active)
        penalty += float(np.sum(signs[1:] * signs[:-1] < 0.0) * 90.0)
    if len(normal_offsets) >= 4:
        second = np.diff(normal_offsets, n=2)
        limit = max(tol * 2.8, float(local_len) * 0.018)
        penalty += float(np.sum(np.clip(np.abs(second) - limit, 0.0, None)) / max(limit, 1e-6) * 34.0)

    if len(handle_proj) >= 2 and handle_proj[0] > 1e-6:
        ratio = handle_proj[1] / max(handle_proj[0], 1e-6)
        if ratio < 0.95:
            penalty += (0.95 - ratio) * 95.0
        elif ratio > 5.8:
            penalty += min(ratio - 5.8, 5.0) * 18.0
    return min(float(penalty), 420.0)


def _smooth_endpoint_bridge_cv(curve: NURBSCurve, *, side: str, local_len: float) -> None:
    cvs = curve.cvs
    if len(cvs) < 7:
        return
    if side == "start":
        origin = cvs[0, :2]
        tangent = _unit(_bezier_d1_start(curve)[:2])
        local = cvs[: min(len(cvs), 5), :2]
        rel = local - origin
        bridge_index = 3
        to_world_sign = 1.0
    else:
        origin = cvs[-1, :2]
        tangent = _unit(_bezier_d1_end(curve)[:2])
        local = cvs[max(0, len(cvs) - 5) :, :2][::-1]
        rel = origin - local
        bridge_index = len(cvs) - 4
        to_world_sign = -1.0
    if np.linalg.norm(tangent) < 1e-9 or len(local) < 4:
        return

    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    proj = rel @ tangent
    normal_offsets = rel @ normal
    step_1 = max(float(proj[1] - proj[0]), 1.0)
    step_2 = max(float(proj[2] - proj[1]), step_1 * 0.55, 1.0)
    min_step = max(step_2 * 0.55, float(local_len) * 0.008, 0.8)
    max_step = max(step_2 * 2.25, float(local_len) * 0.095, 4.0)
    current_proj = float(proj[3])
    target_proj = float(proj[2] + min(max(step_2 * 1.18, min_step), max_step))
    if current_proj < proj[2] + min_step or current_proj > proj[2] + max_step * 1.35:
        new_proj = target_proj * 0.78 + current_proj * 0.22
    else:
        new_proj = target_proj * 0.35 + current_proj * 0.65
    new_proj = max(new_proj, float(proj[2] + min_step))

    current_normal = float(normal_offsets[3])
    target_normal = float(normal_offsets[2] + (normal_offsets[2] - normal_offsets[1]) * 0.62)
    normal_limit = max(float(local_len) * 0.055, 5.0)
    target_normal = float(np.clip(target_normal, normal_offsets[2] - normal_limit, normal_offsets[2] + normal_limit))
    sign_flip = (
        abs(current_normal) > max(float(local_len) * 0.006, 0.55)
        and abs(float(normal_offsets[2])) > max(float(local_len) * 0.006, 0.55)
        and current_normal * float(normal_offsets[2]) < 0.0
    )
    if sign_flip or abs(current_normal - target_normal) > normal_limit * 1.25:
        new_normal = target_normal * 0.82 + current_normal * 0.18
    else:
        new_normal = target_normal * 0.42 + current_normal * 0.58

    xy = origin + to_world_sign * (tangent * new_proj + normal * new_normal)
    curve.cvs[bridge_index, 0] = xy[0]
    curve.cvs[bridge_index, 1] = xy[1]


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


def _class_a_g2_join_constraints(
    segments: list[dict[str, Any]],
    *,
    closed: bool,
    base_curves: list[NURBSCurve] | None = None,
    curvature_scale: float = 1.0,
    handle_scale: float = 1.0,
) -> dict[int, _G2Constraint]:
    count = len(segments)
    join_count = count if closed and count > 2 else count - 1
    constraints: dict[int, _G2Constraint] = {}
    for join_index in range(join_count):
        left_points = np.asarray(segments[join_index]["points"], dtype=float)
        right_points = np.asarray(segments[(join_index + 1) % count]["points"], dtype=float)
        left_len = _polyline_length(left_points)
        right_len = _polyline_length(right_points)
        if left_len <= 1e-6 or right_len <= 1e-6:
            continue
        local_len = max(min(left_len, right_len), 1.0)
        point = 0.5 * (left_points[-1].astype(float) + right_points[0].astype(float))
        tangent = _target_join_tangent(left_points, right_points)
        if np.linalg.norm(tangent) < 1e-9:
            before = _point_before_end(left_points, min(max(left_len * 0.18, 8.0), 48.0))[:2]
            after = _point_after_start(right_points, min(max(right_len * 0.18, 8.0), 48.0))[:2]
            tangent = _unit(after - before)
        if np.linalg.norm(tangent) < 1e-9:
            continue

        target_curvature = _target_join_curvature(left_points, right_points)
        samples = _join_curvature_samples(left_points, right_points)
        sample_curvature = _select_target_join_curvature(samples)
        if abs(sample_curvature) > abs(target_curvature):
            target_curvature = sample_curvature
        if base_curves is not None and len(base_curves) == count:
            base_left = base_curves[join_index]
            base_right = base_curves[(join_index + 1) % count]
            base_left_k = _bezier_curvature_end(base_left)
            base_right_k = _bezier_curvature_start(base_right)
            target_curvature = _desired_join_curvature(
                base_left_k,
                base_right_k,
                target_curvature,
                left_points,
                right_points,
                local_len,
                samples=samples,
            )
        meaningful = _meaningful_join_curvature(local_len)
        if abs(target_curvature) < meaningful:
            target_curvature = 0.0
        max_curvature = min(0.018, 1.65 / max(local_len, 1.0))
        target_curvature = float(np.clip(target_curvature * curvature_scale, -max_curvature, max_curvature))

        base_handle = None
        if base_curves is not None and len(base_curves) == count:
            base_handle = _class_a_base_join_handle(
                base_curves[join_index],
                base_curves[(join_index + 1) % count],
            )
        handle = _class_a_join_handle_length(left_points, right_points, tangent, local_len, base_handle=base_handle)
        handle = float(np.clip(handle * handle_scale, max(local_len * 0.045, 2.5), min(max(local_len * 0.22, 8.0), 42.0)))
        # The derivative is parameter-speed, not CV distance. A 6x reference keeps
        # degree-5/7 handle offsets in a natural Alias editing range.
        speed = handle * 6.0
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        d1 = np.array([tangent[0] * speed, tangent[1] * speed, 0.0], dtype=float)
        d2 = np.array(
            [
                normal[0] * target_curvature * speed * speed,
                normal[1] * target_curvature * speed * speed,
                0.0,
            ],
            dtype=float,
        )
        constraints[join_index] = _G2Constraint(point=point.astype(float), d1=d1, d2=d2)
    return constraints


def _class_a_join_handle_length(
    left_points: np.ndarray,
    right_points: np.ndarray,
    tangent: np.ndarray,
    local_len: float,
    *,
    base_handle: float | None = None,
) -> float:
    left_len = _polyline_length(left_points)
    right_len = _polyline_length(right_points)
    before = _point_before_end(left_points, min(max(left_len * 0.16, 7.0), 42.0))[:2]
    after = _point_after_start(right_points, min(max(right_len * 0.16, 7.0), 42.0))[:2]
    join = 0.5 * (left_points[-1, :2] + right_points[0, :2])
    left_proj = abs(float(np.dot(join - before, tangent[:2])))
    right_proj = abs(float(np.dot(after - join, tangent[:2])))
    observed = 0.5 * (left_proj + right_proj)
    lower = max(local_len * 0.045, 2.5)
    upper = min(max(local_len * 0.22, 8.0), 42.0)
    if base_handle is not None and np.isfinite(base_handle) and base_handle > 1e-6:
        return float(np.clip(base_handle, lower, upper))
    if not np.isfinite(observed) or observed <= 1e-6:
        observed = local_len * 0.13
    return float(np.clip(observed * 0.48, lower, upper))


def _class_a_base_join_handle(left: NURBSCurve, right: NURBSCurve) -> float | None:
    if len(left.cvs) < 2 or len(right.cvs) < 2:
        return None
    left_handle = float(np.linalg.norm(left.cvs[-1, :2] - left.cvs[-2, :2]))
    right_handle = float(np.linalg.norm(right.cvs[1, :2] - right.cvs[0, :2]))
    values = [value for value in (left_handle, right_handle) if np.isfinite(value) and value > 1e-6]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ratio = max(values) / max(min(values), 1e-6)
    if ratio > 3.6:
        return float(np.median(values))
    return float(0.5 * (values[0] + values[1]))


def _repair_class_a_cv_layout(
    curves: list[NURBSCurve],
    target_segments: list[np.ndarray],
    *,
    closed: bool,
) -> None:
    if not curves:
        return
    for index, curve in enumerate(curves):
        base = _clone_curve(curve)
        _fair_free_interior_cvs_guarded(curve, target_segments[index])
        if _class_a_curve_layout_worse(curve, base, target_segments[index]):
            curve.cvs = base.cvs
    _enforce_class_a_join_exactness(curves, closed=closed)


def _class_a_curve_layout_worse(curve: NURBSCurve, base: NURBSCurve, points: np.ndarray) -> bool:
    base_score = (
        _cv_layout_penalty(base.cvs)
        + _cv_dent_penalty(base.cvs) * 1.7
        + _cv_side_consistency_penalty(base, points) * 3.0
        + _cv_target_corridor_penalty(base, points) * 2.0
        + _bezier_mean_error(base, points) * 6.0
    )
    score = (
        _cv_layout_penalty(curve.cvs)
        + _cv_dent_penalty(curve.cvs) * 1.7
        + _cv_side_consistency_penalty(curve, points) * 3.0
        + _cv_target_corridor_penalty(curve, points) * 2.0
        + _bezier_mean_error(curve, points) * 6.0
    )
    if _has_forbidden_cv_side_switch(curve, points) and not _has_forbidden_cv_side_switch(base, points):
        return True
    if _has_forbidden_curvature_sign_change(curve, points) and not _has_forbidden_curvature_sign_change(base, points):
        return True
    return bool(score > base_score + 2.0)


def _enforce_class_a_join_exactness(curves: list[NURBSCurve], *, closed: bool) -> None:
    if len(curves) <= 1:
        return
    join_count = len(curves) if closed and len(curves) > 2 else len(curves) - 1
    for join_index in range(join_count):
        left = curves[join_index]
        right = curves[(join_index + 1) % len(curves)]
        if left.degree < 3 or right.degree < 3 or len(left.cvs) < 4 or len(right.cvs) < 4:
            continue
        point = 0.5 * (left.cvs[-1] + right.cvs[0])
        left_d1 = _bezier_d1_end(left)
        right_d1 = _bezier_d1_start(right)
        d1 = 0.5 * (left_d1 + right_d1)
        if np.linalg.norm(d1[:2]) < 1e-9:
            d1 = left_d1 if np.linalg.norm(left_d1[:2]) > 1e-9 else right_d1
        left_d2 = _bezier_d2_end(left)
        right_d2 = _bezier_d2_start(right)
        d2 = 0.5 * (left_d2 + right_d2)
        _apply_end_derivatives(left, point, d1, d2)
        _apply_start_derivatives(right, point, d1, d2)


def _stamp_class_a_g2_diagnostics(
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
        left_points = fitted[join_index][1].points
        right_points = fitted[(join_index + 1) % len(fitted)][1].points
        local_len = max(min(_polyline_length(left_points), _polyline_length(right_points)), 1.0)
        g01 = _join_g0_g1_metrics(left, right, local_len)
        curvature_delta = abs(_bezier_curvature_end(left) - _bezier_curvature_start(right))
        left.metadata[f"class_a_g2_join_{join_index}_g0_ok"] = bool(g01["g0_ok"])
        left.metadata[f"class_a_g2_join_{join_index}_g1_ok"] = bool(g01["g1_ok"])
        left.metadata[f"class_a_g2_join_{join_index}_tangent_angle_deg"] = float(g01["tangent_angle_deg"])
        left.metadata[f"class_a_g2_join_{join_index}_curvature_delta"] = float(curvature_delta)
        right.metadata[f"class_a_g2_join_{join_index}_g0_ok"] = bool(g01["g0_ok"])
        right.metadata[f"class_a_g2_join_{join_index}_g1_ok"] = bool(g01["g1_ok"])
        right.metadata[f"class_a_g2_join_{join_index}_tangent_angle_deg"] = float(g01["tangent_angle_deg"])
        right.metadata[f"class_a_g2_join_{join_index}_curvature_delta"] = float(curvature_delta)


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
        g01 = _join_g0_g1_metrics(left, right, local_len)
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
        endpoint_jump = (
            _endpoint_cv_jump_penalty(left, side="end", local_len=local_len)
            + _endpoint_cv_jump_penalty(right, side="start", local_len=local_len)
        )
        left.metadata[f"g2_join_{join_index}_gap"] = gap
        left.metadata[f"g2_join_{join_index}_g0_ok"] = bool(g01["g0_ok"])
        left.metadata[f"g2_join_{join_index}_g1_ok"] = bool(g01["g1_ok"])
        left.metadata[f"g2_join_{join_index}_tangent_angle_deg"] = tan
        left.metadata[f"g2_join_{join_index}_curvature_delta"] = curv
        left.metadata[f"g2_join_{join_index}_left_curvature"] = left_curvature
        left.metadata[f"g2_join_{join_index}_right_curvature"] = right_curvature
        left.metadata[f"g2_join_{join_index}_target_curvature"] = target_curvature
        left.metadata[f"g2_join_{join_index}_desired_curvature"] = desired_curvature
        left.metadata[f"g2_join_{join_index}_curvature_flow_penalty"] = flow
        left.metadata[f"g2_join_{join_index}_curvature_collapse_penalty"] = collapse
        left.metadata[f"g2_join_{join_index}_endpoint_cv_jump_penalty"] = endpoint_jump
        right.metadata[f"g2_join_{join_index}_gap"] = gap
        right.metadata[f"g2_join_{join_index}_g0_ok"] = bool(g01["g0_ok"])
        right.metadata[f"g2_join_{join_index}_g1_ok"] = bool(g01["g1_ok"])
        right.metadata[f"g2_join_{join_index}_tangent_angle_deg"] = tan
        right.metadata[f"g2_join_{join_index}_curvature_delta"] = curv
        right.metadata[f"g2_join_{join_index}_left_curvature"] = left_curvature
        right.metadata[f"g2_join_{join_index}_right_curvature"] = right_curvature
        right.metadata[f"g2_join_{join_index}_target_curvature"] = target_curvature
        right.metadata[f"g2_join_{join_index}_desired_curvature"] = desired_curvature
        right.metadata[f"g2_join_{join_index}_curvature_flow_penalty"] = flow
        right.metadata[f"g2_join_{join_index}_curvature_collapse_penalty"] = collapse
        right.metadata[f"g2_join_{join_index}_endpoint_cv_jump_penalty"] = endpoint_jump


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
    source = str(curve.get("source") or "")
    preserve_route_segments = bool(curve.get("preserve_route_segments")) or (
        str(curve.get("span_split_policy") or "") == "intersection_only_junctions"
    )
    if source.startswith("geometry_auto_segment") and not preserve_route_segments:
        points = _curve_points(curve)
        if len(points) >= 2:
            return [
                {
                    "points": points,
                    "start_order": 0,
                    "end_order": max(len(manual) - 1, 1),
                    "segment_count": 1,
                }
            ]
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


def _compact_auto_export_segments(
    segments: list[dict[str, Any]],
    curve: dict[str, Any],
    *,
    image_diag: float,
) -> list[dict[str, Any]]:
    source = str(curve.get("source") or "")
    if not source.startswith("geometry_auto_segment"):
        return segments
    if not segments:
        return []

    min_length = max(18.0, min(36.0, float(image_diag) * 0.014))
    min_chord = max(12.0, min(28.0, float(image_diag) * 0.010))
    current: list[dict[str, Any]] = []
    for segment in segments:
        raw_points = segment.get("points")
        points = np.asarray(raw_points if raw_points is not None else [], dtype=float)
        current.append({**segment, "points": remove_duplicate_points(points, eps=0.5)})
    current = [segment for segment in current if len(segment["points"]) >= 2]
    if not current:
        return []
    if len(current) == 1:
        pts = current[0]["points"]
        if _polyline_length(pts) < min_length and _polyline_chord(pts) < min_chord:
            return []
        return _refresh_segment_counts(current)

    # Tiny spans between auto-generated split points become isolated stray IGES curves.
    # Merge them into the fairer neighboring span instead of exporting them as separate entities.
    max_merges = len(current) * 3
    for _ in range(max_merges):
        tiny_indices = [
            idx
            for idx, segment in enumerate(current)
            if _auto_segment_is_tiny(segment["points"], min_length=min_length, min_chord=min_chord)
        ]
        if not tiny_indices or len(current) <= 1:
            break
        idx = min(tiny_indices, key=lambda item: _polyline_length(current[item]["points"]))
        if idx == 0:
            current[1] = _merge_export_segments(current[0], current[1])
            current.pop(0)
        elif idx == len(current) - 1:
            current[idx - 1] = _merge_export_segments(current[idx - 1], current[idx])
            current.pop(idx)
        else:
            left_angle = _safe_join_angle(current[idx - 1]["points"], current[idx]["points"])
            right_angle = _safe_join_angle(current[idx]["points"], current[idx + 1]["points"])
            if left_angle <= right_angle:
                current[idx - 1] = _merge_export_segments(current[idx - 1], current[idx])
                current.pop(idx)
            else:
                current[idx + 1] = _merge_export_segments(current[idx], current[idx + 1])
                current.pop(idx)

    current = [
        segment
        for segment in current
        if not (
            len(current) > 1
            and _polyline_length(segment["points"]) < min_length * 0.55
            and _polyline_chord(segment["points"]) < min_chord * 0.75
        )
    ]
    return _refresh_segment_counts(current)


def _auto_segment_is_tiny(points: np.ndarray, *, min_length: float, min_chord: float) -> bool:
    return _polyline_length(points) < min_length or _polyline_chord(points) < min_chord


def _polyline_chord(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(points[-1, :2] - points[0, :2]))


def _safe_join_angle(left: np.ndarray, right: np.ndarray) -> float:
    try:
        return _polyline_join_angle(left, right)
    except Exception:
        return 180.0


def _merge_export_segments(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        **left,
        "points": remove_duplicate_points(
            _combine_join_points(np.asarray(left["points"], dtype=float), np.asarray(right["points"], dtype=float)),
            eps=0.5,
        ),
        "end_order": right.get("end_order", left.get("end_order")),
        "export_auto_compacted": True,
        "source_segment_indices": list(left.get("source_segment_indices") or [left.get("start_order", 0)])
        + list(right.get("source_segment_indices") or [right.get("start_order", 0)]),
    }


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


def _has_enough_distinct_fit_points(points: np.ndarray) -> bool:
    pts = remove_duplicate_points(np.asarray(points, dtype=float), eps=0.5)
    if len(pts) >= 4:
        return True
    if len(pts) < 2:
        return False
    chord = float(np.linalg.norm(pts[-1, :2] - pts[0, :2]))
    return chord >= 3.0


def _limit_fit_points(points: np.ndarray, max_fit_points: int | None) -> np.ndarray:
    if max_fit_points is None:
        return points
    limit = max(16, int(max_fit_points))
    if len(points) <= limit:
        return points
    try:
        return resample_polyline(points, limit)
    except Exception:
        idx = np.linspace(0, len(points) - 1, limit).round().astype(int)
        return points[idx]


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


def _curve_alias_overrides(curve: dict[str, Any], *, expected_count: int) -> list[dict[str, Any]]:
    """Return user-edited single-span CV overrides when they match the split chain.

    The web editor stores dynamic G2 CVs separately from the routed skeleton points.
    Export must use those CVs verbatim; otherwise Alias shows a different curve than
    the user approved in the browser.
    """
    if not ENABLE_G2_EDITOR_OVERRIDES:
        return []
    if expected_count <= 0:
        return []
    raw = curve.get("alias_curve_overrides") or curve.get("alias_overrides") or []
    if not isinstance(raw, list) or not raw:
        return []

    slots: list[dict[str, Any] | None] = [None] * expected_count
    loose: list[dict[str, Any]] = []
    for fallback_index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        cvs = _as_points3(item.get("cvs") or item.get("cv") or [])
        if len(cvs) < 4:
            continue
        try:
            degree = int(item.get("degree") or (len(cvs) - 1))
        except Exception:
            degree = len(cvs) - 1
        if len(cvs) != degree + 1:
            degree = len(cvs) - 1
        if degree < 3 or degree > 7 or len(cvs) != degree + 1:
            continue
        clean = {
            **item,
            "degree": degree,
            "cvs": cvs.tolist(),
            "span": 1,
        }
        try:
            segment_index = int(item.get("segment_index", fallback_index))
        except Exception:
            segment_index = fallback_index
        if 0 <= segment_index < expected_count and slots[segment_index] is None:
            slots[segment_index] = clean
        else:
            loose.append(clean)

    loose_iter = iter(loose)
    for index, value in enumerate(slots):
        if value is None:
            slots[index] = next(loose_iter, None)
    if any(value is None for value in slots):
        return []
    return [value for value in slots if value is not None]


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
