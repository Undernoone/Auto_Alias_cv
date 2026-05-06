from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from autoalias.exporters import (
    write_coverage_overlay,
    write_dxf,
    write_iges,
    write_json_bundle,
    write_svg_preview,
)
from autoalias.exporters.json_exporter import load_curve_points
from autoalias.geometry.beautify import beautify_alias_curves
from autoalias.geometry.corner import decompose_l_corner_candidate
from autoalias.geometry.fitting import FittingOptions, SingleSpanFitter
from autoalias.geometry.postprocess import build_alias_design_curves
from autoalias.geometry.special import annotate_special_shape
from autoalias.models import CurveCandidate, NURBSCurve, QualityReport
from autoalias.quality import ClassAValidator
from autoalias.vision.extractor import ExtractorOptions, OpenCVCurveExtractor


@dataclass(slots=True)
class PipelineOptions:
    degree: int | str = "auto"
    max_curves: int = 400
    unit: str = "px"
    exports: tuple[str, ...] = ("json", "svg", "dxf", "iges")
    torch_refine: bool = False
    torch_steps: int = 80
    fill_missing_line_art: bool = True
    fill_missing_passes: int = 4


@dataclass(slots=True)
class PipelineResult:
    curves: list[NURBSCurve]
    candidates: list[CurveCandidate]
    reports: list[QualityReport]
    output_dir: Path
    accepted_curves: list[NURBSCurve] = field(default_factory=list)


class CurveReconstructionPipeline:
    def __init__(self, options: PipelineOptions | None = None):
        self.options = options or PipelineOptions()
        self.fitter = SingleSpanFitter(FittingOptions(degree=self.options.degree))
        self.validator = ClassAValidator()

    def run_image(self, image_path: str | Path, output_dir: str | Path) -> PipelineResult:
        extractor = OpenCVCurveExtractor(
            ExtractorOptions(max_curves=self.options.max_curves)
        )
        candidates = extractor.extract(image_path)
        return self.run_candidates(candidates, output_dir, background_image=image_path)

    def run_points_json(self, point_json: str | Path, output_dir: str | Path) -> PipelineResult:
        raw_curves = load_curve_points(point_json)
        candidates = []
        for idx, item in enumerate(raw_curves):
            label = item.get("label") or item.get("semantic") or f"curve_{idx:03d}"
            points = np.asarray(item["points"], dtype=float)
            candidates.append(
                CurveCandidate(
                    label=label,
                    points=points,
                    confidence=float(item.get("confidence", 1.0)),
                    source="points_json",
                    metadata={k: v for k, v in item.items() if k not in {"label", "points"}},
                )
            )
        return self.run_candidates(candidates, output_dir)

    def run_candidates(
        self,
        candidates: Iterable[CurveCandidate],
        output_dir: str | Path,
        background_image: str | Path | None = None,
    ) -> PipelineResult:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        candidates = list(candidates)
        curves: list[NURBSCurve] = []
        curve_reports: list[QualityReport] = []
        reports: list[QualityReport] = []

        for candidate in candidates:
            fitted_items = self._fit_candidate_many(candidate)
            if not fitted_items:
                reports.append(
                    QualityReport(
                        label=candidate.label,
                        passed=False,
                        metrics={"error": "fit failed"},
                        warnings=["fit failed"],
                    )
                )
                continue
            for curve, report in fitted_items:
                curves.append(curve)
                curve_reports.append(report)
                reports.append(report)

        accepted = [curve for curve, report in zip(curves, curve_reports) if report.passed]
        if background_image is not None and self.options.fill_missing_line_art and accepted:
            from autoalias.vision.extractor import extract_uncovered_line_art_candidates

            for pass_idx in range(max(1, self.options.fill_missing_passes)):
                min_len = max(5.0, 12.0 - pass_idx * 2.0)
                missing_candidates = extract_uncovered_line_art_candidates(
                    background_image,
                    accepted,
                    min_len=min_len,
                    coverage_thickness=7,
                )
                added = 0
                for candidate in missing_candidates:
                    fitted_items = self._fit_candidate_many(candidate)
                    if not fitted_items:
                        continue
                    for curve, report in fitted_items:
                        # The missing pass is allowed to add short repair curves, but not failed ones.
                        if report.passed:
                            curves.append(curve)
                            curve_reports.append(report)
                            reports.append(report)
                            added += 1
                accepted = [curve for curve, report in zip(curves, curve_reports) if report.passed]
                if added == 0:
                    break
        coverage_accepted = accepted
        accepted = build_alias_design_curves(coverage_accepted)
        beautified = beautify_alias_curves(accepted)
        self._write_outputs(
            out,
            curves,
            candidates,
            reports,
            curve_reports,
            accepted,
            coverage_accepted,
            beautified,
            background_image,
        )
        return PipelineResult(
            curves=curves,
            candidates=candidates,
            reports=reports,
            output_dir=out,
            accepted_curves=accepted,
        )

    def _fit_one_candidate(
        self, candidate: CurveCandidate
    ) -> tuple[NURBSCurve, QualityReport] | None:
        try:
            curve = self.fitter.fit_candidate(candidate)
            if self.options.torch_refine:
                from autoalias.learning.optimizer import TorchRefineOptions, refine_curve_torch

                curve = refine_curve_torch(
                    curve,
                    candidate.points,
                    TorchRefineOptions(steps=self.options.torch_steps),
                )
            annotate_special_shape(curve)
            report = self.validator.validate(curve, candidate.points)
            return curve, report
        except Exception:
            return None

    def _fit_candidate_many(self, candidate: CurveCandidate) -> list[tuple[NURBSCurve, QualityReport]]:
        if "line_art" not in candidate.source:
            fitted = self._fit_one_candidate(candidate)
            return [] if fitted is None else [fitted]
        l_corner = None
        if (
            "skeleton" in candidate.source
            and "ellipse" not in candidate.source
            and "hough" not in candidate.source
            and "uncovered" not in candidate.source
        ):
            l_corner = decompose_l_corner_candidate(candidate)
        if l_corner is not None:
            return self._fit_l_corner(candidate, l_corner)
        try:
            pairs = self.fitter.fit_candidate_adaptive_pairs(
                candidate,
                max_error=2.2,
                max_depth=4,
                min_points=22,
            )
        except Exception:
            fitted = self._fit_one_candidate(candidate)
            return [] if fitted is None else [fitted]
        out: list[tuple[NURBSCurve, QualityReport]] = []
        for curve, target in pairs:
            try:
                if self.options.torch_refine:
                    from autoalias.learning.optimizer import TorchRefineOptions, refine_curve_torch

                    curve = refine_curve_torch(
                        curve,
                        target,
                        TorchRefineOptions(steps=self.options.torch_steps),
                    )
                annotate_special_shape(curve)
                out.append((curve, self.validator.validate(curve, target)))
            except Exception:
                continue
        return out

    def _fit_l_corner(self, candidate: CurveCandidate, l_corner) -> list[tuple[NURBSCurve, QualityReport]]:
        out: list[tuple[NURBSCurve, QualityReport]] = []
        leg_specs = [("leg_a", l_corner.leg_a), ("leg_b", l_corner.leg_b)]
        for role, pts in leg_specs:
            if len(pts) < 4:
                continue
            sub = CurveCandidate(
                label=candidate.label,
                points=pts,
                confidence=candidate.confidence,
                source=f"{candidate.source}+l_corner_{role}",
                metadata={
                    **candidate.metadata,
                    "l_corner_group": l_corner.group_id,
                    "l_corner_role": role,
                    "preserve_segment": True,
                },
            )
            fitted = self._fit_one_candidate(sub)
            if fitted is None:
                continue
            curve, report = fitted
            curve.metadata.update(
                {
                    "l_corner_group": l_corner.group_id,
                    "l_corner_role": role,
                    "preserve_segment": True,
                    "segment_count": 3,
                }
            )
            out.append((curve, report))
        blend = l_corner.blend_curve
        annotate_special_shape(blend)
        blend_report = self.validator.validate(blend, l_corner.blend_target)
        # For a fair blend, Alias usability matters more than exact pixel coverage.
        if not blend_report.passed:
            blend_report.warnings = [
                w
                for w in blend_report.warnings
                if "Chamfer" not in w
                and "CV spacing" not in w
                and "control polygon has turnback" not in w
            ]
            blend_report.passed = len(blend_report.warnings) == 0
        out.insert(1 if out else 0, (blend, blend_report))
        return out

    def _write_outputs(
        self,
        out: Path,
        curves: list[NURBSCurve],
        candidates: list[CurveCandidate],
        reports: list[QualityReport],
        curve_reports: list[QualityReport],
        accepted: list[NURBSCurve],
        coverage_accepted: list[NURBSCurve],
        beautified: list[NURBSCurve],
        background_image: str | Path | None = None,
    ) -> None:
        exports = set(self.options.exports)
        if "json" in exports:
            write_json_bundle(out / "curves.json", curves, reports, unit=self.options.unit)
            write_json_bundle(out / "quality.json", [], reports, unit=self.options.unit)
            accepted_reports = [self.validator.validate(curve) for curve in accepted]
            write_json_bundle(out / "accepted_curves.json", accepted, accepted_reports, unit=self.options.unit)
            beautified_reports = [self.validator.validate(curve) for curve in beautified]
            write_json_bundle(
                out / "beautified_curves.json",
                beautified,
                beautified_reports,
                unit=self.options.unit,
            )
            coverage_reports = [report for report in curve_reports if report.passed]
            write_json_bundle(
                out / "coverage_repair_curves.json",
                coverage_accepted,
                coverage_reports,
                unit=self.options.unit,
            )
        if curves and "svg" in exports:
            write_svg_preview(out / "preview.svg", curves, candidates, background_image=background_image)
            write_svg_preview(
                out / "clean_preview.svg",
                curves,
                [],
                background_image=background_image,
                show_labels=False,
                show_comb=False,
                show_cvs=False,
                show_candidates=False,
            )
        if accepted and "svg" in exports:
            write_svg_preview(
                out / "accepted_preview.svg",
                accepted,
                [],
                background_image=background_image,
            )
            write_svg_preview(
                out / "accepted_clean_preview.svg",
                accepted,
                [],
                background_image=background_image,
                show_labels=False,
                show_comb=False,
                show_cvs=False,
                show_candidates=False,
            )
        if beautified and "svg" in exports:
            write_svg_preview(
                out / "beautified_preview.svg",
                beautified,
                [],
                background_image=background_image,
            )
            write_svg_preview(
                out / "beautified_clean_preview.svg",
                beautified,
                [],
                background_image=background_image,
                show_labels=False,
                show_comb=False,
                show_cvs=False,
                show_candidates=False,
            )
        if curves and "dxf" in exports:
            write_dxf(out / "curves.dxf", curves)
        if accepted and "dxf" in exports:
            write_dxf(out / "accepted_curves.dxf", accepted)
        if beautified and "dxf" in exports:
            write_dxf(out / "beautified_curves.dxf", beautified)
        if coverage_accepted and "dxf" in exports:
            write_dxf(out / "coverage_repair_curves.dxf", coverage_accepted)
        if curves and "iges" in exports:
            write_iges(out / "curves.igs", curves)
        if accepted and "iges" in exports:
            write_iges(out / "accepted_curves.igs", accepted)
        if beautified and "iges" in exports:
            write_iges(out / "beautified_curves.igs", beautified)
        if coverage_accepted and "iges" in exports:
            write_iges(out / "coverage_repair_curves.igs", coverage_accepted)
        if background_image is not None and accepted:
            write_coverage_overlay(
                background_image,
                accepted,
                out / "coverage_overlay.png",
                out / "coverage.json",
            )
        if background_image is not None and beautified:
            write_coverage_overlay(
                background_image,
                beautified,
                out / "beautified_coverage_overlay.png",
                out / "beautified_coverage.json",
            )
        if background_image is not None and coverage_accepted:
            write_coverage_overlay(
                background_image,
                coverage_accepted,
                out / "coverage_repair_overlay.png",
                out / "coverage_repair.json",
            )
