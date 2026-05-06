from __future__ import annotations

import numpy as np

from autoalias.geometry.bezier import evaluate_bezier, signed_curvature_2d
from autoalias.geometry.fitting import FittingOptions, SingleSpanFitter
from autoalias.models import CurveCandidate
from autoalias.quality import ClassAValidator


def test_degree7_s_curve_is_single_span() -> None:
    x = np.linspace(0, 300, 120)
    y = 35 * np.sin((x / 300 - 0.5) * np.pi)
    pts = np.column_stack([x, y])
    candidate = CurveCandidate("beltline_s_curve", pts)

    fitter = SingleSpanFitter(FittingOptions(degree=7))
    curve = fitter.fit_candidate(candidate)

    assert curve.degree == 7
    assert curve.is_single_span
    assert len(curve.cvs) == 8
    assert curve.span_count == 1

    samples = evaluate_bezier(curve.cvs, np.linspace(0, 1, 120))
    assert np.mean(np.linalg.norm(samples[:, :2] - pts[:, :2], axis=1)) < 8.0

    k = signed_curvature_2d(curve.cvs, np.linspace(0.02, 0.98, 160))
    assert np.any(k > 0) and np.any(k < 0)


def test_validator_accepts_simple_single_span_curve() -> None:
    x = np.linspace(0, 200, 90)
    y = 0.0015 * (x - 100) ** 2
    pts = np.column_stack([x, y])
    curve = SingleSpanFitter(FittingOptions(degree=5)).fit_candidate(
        CurveCandidate("roofline", pts)
    )
    report = ClassAValidator(max_chamfer_px=10.0).validate(curve, pts)
    assert report.metrics["single_span"] is True
    assert report.metrics["degree"] == 5

