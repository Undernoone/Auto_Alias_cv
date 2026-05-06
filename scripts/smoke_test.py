from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from autoalias.exporters.dxf_exporter import write_dxf
from autoalias.exporters.iges_exporter import write_iges
from autoalias.exporters.json_exporter import write_json_bundle
from autoalias.exporters.svg_exporter import write_svg_preview
from autoalias.geometry.bezier import evaluate_bezier
from autoalias.geometry.fitting import FittingOptions, SingleSpanFitter
from autoalias.models import CurveCandidate
from autoalias.quality import ClassAValidator


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out = root / "out_smoke"
    out.mkdir(exist_ok=True)

    x = np.linspace(0.0, 330.0, 120)
    y = 28.0 * np.sin((x / 330.0 - 0.5) * np.pi)
    points = np.column_stack([x, y])
    candidate = CurveCandidate("smoke_s_curve", points)
    curve = SingleSpanFitter(FittingOptions(degree=7)).fit_candidate(candidate)
    report = ClassAValidator(max_chamfer_px=10.0).validate(curve, points)

    assert curve.is_single_span
    assert curve.span_count == 1
    assert curve.degree == 7
    assert len(curve.cvs) == 8
    assert report.metrics["single_span"] is True

    samples = evaluate_bezier(curve.cvs, np.linspace(0, 1, 120))
    assert float(np.mean(np.linalg.norm(samples[:, :2] - points[:, :2], axis=1))) < 10.0

    write_json_bundle(out / "curves.json", [curve], [report])
    write_svg_preview(out / "preview.svg", [curve], [candidate])
    write_dxf(out / "curves.dxf", [curve])
    write_iges(out / "curves.igs", [curve])

    summary = {
        "curve": curve.label,
        "degree": curve.degree,
        "span": curve.span_count,
        "single_span": curve.is_single_span,
        "passed": report.passed,
        "outputs": sorted(p.name for p in out.iterdir()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

