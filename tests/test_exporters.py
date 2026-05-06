from __future__ import annotations

import numpy as np

from autoalias.exporters.dxf_exporter import write_dxf
from autoalias.exporters.iges_exporter import write_iges
from autoalias.exporters.json_exporter import write_json_bundle
from autoalias.exporters.svg_exporter import write_svg_preview
from autoalias.models import NURBSCurve


def test_exporters_write_files(tmp_path) -> None:
    curve = NURBSCurve.single_span(
        "test_curve",
        3,
        np.array([[0, 0, 0], [40, 15, 0], [80, 15, 0], [120, 0, 0]], dtype=float),
    )
    write_json_bundle(tmp_path / "curves.json", [curve])
    write_svg_preview(tmp_path / "preview.svg", [curve])
    write_dxf(tmp_path / "curves.dxf", [curve])
    write_iges(tmp_path / "curves.igs", [curve])

    assert (tmp_path / "curves.json").read_text(encoding="utf-8")
    assert "SPLINE" in (tmp_path / "curves.dxf").read_text(encoding="ascii")
    assert "126" in (tmp_path / "curves.igs").read_text(encoding="ascii", errors="ignore")

