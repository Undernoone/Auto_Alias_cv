from __future__ import annotations

from pathlib import Path
from typing import Iterable

from autoalias.models import NURBSCurve


def write_dxf(path: str | Path, curves: Iterable[NURBSCurve]) -> None:
    lines: list[str] = ["0", "SECTION", "2", "ENTITIES"]
    for curve in curves:
        flags = 8  # planar
        lines.extend(
            [
                "0",
                "SPLINE",
                "100",
                "AcDbEntity",
                "8",
                _safe_layer(curve.label),
                "100",
                "AcDbSpline",
                "70",
                str(flags),
                "71",
                str(curve.degree),
                "72",
                str(len(curve.knots)),
                "73",
                str(len(curve.cvs)),
                "74",
                "0",
            ]
        )
        for knot in curve.knots:
            lines.extend(["40", _fmt(knot)])
        for weight in curve.weights:
            lines.extend(["41", _fmt(weight)])
        for p in curve.cvs:
            lines.extend(["10", _fmt(p[0]), "20", _fmt(-p[1]), "30", _fmt(p[2])])
    lines.extend(["0", "ENDSEC", "0", "EOF"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def _fmt(value: float) -> str:
    return f"{float(value):.12g}"


def _safe_layer(label: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in label)
    return out[:64] or "curve"

