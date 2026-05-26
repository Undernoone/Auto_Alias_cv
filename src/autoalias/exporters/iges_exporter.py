from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from autoalias.models import NURBSCurve


def write_iges(path: str | Path, curves: Iterable[NURBSCurve]) -> None:
    curves = list(curves)
    if _try_occ_iges(path, curves):
        return
    _write_manual_iges(path, curves)


def _try_occ_iges(path: str | Path, curves: list[NURBSCurve]) -> bool:
    try:
        from OCC.Core.Geom import Geom_BSplineCurve
        from OCC.Core.gp import gp_Pnt
        from OCC.Core.IGESControl import IGESControl_Writer
        from OCC.Core.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal
        from OCC.Core.TColgp import TColgp_Array1OfPnt
    except Exception:
        return False

    writer = IGESControl_Writer()
    for curve in curves:
        poles = TColgp_Array1OfPnt(1, len(curve.cvs))
        weights = TColStd_Array1OfReal(1, len(curve.weights))
        for i, p in enumerate(curve.cvs, start=1):
            poles.SetValue(i, gp_Pnt(float(p[0]), float(-p[1]), float(p[2])))
        for i, w in enumerate(curve.weights, start=1):
            weights.SetValue(i, float(w))
        knots = TColStd_Array1OfReal(1, 2)
        knots.SetValue(1, 0.0)
        knots.SetValue(2, 1.0)
        mults = TColStd_Array1OfInteger(1, 2)
        mults.SetValue(1, curve.degree + 1)
        mults.SetValue(2, curve.degree + 1)
        occ_curve = Geom_BSplineCurve(poles, weights, knots, mults, curve.degree, False)
        writer.AddGeom(occ_curve)
    writer.Write(str(path))
    return True


def _write_manual_iges(path: str | Path, curves: list[NURBSCurve]) -> None:
    start_records = _section_records(["AutoAlias single-span NURBS export"], "S")
    global_records = _global_records()

    directory_records: list[str] = []
    parameter_records: list[str] = []
    p_seq = 1
    d_seq = 1
    for curve in curves:
        pdata = _curve_parameter_data(curve)
        chunks = _parameter_chunks(pdata, 64)
        first_p = p_seq
        line_count = len(chunks)
        parameter_records.extend(_parameter_records(chunks, d_seq, p_seq))
        p_seq += line_count
        directory_records.extend(_directory_records(first_p, line_count, curve.label, d_seq))
        d_seq += 2

    terminate = _terminate_record(
        len(start_records), len(global_records), len(directory_records), len(parameter_records)
    )
    text = "\n".join(start_records + global_records + directory_records + parameter_records + [terminate])
    Path(path).write_text(text + "\n", encoding="ascii", errors="ignore")


def _curve_parameter_data(curve: NURBSCurve) -> str:
    # IGES Type 126 rational B-spline curve. K is upper pole index, M is degree.
    k = len(curve.cvs) - 1
    m = curve.degree
    planar = 1
    closed = 0
    polynomial = 1 if np.allclose(curve.weights, 1.0) else 0
    periodic = 0
    values: list[str] = [
        "126",
        str(k),
        str(m),
        str(planar),
        str(closed),
        str(polynomial),
        str(periodic),
    ]
    values.extend(_fmt(x) for x in curve.knots)
    values.extend(_fmt(x) for x in curve.weights)
    for p in curve.cvs:
        values.extend([_fmt(p[0]), _fmt(-p[1]), _fmt(p[2])])
    values.extend(["0.0", "1.0", "0.0", "0.0", "1.0"])
    return ",".join(values) + ";"


def _directory_records(first_p: int, p_line_count: int, label: str, d_seq: int) -> list[str]:
    safe_label = "".join(ch if ch.isalnum() else "_" for ch in label)[:8] or "CURVE"
    line1_fields = [126, first_p, 0, 0, 0, 0, 0, 0, 0]
    line2_fields = [126, 0, 0, p_line_count, 0, 0, 0, safe_label, 0]
    return [
        _dir_line(line1_fields, d_seq),
        _dir_line(line2_fields, d_seq + 1),
    ]


def _dir_line(fields: list[object], seq: int) -> str:
    data = "".join(str(f)[:8].rjust(8) for f in fields)
    return f"{data[:72]}D{seq:7d}"


def _parameter_records(chunks: list[str], d_seq: int, first_seq: int) -> list[str]:
    out = []
    for i, chunk in enumerate(chunks):
        out.append(f"{chunk:<64}{d_seq:8d}P{first_seq + i:7d}")
    return out


def _global_records() -> list[str]:
    # Minimal but standards-shaped global section.
    params = [
        _h(","),
        _h(";"),
        _h("AutoAlias Curves"),
        _h("AutoAlias Curves"),
        _h("AutoAlias single-span NURBS export"),
        _h("1"),
        "32",
        "38",
        "6",
        "308",
        "15",
        _h("AutoAlias Curves"),
        "1.0",
        "1",
        _h("INCH"),
        "1",
        "0.01",
        _h("20260430.000000"),
        "1E-06",
        "1000.0",
        _h("user"),
        _h("AutoAlias"),
        "11",
        "0",
        _h("20260430.000000"),
    ]
    return _section_records_from_chunks(_parameter_chunks(",".join(params) + ";", 72), "G")


def _section_records(payloads: list[str], section: str) -> list[str]:
    records = []
    seq = 1
    for payload in payloads:
        for chunk in _chunks(payload, 72):
            records.append(f"{chunk:<72}{section}{seq:7d}")
            seq += 1
    return records


def _section_records_from_chunks(chunks: list[str], section: str) -> list[str]:
    return [f"{chunk:<72}{section}{seq:7d}" for seq, chunk in enumerate(chunks, start=1)]


def _terminate_record(s_count: int, g_count: int, d_count: int, p_count: int) -> str:
    data = f"S{s_count:7d}G{g_count:7d}D{d_count:7d}P{p_count:7d}"
    return f"{data:<72}T{1:7d}"


def _chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def _parameter_chunks(text: str, size: int) -> list[str]:
    """Split IGES parameter data without cutting numeric values in half."""
    chunks: list[str] = []
    current = ""
    for token in _parameter_tokens(text):
        if current and len(current) + len(token) > size:
            chunks.append(current)
            current = ""
        if len(token) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_chunks(token, size))
            token = ""
            continue
        current += token
    if current:
        chunks.append(current)
    return chunks or [""]


def _parameter_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        start = i
        j = i
        while j < n and text[j].isdigit():
            j += 1
        if j > i and j < n and text[j] == "H":
            length = int(text[i:j])
            i = j + 1 + length
            if i < n and text[i] in ",;":
                i += 1
            tokens.append(text[start:i])
            continue
        while i < n and text[i] not in ",;":
            i += 1
        if i < n:
            i += 1
        tokens.append(text[start:i])
    return [token for token in tokens if token]


def _fmt(value: float) -> str:
    return f"{float(value):.12g}"


def _h(text: str) -> str:
    return f"{len(text)}H{text}"
