from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from autoalias.models import NURBSCurve, QualityReport


def write_json_bundle(
    path: str | Path,
    curves: Iterable[NURBSCurve],
    reports: Iterable[QualityReport] | None = None,
    unit: str = "px",
) -> None:
    data = {
        "schema": "autoalias.curves.v1",
        "unit": unit,
        "curves": [curve.to_dict() for curve in curves],
    }
    if reports is not None:
        data["quality"] = [report.to_dict() for report in reports]
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_curve_points(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if "curves" in data:
        return data["curves"]
    if "points" in data:
        return [{"label": data.get("label", "curve"), "points": data["points"]}]
    raise ValueError("point JSON must contain `curves` or `points`")

