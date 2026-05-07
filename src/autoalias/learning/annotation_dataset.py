from __future__ import annotations

import json
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autoalias.geometry.fitting import FittingOptions, SingleSpanFitter
from autoalias.geometry.polyline import remove_duplicate_points
from autoalias.models import CurveCandidate


@dataclass(slots=True)
class AnnotationBuildResult:
    out: Path
    item_count: int
    skipped_count: int


def build_supervision_from_annotations(
    annotation_paths: list[str | Path],
    out: str | Path,
    *,
    degree: int | str = "auto",
    min_points: int = 8,
) -> AnnotationBuildResult:
    """Convert interactive manual curve annotations into supervised decoder data.

    The review UI stores design intent as manual split points plus routed skeleton points.
    This builder turns each saved design curve into a single-span fitted target so the
    neural decoder can learn from the user's topology and point-placement decisions.
    """
    resolved_paths = _expand_annotation_paths(annotation_paths)
    out_path = Path(out).resolve()
    fitter = SingleSpanFitter(FittingOptions(degree=degree))
    items: list[dict[str, Any]] = []
    skipped = 0

    for path in resolved_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        image_path = data.get("graph", {}).get("image")
        for curve in data.get("design_curves", []):
            points = _curve_points(curve)
            if len(points) < min_points:
                skipped += 1
                continue
            try:
                points = remove_duplicate_points(points, eps=0.5)
                if len(points) < min_points:
                    skipped += 1
                    continue
                label = str(curve.get("semantic") or curve.get("id") or "manual_design_curve")
                candidate = CurveCandidate(
                    label=label,
                    points=points,
                    confidence=1.0,
                    source=str(path),
                    metadata={
                        "annotation_id": curve.get("id"),
                        "semantic": curve.get("semantic"),
                        "closed": bool(curve.get("closed", False)),
                        "image": image_path,
                    },
                )
                fitted = fitter.fit_candidate(candidate)
            except Exception:
                skipped += 1
                continue

            items.append(
                {
                    "label": fitted.label,
                    "points": points[:, :2].round(3).tolist(),
                    "degree": fitted.degree,
                    "cv": fitted.cvs.round(6).tolist(),
                    "weights": fitted.weights.round(6).tolist(),
                    "knots": fitted.knots.round(6).tolist(),
                    "semantic": curve.get("semantic"),
                    "closed": bool(curve.get("closed", False)),
                    "manual_points": curve.get("manual_points", curve.get("cut_points", [])),
                    "annotation_id": curve.get("id"),
                    "annotation_path": str(path),
                    "image": image_path,
                    "source": "interactive_annotation",
                }
            )

    payload = {
        "version": 1,
        "task": "autoalias_curve_supervision",
        "curves": items,
        "source_annotations": [str(path) for path in resolved_paths],
        "skipped_count": skipped,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return AnnotationBuildResult(out=out_path, item_count=len(items), skipped_count=skipped)


def _expand_annotation_paths(paths: list[str | Path]) -> list[Path]:
    expanded: list[Path] = []
    for path_like in paths:
        text = str(path_like)
        if any(ch in text for ch in "*?["):
            expanded.extend(Path(match).resolve() for match in sorted(glob.glob(text)))
        else:
            expanded.append(Path(text).resolve())
    seen = set()
    out: list[Path] = []
    for path in expanded:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            out.append(path)
    if not out:
        raise FileNotFoundError("no annotation JSON files matched the input path(s)")
    return out


def _curve_points(curve: dict[str, Any]) -> np.ndarray:
    routed = curve.get("routed_points") or []
    if len(routed) >= 4:
        return _as_points3(routed)
    manual = curve.get("manual_points") or curve.get("cut_points") or []
    return _as_points3([[p["x"], p["y"]] if isinstance(p, dict) else p for p in manual])


def _as_points3(points: Any) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        return np.zeros((0, 3), dtype=float)
    if arr.shape[1] == 2:
        arr = np.column_stack([arr, np.zeros(len(arr), dtype=float)])
    return arr
