from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from autoalias.geometry.bezier import evaluate_bezier
from autoalias.models import NURBSCurve
from autoalias.vision.extractor import _skeletonize_zhang_suen


def write_coverage_overlay(
    image_path: str | Path,
    curves: list[NURBSCurve],
    overlay_path: str | Path,
    report_path: str | Path,
) -> None:
    cv2 = _require_cv2()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    skeleton = _skeletonize_zhang_suen(ink > 0)

    covered = np.zeros(skeleton.shape, np.uint8)
    for curve in curves:
        pts = evaluate_bezier(curve.cvs, np.linspace(0.0, 1.0, 220), curve.weights)
        pts_i = np.round(pts[:, :2]).astype(np.int32)
        if len(pts_i) >= 2:
            cv2.polylines(covered, [pts_i], False, 255, thickness=7, lineType=cv2.LINE_AA)

    skel_u8 = (skeleton.astype(np.uint8) * 255)
    covered_skel = (covered > 0) & skeleton
    missed = skeleton & ~(covered > 0)
    total = int(np.sum(skeleton))
    missed_count = int(np.sum(missed))
    coverage = 1.0 - missed_count / max(total, 1)

    overlay = cv2.addWeighted(image, 0.28, np.full_like(image, 255), 0.72, 0)
    overlay[skel_u8 > 0] = (175, 175, 175)
    overlay[covered_skel] = (255, 80, 0)  # BGR blue
    overlay[missed] = (0, 0, 255)  # red missed skeleton pixels
    for curve in curves:
        pts = evaluate_bezier(curve.cvs, np.linspace(0.0, 1.0, 220), curve.weights)
        pts_i = np.round(pts[:, :2]).astype(np.int32)
        if len(pts_i) >= 2:
            cv2.polylines(overlay, [pts_i], False, (255, 80, 0), thickness=2, lineType=cv2.LINE_AA)

    cv2.imwrite(str(overlay_path), overlay)
    Path(report_path).write_text(
        json.dumps(
            {
                "skeleton_pixels": total,
                "missed_skeleton_pixels": missed_count,
                "coverage_ratio": coverage,
                "note": "Blue is generated curve coverage; red is uncovered line-art skeleton.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _require_cv2():
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for coverage overlay") from exc
    return cv2

