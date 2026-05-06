from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from autoalias.geometry.bezier import evaluate_bezier, signed_curvature_2d
from autoalias.models import CurveCandidate, NURBSCurve


def write_svg_preview(
    path: str | Path,
    curves: Iterable[NURBSCurve],
    candidates: Iterable[CurveCandidate] | None = None,
    width: int | None = None,
    height: int | None = None,
    background_image: str | Path | None = None,
    show_labels: bool = True,
    show_comb: bool = True,
    show_cvs: bool = True,
    show_candidates: bool = True,
) -> None:
    curves = list(curves)
    candidates = list(candidates or [])
    all_pts = []
    for c in curves:
        all_pts.append(c.cvs[:, :2])
        all_pts.append(evaluate_bezier(c.cvs, np.linspace(0, 1, 120))[:, :2])
    for c in candidates:
        all_pts.append(c.points[:, :2])
    if not all_pts:
        raise ValueError("nothing to preview")
    pts = np.vstack(all_pts)
    min_xy = np.min(pts, axis=0)
    max_xy = np.max(pts, axis=0)
    image_size = _image_size(background_image) if background_image is not None else None
    if image_size is not None:
        max_xy = np.maximum(max_xy, np.array(image_size, dtype=float))
        min_xy = np.minimum(min_xy, np.array([0.0, 0.0], dtype=float))
    pad = 40.0
    vb = [min_xy[0] - pad, min_xy[1] - pad, max_xy[0] - min_xy[0] + 2 * pad, max_xy[1] - min_xy[1] + 2 * pad]
    width = width or int(max(640, vb[2]))
    height = height or int(max(360, vb[3]))

    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="{vb[0]:.3f} {vb[1]:.3f} {vb[2]:.3f} {vb[3]:.3f}">',
        "<style>",
        ".target{fill:none;stroke:#9aa3ad;stroke-width:1;stroke-opacity:.45}",
        ".curve{fill:none;stroke:#0067ff;stroke-width:2.4;stroke-linecap:round}",
        ".cv{fill:none;stroke:#ff7a00;stroke-width:1.2;stroke-dasharray:6 4}",
        ".pt{fill:#ff7a00;stroke:white;stroke-width:1}",
        ".comb{stroke:#00a36c;stroke-width:.8;stroke-opacity:.65}",
        ".label{font:12px sans-serif;fill:#1f2937}",
        "</style>",
    ]
    body.append(
        f'<rect x="{vb[0]:.3f}" y="{vb[1]:.3f}" width="{vb[2]:.3f}" '
        f'height="{vb[3]:.3f}" fill="#ffffff"/>'
    )
    if background_image is not None:
        href = Path(background_image).resolve().as_uri()
        img_w, img_h = image_size if image_size is not None else (max_xy[0], max_xy[1])
        body.append(
            f'<image href="{href}" x="0" y="0" width="{img_w:.3f}" '
            f'height="{img_h:.3f}" opacity="0.32" preserveAspectRatio="none"/>'
        )
    if show_candidates:
        for cand in candidates:
            body.append(f'<polyline class="target" points="{_points(cand.points)}"/>')
    for curve in curves:
        u = np.linspace(0, 1, 180)
        samples = evaluate_bezier(curve.cvs, u, curve.weights)
        body.append(f'<polyline class="curve" points="{_points(samples)}"/>')
        if show_cvs:
            body.append(f'<polyline class="cv" points="{_points(curve.cvs)}"/>')
            for p in curve.cvs:
                body.append(f'<circle class="pt" cx="{p[0]:.3f}" cy="{p[1]:.3f}" r="3"/>')
        if show_comb:
            body.extend(_curvature_comb(curve, samples, u))
        if show_labels:
            p = samples[min(8, len(samples) - 1)]
            body.append(
                f'<text class="label" x="{p[0]:.3f}" y="{p[1] - 8:.3f}">'
                f'{_escape(curve.label)} d{curve.degree} span={curve.span_count}</text>'
            )
    body.append("</svg>")
    Path(path).write_text("\n".join(body), encoding="utf-8")


def _points(points: np.ndarray) -> str:
    return " ".join(f"{p[0]:.3f},{p[1]:.3f}" for p in points)


def _curvature_comb(curve: NURBSCurve, samples: np.ndarray, u: np.ndarray) -> list[str]:
    if len(samples) < 5:
        return []
    k = signed_curvature_2d(curve.cvs, u)
    d = np.gradient(samples[:, :2], axis=0)
    n = np.column_stack([-d[:, 1], d[:, 0]])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(norm, 1e-9)
    scale = 28.0 / max(np.max(np.abs(k)), 1e-9)
    lines = []
    for i in range(0, len(samples), 6):
        a = samples[i, :2]
        b = a + n[i] * k[i] * scale
        lines.append(f'<line class="comb" x1="{a[0]:.3f}" y1="{a[1]:.3f}" x2="{b[0]:.3f}" y2="{b[1]:.3f}"/>')
    return lines


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _image_size(path: str | Path | None) -> tuple[int, int] | None:
    if path is None:
        return None
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        h, w = img.shape[:2]
        return int(w), int(h)
    except Exception:
        return None
