from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autoalias.models import CurveCandidate


@dataclass(slots=True)
class ExtractorOptions:
    max_curves: int = 400
    min_component_points: int = 24
    canny_low: int = 45
    canny_high: int = 135
    body_mask: bool = True
    include_silhouette: bool = True
    include_internal_edges: bool = True


class OpenCVCurveExtractor:
    """Local curve candidate extractor.

    This is the runnable fallback path. Industrial deployments should feed this stage with
    SAM2/Grounded-SAM/DINOv2 masks and ridges, but the downstream fitting/export code is identical.
    """

    def __init__(self, options: ExtractorOptions | None = None):
        self.options = options or ExtractorOptions()

    def extract(self, image_path: str | Path) -> list[CurveCandidate]:
        cv2 = _require_cv2()
        path = Path(image_path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot read image: {path}")

        if _is_line_art(image):
            candidates = self._line_art_candidates(image)
            if candidates:
                return sorted(
                    candidates,
                    key=lambda c: (c.confidence, _curve_length(c.points)),
                    reverse=True,
                )[: self.options.max_curves]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 45, 45)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        mask = self._body_mask(image) if self.options.body_mask else np.ones(gray.shape, np.uint8) * 255
        candidates: list[CurveCandidate] = []

        if self.options.include_silhouette:
            candidates.extend(self._silhouette_candidates(mask))
        if self.options.include_internal_edges:
            candidates.extend(self._edge_candidates(gray, mask))

        candidates = sorted(candidates, key=lambda c: (c.confidence, len(c.points)), reverse=True)
        return candidates[: self.options.max_curves]

    def _line_art_candidates(self, image: np.ndarray) -> list[CurveCandidate]:
        cv2 = _require_cv2()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Keep anti-aliased black strokes connected, remove isolated specks.
        small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, small, iterations=1)
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, small, iterations=1)

        skeleton = _skeletonize_zhang_suen(ink > 0)
        chains = _trace_skeleton_chains(skeleton)
        candidates: list[CurveCandidate] = []
        image_diag = float(np.hypot(*gray.shape[:2]))
        min_len = max(14.0, image_diag * 0.008)
        for chain in chains:
            if len(chain) < 6:
                continue
            length = _curve_length(chain)
            if length < min_len:
                continue
            bbox_w = float(np.ptp(chain[:, 0]))
            bbox_h = float(np.ptp(chain[:, 1]))
            if max(bbox_w, bbox_h) < 18:
                continue
            # Avoid sending almost full circles as one single-span line; those should be split by
            # loop tracing or handled as rational arcs in a later stage.
            label = _semantic_guess(chain)
            confidence = min(0.98, 0.45 + length / max(image_diag * 1.5, 1.0))
            candidates.append(CurveCandidate(label, chain, confidence, "line_art_skeleton"))
        candidates = _stitch_candidates(candidates)
        candidates.extend(_hough_line_candidates(skeleton, image_diag))
        candidates.extend(_component_ellipse_candidates(ink, image_diag))
        candidates = _deduplicate_candidates(candidates)
        return _trim_overlapping_candidates(candidates)

    def _body_mask(self, image: np.ndarray) -> np.ndarray:
        cv2 = _require_cv2()
        h, w = image.shape[:2]
        rect = (int(w * 0.05), int(h * 0.12), int(w * 0.90), int(h * 0.76))
        mask = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(image, mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
            out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except Exception:
            out = np.ones((h, w), np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=2)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)
        return out

    def _silhouette_candidates(self, mask: np.ndarray) -> list[CurveCandidate]:
        cv2 = _require_cv2()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return []
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contour = contours[0].reshape(-1, 2).astype(float)
        if len(contour) < self.options.min_component_points:
            return []

        # Split silhouette into upper/lower/left/right runs in image coordinates.
        x = contour[:, 0]
        y = contour[:, 1]
        order_x = np.argsort(x)
        sorted_pts = contour[order_x]
        bins = max(48, min(220, int(np.ptp(x) / 4)))
        if bins < 8:
            return []
        xs = np.linspace(np.min(x), np.max(x), bins)
        upper = []
        lower = []
        for i in range(len(xs) - 1):
            m = (sorted_pts[:, 0] >= xs[i]) & (sorted_pts[:, 0] < xs[i + 1])
            group = sorted_pts[m]
            if len(group) < 2:
                continue
            upper.append(group[np.argmin(group[:, 1])])
            lower.append(group[np.argmax(group[:, 1])])

        out: list[CurveCandidate] = []
        if len(upper) >= 8:
            out.append(CurveCandidate("silhouette_upper_roofline", np.asarray(upper), 0.95, "opencv"))
        if len(lower) >= 8:
            out.append(CurveCandidate("silhouette_lower_sill", np.asarray(lower), 0.75, "opencv"))

        # Front/rear vertical profiles.
        order_y = np.argsort(y)
        sorted_y = contour[order_y]
        ys = np.linspace(np.min(y), np.max(y), max(24, min(160, int(np.ptp(y) / 4))))
        left = []
        right = []
        for i in range(len(ys) - 1):
            m = (sorted_y[:, 1] >= ys[i]) & (sorted_y[:, 1] < ys[i + 1])
            group = sorted_y[m]
            if len(group) < 2:
                continue
            left.append(group[np.argmin(group[:, 0])])
            right.append(group[np.argmax(group[:, 0])])
        if len(left) >= 8:
            out.append(CurveCandidate("silhouette_rear_or_left_profile", np.asarray(left), 0.70, "opencv"))
        if len(right) >= 8:
            out.append(CurveCandidate("silhouette_front_or_right_profile", np.asarray(right), 0.70, "opencv"))
        return out

    def _edge_candidates(self, gray: np.ndarray, mask: np.ndarray) -> list[CurveCandidate]:
        cv2 = _require_cv2()
        edges = cv2.Canny(gray, self.options.canny_low, self.options.canny_high)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(edges, 8)
        candidates: list[CurveCandidate] = []
        for idx in range(1, num):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < self.options.min_component_points:
                continue
            ys, xs = np.where(labels == idx)
            pts = np.column_stack([xs, ys]).astype(float)
            ordered = _order_component_points(pts)
            if len(ordered) < self.options.min_component_points:
                continue
            bbox_w = float(np.ptp(ordered[:, 0]))
            bbox_h = float(np.ptp(ordered[:, 1]))
            extent = max(bbox_w, bbox_h)
            if extent < 40:
                continue
            label = _semantic_guess(ordered)
            confidence = min(0.90, 0.35 + len(ordered) / 1500.0 + extent / 2000.0)
            candidates.append(CurveCandidate(label, ordered, confidence, "opencv_edges"))
        return candidates


def extract_uncovered_line_art_candidates(
    image_path: str | Path,
    curves,
    min_len: float = 12.0,
    coverage_thickness: int = 7,
) -> list[CurveCandidate]:
    """Second pass: trace line-art centerline pixels not covered by existing curves."""
    cv2 = _require_cv2()
    from autoalias.geometry.bezier import evaluate_bezier

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or not _is_line_art(image):
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    skeleton = _skeletonize_zhang_suen(ink > 0)
    covered = np.zeros(skeleton.shape, np.uint8)
    for curve in curves:
        pts = evaluate_bezier(curve.cvs, np.linspace(0.0, 1.0, 220), curve.weights)
        pts_i = np.round(pts[:, :2]).astype(np.int32)
        if len(pts_i) >= 2:
            cv2.polylines(
                covered,
                [pts_i],
                False,
                255,
                thickness=coverage_thickness,
                lineType=cv2.LINE_AA,
            )

    missed = skeleton & ~(covered > 0)
    candidates: list[CurveCandidate] = []
    for chain in _trace_skeleton_chains(missed):
        length = _curve_length(chain)
        if length < min_len or len(chain) < 5:
            continue
        if max(float(np.ptp(chain[:, 0])), float(np.ptp(chain[:, 1]))) < 6:
            continue
        candidates.append(
            CurveCandidate(
                "line_art_missing_segment",
                chain,
                confidence=0.62,
                source="line_art_uncovered_pass",
            )
        )
    candidates.extend(_uncovered_component_repair_candidates(missed, min_area=4))
    candidates = _stitch_candidates(candidates, max_gap=12.0, max_angle_deg=24.0)
    return _trim_overlapping_candidates(_deduplicate_candidates(candidates), threshold_px=3.5)


def _order_component_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 2:
        return pts
    centered = pts[:, :2] - np.mean(pts[:, :2], axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    t = centered @ axis
    ordered = pts[np.argsort(t)]
    # If this is more vertical than horizontal, keep top-to-bottom consistency.
    if abs(axis[1]) > abs(axis[0]) and ordered[0, 1] > ordered[-1, 1]:
        ordered = ordered[::-1]
    elif abs(axis[0]) >= abs(axis[1]) and ordered[0, 0] > ordered[-1, 0]:
        ordered = ordered[::-1]
    return ordered


def _is_line_art(image: np.ndarray) -> bool:
    cv2 = _require_cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_ratio = float(np.mean(gray < 120))
    bright_ratio = float(np.mean(gray > 225))
    color_delta = float(np.mean(np.std(image.astype(float), axis=2)))
    return bright_ratio > 0.55 and 0.003 < dark_ratio < 0.35 and color_delta < 8.0


def _skeletonize_zhang_suen(binary: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    binary = binary.astype(np.uint8)
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is not None and hasattr(ximgproc, "thinning"):
        return ximgproc.thinning(binary * 255, thinningType=ximgproc.THINNING_ZHANGSUEN) > 0

    img = np.pad(binary, 1, mode="constant")
    changed = True
    iteration = 0
    while changed and iteration < 80:
        changed = False
        iteration += 1
        for phase in (0, 1):
            p2 = img[:-2, 1:-1]
            p3 = img[:-2, 2:]
            p4 = img[1:-1, 2:]
            p5 = img[2:, 2:]
            p6 = img[2:, 1:-1]
            p7 = img[2:, :-2]
            p8 = img[1:-1, :-2]
            p9 = img[:-2, :-2]
            center = img[1:-1, 1:-1]
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            transitions = np.zeros_like(center, dtype=np.uint8)
            for a, b in zip(neighbors, neighbors[1:] + neighbors[:1]):
                transitions += ((a == 0) & (b == 1)).astype(np.uint8)
            count = sum(neighbors)
            if phase == 0:
                m1 = (p2 * p4 * p6) == 0
                m2 = (p4 * p6 * p8) == 0
            else:
                m1 = (p2 * p4 * p8) == 0
                m2 = (p2 * p6 * p8) == 0
            remove = (center == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & m1 & m2
            if np.any(remove):
                center[remove] = 0
                changed = True
    return img[1:-1, 1:-1].astype(bool)


def _trace_skeleton_chains(skeleton: np.ndarray) -> list[np.ndarray]:
    ys, xs = np.where(skeleton)
    pixels = set(zip(xs.tolist(), ys.tolist()))
    if not pixels:
        return []

    degree: dict[tuple[int, int], int] = {p: len(_neighbors(p, pixels)) for p in pixels}
    nodes = {p for p, d in degree.items() if d != 2}
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    chains: list[np.ndarray] = []

    for node in sorted(nodes):
        for nb in _neighbors(node, pixels):
            edge = _edge_key(node, nb)
            if edge in visited_edges:
                continue
            chain = _walk_chain(node, nb, pixels, nodes, visited_edges)
            if len(chain) >= 2:
                chains.append(_chain_to_points(chain))

    # Closed loops have no endpoint/junction nodes.
    remaining_edges = []
    for p in pixels:
        for nb in _neighbors(p, pixels):
            edge = _edge_key(p, nb)
            if edge not in visited_edges:
                remaining_edges.append((p, nb))
    for start, nb in remaining_edges:
        edge = _edge_key(start, nb)
        if edge in visited_edges:
            continue
        chain = _walk_loop(start, nb, pixels, visited_edges)
        if len(chain) >= 8:
            chains.extend(_split_loop_chain(_chain_to_points(chain)))
    return [_smooth_ordered_chain(c) for c in chains]


def _walk_chain(
    start: tuple[int, int],
    first: tuple[int, int],
    pixels: set[tuple[int, int]],
    nodes: set[tuple[int, int]],
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]],
) -> list[tuple[int, int]]:
    chain = [start]
    prev = start
    curr = first
    while True:
        visited_edges.add(_edge_key(prev, curr))
        chain.append(curr)
        if curr in nodes and curr != start:
            break
        options = [p for p in _neighbors(curr, pixels) if p != prev]
        options = [p for p in options if _edge_key(curr, p) not in visited_edges]
        if not options:
            break
        # Continue as straight as possible through tiny skeleton artifacts.
        direction = np.array([curr[0] - prev[0], curr[1] - prev[1]], dtype=float)
        curr_arr = np.array(curr, dtype=float)
        nxt = max(
            options,
            key=lambda p: float(np.dot(direction, np.array(p, dtype=float) - curr_arr)),
        )
        prev, curr = curr, nxt
    return chain


def _walk_loop(
    start: tuple[int, int],
    first: tuple[int, int],
    pixels: set[tuple[int, int]],
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]],
) -> list[tuple[int, int]]:
    chain = [start]
    prev = start
    curr = first
    while True:
        edge = _edge_key(prev, curr)
        if edge in visited_edges:
            break
        visited_edges.add(edge)
        chain.append(curr)
        options = [p for p in _neighbors(curr, pixels) if p != prev]
        if not options:
            break
        if start in options and len(chain) > 8:
            chain.append(start)
            visited_edges.add(_edge_key(curr, start))
            break
        direction = np.array([curr[0] - prev[0], curr[1] - prev[1]], dtype=float)
        curr_arr = np.array(curr, dtype=float)
        nxt = max(
            options,
            key=lambda p: float(np.dot(direction, np.array(p, dtype=float) - curr_arr)),
        )
        prev, curr = curr, nxt
    return chain


def _neighbors(p: tuple[int, int], pixels: set[tuple[int, int]]) -> list[tuple[int, int]]:
    x, y = p
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            q = (x + dx, y + dy)
            if q in pixels:
                out.append(q)
    return out


def _edge_key(
    a: tuple[int, int], b: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _chain_to_points(chain: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray([[x, y] for x, y in chain], dtype=float)


def _split_loop_chain(points: np.ndarray) -> list[np.ndarray]:
    if len(points) < 24:
        return [points]
    if np.linalg.norm(points[0] - points[-1]) < 2:
        points = points[:-1]
    chunks = []
    count = max(3, min(6, int(_curve_length(points) / 180)))
    for i in range(count):
        a = int(i * len(points) / count)
        b = int((i + 1) * len(points) / count) + 1
        if b <= len(points):
            chunk = points[a:b]
        else:
            chunk = np.vstack([points[a:], points[: b - len(points)]])
        if len(chunk) >= 8:
            chunks.append(chunk)
    return chunks


def _hough_line_candidates(skeleton: np.ndarray, image_diag: float) -> list[CurveCandidate]:
    cv2 = _require_cv2()
    skel_u8 = (skeleton.astype(np.uint8) * 255)
    min_len = max(18, int(image_diag * 0.018))
    lines = cv2.HoughLinesP(
        skel_u8,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(14, int(min_len * 0.55)),
        minLineLength=min_len,
        maxLineGap=7,
    )
    if lines is None:
        return []

    raw = []
    for item in lines.reshape(-1, 4):
        x1, y1, x2, y2 = [float(v) for v in item]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_len:
            continue
        angle = float(np.arctan2(y2 - y1, x2 - x1))
        midpoint = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
        raw.append((length, angle, midpoint, np.array([x1, y1]), np.array([x2, y2])))

    raw.sort(key=lambda x: x[0], reverse=True)
    kept: list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray]] = []
    for line in raw:
        length, angle, midpoint, p0, p1 = line
        duplicate = False
        for other in kept:
            other_len, other_angle, other_mid, q0, q1 = other
            angle_delta = abs(np.arctan2(np.sin(angle - other_angle), np.cos(angle - other_angle)))
            endpoint_delta = min(
                np.linalg.norm(p0 - q0) + np.linalg.norm(p1 - q1),
                np.linalg.norm(p0 - q1) + np.linalg.norm(p1 - q0),
            )
            if angle_delta < np.deg2rad(3.0) and (
                endpoint_delta < 16.0 or np.linalg.norm(midpoint - other_mid) < 8.0
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(line)
        if len(kept) >= 180:
            break

    out: list[CurveCandidate] = []
    for length, _angle, _mid, p0, p1 in kept:
        n = max(6, min(80, int(length / 6)))
        t = np.linspace(0.0, 1.0, n)
        pts = (1.0 - t[:, None]) * p0 + t[:, None] * p1
        out.append(
            CurveCandidate(
                "line_art_straight_segment",
                pts,
                confidence=min(0.92, 0.50 + length / max(image_diag * 2.0, 1.0)),
                source="line_art_hough",
            )
        )
    return out


def _component_ellipse_candidates(ink: np.ndarray, image_diag: float) -> list[CurveCandidate]:
    cv2 = _require_cv2()
    num, labels, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    if num <= 1:
        return []
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)]
    largest = 1 + int(np.argmax(areas))
    out: list[CurveCandidate] = []
    for idx in range(1, num):
        if idx == largest:
            continue
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < 80 or area > 7000:
            continue
        if max(w, h) < 12 or max(w, h) > 260:
            continue
        ys, xs = np.where(labels == idx)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        if len(pts) < 12:
            continue
        try:
            ellipse = cv2.fitEllipse(pts.reshape(-1, 1, 2))
        except Exception:
            continue
        (cx, cy), (major, minor), angle_deg = ellipse
        if major < 8 or minor < 8:
            continue
        # fitEllipse may return swapped axes; both are diameters in pixels.
        axes = np.array([major * 0.5, minor * 0.5], dtype=float)
        if np.max(axes) / max(np.min(axes), 1e-6) > 4.5:
            continue
        # Slightly shrink thick-stroke contours toward their centerline.
        shrink = max(1.5, min(4.0, np.sqrt(area) * 0.035))
        axes = np.maximum(axes - shrink, 4.0)
        angle = np.deg2rad(angle_deg)
        arc_count = 4
        if max(w, h) < 45:
            arc_count = 3
        for k in range(arc_count):
            a0 = 2.0 * np.pi * k / arc_count
            a1 = 2.0 * np.pi * (k + 1) / arc_count
            theta = np.linspace(a0, a1, 48)
            local = np.column_stack([axes[0] * np.cos(theta), axes[1] * np.sin(theta)])
            rot = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
                dtype=float,
            )
            arc = local @ rot.T + np.array([cx, cy], dtype=float)
            out.append(
                CurveCandidate(
                    "line_art_closed_loop_arc",
                    arc,
                    confidence=0.78,
                    source="line_art_component_ellipse",
                )
            )
    return out


def _uncovered_component_repair_candidates(
    missed: np.ndarray,
    min_area: int = 4,
) -> list[CurveCandidate]:
    cv2 = _require_cv2()
    num, labels, stats, _ = cv2.connectedComponentsWithStats(missed.astype(np.uint8), 8)
    out: list[CurveCandidate] = []
    for idx in range(1, num):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < min_area:
            continue
        if max(w, h) < 2:
            continue
        ys, xs = np.where(labels == idx)
        pts = np.column_stack([xs, ys]).astype(float)
        ordered = _order_repair_component_points(pts)
        if len(ordered) < 4:
            continue
        # Tiny blobs are better represented as one short design repair segment than ignored.
        if _curve_length(ordered) < 2.0:
            continue
        out.append(
            CurveCandidate(
                "line_art_repair_segment",
                ordered,
                confidence=0.50,
                source="line_art_uncovered_component",
            )
        )
    return out


def _order_repair_component_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 2:
        return pts
    if len(pts) <= 12:
        return _nearest_neighbor_order(pts)
    # Prefer a farthest-pair path for small arc fragments. It is more stable than raw PCA when
    # the uncovered component is a curved wheel or mirror residue.
    sample = pts[:: max(1, len(pts) // 80), :2]
    diff = sample[:, None, :] - sample[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    i, j = np.unravel_index(int(np.argmax(dist2)), dist2.shape)
    start = sample[i]
    end = sample[j]
    axis = end - start
    norm = np.linalg.norm(axis)
    if norm <= 1e-6:
        return _nearest_neighbor_order(pts)
    axis = axis / norm
    proj = pts[:, :2] @ axis
    ordered = pts[np.argsort(proj)]
    # If projection collapses many points, nearest-neighbor gives a better short repair.
    if np.ptp(proj) < max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) * 0.55:
        ordered = _nearest_neighbor_order(pts)
    return _smooth_ordered_chain(ordered)


def _nearest_neighbor_order(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 2:
        return pts
    diff = pts[:, None, :2] - pts[None, :, :2]
    dist2 = np.sum(diff * diff, axis=2)
    i, j = np.unravel_index(int(np.argmax(dist2)), dist2.shape)
    start = int(i)
    remaining = set(range(len(pts)))
    order = [start]
    remaining.remove(start)
    while remaining:
        last = order[-1]
        nxt = min(remaining, key=lambda k: float(dist2[last, k]))
        order.append(nxt)
        remaining.remove(nxt)
    ordered = pts[order]
    # Make direction deterministic.
    if tuple(ordered[-1, :2]) < tuple(ordered[0, :2]):
        ordered = ordered[::-1]
    return _smooth_ordered_chain(ordered)


def _stitch_candidates(
    candidates: list[CurveCandidate],
    max_gap: float = 18.0,
    max_angle_deg: float = 28.0,
) -> list[CurveCandidate]:
    """Join skeleton fragments that are separated by tiny gaps and have matching tangents."""
    items = [c for c in candidates]
    max_angle = float(np.deg2rad(max_angle_deg))
    changed = True
    while changed:
        changed = False
        best: tuple[float, int, int, np.ndarray] | None = None
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                merged = _try_merge_pair(items[i].points, items[j].points, max_gap, max_angle)
                if merged is None:
                    continue
                gap = _merge_gap_score(items[i].points, items[j].points)
                if best is None or gap < best[0]:
                    best = (gap, i, j, merged)
        if best is None:
            break
        _, i, j, merged_points = best
        a = items[i]
        b = items[j]
        merged_candidate = CurveCandidate(
            label=a.label if _curve_length(a.points) >= _curve_length(b.points) else b.label,
            points=merged_points,
            confidence=max(a.confidence, b.confidence),
            source=f"{a.source}+stitched",
            metadata={"stitched_from": [a.label, b.label]},
        )
        items = [c for k, c in enumerate(items) if k not in (i, j)]
        items.append(merged_candidate)
        changed = True
    return items


def _try_merge_pair(
    a: np.ndarray,
    b: np.ndarray,
    max_gap: float,
    max_angle: float,
) -> np.ndarray | None:
    combos = [
        (np.linalg.norm(a[-1, :2] - b[0, :2]), _end_tangent(a), _start_tangent(b), a, b),
        (np.linalg.norm(b[-1, :2] - a[0, :2]), _end_tangent(b), _start_tangent(a), b, a),
        (np.linalg.norm(a[0, :2] - b[0, :2]), -_start_tangent(a), _start_tangent(b), a[::-1], b),
        (np.linalg.norm(a[-1, :2] - b[-1, :2]), _end_tangent(a), -_end_tangent(b), a, b[::-1]),
    ]
    best = None
    for gap, t0, t1, p0, p1 in combos:
        if gap > max_gap:
            continue
        if not np.all(np.isfinite(t0)) or not np.all(np.isfinite(t1)):
            continue
        angle = _angle_between(t0, t1)
        if angle > max_angle:
            continue
        score = gap + 25.0 * angle
        if best is None or score < best[0]:
            best = (score, p0, p1)
    if best is None:
        return None
    _, p0, p1 = best
    if np.linalg.norm(p0[-1, :2] - p1[0, :2]) <= 1.5:
        return np.vstack([p0, p1[1:]])
    return np.vstack([p0, p1])


def _merge_gap_score(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        min(
            np.linalg.norm(a[-1, :2] - b[0, :2]),
            np.linalg.norm(b[-1, :2] - a[0, :2]),
            np.linalg.norm(a[0, :2] - b[0, :2]),
            np.linalg.norm(a[-1, :2] - b[-1, :2]),
        )
    )


def _start_tangent(points: np.ndarray) -> np.ndarray:
    k = min(8, len(points) - 1)
    v = points[k, :2] - points[0, :2]
    n = np.linalg.norm(v)
    return v / max(n, 1e-9)


def _end_tangent(points: np.ndarray) -> np.ndarray:
    k = min(8, len(points) - 1)
    v = points[-1, :2] - points[-1 - k, :2]
    n = np.linalg.norm(v)
    return v / max(n, 1e-9)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.arccos(dot))


def _suppress_overlapping_candidates(
    candidates: list[CurveCandidate],
    threshold_px: float = 3.5,
    coverage_ratio: float = 0.72,
) -> list[CurveCandidate]:
    """Remove duplicate curves that lie on top of already-kept longer curves."""
    ordered = sorted(candidates, key=lambda c: (_curve_length(c.points), c.confidence), reverse=True)
    kept: list[CurveCandidate] = []
    for cand in ordered:
        if not kept:
            kept.append(cand)
            continue
        covered = False
        for other in kept:
            if _candidate_overlap_fraction(cand.points, other.points, threshold_px) >= coverage_ratio:
                covered = True
                break
        if not covered:
            kept.append(cand)
    return kept


def _trim_overlapping_candidates(
    candidates: list[CurveCandidate],
    threshold_px: float = 3.5,
    min_run_points: int = 6,
) -> list[CurveCandidate]:
    """Keep long curves, trim later candidates to only their non-overlapping portions.

    This is stricter than duplicate suppression: it prevents partial overlaps from producing two
    Alias curves on top of each other, while preserving true nearby parallel design lines.
    """
    ordered = sorted(candidates, key=lambda c: (_curve_length(c.points), c.confidence), reverse=True)
    kept: list[CurveCandidate] = []
    kept_clouds: list[np.ndarray] = []
    for cand in ordered:
        pts = cand.points
        if len(pts) < min_run_points:
            continue
        if not kept_clouds:
            kept.append(cand)
            kept_clouds.append(_sample_points_for_overlap(pts))
            continue
        cloud = np.vstack(kept_clouds)
        step = max(1, len(pts) // 180)
        probe = pts[::step, :2]
        dmin = _min_distances(probe, cloud)
        covered_probe = dmin <= threshold_px
        if np.mean(covered_probe) < 0.08:
            kept.append(cand)
            kept_clouds.append(_sample_points_for_overlap(pts))
            continue
        covered = np.interp(
            np.arange(len(pts)),
            np.arange(0, len(pts), step)[: len(covered_probe)],
            covered_probe.astype(float),
        ) >= 0.5
        runs = _false_runs(covered)
        added_any = False
        for start, end in runs:
            if end - start < min_run_points:
                continue
            segment = pts[start:end]
            if _curve_length(segment) < 10.0:
                continue
            trimmed = CurveCandidate(
                cand.label,
                segment,
                confidence=max(0.1, cand.confidence - 0.04),
                source=f"{cand.source}+trimmed",
                metadata={**cand.metadata, "trimmed_from_overlap": True},
            )
            kept.append(trimmed)
            kept_clouds.append(_sample_points_for_overlap(segment))
            added_any = True
        if not added_any and np.mean(covered) < 0.45:
            kept.append(cand)
            kept_clouds.append(_sample_points_for_overlap(pts))
    return kept


def _sample_points_for_overlap(points: np.ndarray) -> np.ndarray:
    step = max(1, len(points) // 180)
    return points[::step, :2]


def _false_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask):
        if not value and start is None:
            start = i
        elif value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _candidate_overlap_fraction(a: np.ndarray, b: np.ndarray, threshold_px: float) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    step_a = max(1, len(a) // 90)
    step_b = max(1, len(b) // 160)
    pa = a[::step_a, :2]
    pb = b[::step_b, :2]
    dmin = _min_distances(pa, pb)
    return float(np.mean(dmin <= threshold_px))


def _min_distances(points: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    out = np.full(len(points), np.inf, dtype=float)
    for start in range(0, len(cloud), 512):
        chunk = cloud[start : start + 512]
        diff = points[:, None, :] - chunk[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        out = np.minimum(out, np.sqrt(np.min(dist2, axis=1)))
    return out


def _smooth_ordered_chain(points: np.ndarray) -> np.ndarray:
    if len(points) < 7:
        return points
    # Keep original end points, lightly smooth pixel stair-steps.
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    pad = len(kernel) // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    out = np.vstack(
        [np.convolve(padded[:, j], kernel, mode="valid") for j in range(points.shape[1])]
    ).T
    out[0] = points[0]
    out[-1] = points[-1]
    return out


def _curve_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)))


def _deduplicate_candidates(candidates: list[CurveCandidate]) -> list[CurveCandidate]:
    out: list[CurveCandidate] = []
    for cand in sorted(candidates, key=lambda c: _curve_length(c.points), reverse=True):
        c0 = cand.points[0, :2]
        c1 = cand.points[-1, :2]
        duplicate = False
        for other in out:
            o0 = other.points[0, :2]
            o1 = other.points[-1, :2]
            if (
                min(np.linalg.norm(c0 - o0) + np.linalg.norm(c1 - o1), np.linalg.norm(c0 - o1) + np.linalg.norm(c1 - o0))
                < 8.0
                and abs(_curve_length(cand.points) - _curve_length(other.points)) < 16.0
            ):
                duplicate = True
                break
        if not duplicate:
            out.append(cand)
    return out


def _semantic_guess(points: np.ndarray) -> str:
    w = float(np.ptp(points[:, 0]))
    h = float(np.ptp(points[:, 1]))
    if w > 2.5 * max(h, 1.0):
        y_mean = float(np.mean(points[:, 1]))
        y_min = float(np.min(points[:, 1]))
        y_max = float(np.max(points[:, 1]))
        pos = (y_mean - y_min) / max(y_max - y_min, 1.0)
        if pos < 0.35:
            return "candidate_roofline_or_beltline"
        if pos > 0.65:
            return "candidate_side_skirt_or_bumper"
        return "candidate_character_or_beltline"
    if h > 2.0 * max(w, 1.0):
        return "candidate_front_rear_profile"
    return "candidate_lamp_grille_wheel_arch"


def _require_cv2():
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("OpenCV is required. Install with `pip install opencv-python`.") from exc
    return cv2
