from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autoalias.vision.extractor import _require_cv2


@dataclass(slots=True)
class RawFeatureLinePreprocessResult:
    source_path: Path
    output_path: Path
    crop_bbox: tuple[int, int, int, int]
    line_pixels: int


def preprocess_raw_feature_lines(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    canny_low: int = 28,
    canny_high: int = 78,
    background_lab_distance: float = 9.0,
    min_component_area: int = 45,
    min_component_extent: int = 24,
    crop_padding: int = 35,
) -> RawFeatureLinePreprocessResult:
    """Convert an unprocessed photo/render into black-line-on-white line art.

    This is intended as a front door for images that have not already gone
    through ControlNet/Canny/line-art preprocessing. The returned image is the
    one AutoAlias should use for skeleton extraction, so cropping is safe because
    all downstream coordinates live in this processed image space.
    """
    cv2 = _require_cv2()
    source = Path(input_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{source.stem}_feature_lines_white.png"

    img_bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {source}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    smooth = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=65, sigmaSpace=65)
    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=int(canny_low), threshold2=int(canny_high), L2gradient=True)

    border = max(2, min(25, h // 4, w // 4))
    border_pixels = np.concatenate(
        [
            img_rgb[:border].reshape(-1, 3),
            img_rgb[-border:].reshape(-1, 3),
            img_rgb[:, :border].reshape(-1, 3),
            img_rgb[:, -border:].reshape(-1, 3),
        ]
    )
    border_lab = cv2.cvtColor(
        border_pixels.reshape(-1, 1, 3).astype(np.uint8),
        cv2.COLOR_RGB2LAB,
    ).reshape(-1, 3)
    bg_lab = np.median(border_lab, axis=0)

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    dist = np.linalg.norm(lab - bg_lab, axis=2)
    obj_mask = (dist > float(background_lab_distance)).astype(np.uint8) * 255
    obj_mask = cv2.medianBlur(obj_mask, 7)
    obj_mask = cv2.morphologyEx(
        obj_mask,
        cv2.MORPH_CLOSE,
        np.ones((17, 17), np.uint8),
        iterations=2,
    )
    obj_mask = cv2.morphologyEx(
        obj_mask,
        cv2.MORPH_OPEN,
        np.ones((7, 7), np.uint8),
        iterations=1,
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obj_mask)
    if num_labels > 1:
        largest_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        obj_mask = (labels == largest_id).astype(np.uint8) * 255

    flood = obj_mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, seedPoint=(0, 0), newVal=255)
    holes = cv2.bitwise_not(flood)
    obj_filled = cv2.bitwise_or(obj_mask, holes)
    obj_dilated = cv2.dilate(obj_filled, np.ones((25, 25), np.uint8), iterations=1)
    edges_obj = cv2.bitwise_and(edges, edges, mask=obj_dilated)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges_obj, connectivity=8)
    filtered = np.zeros_like(edges_obj)
    for index in range(1, num_labels):
        area = int(stats[index, cv2.CC_STAT_AREA])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if area >= int(min_component_area) and max(width, height) >= int(min_component_extent):
            filtered[labels == index] = 255

    line_mask = cv2.dilate(filtered, np.ones((2, 2), np.uint8), iterations=1)
    ys, xs = np.where(line_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError(
            "原图预处理没有找到可用线条像素。"
            "这通常表示当前图片已经是线稿/ControlNet 结果/黑底白线图，"
            "或者线条太淡、对比度太低，被主体遮罩和噪声过滤步骤过滤掉了。"
            "请取消“原图预处理”，或改用“黑底白线草图”/“铅笔弱线增强”，"
            "并适当降低弱线阈值后重新提取。"
        )

    pad = int(max(crop_padding, 0))
    x1 = max(int(xs.min()) - pad, 0)
    x2 = min(int(xs.max()) + pad, w)
    y1 = max(int(ys.min()) - pad, 0)
    y2 = min(int(ys.max()) + pad, h)
    line_crop = line_mask[y1:y2, x1:x2]

    white_bg = np.full((y2 - y1, x2 - x1, 3), 255, dtype=np.uint8)
    white_bg[line_crop > 0] = 0
    if not cv2.imwrite(str(output), white_bg):
        raise OSError(f"Cannot write preprocessed image: {output}")

    return RawFeatureLinePreprocessResult(
        source_path=source,
        output_path=output,
        crop_bbox=(x1, y1, x2, y2),
        line_pixels=int(np.count_nonzero(line_crop)),
    )


def preprocess_thick_stroke_contours(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    background_lab_distance: float = 10.0,
    min_component_area: int = 180,
    min_contour_area: float = 120.0,
    crop_padding: int = 35,
) -> RawFeatureLinePreprocessResult:
    """Extract outline curves from a thick filled logo/marker stroke image.

    Thick strokes are not line art: their medial skeleton is usually the wrong
    design curve. This mode converts dark/foreground filled regions into one-pixel
    contour lines, including inner holes, then hands that contour drawing to the
    normal AutoAlias graph builder.
    """
    cv2 = _require_cv2()
    source = Path(input_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{source.stem}_thick_stroke_contours_white.png"

    img_bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {source}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _otsu, dark_mask = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    border = max(2, min(25, h // 4, w // 4))
    border_pixels = np.concatenate(
        [
            img_rgb[:border].reshape(-1, 3),
            img_rgb[-border:].reshape(-1, 3),
            img_rgb[:, :border].reshape(-1, 3),
            img_rgb[:, -border:].reshape(-1, 3),
        ]
    )
    border_lab = cv2.cvtColor(
        border_pixels.reshape(-1, 1, 3).astype(np.uint8),
        cv2.COLOR_RGB2LAB,
    ).reshape(-1, 3)
    bg_lab = np.median(border_lab, axis=0)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    dist = np.linalg.norm(lab - bg_lab, axis=2)
    color_mask = ((dist > float(background_lab_distance)) & (gray < 248)).astype(np.uint8) * 255

    mask = cv2.bitwise_or(dark_mask, color_mask)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = int(np.max(areas)) if len(areas) else 0
        area_floor = max(int(min_component_area), int(largest * 0.015))
        for index in range(1, num_labels):
            area = int(stats[index, cv2.CC_STAT_AREA])
            width = int(stats[index, cv2.CC_STAT_WIDTH])
            height = int(stats[index, cv2.CC_STAT_HEIGHT])
            touches_corner = (
                int(stats[index, cv2.CC_STAT_LEFT]) <= 1
                and int(stats[index, cv2.CC_STAT_TOP]) >= h - max(6, h // 12)
            )
            if area >= area_floor and max(width, height) >= 16 and not touches_corner:
                filtered[labels == index] = 255
    mask = filtered if np.count_nonzero(filtered) else mask

    contours, _hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_mask = np.zeros_like(mask)
    for contour in contours:
        area = abs(float(cv2.contourArea(contour)))
        perimeter = float(cv2.arcLength(contour, closed=True))
        if area >= float(min_contour_area) or perimeter >= 45.0:
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=1, lineType=cv2.LINE_8)

    ys, xs = np.where(contour_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("thick-stroke contour preprocessing found no usable contours")

    pad = int(max(crop_padding, 0))
    x1 = max(int(xs.min()) - pad, 0)
    x2 = min(int(xs.max()) + pad, w)
    y1 = max(int(ys.min()) - pad, 0)
    y2 = min(int(ys.max()) + pad, h)
    contour_crop = contour_mask[y1:y2, x1:x2]

    white_bg = np.full((y2 - y1, x2 - x1, 3), 255, dtype=np.uint8)
    white_bg[contour_crop > 0] = 0
    if not cv2.imwrite(str(output), white_bg):
        raise OSError(f"Cannot write preprocessed image: {output}")

    return RawFeatureLinePreprocessResult(
        source_path=source,
        output_path=output,
        crop_bbox=(x1, y1, x2, y2),
        line_pixels=int(np.count_nonzero(contour_crop)),
    )
