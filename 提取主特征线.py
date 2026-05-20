import cv2
import numpy as np
from pathlib import Path


# ========= 路径设置 =========
input_path = Path(r"F:\430AutoAlias\image(225).png")
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)


# ========= 读取图片 =========
img = cv2.imread(str(input_path))
if img is None:
    raise FileNotFoundError(f"Cannot read image: {input_path}")

# 这张图的鼠标主体区域裁剪框
# 如果换图片，需要重新调这几个值
x0, y0, x1, y1 = 106, 105, 677, 876
crop = img[y0:y1, x0:x1].copy()

h, w = crop.shape[:2]


# ========= 1. 灰度 + 去噪 =========
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

# 双边滤波：去掉细小噪声，同时保留边缘
blur = cv2.bilateralFilter(gray, d=9, sigmaColor=40, sigmaSpace=40)

# 轻微增强对比度，让缝隙线更明显
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(blur)


# ========= 2. Canny 边缘检测 =========
edges = cv2.Canny(enhanced, threshold1=35, threshold2=95)

# 连接断裂的边缘
kernel = np.ones((3, 3), np.uint8)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)


# ========= 3. 轮廓提取和过滤 =========
contours, _ = cv2.findContours(
    edges,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_NONE
)

line_mask = np.zeros((h, w), dtype=np.uint8)

for cnt in contours:
    length = cv2.arcLength(cnt, closed=False)
    x, y, cw, ch = cv2.boundingRect(cnt)

    # 过滤很短、很碎的小边缘
    if length < 90:
        continue

    # 过滤太小的纹理噪声
    if cw < 10 or ch < 10:
        continue

    # 简化轮廓，减少锯齿
    approx = cv2.approxPolyDP(cnt, epsilon=1.2, closed=False)

    cv2.polylines(
        line_mask,
        [approx],
        isClosed=False,
        color=255,
        thickness=2,
        lineType=cv2.LINE_AA
    )


# ========= 4. 额外提取鼠标外轮廓 =========
# 用 GrabCut 分割主体，补充外轮廓
grab_mask = np.zeros((h, w), np.uint8)

rect = (5, 5, w - 10, h - 10)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

cv2.grabCut(
    crop,
    grab_mask,
    rect,
    bgd_model,
    fgd_model,
    5,
    cv2.GC_INIT_WITH_RECT
)

fg_mask = np.where(
    (grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD),
    255,
    0
).astype("uint8")

fg_mask = cv2.morphologyEx(
    fg_mask,
    cv2.MORPH_CLOSE,
    np.ones((9, 9), np.uint8),
    iterations=2
)

outer_contours, _ = cv2.findContours(
    fg_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if outer_contours:
    largest = max(outer_contours, key=cv2.contourArea)
    approx_outer = cv2.approxPolyDP(largest, epsilon=2.0, closed=True)

    cv2.polylines(
        line_mask,
        [approx_outer],
        isClosed=True,
        color=255,
        thickness=2,
        lineType=cv2.LINE_AA
    )


# ========= 5. 输出白底线稿 =========
white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
white_bg[line_mask > 0] = (0, 0, 0)

cv2.imwrite(str(out_dir / "feature_lines_white.png"), white_bg)


# ========= 6. 输出透明背景线稿 =========
transparent = np.zeros((h, w, 4), dtype=np.uint8)
transparent[..., 0] = 0
transparent[..., 1] = 0
transparent[..., 2] = 0
transparent[..., 3] = line_mask

cv2.imwrite(str(out_dir / "feature_lines_transparent.png"), transparent)


# ========= 7. 输出叠加预览 =========
overlay = crop.copy()
red_layer = crop.copy()

# OpenCV 是 BGR，红色是 (0, 0, 255)
red_layer[line_mask > 0] = (0, 0, 255)

overlay = cv2.addWeighted(crop, 0.75, red_layer, 0.25, 0)
cv2.imwrite(str(out_dir / "feature_lines_overlay.png"), overlay)


# ========= 8. 输出 SVG 矢量线稿 =========
def contour_to_svg_path(cnt):
    pts = cnt.reshape(-1, 2)
    if len(pts) < 2:
        return ""

    d = f"M {pts[0][0]} {pts[0][1]} "
    for x, y in pts[1:]:
        d += f"L {x} {y} "
    return d.strip()


svg_contours, _ = cv2.findContours(
    line_mask,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_NONE
)

svg_paths = []

for cnt in svg_contours:
    length = cv2.arcLength(cnt, closed=False)
    if length < 60:
        continue

    approx = cv2.approxPolyDP(cnt, epsilon=1.5, closed=False)
    path = contour_to_svg_path(approx)

    if path:
        svg_paths.append(path)

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<rect width="100%" height="100%" fill="white"/>
<g fill="none" stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
'''

for path in svg_paths:
    svg += f'<path d="{path}"/>\n'

svg += '''</g>
</svg>
'''

with open(out_dir / "feature_lines.svg", "w", encoding="utf-8") as f:
    f.write(svg)

print("Done.")
print(f"Saved to: {out_dir.resolve()}")