# import cv2
# import numpy as np
# from pathlib import Path
#
#
# # ========= 路径设置 =========
# input_path = Path(r"F:\430AutoAlias\image(225).png")
# out_dir = Path("outputs")
# out_dir.mkdir(exist_ok=True)
#
#
# # ========= 读取图片 =========
# img = cv2.imread(str(input_path))
# if img is None:
#     raise FileNotFoundError(f"Cannot read image: {input_path}")
#
# # 这张图的鼠标主体区域裁剪框
# # 如果换图片，需要重新调这几个值
# x0, y0, x1, y1 = 106, 105, 677, 876
# crop = img[y0:y1, x0:x1].copy()
#
# h, w = crop.shape[:2]
#
#
# # ========= 1. 灰度 + 去噪 =========
# gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#
# # 双边滤波：去掉细小噪声，同时保留边缘
# blur = cv2.bilateralFilter(gray, d=9, sigmaColor=40, sigmaSpace=40)
#
# # 轻微增强对比度，让缝隙线更明显
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# enhanced = clahe.apply(blur)
#
#
# # ========= 2. Canny 边缘检测 =========
# edges = cv2.Canny(enhanced, threshold1=35, threshold2=95)
#
# # 连接断裂的边缘
# kernel = np.ones((3, 3), np.uint8)
# edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
#
#
# # ========= 3. 轮廓提取和过滤 =========
# contours, _ = cv2.findContours(
#     edges,
#     cv2.RETR_LIST,
#     cv2.CHAIN_APPROX_NONE
# )
#
# line_mask = np.zeros((h, w), dtype=np.uint8)
#
# for cnt in contours:
#     length = cv2.arcLength(cnt, closed=False)
#     x, y, cw, ch = cv2.boundingRect(cnt)
#
#     # 过滤很短、很碎的小边缘
#     if length < 90:
#         continue
#
#     # 过滤太小的纹理噪声
#     if cw < 10 or ch < 10:
#         continue
#
#     # 简化轮廓，减少锯齿
#     approx = cv2.approxPolyDP(cnt, epsilon=1.2, closed=False)
#
#     cv2.polylines(
#         line_mask,
#         [approx],
#         isClosed=False,
#         color=255,
#         thickness=2,
#         lineType=cv2.LINE_AA
#     )
#
#
# # ========= 4. 额外提取鼠标外轮廓 =========
# # 用 GrabCut 分割主体，补充外轮廓
# grab_mask = np.zeros((h, w), np.uint8)
#
# rect = (5, 5, w - 10, h - 10)
# bgd_model = np.zeros((1, 65), np.float64)
# fgd_model = np.zeros((1, 65), np.float64)
#
# cv2.grabCut(
#     crop,
#     grab_mask,
#     rect,
#     bgd_model,
#     fgd_model,
#     5,
#     cv2.GC_INIT_WITH_RECT
# )
#
# fg_mask = np.where(
#     (grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD),
#     255,
#     0
# ).astype("uint8")
#
# fg_mask = cv2.morphologyEx(
#     fg_mask,
#     cv2.MORPH_CLOSE,
#     np.ones((9, 9), np.uint8),
#     iterations=2
# )
#
# outer_contours, _ = cv2.findContours(
#     fg_mask,
#     cv2.RETR_EXTERNAL,
#     cv2.CHAIN_APPROX_SIMPLE
# )
#
# if outer_contours:
#     largest = max(outer_contours, key=cv2.contourArea)
#     approx_outer = cv2.approxPolyDP(largest, epsilon=2.0, closed=True)
#
#     cv2.polylines(
#         line_mask,
#         [approx_outer],
#         isClosed=True,
#         color=255,
#         thickness=2,
#         lineType=cv2.LINE_AA
#     )
#
#
# # ========= 5. 输出白底线稿 =========
# white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
# white_bg[line_mask > 0] = (0, 0, 0)
#
# cv2.imwrite(str(out_dir / "feature_lines_white.png"), white_bg)
#
#
# # ========= 6. 输出透明背景线稿 =========
# transparent = np.zeros((h, w, 4), dtype=np.uint8)
# transparent[..., 0] = 0
# transparent[..., 1] = 0
# transparent[..., 2] = 0
# transparent[..., 3] = line_mask
#
# cv2.imwrite(str(out_dir / "feature_lines_transparent.png"), transparent)
#
#
# # ========= 7. 输出叠加预览 =========
# overlay = crop.copy()
# red_layer = crop.copy()
#
# # OpenCV 是 BGR，红色是 (0, 0, 255)
# red_layer[line_mask > 0] = (0, 0, 255)
#
# overlay = cv2.addWeighted(crop, 0.75, red_layer, 0.25, 0)
# cv2.imwrite(str(out_dir / "feature_lines_overlay.png"), overlay)
#
#
# # ========= 8. 输出 SVG 矢量线稿 =========
# def contour_to_svg_path(cnt):
#     pts = cnt.reshape(-1, 2)
#     if len(pts) < 2:
#         return ""
#
#     d = f"M {pts[0][0]} {pts[0][1]} "
#     for x, y in pts[1:]:
#         d += f"L {x} {y} "
#     return d.strip()
#
#
# svg_contours, _ = cv2.findContours(
#     line_mask,
#     cv2.RETR_LIST,
#     cv2.CHAIN_APPROX_NONE
# )
#
# svg_paths = []
#
# for cnt in svg_contours:
#     length = cv2.arcLength(cnt, closed=False)
#     if length < 60:
#         continue
#
#     approx = cv2.approxPolyDP(cnt, epsilon=1.5, closed=False)
#     path = contour_to_svg_path(approx)
#
#     if path:
#         svg_paths.append(path)
#
# svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
# <rect width="100%" height="100%" fill="white"/>
# <g fill="none" stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
# '''
#
# for path in svg_paths:
#     svg += f'<path d="{path}"/>\n'
#
# svg += '''</g>
# </svg>
# '''
#
# with open(out_dir / "feature_lines.svg", "w", encoding="utf-8") as f:
#     f.write(svg)
#
# print("Done.")
# print(f"Saved to: {out_dir.resolve()}")

import cv2
import numpy as np
from PIL import Image
from pathlib import Path


# ========= 输入输出 =========
input_path = Path(r"F:\430AutoAlias\image(225).png")  # 改成你的原图路径
output_path = Path("mouse_feature_lines_white.png")


# ========= 读取图片 =========
img_bgr = cv2.imread(str(input_path))

if img_bgr is None:
    raise FileNotFoundError(f"Cannot read image: {input_path}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]


# ========= 1. 保边去噪 =========
# 双边滤波可以去掉表面细噪声，同时保留结构边缘
smooth = cv2.bilateralFilter(
    img_rgb,
    d=9,
    sigmaColor=65,
    sigmaSpace=65
)

gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)


# ========= 2. Canny 提取边缘 =========
edges = cv2.Canny(
    gray,
    threshold1=28,
    threshold2=78,
    L2gradient=True
)


# ========= 3. 生成主体区域 mask，去掉背景边缘 =========
# 用图像边缘区域估计背景颜色
border_pixels = np.concatenate([
    img_rgb[:25].reshape(-1, 3),
    img_rgb[-25:].reshape(-1, 3),
    img_rgb[:, :25].reshape(-1, 3),
    img_rgb[:, -25:].reshape(-1, 3)
])

border_lab = cv2.cvtColor(
    border_pixels.reshape(-1, 1, 3).astype(np.uint8),
    cv2.COLOR_RGB2LAB
).reshape(-1, 3)

bg_lab = np.median(border_lab, axis=0)

# 计算每个像素和背景颜色的 LAB 距离
lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
dist = np.linalg.norm(lab - bg_lab, axis=2)

# 距离背景足够远的区域认为是鼠标主体和阴影区域
obj_mask = (dist > 9).astype(np.uint8) * 255

# 平滑 mask
obj_mask = cv2.medianBlur(obj_mask, 7)

obj_mask = cv2.morphologyEx(
    obj_mask,
    cv2.MORPH_CLOSE,
    np.ones((17, 17), np.uint8),
    iterations=2
)

obj_mask = cv2.morphologyEx(
    obj_mask,
    cv2.MORPH_OPEN,
    np.ones((7, 7), np.uint8),
    iterations=1
)


# ========= 4. 只保留最大的主体连通区域 =========
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obj_mask)

if num_labels > 1:
    largest_id = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    obj_mask = (labels == largest_id).astype(np.uint8) * 255


# ========= 5. 填充主体内部孔洞 =========
flood = obj_mask.copy()
flood_mask = np.zeros((h + 2, w + 2), np.uint8)

cv2.floodFill(
    flood,
    flood_mask,
    seedPoint=(0, 0),
    newVal=255
)

holes = cv2.bitwise_not(flood)
obj_filled = cv2.bitwise_or(obj_mask, holes)

# 稍微膨胀主体 mask，确保边缘不会被裁掉
obj_dilated = cv2.dilate(
    obj_filled,
    np.ones((25, 25), np.uint8),
    iterations=1
)


# ========= 6. 用主体 mask 限制 Canny 边缘 =========
edges_obj = cv2.bitwise_and(
    edges,
    edges,
    mask=obj_dilated
)


# ========= 7. 连通域过滤，去掉碎噪声 =========
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    edges_obj,
    connectivity=8
)

filtered = np.zeros_like(edges_obj)

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    ww = stats[i, cv2.CC_STAT_WIDTH]
    hh = stats[i, cv2.CC_STAT_HEIGHT]

    # 保留较长的结构线，去掉小纹理、小噪点
    if area >= 45 and max(ww, hh) >= 24:
        filtered[labels == i] = 255


# ========= 8. 线条加粗成适合展示的黑线 =========
line_mask = cv2.dilate(
    filtered,
    np.ones((2, 2), np.uint8),
    iterations=1
)


# ========= 9. 裁剪到线条区域 =========
ys, xs = np.where(line_mask > 0)

pad = 35

x1 = max(xs.min() - pad, 0)
x2 = min(xs.max() + pad, w)
y1 = max(ys.min() - pad, 0)
y2 = min(ys.max() + pad, h)

line_crop = line_mask[y1:y2, x1:x2]


# ========= 10. 黑线白底输出 =========
white_bg = np.full(
    (y2 - y1, x2 - x1, 3),
    255,
    dtype=np.uint8
)

white_bg[line_crop > 0] = 0

Image.fromarray(white_bg).save(output_path)

print(f"Saved: {output_path}")