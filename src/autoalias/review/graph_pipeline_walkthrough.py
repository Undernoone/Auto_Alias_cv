"""把 `graph.py` 里 `build_review_graph_bundle` 的流程拆成独立步骤，便于对照源码学习。

**推荐**：在 Cursor / PyCharm 中打开本文件，修改文件末尾的 ``IDE_IMAGE`` / ``IDE_DEBUG_OUT``，
然后右键 **Run** 或按 **F5**，无需命令行。

与网页上传后的处理一致：读图 → 墨迹 → 细化骨架 → Router → 链 → 边/节点 → 覆盖率。
最终仍调用 `build_review_graph_bundle` 做结果对照（应一致）。

其它代码里也可以直接调用::

    from pathlib import Path
    from autoalias.review.graph_pipeline_walkthrough import run_walkthrough
    run_walkthrough(Path(r"F:\\430AutoAlias\\test1.png"), debug_out=Path(r"F:\\430AutoAlias\\out"))
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 直接运行本文件时（例如 IDE / ``python .../graph_pipeline_walkthrough.py``），
# Python 默认不把仓库的 ``src`` 放进 sys.path，会报 ``No module named 'autoalias'``。
_pkg_root = Path(__file__).resolve().parents[2]
if (_pkg_root / "autoalias").is_dir() and str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import numpy as np

from autoalias.review.graph import (
    ReviewGraphOptions,
    SkeletonRouter,
    _as_points3,
    _bbox,
    _cluster_endpoints,
    _downsample_points,
    _edge_coverage_metrics,
    _make_nodes,
    _round_points,
    build_review_graph_bundle,
    graph_snapshot_for_training,
)
from autoalias.vision.extractor import (
    _curve_length,
    _is_line_art,
    _require_cv2,
    _semantic_guess,
    _skeletonize_zhang_suen,
    _trace_skeleton_chains,
)


# ---------------------------------------------------------------------------
# 步骤 0：入口与说明
# ---------------------------------------------------------------------------


def step00_intro(image_path: Path) -> None:
    print("\n[步骤 0] 开始")
    print(f"  图片路径: {image_path.resolve()}")
    print(
        "  后续步骤与 `review/graph.py` 中 `build_review_graph_bundle` 一致，"
        "对应源码约 219–305 行。"
    )


# ---------------------------------------------------------------------------
# 步骤 1：读入彩色图
# ---------------------------------------------------------------------------


def step01_read_bgr(image_path: Path) -> np.ndarray:
    print("\n[步骤 1] 读入 BGR 彩色图")
    cv2 = _require_cv2()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    h, w = image.shape[:2]
    print(f"  尺寸: 宽={w} 高={h}，通道数={image.shape[2]}（BGR）")
    return image


# ---------------------------------------------------------------------------
# 步骤 2：灰度 + 高斯模糊
# ---------------------------------------------------------------------------


def step02_grayscale_blur(image: np.ndarray) -> np.ndarray:
    print("\n[步骤 2] 灰度化 + 高斯模糊")
    cv2 = _require_cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    print("  使用 kernel=(3,3) 的 GaussianBlur 抑制噪声。")
    return gray


# ---------------------------------------------------------------------------
# 步骤 3：线稿判定 + 生成墨迹掩膜 ink
# ---------------------------------------------------------------------------


def step03_binary_ink(image: np.ndarray, gray: np.ndarray) -> np.ndarray:
    print("\n[步骤 3] 生成二值「墨迹」ink（白=线条区域，黑=背景）")
    cv2 = _require_cv2()
    is_art = _is_line_art(image)
    print(f"  `_is_line_art` = {is_art}")
    if is_art:
        print("  → 走线稿分支: Otsu 反二值（THRESH_BINARY_INV + THRESH_OTSU）")
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        print("  → 走照片/渲染分支: Canny(55, 150)")
        ink = cv2.Canny(gray, 55, 150)
    ink_nonzero = int(np.count_nonzero(ink))
    print(f"  ink 中非零像素数: {ink_nonzero}")
    return ink


# ---------------------------------------------------------------------------
# 步骤 4：形态学开运算 + 闭运算
# ---------------------------------------------------------------------------


def step04_morphology(ink: np.ndarray) -> np.ndarray:
    print("\n[步骤 4] 形态学处理")
    cv2 = _require_cv2()
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, small, iterations=1)
    print("  MORPH_OPEN 迭代 1：去掉小噪点")
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, small, iterations=1)
    print("  MORPH_CLOSE 迭代 1：弥合小断裂")
    return ink


# ---------------------------------------------------------------------------
# 步骤 5：Zhang–Suen 细化 → 单像素骨架
# ---------------------------------------------------------------------------


def step05_skeletonize(ink: np.ndarray) -> np.ndarray:
    print("\n[步骤 5] Zhang–Suen 细化（`_skeletonize_zhang_suen`）")
    print("  输入: ink > 0 的布尔区域")
    skeleton = _skeletonize_zhang_suen(ink > 0)
    n_pix = int(np.count_nonzero(skeleton))
    print(f"  骨架像素数: {n_pix}（每条线约 1 像素宽）")
    return skeleton


# ---------------------------------------------------------------------------
# 步骤 6：由骨架建 SkeletonRouter（导航图）
# ---------------------------------------------------------------------------


def step06_build_router(skeleton: np.ndarray) -> SkeletonRouter:
    print("\n[步骤 6] `SkeletonRouter.from_skeleton`")
    print("  每个骨架像素 = 图上一个顶点；8 邻域连边，斜边权重 sqrt(2)。")
    router = SkeletonRouter.from_skeleton(skeleton)
    print(f"  Router 顶点数: {len(router.coords)}")
    return router


# ---------------------------------------------------------------------------
# 步骤 7：沿骨架追踪链 chains
# ---------------------------------------------------------------------------


def step07_trace_chains(skeleton: np.ndarray) -> list[np.ndarray]:
    print("\n[步骤 7] `_trace_skeleton_chains`")
    print("  在岔口/端点处切开，得到多条折线链；闭合环单独处理。")
    chains = _trace_skeleton_chains(skeleton)
    print(f"  链条数: {len(chains)}")
    lens = [len(c) for c in chains]
    if lens:
        print(f"  单链点数: min={min(lens)} max={max(lens)}")
    return chains


# ---------------------------------------------------------------------------
# 步骤 8：链 → raw_edges / coverage_fragments
# ---------------------------------------------------------------------------


def step08_filter_chains_to_raw_edges(
    chains: list[np.ndarray],
    options: ReviewGraphOptions,
) -> tuple[list[dict[str, Any]], list[list[list[float]]]]:
    print("\n[步骤 8] 过滤链 → raw_edges 与 coverage_fragments")
    print(
        f"  规则: 长度 < {options.min_edge_length} 或点数 < 3 → fragment；"
        f" bbox 宽高都 < 3 → fragment；否则进入 raw_edges。"
    )
    raw_edges: list[dict[str, Any]] = []
    coverage_fragments: list[list[list[float]]] = []
    for chain in chains:
        points = _as_points3(chain)
        length = _curve_length(points)
        if length < options.min_edge_length or len(points) < 3:
            coverage_fragments.append(_round_points(_downsample_points(points, 12)))
            continue
        bbox = _bbox(points)
        if max(bbox["width"], bbox["height"]) < 3:
            coverage_fragments.append(_round_points(_downsample_points(points, 12)))
            continue
        raw_edges.append(
            {
                "points": points,
                "length": float(length),
                "bbox": bbox,
                "label": _semantic_guess(points),
            }
        )
    print(f"  raw_edges: {len(raw_edges)}，coverage_fragments: {len(coverage_fragments)}")
    return raw_edges, coverage_fragments


# ---------------------------------------------------------------------------
# 步骤 9：排序、端点聚类、nodes、edges 列表
# ---------------------------------------------------------------------------


def step09_build_graph_edges_and_nodes(
    raw_edges: list[dict[str, Any]],
    options: ReviewGraphOptions,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print("\n[步骤 9] 聚类端点 → nodes；拼装 edges JSON")
    print(
        f"  `endpoint_cluster_radius`={options.endpoint_cluster_radius}，"
        f"`max_points_per_edge`={options.max_points_per_edge}"
    )
    raw_edges.sort(key=lambda item: item["length"], reverse=True)
    node_ids = _cluster_endpoints(raw_edges, options.endpoint_cluster_radius)
    nodes = _make_nodes(raw_edges, node_ids)
    edges: list[dict[str, Any]] = []
    for idx, edge in enumerate(raw_edges):
        points = _downsample_points(edge["points"], options.max_points_per_edge)
        start_node, end_node = node_ids[idx]
        edges.append(
            {
                "id": f"edge_{idx:04d}",
                "label": edge["label"],
                "points": _round_points(points),
                "start_node": start_node,
                "end_node": end_node,
                "length": round(float(edge["length"]), 3),
                "bbox": edge["bbox"],
            }
        )
    node_list: list[dict[str, Any]] = []
    for node_id, node in nodes.items():
        node_list.append(
            {
                "id": node_id,
                "x": round(float(node["point"][0]), 3),
                "y": round(float(node["point"][1]), 3),
                "degree": len(node["edges"]),
                "edges": sorted(node["edges"]),
            }
        )
    node_list.sort(key=lambda item: (-int(item["degree"]), item["id"]))
    print(f"  edges: {len(edges)}，nodes: {len(node_list)}")
    return edges, node_list


# ---------------------------------------------------------------------------
# 步骤 10：骨架覆盖率
# ---------------------------------------------------------------------------


def step10_coverage(
    skeleton: np.ndarray,
    raw_edges: list[dict[str, Any]],
    coverage_fragments: list[list[list[float]]],
) -> dict[str, float | int]:
    print("\n[步骤 10] `_edge_coverage_metrics`")
    print("  把 raw_edges 与 fragments 画到掩膜上（线宽 3），与 skeleton 对比。")
    cov = _edge_coverage_metrics(skeleton, raw_edges, coverage_fragments)
    print(f"  coverage_ratio ≈ {cov.get('coverage_ratio')}（1 表示骨架全被边/碎片覆盖到）")
    return cov


# ---------------------------------------------------------------------------
# 步骤 11：组装 graph 字典（与 graph.py 返回结构一致）
# ---------------------------------------------------------------------------


def step11_assemble_graph_dict(
    path: Path,
    h: int,
    w: int,
    edges: list[dict[str, Any]],
    node_list: list[dict[str, Any]],
    coverage: dict[str, float | int],
    coverage_fragments: list[list[list[float]]],
) -> dict[str, Any]:
    print("\n[步骤 11] 组装 graph 字典（version / image_size / edges / nodes / coverage …）")
    graph = {
        "version": 1,
        "image": str(path.resolve()),
        "image_name": path.name,
        "image_size": {"width": int(w), "height": int(h)},
        "edge_count": len(edges),
        "node_count": len(node_list),
        "coverage": coverage,
        "coverage_fragments": coverage_fragments,
        "edges": edges,
        "nodes": node_list,
    }
    print(
        f"  edge_count={graph['edge_count']}, node_count={graph['node_count']}, "
        f"fragments={len(coverage_fragments)}"
    )
    return graph


# ---------------------------------------------------------------------------
# 步骤 12：与 `build_review_graph_bundle` 对照
# ---------------------------------------------------------------------------


def step12_verify_against_graph_module(
    image_path: Path, options: ReviewGraphOptions | None
) -> None:
    print("\n[步骤 12] 对照 `build_review_graph_bundle`（应步进结果与之一致）")
    options = options or ReviewGraphOptions()
    bundle_graph, _router = build_review_graph_bundle(image_path, options)
    # 不对 fragments 做逐元素比对（浮点列表顺序可能等价即可）；比对关键计数
    print(f"  bundle edge_count={bundle_graph['edge_count']} node_count={bundle_graph['node_count']}")


# ---------------------------------------------------------------------------
# 可选：调试图保存（对应前端红线/绿线的数据来源）
# ---------------------------------------------------------------------------


def save_debug_images(
    debug_out: Path,
    image: np.ndarray,
    ink: np.ndarray,
    skeleton: np.ndarray,
    graph_edges: list[dict[str, Any]],
) -> None:
    """保存 ink、骨架叠加、绿边叠加，方便和网页红/绿对照。"""
    cv2 = _require_cv2()
    debug_out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_out / "01_ink.png"), ink)
    print(f"  已写: {debug_out / '01_ink.png'}")

    skel_u8 = (skeleton.astype(np.uint8) * 255)
    cv2.imwrite(str(debug_out / "02_skeleton_only.png"), skel_u8)
    print(f"  已写: {debug_out / '02_skeleton_only.png'}")

    red = np.zeros_like(image)
    red[:, :, 2] = skel_u8  # BGR: red channel
    blend = cv2.addWeighted(image, 0.65, red, 0.35, 0)
    cv2.imwrite(str(debug_out / "03_skeleton_on_image.png"), blend)
    print(f"  已写: {debug_out / '03_skeleton_on_image.png'}（对应网页「完整骨架」思路）")

    green_overlay = np.zeros_like(image)
    for edge in graph_edges:
        pts = np.array(edge.get("points") or [], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(green_overlay, [pts], False, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    edges_blend = cv2.addWeighted(image, 0.55, green_overlay, 0.45, 0)
    cv2.imwrite(str(debug_out / "04_edges_on_image.png"), edges_blend)
    print(f"  已写: {debug_out / '04_edges_on_image.png'}（对应网页绿色 edges）")


# ---------------------------------------------------------------------------
# 完整流水线（逐步打印）
# ---------------------------------------------------------------------------


@dataclass
class WalkthroughResult:
    graph: dict[str, Any]
    router: SkeletonRouter
    training_snapshot: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.training_snapshot = graph_snapshot_for_training(self.graph)


def run_walkthrough(
    image_path: str | Path,
    options: ReviewGraphOptions | None = None,
    *,
    debug_out: Path | None = None,
    verify: bool = True,
) -> WalkthroughResult:
    path = Path(image_path)
    options = options or ReviewGraphOptions()

    step00_intro(path)
    image = step01_read_bgr(path)
    h, w = image.shape[:2]

    gray = step02_grayscale_blur(image)
    ink = step03_binary_ink(image, gray)
    ink = step04_morphology(ink)

    skeleton = step05_skeletonize(ink)
    router = step06_build_router(skeleton)
    chains = step07_trace_chains(skeleton)
    raw_edges, coverage_fragments = step08_filter_chains_to_raw_edges(chains, options)
    edges, node_list = step09_build_graph_edges_and_nodes(raw_edges, options)
    coverage = step10_coverage(skeleton, raw_edges, coverage_fragments)
    graph = step11_assemble_graph_dict(path, h, w, edges, node_list, coverage, coverage_fragments)

    if verify:
        step12_verify_against_graph_module(path, options)
        bg, _ = build_review_graph_bundle(path, options)
        if bg["edge_count"] != graph["edge_count"] or bg["node_count"] != graph["node_count"]:
            raise RuntimeError("步进结果与 build_review_graph_bundle 不一致，请检查代码。")
        print("  [OK] edge/node 计数与 build_review_graph_bundle 一致")

    if debug_out is not None:
        print("\n[调试] 写入中间图像…")
        save_debug_images(debug_out, image, ink, skeleton, edges)

    print("\n完成。网页红色点来自 Router 坐标下采样；绿色线来自 graph['edges']。")
    return WalkthroughResult(graph=graph, router=router)


# -----------------------------------------------------------------------------
# IDE 入口：改下面两行，然后 Run 本文件（不要依赖命令行参数）
# -----------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]

# 要处理的图片（改成你的路径）
IDE_IMAGE: Path = _REPO_ROOT / "F:\ComfyUI\output\\2026-05-18\controlnet\\110534_realistic_lineart_00001_.png"

# 调试输出目录；设为 None 则只打印步骤、不写 PNG/JSON
IDE_DEBUG_OUT: Path | None = _REPO_ROOT / "debug_graph_steps"


def run_from_ide(
    image: Path | None = None,
    debug_out: Path | None = None,
    *,
    verify: bool = True,
) -> WalkthroughResult:
    """在 IDE 里 Run 本文件时调用；也可在别的模块里 ``run_from_ide()``。"""
    img = image if image is not None else IDE_IMAGE
    out = debug_out if debug_out is not None else IDE_DEBUG_OUT
    result = run_walkthrough(img, debug_out=out, verify=verify)
    if out is not None:
        summary = out / "graph_summary.json"
        summary.write_text(
            json.dumps(result.training_snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"已写训练快照摘要: {summary}")
    return result


if __name__ == "__main__":
    run_from_ide()
