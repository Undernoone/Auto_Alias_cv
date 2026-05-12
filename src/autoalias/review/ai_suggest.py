from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from autoalias.vision.extractor import _require_cv2


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
_MODEL_CACHE: dict[tuple[str, str, bool], tuple[Any, Any]] = {}
_MODEL_LOCK = threading.Lock()


@dataclass(slots=True)
class VlmSuggestOptions:
    model: str = DEFAULT_MODEL
    device: str = "auto"
    local_files_only: bool = True
    max_curves: int = 12
    max_points_per_curve: int = 12
    max_new_tokens: int = 1800
    snap_max_distance: float = 96.0


def suggest_design_curves(
    image_path: str | Path,
    *,
    router: Any,
    output_dir: str | Path,
    options: VlmSuggestOptions | None = None,
    progress: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    """Ask a local vision-language model to propose manual split points.

    The VLM only proposes sparse designer-style cut points. The existing AutoAlias
    skeleton router still owns snapping, path finding, NURBS fitting and IGES export.
    """
    options = options or _options_from_env()
    image = Path(image_path).resolve()
    out = Path(output_dir).resolve() / "ai_context"
    out.mkdir(parents=True, exist_ok=True)

    _report(progress, 5, "正在生成 AI 识别用的骨架叠加图")
    context_path, image_size = _make_context_image(image, router.coords, out)
    _report(progress, 15, "骨架叠加图已生成")
    prompt = _build_prompt(
        width=image_size[0],
        height=image_size[1],
        max_curves=options.max_curves,
        max_points=options.max_points_per_curve,
    )
    raw_text = _run_qwen_vl(
        context_path,
        prompt,
        model_name=options.model,
        device=options.device,
        local_files_only=options.local_files_only,
        max_new_tokens=options.max_new_tokens,
        progress=progress,
    )
    _report(progress, 84, "正在解析 AI 返回的分线点")
    parsed = _extract_json_object(raw_text)
    _report(progress, 90, "正在把 AI 点吸附到骨架")
    curves = _normalize_curves(
        parsed,
        router=router,
        width=image_size[0],
        height=image_size[1],
        max_curves=options.max_curves,
        max_points_per_curve=options.max_points_per_curve,
        snap_max_distance=options.snap_max_distance,
    )
    _report(progress, 96, "AI 分线点已生成，正在写入曲线列表")
    return {
        "ok": True,
        "model": options.model,
        "context_image": str(context_path),
        "raw_text": raw_text,
        "curves": curves,
        "curve_count": len(curves),
    }


def dependency_status() -> dict[str, Any]:
    required = ["torch", "transformers", "qwen_vl_utils", "PIL", "accelerate"]
    status: dict[str, Any] = {"ok": True, "packages": {}}
    for name in required:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "")
            status["packages"][name] = {"ok": True, "version": version}
        except Exception as exc:
            status["ok"] = False
            status["packages"][name] = {"ok": False, "error": str(exc)}
    return status


def _options_from_env() -> VlmSuggestOptions:
    return VlmSuggestOptions(
        model=os.environ.get("AUTOALIAS_VLM_MODEL", DEFAULT_MODEL),
        device=os.environ.get("AUTOALIAS_VLM_DEVICE", "auto"),
        local_files_only=os.environ.get("AUTOALIAS_VLM_LOCAL_FILES_ONLY", "1").lower()
        not in {"0", "false", "no", "off"},
        max_curves=_env_int("AUTOALIAS_VLM_MAX_CURVES", 12),
        max_points_per_curve=_env_int("AUTOALIAS_VLM_MAX_POINTS", 12),
        max_new_tokens=_env_int("AUTOALIAS_VLM_MAX_NEW_TOKENS", 1800),
        snap_max_distance=float(os.environ.get("AUTOALIAS_VLM_SNAP_MAX_DISTANCE", "96")),
    )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _make_context_image(
    image_path: Path,
    skeleton_coords: np.ndarray,
    output_dir: Path,
) -> tuple[Path, tuple[int, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2 = _require_cv2()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    h, w = image.shape[:2]
    overlay = image.copy()

    coords = np.asarray(skeleton_coords, dtype=float)
    if coords.ndim == 2 and coords.shape[1] >= 2 and len(coords):
        if len(coords) > 36000:
            idx = np.linspace(0, len(coords) - 1, 36000).round().astype(int)
            coords = coords[idx]
        radius = max(1, int(round(max(w, h) / 1200)))
        for x_f, y_f in coords[:, :2]:
            x = int(round(float(x_f)))
            y = int(round(float(y_f)))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(overlay, (x, y), radius, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    blended = cv2.addWeighted(overlay, 0.62, image, 0.38, 0.0)
    grid_step = 100 if max(w, h) <= 1800 else 200
    for x in range(0, w, grid_step):
        cv2.line(blended, (x, 0), (x, h - 1), (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(blended, str(x), (x + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    for y in range(0, h, grid_step):
        cv2.line(blended, (0, y), (w - 1, y), (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(blended, str(y), (4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    cv2.putText(
        blended,
        f"AutoAlias VLM context: red=skeleton, coordinate size={w}x{h}",
        (18, max(32, int(h * 0.035))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    target = output_dir / f"{image_path.stem}_vlm_context_{int(time.time())}.png"
    cv2.imwrite(str(target), blended)
    return target, (int(w), int(h))


def _build_prompt(width: int, height: int, max_curves: int, max_points: int) -> str:
    return f"""
You are an expert automotive Alias Class-A curve layout assistant.

The image is an automotive side-view line drawing. Red pixels are the extracted skeleton.
Your job is NOT to trace dense pixels. Your job is to place sparse manual split points
the way an Alias designer would: enough points to define each intended design curve,
but not so many that the downstream single-span NURBS becomes ugly.

Coordinate system:
- Output pixel coordinates in the original image coordinate system.
- Image width = {width}, height = {height}.
- x grows to the right, y grows downward.
- Points should lie on visible black/red line structure as much as possible.

Rules:
- Return JSON only, no markdown, no comments.
- Use at most {max_curves} curves.
- Use 2 to {max_points} points per curve.
- Prefer important automotive curves: outer body profile, roofline, door/window opening,
  beltline, wheel arches, lamp outlines, bumper/skirt/detail lines.
- For long smooth curves, use few points at meaningful design transition locations.
- For L-like corners, put points before and after the blend region so AutoAlias can create
  a smooth corner segment.
- Do not output duplicate or nearly duplicate points.
- Do not output random reflections, labels, background, or grid lines.

JSON schema:
{{
  "curves": [
    {{
      "semantic": "outer_profile|door_opening|wheel_arch|beltline|roofline|lamp|bumper|detail_line",
      "closed": false,
      "confidence": 0.0,
      "reason": "short reason",
      "points": [[x1, y1], [x2, y2]]
    }}
  ]
}}
""".strip()


def _run_qwen_vl(
    image_path: Path,
    prompt: str,
    *,
    model_name: str,
    device: str,
    local_files_only: bool,
    max_new_tokens: int,
    progress: Callable[[int, str], None] | None = None,
) -> str:
    try:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except Exception as exc:
        raise RuntimeError(
            "AI dependencies are missing. Install: python -m pip install "
            "torch accelerate transformers qwen-vl-utils pillow"
        ) from exc

    resolved_device = _resolve_device(device, torch)
    cache_key = (model_name, resolved_device, local_files_only)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is None:
            _report(
                progress,
                22,
                "正在加载 Qwen-VL 模型；第一次使用会自动下载权重，时间取决于网络",
            )
            model_kwargs: dict[str, Any] = {
                "torch_dtype": "auto",
                "local_files_only": local_files_only,
            }
            if resolved_device.startswith("cuda"):
                model_kwargs["device_map"] = {"": resolved_device}
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
            if not resolved_device.startswith("cuda"):
                model = model.to(resolved_device)
            model.eval()
            processor = AutoProcessor.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
            _MODEL_CACHE[cache_key] = (model, processor)
            _report(progress, 62, "Qwen-VL 模型已加载")
        else:
            model, processor = cached
            _report(progress, 62, "已复用内存中的 Qwen-VL 模型")

    _report(progress, 70, "正在让视觉大模型理解汽车线条并建议分段点")
    messages = [
        {
            "role": "user",
            "content": [
                # qwen-vl-utils strips "file://" by slicing, which leaves "/F:/..."
                # on Windows and makes PIL raise Errno 22. A plain absolute path is
                # accepted by the same utility and works reliably on Windows.
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    if hasattr(model, "device"):
        inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    _report(progress, 82, "视觉大模型推理完成")
    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated, strict=False)
    ]
    decoded = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip() if decoded else ""


def _report(progress: Callable[[int, str], None] | None, percent: int, message: str) -> None:
    if progress is None:
        return
    try:
        progress(max(0, min(int(percent), 100)), message)
    except Exception:
        pass


def _resolve_device(device: str, torch_module: Any) -> str:
    if device and device != "auto":
        return device
    if torch_module.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"VLM did not return JSON: {text[:500]}")
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("VLM JSON must be an object")
    return parsed


def _normalize_curves(
    payload: dict[str, Any],
    *,
    router: Any,
    width: int,
    height: int,
    max_curves: int,
    max_points_per_curve: int,
    snap_max_distance: float,
) -> list[dict[str, Any]]:
    raw_curves = payload.get("curves", [])
    if isinstance(raw_curves, dict):
        raw_curves = [raw_curves]
    if not isinstance(raw_curves, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in raw_curves[:max_curves]:
        if not isinstance(raw, dict):
            continue
        points = _coerce_points(raw.get("points", []), width=width, height=height)
        points = _dedupe_points(points, min_distance=4.0)
        points = _limit_points(points, max_points_per_curve)
        snapped = _snap_points(points, router, max_distance=snap_max_distance)
        snapped = _dedupe_points(snapped, min_distance=4.0)
        if len(snapped) < 2:
            continue
        out.append(
            {
                "semantic": _clean_semantic(str(raw.get("semantic", "detail_line"))),
                "closed": bool(raw.get("closed", False)),
                "confidence": _safe_float(raw.get("confidence", 0.5), 0.5),
                "reason": str(raw.get("reason", ""))[:240],
                "manual_points": [
                    {"x": round(float(x), 3), "y": round(float(y), 3), "order": i}
                    for i, (x, y) in enumerate(snapped)
                ],
            }
        )
    return out


def _coerce_points(points: Any, *, width: int, height: int) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    if not isinstance(points, list):
        return out
    for item in points:
        x: Any
        y: Any
        if isinstance(item, dict):
            x = item.get("x")
            y = item.get("y")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = item[0], item[1]
        else:
            continue
        try:
            xf = min(max(float(x), 0.0), float(width - 1))
            yf = min(max(float(y), 0.0), float(height - 1))
        except (TypeError, ValueError):
            continue
        out.append((xf, yf))
    return out


def _snap_points(
    points: list[tuple[float, float]],
    router: Any,
    *,
    max_distance: float,
) -> list[tuple[float, float]]:
    snapped: list[tuple[float, float]] = []
    for point in points:
        try:
            index, distance = router.nearest_index(point)
            if distance <= max_distance:
                xy = router.coords[index]
                snapped.append((float(xy[0]), float(xy[1])))
            else:
                snapped.append(point)
        except Exception:
            snapped.append(point)
    return snapped


def _dedupe_points(
    points: list[tuple[float, float]],
    *,
    min_distance: float,
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for point in points:
        if out and np.linalg.norm(np.asarray(point) - np.asarray(out[-1])) < min_distance:
            continue
        out.append(point)
    if len(out) > 2 and np.linalg.norm(np.asarray(out[0]) - np.asarray(out[-1])) < min_distance:
        out.pop()
    return out


def _limit_points(points: list[tuple[float, float]], max_count: int) -> list[tuple[float, float]]:
    if len(points) <= max_count or max_count < 2:
        return points
    indices = np.linspace(0, len(points) - 1, max_count).round().astype(int)
    return [points[int(i)] for i in indices]


def _clean_semantic(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
    allowed = {
        "outer_profile",
        "door_opening",
        "wheel_arch",
        "beltline",
        "roofline",
        "lamp",
        "bumper",
        "detail_line",
    }
    aliases = {
        "window": "door_opening",
        "window_opening": "door_opening",
        "side_window": "door_opening",
        "body": "outer_profile",
        "silhouette": "outer_profile",
        "front_lamp": "lamp",
        "rear_lamp": "lamp",
        "side_skirt": "bumper",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in allowed else "detail_line"


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
