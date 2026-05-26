from __future__ import annotations

import json
import mimetypes
import socket
import time
import webbrowser
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from autoalias.review.graph import ReviewGraphOptions, build_review_graph_bundle, graph_snapshot_for_training
from autoalias.review.preprocess import preprocess_raw_feature_lines, preprocess_thick_stroke_contours


@dataclass(slots=True)
class ReviewSession:
    image_path: Path
    output_dir: Path
    graph: dict[str, Any]
    router: Any
    corrections_path: Path
    corrections: list[dict[str, Any]] = field(default_factory=list)
    design_curves: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        image_path: str | Path,
        output_dir: str | Path,
        graph_options: ReviewGraphOptions | None = None,
    ) -> "ReviewSession":
        image = Path(image_path).resolve()
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        graph_options = graph_options or ReviewGraphOptions()
        source_image = image
        preprocess_mode = _clean_input_preprocess(graph_options.input_preprocess)
        preprocess_meta: dict[str, Any] = {}
        if preprocess_mode == "raw_feature_lines":
            preprocessed = preprocess_raw_feature_lines(image, out / "preprocessed")
            image = preprocessed.output_path
            graph_options = replace(
                graph_options,
                input_preprocess=preprocess_mode,
                extraction_mode="black_on_white_line_art",
            )
            preprocess_meta = {
                "source_image": str(source_image),
                "preprocessed_image": str(preprocessed.output_path),
                "preprocess_crop_bbox": list(preprocessed.crop_bbox),
                "preprocess_line_pixels": preprocessed.line_pixels,
            }
        elif preprocess_mode == "thick_stroke_contours":
            preprocessed = preprocess_thick_stroke_contours(image, out / "preprocessed")
            image = preprocessed.output_path
            graph_options = replace(
                graph_options,
                input_preprocess=preprocess_mode,
                extraction_mode="black_on_white_line_art",
            )
            preprocess_meta = {
                "source_image": str(source_image),
                "preprocessed_image": str(preprocessed.output_path),
                "preprocess_crop_bbox": list(preprocessed.crop_bbox),
                "preprocess_line_pixels": preprocessed.line_pixels,
            }
        else:
            graph_options = replace(graph_options, input_preprocess="none")
        graph, router = build_review_graph_bundle(image, graph_options)
        graph.update(preprocess_meta)
        corrections_path = out / f"{image.stem}.topology_corrections.json"
        corrections: list[dict[str, Any]] = []
        design_curves: list[dict[str, Any]] = []
        if corrections_path.exists():
            try:
                data = json.loads(corrections_path.read_text(encoding="utf-8"))
                corrections = list(data.get("corrections", []))
                design_curves = list(data.get("design_curves", []))
            except Exception:
                corrections = []
                design_curves = []
        return cls(image, out, graph, router, corrections_path, corrections, design_curves)

    def state(self) -> dict[str, Any]:
        return {
            "graph": self.graph,
            "corrections": self.corrections,
            "design_curves": self.design_curves,
            "corrections_path": str(self.corrections_path),
            "saved_count": len(self.corrections) + len(self.design_curves),
        }

    def save(
        self,
        corrections: list[dict[str, Any]],
        design_curves: list[dict[str, Any]] | None = None,
    ) -> None:
        self.corrections = corrections
        if design_curves is not None:
            self.design_curves = design_curves
        payload = {
            "version": 1,
            "task": "autoalias_topology_correction",
            "created_or_updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "graph": graph_snapshot_for_training(self.graph),
            "corrections": self.corrections,
            "design_curves": self.design_curves,
        }
        self.corrections_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def route_points(self, points: list[Any], closed: bool = False) -> dict[str, Any]:
        clean = [_coerce_point(item) for item in points]
        clean = [item for item in clean if item is not None]
        if len(clean) < 2:
            return {"ok": False, "reason": "need at least two points", "segments": [], "points": []}
        segments: list[dict[str, Any]] = []
        combined: list[list[float]] = []
        all_ok = True
        pairs = list(zip(clean, clean[1:]))
        is_closed = bool(closed and len(clean) >= 3)
        if is_closed:
            pairs.append((clean[-1], clean[0]))
        for start, end in pairs:
            segment = self.router.route(start, end)
            segments.append(segment)
            all_ok = bool(segment.get("ok")) and all_ok
            segment_points = segment.get("points") or [list(start), list(end)]
            if combined and segment_points:
                combined.extend(segment_points[1:])
            else:
                combined.extend(segment_points)
        return {
            "ok": all_ok,
            "closed": is_closed,
            "segments": segments,
            "points": combined,
            "point_count": len(combined),
        }


def _coerce_point(item: Any) -> tuple[float, float] | None:
    if isinstance(item, dict):
        if "x" in item and "y" in item:
            return (float(item["x"]), float(item["y"]))
        if "point" in item:
            return _coerce_point(item["point"])
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return (float(item[0]), float(item[1]))
    return None


def _clean_input_preprocess(value: str | None) -> str:
    clean = str(value or "none").strip().lower()
    return clean if clean in {"none", "raw_feature_lines", "thick_stroke_contours"} else "none"


def run_review_app(
    image_path: str | Path,
    output_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> str:
    session = ReviewSession.create(image_path, output_dir)
    port = _find_available_port(host, port)
    handler = _make_handler(session)
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"AutoAlias review tool: {url}")
    print(f"Corrections file: {session.corrections_path}")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return url


def _find_available_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port found near {preferred}")


def _make_handler(session: ReviewSession):
    class ReviewHandler(BaseHTTPRequestHandler):
        server_version = "AutoAliasReview/0.1"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_text(_html(), "text/html; charset=utf-8")
            elif parsed.path == "/api/state":
                self._send_json(session.state())
            elif parsed.path == "/api/export":
                self._send_json(
                    {
                        "version": 1,
                        "task": "autoalias_topology_correction",
                        "graph": graph_snapshot_for_training(session.graph),
                        "corrections": session.corrections,
                        "design_curves": session.design_curves,
                    }
                )
            elif parsed.path == "/image":
                self._send_file(session.image_path)
            else:
                self.send_error(404, "not found")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/route":
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw.decode("utf-8"))
                    points = payload.get("points", [])
                    if not isinstance(points, list):
                        raise ValueError("points must be a list")
                    closed = bool(payload.get("closed", False))
                    self._send_json(session.route_points(points, closed=closed))
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            if parsed.path != "/api/corrections":
                self.send_error(404, "not found")
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
                corrections = payload.get("corrections", [])
                design_curves = payload.get("design_curves", None)
                if not isinstance(corrections, list):
                    raise ValueError("corrections must be a list")
                if design_curves is not None and not isinstance(design_curves, list):
                    raise ValueError("design_curves must be a list")
                session.save(corrections, design_curves)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            self._send_json(
                {
                    "ok": True,
                    "saved_count": len(session.corrections) + len(session.design_curves),
                    "design_curve_count": len(session.design_curves),
                    "corrections_path": str(session.corrections_path),
                }
            )

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_text(self, text: str, content_type: str, status: int = 200) -> None:
            data = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_file(self, path: Path) -> None:
            data = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ReviewHandler


def _html() -> str:
    return r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AutoAlias 分线纠错</title>
<style>
:root {
  color-scheme: light;
  --bg: #f4f5f2;
  --panel: #ffffff;
  --ink: #202323;
  --muted: #6c7370;
  --line: #d6dbd8;
  --blue: #0b6dff;
  --orange: #f47b20;
  --green: #188a5a;
  --red: #cc3c3c;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  overflow: hidden;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
}
.app {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 320px;
  height: 100vh;
}
.stage {
  position: relative;
  min-width: 0;
}
canvas {
  display: block;
  width: 100%;
  height: 100%;
  cursor: crosshair;
  background: #fff;
}
.panel {
  border-left: 1px solid var(--line);
  background: var(--panel);
  padding: 14px;
  overflow: auto;
}
.title {
  font-size: 18px;
  font-weight: 650;
  margin-bottom: 10px;
}
.metric {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 8px;
  padding: 7px 0;
  border-bottom: 1px solid #edf0ee;
}
.metric span:first-child { color: var(--muted); }
.section {
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid var(--line);
}
.section h2 {
  margin: 0 0 10px;
  font-size: 13px;
  letter-spacing: 0;
  color: var(--muted);
  font-weight: 650;
}
.buttons {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
button {
  min-height: 34px;
  border: 1px solid #cbd2ce;
  background: #f9faf8;
  color: var(--ink);
  border-radius: 6px;
  font: inherit;
  cursor: pointer;
}
button:hover { border-color: #8fa09a; background: #fff; }
button.primary { border-color: #7a9fe0; background: #eaf2ff; color: #0b4cad; }
button.good { border-color: #8fbea9; background: #eef9f3; color: #126d47; }
button.warn { border-color: #e1b179; background: #fff4e8; color: #8a4a0c; }
button.bad { border-color: #df9b9b; background: #fff0f0; color: #a52424; }
button:disabled { opacity: 0.45; cursor: default; }
.selected {
  min-height: 46px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fbfcfb;
  padding: 8px;
  color: var(--muted);
  word-break: break-all;
}
.list {
  display: grid;
  gap: 8px;
}
.item {
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
  background: #fbfcfb;
}
.item.active {
  border-color: var(--orange);
  box-shadow: 0 0 0 2px rgba(244,123,32,0.14);
}
.item .kind {
  font-weight: 650;
}
.item .meta {
  color: var(--muted);
  font-size: 12px;
  margin-top: 3px;
  word-break: break-all;
}
.path {
  color: var(--muted);
  word-break: break-all;
  font-size: 12px;
}
.floating {
  position: absolute;
  left: 12px;
  top: 12px;
  background: rgba(255,255,255,0.92);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px 10px;
  color: var(--muted);
  pointer-events: none;
}
</style>
</head>
<body>
<div class="app">
  <div class="stage">
    <canvas id="canvas"></canvas>
    <div class="floating" id="hoverLabel">加载中</div>
  </div>
  <aside class="panel">
    <div class="title">AutoAlias 分线纠错</div>
    <div class="metric"><span>线段</span><strong id="edgeCount">0</strong></div>
    <div class="metric"><span>节点</span><strong id="nodeCount">0</strong></div>
    <div class="metric"><span>覆盖</span><strong id="coverageCount">0%</strong></div>
    <div class="metric"><span>标注</span><strong id="corrCount">0</strong></div>

    <div class="section">
      <h2>已选线段</h2>
      <div class="selected" id="selectedBox">未选择</div>
    </div>

    <div class="section">
      <h2>设计曲线</h2>
      <div class="metric"><span>切割点</span><strong id="cutCount">0</strong></div>
      <div class="metric"><span>设计曲线</span><strong id="curveCount">0</strong></div>
      <div class="buttons">
        <button class="primary" id="btnSaveCurve">保存设计曲线</button>
        <button class="good" id="btnSaveNextCurve">保存并下一条</button>
        <button id="btnManualMode">手动分段</button>
        <button id="btnShowEdges">显示自动边</button>
      </div>
      <div class="buttons" style="margin-top:8px">
        <button id="btnUndoPoint">撤回上一点</button>
        <button id="btnDeletePoint">删除选中点</button>
        <button id="btnCloseCurve">闭合曲线</button>
        <button id="btnClearCuts">清空切点</button>
      </div>
      <div class="buttons" style="margin-top:8px">
        <button id="btnCutStart">起点</button>
        <button id="btnCutPoint">切分点</button>
        <button class="warn" id="btnCutBlend">圆滑倒角</button>
        <button id="btnCutBreak">硬角/断开</button>
      </div>
      <div class="buttons" style="margin-top:8px">
        <button id="btnCurveOuter">外轮廓</button>
        <button id="btnCurveDoor">门洞/车窗</button>
        <button id="btnCurveWheel">轮拱</button>
        <button id="btnCurveDetail">细节线</button>
      </div>
      <div class="selected" id="curveBox" style="margin-top:8px">手动分段模式：直接在图上点分段点，系统只负责吸附到附近线稿。</div>
    </div>

    <div class="section">
      <h2>关系</h2>
      <div class="buttons">
        <button class="good" id="btnConnect">同一条线</button>
        <button class="warn" id="btnBlend">圆滑倒角</button>
        <button id="btnBreak">断开</button>
        <button class="bad" id="btnReject">不要连接</button>
      </div>
    </div>

    <div class="section">
      <h2>语义</h2>
      <div class="buttons">
        <button id="btnOuter">外轮廓</button>
        <button id="btnDoor">门洞</button>
        <button id="btnWheel">轮拱</button>
        <button id="btnDetail">细节线</button>
      </div>
    </div>

    <div class="section">
      <h2>操作</h2>
      <div class="buttons">
        <button class="primary" id="btnSave">保存</button>
        <button id="btnClear">清空选择</button>
        <button id="btnReset">重置视图</button>
        <button class="bad" id="btnDelete">删除标注</button>
      </div>
    </div>

    <div class="section">
      <h2>标注列表</h2>
      <div class="list" id="corrList"></div>
    </div>

    <div class="section">
      <h2>设计曲线列表</h2>
      <div class="list" id="curveList"></div>
    </div>

    <div class="section">
      <h2>文件</h2>
      <div class="path" id="pathBox"></div>
    </div>
  </aside>
</div>

<script>
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const img = new Image();
let state = null;
let edges = [];
let nodes = [];
let coverageFragments = [];
let edgeMap = new Map();
let nodeMap = new Map();
let corrections = [];
let designCurves = [];
let selectedEdges = [];
let cutPoints = [];
let selectedCutIndex = null;
let closedCurve = false;
let routePreview = null;
let routeStatus = "";
let routeRequestId = 0;
let activeCorrection = null;
let activeDesignCurve = null;
let hoverEdge = null;
let hoverNode = null;
let transform = { scale: 1, x: 0, y: 0 };
let dragging = false;
let lastMouse = null;
let cutMode = true;
let manualMode = true;
let showAutoEdges = false;
let cutRole = "cut";
let designSemantic = "detail_line";

const typeName = {
  connect: "同一条线",
  blend: "圆滑倒角",
  break: "断开",
  reject: "不要连接",
  semantic: "语义"
};
const semanticName = {
  outer_profile: "外轮廓",
  door_opening: "门洞",
  wheel_arch: "轮拱",
  detail_line: "细节线"
};

async function boot() {
  const res = await fetch("/api/state");
  state = await res.json();
  edges = state.graph.edges;
  nodes = state.graph.nodes;
  coverageFragments = state.graph.coverage_fragments || [];
  corrections = state.corrections || [];
  designCurves = state.design_curves || [];
  edgeMap = new Map(edges.map(e => [e.id, e]));
  nodeMap = new Map(nodes.map(n => [n.id, n]));
  img.onload = () => {
    resize();
    resetView();
    render();
  };
  img.src = "/image";
  document.getElementById("edgeCount").textContent = edges.length;
  document.getElementById("nodeCount").textContent = nodes.length;
  const coverage = state.graph.coverage ? state.graph.coverage.coverage_ratio : 0;
  document.getElementById("coverageCount").textContent = Math.round(coverage * 1000) / 10 + "%";
  document.getElementById("pathBox").textContent = state.corrections_path;
  updatePanel();
}

function resize() {
  const rect = canvas.parentElement.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(300, Math.floor(rect.width * ratio));
  canvas.height = Math.max(300, Math.floor(rect.height * ratio));
  canvas.style.width = rect.width + "px";
  canvas.style.height = rect.height + "px";
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function resetView() {
  const rect = canvas.getBoundingClientRect();
  const s = Math.min(rect.width / img.width, rect.height / img.height) * 0.96;
  transform.scale = s;
  transform.x = (rect.width - img.width * s) * 0.5;
  transform.y = (rect.height - img.height * s) * 0.5;
  render();
}

function worldToScreen(p) {
  return [p[0] * transform.scale + transform.x, p[1] * transform.scale + transform.y];
}

function screenToWorld(x, y) {
  return [(x - transform.x) / transform.scale, (y - transform.y) / transform.scale];
}

function render() {
  if (!state || !img.complete) return;
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.save();
  ctx.translate(transform.x, transform.y);
  ctx.scale(transform.scale, transform.scale);
  ctx.drawImage(img, 0, 0);
  if (!manualMode || showAutoEdges) {
    drawCoverageFragments();
    drawEdges();
    drawNodes();
  }
  drawCutPoints();
  ctx.restore();
}

function drawCoverageFragments() {
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(11,109,255,0.42)";
  ctx.fillStyle = "rgba(11,109,255,0.42)";
  ctx.lineWidth = 3.2 / transform.scale;
  const r = Math.max(1.7 / transform.scale, 0.9);
  for (const frag of coverageFragments) {
    if (!frag.length) continue;
    if (frag.length === 1) {
      ctx.beginPath();
      ctx.arc(frag[0][0], frag[0][1], r, 0, Math.PI * 2);
      ctx.fill();
      continue;
    }
    ctx.beginPath();
    frag.forEach((p, i) => {
      if (i === 0) ctx.moveTo(p[0], p[1]);
      else ctx.lineTo(p[0], p[1]);
    });
    ctx.stroke();
  }
}

function drawEdges() {
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const edge of edges) {
    const isSel = selectedEdges.includes(edge.id);
    const isHover = hoverEdge && hoverEdge.id === edge.id;
    ctx.beginPath();
    edge.points.forEach((p, i) => {
      if (i === 0) ctx.moveTo(p[0], p[1]);
      else ctx.lineTo(p[0], p[1]);
    });
    ctx.strokeStyle = isSel ? "#f47b20" : isHover ? "#00a4a8" : (manualMode ? "rgba(11,109,255,0.28)" : "rgba(11,109,255,0.82)");
    ctx.lineWidth = (isSel || isHover) ? 5.2 / transform.scale : (manualMode ? 2.0 / transform.scale : 4.0 / transform.scale);
    ctx.stroke();
  }
}

function drawNodes() {
  for (const node of nodes) {
    if (node.degree < 2) continue;
    const r = Math.max(3.5 / transform.scale, 1.6);
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
    ctx.fillStyle = hoverNode && hoverNode.id === node.id ? "#f47b20" : "rgba(204,60,60,0.78)";
    ctx.fill();
  }
}

function drawCutPoints() {
  const roleColor = {
    start: "#188a5a",
    end: "#188a5a",
    cut: "#7b61ff",
    blend: "#f47b20",
    break: "#cc3c3c"
  };
  if (routePreview && routePreview.points && routePreview.points.length >= 2) {
    drawSmoothPolyline(routePreview.points, "#0b6dff", 4.2 / transform.scale);
    drawSmoothPolyline(routePreview.points, "rgba(255,255,255,0.65)", 1.2 / transform.scale);
  } else if (cutPoints.length >= 2) {
    ctx.beginPath();
    cutPoints.forEach((p, i) => {
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    if (closedCurve && cutPoints.length >= 3) {
      ctx.lineTo(cutPoints[0].x, cutPoints[0].y);
    }
    ctx.strokeStyle = "rgba(244,123,32,0.82)";
    ctx.lineWidth = 3.0 / transform.scale;
    ctx.setLineDash([10 / transform.scale, 5 / transform.scale]);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  for (let i = 0; i < cutPoints.length; i++) {
    const p = cutPoints[i];
    const r = Math.max(6.5 / transform.scale, 2.8);
    const isSelected = selectedCutIndex === i;
    if (isSelected) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, r * 1.65, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,210,80,0.72)";
      ctx.fill();
    }
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = roleColor[p.role] || "#7b61ff";
    ctx.fill();
    ctx.lineWidth = 1.8 / transform.scale;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
    ctx.fillStyle = "#202323";
    ctx.font = `${Math.max(10 / transform.scale, 5)}px Segoe UI`;
    ctx.fillText(String(i + 1), p.x + r * 1.2, p.y - r * 1.2);
  }
}

function drawSmoothPolyline(points, color, width) {
  if (!points || points.length < 2) return;
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  if (points.length === 2) {
    ctx.lineTo(points[1][0], points[1][1]);
  } else {
    for (let i = 1; i < points.length - 1; i++) {
      const current = points[i];
      const next = points[i + 1];
      const midX = (current[0] + next[0]) * 0.5;
      const midY = (current[1] + next[1]) * 0.5;
      ctx.quadraticCurveTo(current[0], current[1], midX, midY);
    }
    const last = points[points.length - 1];
    ctx.lineTo(last[0], last[1]);
  }
  ctx.stroke();
  ctx.restore();
}

function pickEdge(wx, wy) {
  const threshold = Math.max(7 / transform.scale, 2.2);
  let best = null;
  for (const edge of edges) {
    const box = edge.bbox;
    if (wx < box.x - threshold || wy < box.y - threshold ||
        wx > box.x + box.width + threshold || wy > box.y + box.height + threshold) continue;
    const d = pointPolylineDistance(wx, wy, edge.points);
    if (d < threshold && (!best || d < best.d)) best = { edge, d };
  }
  return best ? best.edge : null;
}

function pickNode(wx, wy) {
  const threshold = Math.max(10 / transform.scale, 3.0);
  let best = null;
  for (const node of nodes) {
    const d = Math.hypot(wx - node.x, wy - node.y);
    if (d < threshold && (!best || d < best.d)) best = { node, d };
  }
  return best ? best.node : null;
}

function pointPolylineDistance(x, y, points) {
  let best = Infinity;
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i], b = points[i + 1];
    const vx = b[0] - a[0], vy = b[1] - a[1];
    const wx = x - a[0], wy = y - a[1];
    const len2 = vx * vx + vy * vy || 1;
    const t = Math.max(0, Math.min(1, (wx * vx + wy * vy) / len2));
    const px = a[0] + t * vx, py = a[1] + t * vy;
    best = Math.min(best, Math.hypot(x - px, y - py));
  }
  return best;
}

function nearestPointOnEdge(wx, wy, edge) {
  let best = null;
  for (let i = 0; i < edge.points.length - 1; i++) {
    const a = edge.points[i], b = edge.points[i + 1];
    const vx = b[0] - a[0], vy = b[1] - a[1];
    const len2 = vx * vx + vy * vy || 1;
    const t = Math.max(0, Math.min(1, ((wx - a[0]) * vx + (wy - a[1]) * vy) / len2));
    const x = a[0] + t * vx, y = a[1] + t * vy;
    const d = Math.hypot(wx - x, wy - y);
    if (!best || d < best.d) best = { x, y, d, segment_index: i, segment_t: t };
  }
  return best;
}

function pickCutPoint(wx, wy) {
  const threshold = Math.max(10 / transform.scale, 3.6);
  let best = null;
  for (let i = 0; i < cutPoints.length; i++) {
    const p = cutPoints[i];
    const d = Math.hypot(wx - p.x, wy - p.y);
    if (d <= threshold && (!best || d < best.d)) best = { index: i, d };
  }
  return best ? best.index : null;
}

function addCutPoint(wx, wy) {
  const pool = (!manualMode && selectedEdges.length) ? selectedEdges.map(id => edgeMap.get(id)).filter(Boolean) : edges;
  let best = null;
  for (const edge of pool) {
    const p = nearestPointOnEdge(wx, wy, edge);
    if (p && (!best || p.d < best.d)) best = { ...p, edge_id: edge.id };
  }
  const snapThreshold = Math.max(18 / transform.scale, 4.0);
  if (!best || best.d > snapThreshold) {
    best = { x: wx, y: wy, d: Infinity, edge_id: null, segment_index: null, segment_t: null };
  }
  if (!best) return;
  cutPoints.push({
    x: Math.round(best.x * 1000) / 1000,
    y: Math.round(best.y * 1000) / 1000,
    edge_id: best.edge_id,
    role: cutRole,
    segment_index: best.segment_index,
    segment_t: best.segment_t == null ? null : Math.round(best.segment_t * 1000) / 1000
  });
  selectedCutIndex = cutPoints.length - 1;
  updatePanel();
  refreshRoutePreview();
}

function clearRoutePreview() {
  routePreview = null;
  routeStatus = "";
  routeRequestId++;
}

function clearCurrentCurve() {
  selectedEdges = [];
  cutPoints = [];
  selectedCutIndex = null;
  closedCurve = false;
  activeDesignCurve = null;
  clearRoutePreview();
  updatePanel();
  render();
}

function undoCutPoint() {
  if (!cutPoints.length) return;
  cutPoints.pop();
  selectedCutIndex = cutPoints.length ? cutPoints.length - 1 : null;
  if (cutPoints.length < 3) closedCurve = false;
  clearRoutePreview();
  refreshRoutePreview();
}

function deleteSelectedCutPoint() {
  if (selectedCutIndex == null || selectedCutIndex < 0 || selectedCutIndex >= cutPoints.length) return;
  cutPoints.splice(selectedCutIndex, 1);
  selectedCutIndex = cutPoints.length ? Math.min(selectedCutIndex, cutPoints.length - 1) : null;
  if (cutPoints.length < 3) closedCurve = false;
  clearRoutePreview();
  refreshRoutePreview();
}

function toggleClosedCurve() {
  if (cutPoints.length < 3) return;
  closedCurve = !closedCurve;
  clearRoutePreview();
  refreshRoutePreview();
}

async function refreshRoutePreview() {
  const requestId = ++routeRequestId;
  if (cutPoints.length < 2) {
    routePreview = null;
    routeStatus = "";
    updatePanel();
    render();
    return;
  }
  routeStatus = "正在沿骨架生成曲线...";
  updatePanel();
  render();
  try {
    const res = await fetch("/api/route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        points: cutPoints.map(p => ({ x: p.x, y: p.y })),
        closed: closedCurve
      })
    });
    const result = await res.json();
    if (requestId !== routeRequestId) return;
    routePreview = result;
    routeStatus = result.ok
      ? `已生成${result.closed ? "闭合" : ""}骨架曲线：${result.point_count || 0} 个骨架点`
      : `骨架没有连通：请在目标线中间再加一个引导点`;
  } catch (_err) {
    if (requestId !== routeRequestId) return;
    routePreview = null;
    routeStatus = "骨架路径生成失败";
  }
  updatePanel();
  render();
}

function toggleEdge(edge) {
  if (!edge) return;
  activeCorrection = null;
  activeDesignCurve = null;
  if (selectedEdges.includes(edge.id)) {
    selectedEdges = selectedEdges.filter(id => id !== edge.id);
  } else {
    selectedEdges.push(edge.id);
  }
  updatePanel();
  render();
}

function updatePanel() {
  document.getElementById("corrCount").textContent = corrections.length;
  document.getElementById("cutCount").textContent = cutPoints.length;
  document.getElementById("curveCount").textContent = designCurves.length;
  const box = document.getElementById("selectedBox");
  box.textContent = selectedEdges.length ? `${selectedEdges.length} 条：` + selectedEdges.join(" + ") : "未选择";
  const curveBox = document.getElementById("curveBox");
  const routeText = routeStatus ? `；${routeStatus}` : "";
  const pointText = selectedCutIndex == null ? "" : `；选中第 ${selectedCutIndex + 1} 点`;
  const closedText = closedCurve ? "；闭合" : "";
  curveBox.textContent = manualMode
    ? `手动分段：当前语义 ${semanticName[designSemantic]}；点角色 ${cutRoleName(cutRole)}；已点 ${cutPoints.length} 个点${closedText}${pointText}${routeText}。`
    : `半自动：当前语义 ${semanticName[designSemantic]}；点角色 ${cutRoleName(cutRole)}；${cutMode ? "正在点切割点" : "正在选择线段"}${closedText}${pointText}${routeText}`;
  const two = selectedEdges.length === 2;
  document.getElementById("btnConnect").disabled = !two;
  document.getElementById("btnBlend").disabled = !two;
  document.getElementById("btnBreak").disabled = !two;
  document.getElementById("btnReject").disabled = !two;
  const one = selectedEdges.length >= 1;
  document.getElementById("btnOuter").disabled = !one;
  document.getElementById("btnDoor").disabled = !one;
  document.getElementById("btnWheel").disabled = !one;
  document.getElementById("btnDetail").disabled = !one;
  const canSaveCurve = selectedEdges.length > 0 || cutPoints.length >= 2;
  document.getElementById("btnSaveCurve").disabled = !canSaveCurve;
  document.getElementById("btnSaveNextCurve").disabled = !canSaveCurve;
  document.getElementById("btnUndoPoint").disabled = cutPoints.length === 0;
  document.getElementById("btnDeletePoint").disabled = selectedCutIndex == null;
  document.getElementById("btnCloseCurve").disabled = cutPoints.length < 3;
  document.getElementById("btnCloseCurve").classList.toggle("primary", closedCurve);
  document.getElementById("btnCloseCurve").textContent = closedCurve ? "闭合中" : "闭合曲线";
  document.getElementById("btnManualMode").classList.toggle("primary", manualMode);
  document.getElementById("btnManualMode").textContent = manualMode ? "手动分段中" : "手动分段";
  document.getElementById("btnShowEdges").textContent = showAutoEdges ? "隐藏自动边" : "显示自动边";
  renderCorrections();
  renderDesignCurves();
}

function cutRoleName(role) {
  return {
    start: "起点",
    end: "终点",
    cut: "切分点",
    blend: "圆滑倒角",
    break: "硬角/断开"
  }[role] || role;
}

function renderCorrections() {
  const list = document.getElementById("corrList");
  list.innerHTML = "";
  for (const corr of corrections.slice().reverse()) {
    const item = document.createElement("div");
    item.className = "item" + (activeCorrection === corr.id ? " active" : "");
    const kind = document.createElement("div");
    kind.className = "kind";
    kind.textContent = corr.type === "semantic" ? semanticName[corr.semantic] : typeName[corr.type];
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = corr.edge_ids.join(" + ") + (corr.node_id ? " @ " + corr.node_id : "");
    item.append(kind, meta);
    item.onclick = () => {
      activeCorrection = corr.id;
      selectedEdges = corr.edge_ids.slice(0, 2);
      updatePanel();
      render();
    };
    list.appendChild(item);
  }
}

function renderDesignCurves() {
  const list = document.getElementById("curveList");
  list.innerHTML = "";
  for (const curve of designCurves.slice().reverse()) {
    const item = document.createElement("div");
    item.className = "item" + (activeDesignCurve === curve.id ? " active" : "");
    const kind = document.createElement("div");
    kind.className = "kind";
    kind.textContent = (curve.type === "manual_design_curve" ? "手动 " : "") + (semanticName[curve.semantic] || curve.semantic || "设计曲线");
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${(curve.edge_ids || []).length} 条自动线，${(curve.manual_points || curve.cut_points || []).length} 个手动点${curve.closed ? "，闭合" : ""}`;
    item.append(kind, meta);
    item.onclick = () => {
      activeDesignCurve = curve.id;
      activeCorrection = null;
      selectedEdges = (curve.edge_ids || []).slice();
      cutPoints = (curve.manual_points || curve.cut_points || []).map(p => ({ ...p }));
      selectedCutIndex = null;
      closedCurve = !!curve.closed;
      routePreview = curve.routed_points && curve.routed_points.length >= 2
        ? { ok: !!curve.route_ok, points: curve.routed_points, segments: curve.route_segments || [] }
        : null;
      routeStatus = routePreview ? `已加载骨架曲线：${routePreview.points.length} 个骨架点` : "";
      designSemantic = curve.semantic || "detail_line";
      manualMode = curve.type === "manual_design_curve" || selectedEdges.length === 0;
      cutMode = true;
      updatePanel();
      render();
      refreshRoutePreview();
    };
    list.appendChild(item);
  }
}

function addRelation(type) {
  if (selectedEdges.length !== 2) return;
  corrections.push({
    id: makeId(),
    type,
    edge_ids: selectedEdges.slice(0, 2),
    node_id: commonNode(selectedEdges[0], selectedEdges[1]),
    blend_style: type === "blend" ? "fair_large_radius" : undefined,
    created_at: new Date().toISOString()
  });
  saveCorrections();
}

function addSemantic(semantic) {
  if (!selectedEdges.length) return;
  corrections.push({
    id: makeId(),
    type: "semantic",
    semantic,
    edge_ids: selectedEdges.slice(0, 1),
    created_at: new Date().toISOString()
  });
  saveCorrections();
}

async function saveDesignCurve(startNext = false) {
  if (!(selectedEdges.length > 0 || cutPoints.length >= 2)) return;
  if (cutPoints.length >= 2 && (!routePreview || !routePreview.points || routePreview.points.length < 2)) {
    await refreshRoutePreview();
  }
  const isManual = manualMode || selectedEdges.length === 0;
  const item = {
    id: activeDesignCurve || makeId().replace("corr_", "curve_"),
    type: isManual ? "manual_design_curve" : "design_curve",
    semantic: designSemantic,
    edge_ids: selectedEdges.slice(),
    manual_points: cutPoints.map((p, i) => ({ ...p, order: i })),
    cut_points: cutPoints.map((p, i) => ({ ...p, order: i })),
    closed: closedCurve,
    routed_points: routePreview && routePreview.points ? routePreview.points : [],
    route_segments: routePreview && routePreview.segments ? routePreview.segments : [],
    route_ok: routePreview ? !!routePreview.ok : false,
    created_at: new Date().toISOString()
  };
  const idx = designCurves.findIndex(c => c.id === item.id);
  if (idx >= 0) designCurves[idx] = item;
  else designCurves.push(item);
  activeDesignCurve = item.id;
  await saveCorrections();
  if (startNext) clearCurrentCurve();
}

function setCutRole(role) {
  cutRole = role;
  updatePanel();
}

function setDesignSemantic(semantic) {
  designSemantic = semantic;
  updatePanel();
}

function commonNode(aId, bId) {
  const a = edgeMap.get(aId), b = edgeMap.get(bId);
  if (!a || !b) return null;
  const aNodes = [a.start_node, a.end_node];
  const bNodes = [b.start_node, b.end_node];
  for (const node of aNodes) if (bNodes.includes(node)) return node;
  return nearestEndpointNode(a, b);
}

function nearestEndpointNode(a, b) {
  const pairs = [
    [a.start_node, b.start_node],
    [a.start_node, b.end_node],
    [a.end_node, b.start_node],
    [a.end_node, b.end_node]
  ];
  let best = null;
  for (const [na, nb] of pairs) {
    const pa = nodeMap.get(na), pb = nodeMap.get(nb);
    if (!pa || !pb) continue;
    const d = Math.hypot(pa.x - pb.x, pa.y - pb.y);
    if (!best || d < best.d) best = { d, node: d < 18 ? na + "|" + nb : null };
  }
  return best ? best.node : null;
}

function makeId() {
  return "corr_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 7);
}

async function saveCorrections() {
  const res = await fetch("/api/corrections", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ corrections, design_curves: designCurves })
  });
  const result = await res.json();
  if (result.ok) document.getElementById("pathBox").textContent = result.corrections_path;
  updatePanel();
}

function deleteActive() {
  if (activeDesignCurve) {
    designCurves = designCurves.filter(c => c.id !== activeDesignCurve);
    activeDesignCurve = null;
  } else if (activeCorrection) {
    corrections = corrections.filter(c => c.id !== activeCorrection);
    activeCorrection = null;
  } else {
    return;
  }
  saveCorrections();
}

canvas.addEventListener("mousedown", e => {
  if (e.button === 1 || e.altKey) {
    dragging = true;
    lastMouse = [e.clientX, e.clientY];
    e.preventDefault();
  }
});
canvas.addEventListener("mousemove", e => {
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
  if (dragging && lastMouse) {
    transform.x += e.clientX - lastMouse[0];
    transform.y += e.clientY - lastMouse[1];
    lastMouse = [e.clientX, e.clientY];
    render();
    return;
  }
  const [wx, wy] = screenToWorld(sx, sy);
  hoverEdge = pickEdge(wx, wy);
  hoverNode = pickNode(wx, wy);
  document.getElementById("hoverLabel").textContent = hoverEdge ? hoverEdge.id + "  " + hoverEdge.label : " ";
  render();
});
window.addEventListener("mouseup", () => { dragging = false; lastMouse = null; });
canvas.addEventListener("click", e => {
  if (e.altKey) return;
  const rect = canvas.getBoundingClientRect();
  const [wx, wy] = screenToWorld(e.clientX - rect.left, e.clientY - rect.top);
  if (manualMode || cutMode) {
    const hitCut = pickCutPoint(wx, wy);
    if (hitCut != null) {
      selectedCutIndex = hitCut;
      updatePanel();
      render();
      return;
    }
    addCutPoint(wx, wy);
  } else {
    toggleEdge(pickEdge(wx, wy));
  }
});
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
  const before = screenToWorld(sx, sy);
  const factor = e.deltaY < 0 ? 1.12 : 0.89;
  transform.scale = Math.max(0.05, Math.min(20, transform.scale * factor));
  transform.x = sx - before[0] * transform.scale;
  transform.y = sy - before[1] * transform.scale;
  render();
}, { passive: false });

document.getElementById("btnConnect").onclick = () => addRelation("connect");
document.getElementById("btnBlend").onclick = () => addRelation("blend");
document.getElementById("btnBreak").onclick = () => addRelation("break");
document.getElementById("btnReject").onclick = () => addRelation("reject");
document.getElementById("btnOuter").onclick = () => addSemantic("outer_profile");
document.getElementById("btnDoor").onclick = () => addSemantic("door_opening");
document.getElementById("btnWheel").onclick = () => addSemantic("wheel_arch");
document.getElementById("btnDetail").onclick = () => addSemantic("detail_line");
document.getElementById("btnSaveCurve").onclick = () => saveDesignCurve(false);
document.getElementById("btnSaveNextCurve").onclick = () => saveDesignCurve(true);
document.getElementById("btnManualMode").onclick = () => { manualMode = !manualMode; cutMode = manualMode || cutMode; updatePanel(); render(); };
document.getElementById("btnShowEdges").onclick = () => { showAutoEdges = !showAutoEdges; updatePanel(); render(); };
document.getElementById("btnUndoPoint").onclick = undoCutPoint;
document.getElementById("btnDeletePoint").onclick = deleteSelectedCutPoint;
document.getElementById("btnCloseCurve").onclick = toggleClosedCurve;
document.getElementById("btnClearCuts").onclick = () => { cutPoints = []; selectedCutIndex = null; closedCurve = false; clearRoutePreview(); updatePanel(); render(); };
document.getElementById("btnCutStart").onclick = () => setCutRole("start");
document.getElementById("btnCutPoint").onclick = () => setCutRole("cut");
document.getElementById("btnCutBlend").onclick = () => setCutRole("blend");
document.getElementById("btnCutBreak").onclick = () => setCutRole("break");
document.getElementById("btnCurveOuter").onclick = () => setDesignSemantic("outer_profile");
document.getElementById("btnCurveDoor").onclick = () => setDesignSemantic("door_opening");
document.getElementById("btnCurveWheel").onclick = () => setDesignSemantic("wheel_arch");
document.getElementById("btnCurveDetail").onclick = () => setDesignSemantic("detail_line");
document.getElementById("btnSave").onclick = saveCorrections;
document.getElementById("btnClear").onclick = () => { selectedEdges = []; cutPoints = []; selectedCutIndex = null; closedCurve = false; clearRoutePreview(); activeCorrection = null; activeDesignCurve = null; updatePanel(); render(); };
document.getElementById("btnReset").onclick = resetView;
document.getElementById("btnDelete").onclick = deleteActive;
window.addEventListener("keydown", e => {
  if (e.ctrlKey && e.key.toLowerCase() === "z") {
    e.preventDefault();
    undoCutPoint();
    return;
  }
  if (e.key === "Delete" || e.key === "Backspace") {
    if (selectedCutIndex != null) {
      e.preventDefault();
      deleteSelectedCutPoint();
    }
  }
});
window.addEventListener("resize", () => { resize(); render(); });
boot();
</script>
</body>
</html>"""
