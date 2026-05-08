from __future__ import annotations

import json
import mimetypes
import socket
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from autoalias.geometry.bezier import evaluate_bezier
from autoalias.geometry.fitting import FittingOptions, SingleSpanFitter
from autoalias.geometry.polyline import remove_duplicate_points
from autoalias.models import CurveCandidate
from autoalias.quality import ClassAValidator
from autoalias.review.fit_reviewed import fit_reviewed_annotations
from autoalias.review.server import ReviewSession


def run_skeleton_review_server(
    output_dir: str | Path,
    *,
    host: str = "0.0.0.0",
    port: int = 8765,
    open_browser: bool = True,
) -> str:
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "uploads").mkdir(parents=True, exist_ok=True)
    (out / "alias_exports").mkdir(parents=True, exist_ok=True)

    sessions: dict[str, ReviewSession] = {}
    exports: dict[str, dict[str, Path]] = {}
    actual_port = _find_available_port(host, port)
    handler = _make_handler(out, sessions, exports)
    httpd = ThreadingHTTPServer((host, actual_port), handler)

    local_url = f"http://127.0.0.1:{actual_port}/"
    bind_url = f"http://{host}:{actual_port}/"
    print(f"AutoAlias skeleton review server: {bind_url}", flush=True)
    print(f"Open locally: {local_url}", flush=True)
    for ip in _local_ipv4_addresses():
        print(f"LAN URL: http://{ip}:{actual_port}/", flush=True)
    print(f"Workspace: {out}", flush=True)

    if open_browser:
        webbrowser.open(local_url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return local_url


def _make_handler(
    output_dir: Path,
    sessions: dict[str, ReviewSession],
    exports: dict[str, dict[str, Path]],
):
    class SkeletonReviewHandler(BaseHTTPRequestHandler):
        server_version = "AutoAliasSkeletonReview/0.1"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_text(_html(), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/state":
                session = self._require_session(parsed)
                if session is None:
                    return
                payload = _client_state(session)
                payload["sid"] = _sid(parsed)
                payload["exports"] = _export_payload(exports.get(_sid(parsed), {}), _sid(parsed))
                self._send_json(payload)
                return
            if parsed.path == "/image":
                session = self._require_session(parsed)
                if session is not None:
                    self._send_file(session.image_path)
                return
            if parsed.path == "/download":
                sid = _sid(parsed)
                kind = parse_qs(parsed.query).get("kind", [""])[0]
                path = exports.get(sid, {}).get(kind)
                if path is None:
                    self.send_error(404, "export not found")
                    return
                self._send_file(path, attachment=True)
                return
            self.send_error(404, "not found")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/upload":
                self._handle_upload()
                return
            if parsed.path == "/api/route":
                session = self._require_session(parsed)
                if session is None:
                    return
                self._handle_route(session)
                return
            if parsed.path == "/api/fit-preview":
                session = self._require_session(parsed)
                if session is None:
                    return
                self._handle_fit_preview(session)
                return
            if parsed.path == "/api/snap":
                session = self._require_session(parsed)
                if session is None:
                    return
                self._handle_snap(session)
                return
            if parsed.path == "/api/corrections":
                session = self._require_session(parsed)
                if session is None:
                    return
                self._handle_save(session)
                return
            if parsed.path == "/api/export-iges":
                session = self._require_session(parsed)
                if session is None:
                    return
                self._handle_export(session, _sid(parsed))
                return
            self.send_error(404, "not found")

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _handle_upload(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                self._send_json({"ok": False, "error": "empty upload"}, status=400)
                return
            filename = self.headers.get("X-Filename", "uploaded.png")
            safe_name = _safe_filename(filename)
            target = _unique_path(output_dir / "uploads" / safe_name)
            target.write_bytes(self.rfile.read(length))
            try:
                session = ReviewSession.create(target, output_dir)
            except Exception as exc:
                try:
                    target.unlink()
                except OSError:
                    pass
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            sid = _make_session_id(target)
            sessions[sid] = session
            payload = _client_state(session)
            payload["ok"] = True
            payload["sid"] = sid
            self._send_json(payload)

        def _handle_route(self, session: ReviewSession) -> None:
            try:
                payload = self._read_json()
                points = payload.get("points", [])
                if not isinstance(points, list):
                    raise ValueError("points must be a list")
                closed = bool(payload.get("closed", False))
                branch_choices = payload.get("branch_choices", [])
                if not isinstance(branch_choices, list):
                    branch_choices = []
                self._send_json(
                    _route_points_with_choices(
                        session,
                        points,
                        closed=closed,
                        branch_choices=branch_choices,
                        candidate_count=int(payload.get("candidate_count", 3)),
                    )
                )
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)

        def _handle_fit_preview(self, session: ReviewSession) -> None:
            try:
                payload = self._read_json()
                degree = payload.get("degree", "auto")
                route_segments = payload.get("route_segments", [])
                if not isinstance(route_segments, list):
                    raise ValueError("route_segments must be a list")
                self._send_json(_fit_preview_segments(route_segments, degree=degree))
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)

        def _handle_snap(self, session: ReviewSession) -> None:
            try:
                payload = self._read_json()
                point = _coerce_xy(payload.get("point", payload))
                if point is None:
                    raise ValueError("point must contain x and y")
                max_distance = float(payload.get("max_distance", 24.0))
                index, distance = session.router.nearest_index(point)
                if distance > max_distance:
                    self._send_json(
                        {
                            "ok": False,
                            "reason": "nearest skeleton point is outside snap radius",
                            "distance": round(float(distance), 3),
                            "max_distance": round(float(max_distance), 3),
                        }
                    )
                    return
                snapped = session.router.coords[index]
                self._send_json(
                    {
                        "ok": True,
                        "x": round(float(snapped[0]), 3),
                        "y": round(float(snapped[1]), 3),
                        "distance": round(float(distance), 3),
                    }
                )
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)

        def _handle_save(self, session: ReviewSession) -> None:
            try:
                payload = self._read_json()
                corrections = payload.get("corrections", [])
                design_curves = payload.get("design_curves", [])
                if not isinstance(corrections, list):
                    raise ValueError("corrections must be a list")
                if not isinstance(design_curves, list):
                    raise ValueError("design_curves must be a list")
                session.save(corrections, design_curves)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            self._send_json(
                {
                    "ok": True,
                    "corrections_path": str(session.corrections_path),
                    "design_curve_count": len(session.design_curves),
                }
            )

        def _handle_export(self, session: ReviewSession, sid: str) -> None:
            try:
                payload = self._read_json(required=False)
                corrections = payload.get("corrections", session.corrections)
                design_curves = payload.get("design_curves", session.design_curves)
                session.save(corrections, design_curves)
                degree = payload.get("degree", "auto")
                export_dir = output_dir / "alias_exports" / session.image_path.stem
                result = fit_reviewed_annotations(
                    [session.corrections_path],
                    export_dir,
                    degree=degree,
                    min_points=8,
                )
                exports[sid] = {
                    "iges": export_dir / "reviewed_curves.igs",
                    "json": export_dir / "reviewed_curves.json",
                    "preview": export_dir / "reviewed_preview.svg",
                    "clean_preview": export_dir / "reviewed_clean_preview.svg",
                }
                passed = sum(1 for report in result.reports if report.passed)
                self._send_json(
                    {
                        "ok": True,
                        "curve_count": len(result.curves),
                        "passed_count": passed,
                        "skipped_count": result.skipped_count,
                        "out": str(export_dir),
                        "exports": _export_payload(exports[sid], sid),
                        "warnings": [
                            {
                                "label": report.label,
                                "warnings": report.warnings,
                            }
                            for report in result.reports
                            if report.warnings
                        ],
                    }
                )
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)

        def _require_session(self, parsed) -> ReviewSession | None:
            sid = _sid(parsed)
            session = sessions.get(sid)
            if session is None:
                self._send_json({"ok": False, "error": "missing or expired image session"}, status=404)
            return session

        def _read_json(self, required: bool = True) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                if required:
                    raise ValueError("missing JSON payload")
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

        def _send_json(self, payload: Any, status: int = 200) -> None:
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

        def _send_file(self, path: Path, attachment: bool = False) -> None:
            data = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.end_headers()
            self.wfile.write(data)

    return SkeletonReviewHandler


def _route_points_with_choices(
    session: ReviewSession,
    points: list[Any],
    *,
    closed: bool = False,
    branch_choices: list[Any] | None = None,
    candidate_count: int = 3,
) -> dict[str, Any]:
    clean = [_coerce_xy(item) for item in points]
    clean = [item for item in clean if item is not None]
    if len(clean) < 2:
        return {"ok": False, "reason": "need at least two points", "segments": [], "points": []}
    branch_choices = branch_choices or []
    segments: list[dict[str, Any]] = []
    combined: list[list[float]] = []
    all_ok = True
    pairs = list(zip(clean, clean[1:]))
    is_closed = bool(closed and len(clean) >= 3)
    if is_closed:
        pairs.append((clean[-1], clean[0]))
    for index, (start, end) in enumerate(pairs):
        candidates = session.router.route_candidates(
            start,
            end,
            count=max(1, min(int(candidate_count), 5)),
        )
        choice = _safe_choice(branch_choices, index, len(candidates))
        chosen = candidates[choice] if candidates else {"ok": False, "points": [list(start), list(end)]}
        segment_points = chosen.get("points") or [list(start), list(end)]
        if combined and segment_points:
            combined.extend(segment_points[1:])
        else:
            combined.extend(segment_points)
        all_ok = bool(chosen.get("ok")) and all_ok
        segments.append(
            {
                **chosen,
                "segment_index": index,
                "selected_candidate": choice,
                "alternatives": candidates,
            }
        )
    return {
        "ok": all_ok,
        "closed": is_closed,
        "segments": segments,
        "points": combined,
        "point_count": len(combined),
    }


def _fit_preview_segments(route_segments: list[dict[str, Any]], degree: int | str = "auto") -> dict[str, Any]:
    parsed_degree = _parse_preview_degree(degree)
    validator = ClassAValidator()
    previews: list[dict[str, Any]] = []
    for index, segment in enumerate(route_segments):
        raw_points = segment.get("points") or []
        points = _as_points3(raw_points)
        try:
            points = remove_duplicate_points(points, eps=0.5)
            points = _ensure_preview_points(points)
            if len(points) < 4:
                raise ValueError("not enough points")
            label = f"preview_segment_{index + 1:03d}"
            candidate = CurveCandidate(label=label, points=points, source="fit_preview")
            curve = _fit_preview_lowest_degree(candidate, points, parsed_degree, validator)
            report = validator.validate(curve, points)
            samples = evaluate_bezier(curve.cvs, np.linspace(0.0, 1.0, 120), curve.weights)
            previews.append(
                {
                    "ok": True,
                    "segment_index": index,
                    "degree": curve.degree,
                    "span": curve.span_count,
                    "samples": _round_xy(samples),
                    "cvs": _round_xy(curve.cvs),
                    "passed": report.passed,
                    "color": _quality_color(report.warnings),
                    "warnings": report.warnings,
                    "metrics": report.metrics,
                }
            )
        except Exception as exc:
            previews.append(
                {
                    "ok": False,
                    "segment_index": index,
                    "color": "#d93025",
                    "warnings": [str(exc)],
                }
            )
    passed = sum(1 for item in previews if item.get("passed"))
    return {
        "ok": True,
        "segments": previews,
        "segment_count": len(previews),
        "passed_count": passed,
    }


def _fit_preview_lowest_degree(
    candidate: CurveCandidate,
    target_points: np.ndarray,
    degree: int | str,
    validator: ClassAValidator,
):
    if isinstance(degree, int):
        return SingleSpanFitter(FittingOptions(degree=degree)).fit_candidate(candidate)
    best_curve = None
    best_score = float("inf")
    for candidate_degree in (3, 4, 5, 6, 7):
        curve = SingleSpanFitter(FittingOptions(degree=candidate_degree)).fit_candidate(candidate)
        report = validator.validate(curve, target_points)
        chamfer = float(report.metrics.get("chamfer_mean", 999.0))
        warnings = len(report.warnings)
        score = warnings * 1000.0 + chamfer + candidate_degree * 0.01
        if report.passed:
            return curve
        if score < best_score:
            best_score = score
            best_curve = curve
    if best_curve is None:
        raise ValueError("failed to fit any degree")
    return best_curve


def _safe_choice(branch_choices: list[Any], index: int, candidate_count: int) -> int:
    if candidate_count <= 0:
        return 0
    try:
        value = int(branch_choices[index])
    except Exception:
        value = 0
    return max(0, min(value, candidate_count - 1))


def _parse_preview_degree(value: Any) -> int | str:
    if str(value) == "auto":
        return "auto"
    degree = int(value)
    if degree not in (3, 4, 5, 6, 7):
        return "auto"
    return degree


def _quality_color(warnings: list[str]) -> str:
    if not warnings:
        return "#14a05a"
    joined = " ".join(warnings).lower()
    if "turnback" in joined or "curvature" in joined or "oscillation" in joined:
        return "#d93025"
    return "#f4a000"


def _as_points3(points: Any) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        return np.zeros((0, 3), dtype=float)
    if arr.shape[1] == 2:
        arr = np.column_stack([arr, np.zeros(len(arr), dtype=float)])
    return arr


def _ensure_preview_points(points: np.ndarray) -> np.ndarray:
    if len(points) >= 4:
        return points
    if len(points) < 2:
        return points
    return _line_points(points[0], points[-1])


def _line_points(a: np.ndarray, b: np.ndarray, count: int = 8) -> np.ndarray:
    u = np.linspace(0.0, 1.0, count)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pts = a * (1.0 - u[:, None]) + b * u[:, None]
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts), dtype=float)])
    return pts


def _round_xy(points: np.ndarray) -> list[list[float]]:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return []
    return [[round(float(x), 3), round(float(y), 3)] for x, y in arr[:, :2]]


def _sid(parsed) -> str:
    return parse_qs(parsed.query).get("sid", [""])[0]


def _client_state(session: ReviewSession) -> dict[str, Any]:
    payload = session.state()
    graph = dict(payload.get("graph", {}))
    graph.pop("coverage_fragments", None)
    graph["full_skeleton_points"] = _downsample_xy(session.router.coords, 12000)
    payload["graph"] = graph
    return payload


def _export_payload(paths: dict[str, Path], sid: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind, path in paths.items():
        if path.exists():
            out[kind] = {
                "path": str(path),
                "url": f"/download?sid={sid}&kind={kind}",
            }
    return out


def _find_available_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port found near {preferred}")


def _local_ipv4_addresses() -> list[str]:
    ips: set[str] = set()
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except OSError:
        pass
    return sorted(ips)


def _safe_filename(name: str) -> str:
    raw = Path(name).name or "uploaded.png"
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)
    if "." not in safe:
        safe += ".png"
    return safe


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for idx in range(1, 1000):
        candidate = path.with_name(f"{stem}_{stamp}_{idx:03d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"cannot create unique upload path near {path}")


def _make_session_id(path: Path) -> str:
    return f"{path.stem}_{int(time.time() * 1000):x}"


def _downsample_xy(points: Any, max_count: int) -> list[list[float]]:
    try:
        import numpy as np

        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) == 0:
            return []
        if len(arr) > max_count:
            idx = np.linspace(0, len(arr) - 1, max_count).round().astype(int)
            arr = arr[idx]
        return [[round(float(x), 3), round(float(y), 3)] for x, y in arr[:, :2]]
    except Exception:
        return []


def _coerce_xy(item: Any) -> tuple[float, float] | None:
    if isinstance(item, dict) and "x" in item and "y" in item:
        return (float(item["x"]), float(item["y"]))
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return (float(item[0]), float(item[1]))
    return None


def _html() -> str:
    return r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AutoAlias 骨架分段导出</title>
<style>
:root { --bg:#f5f6f3; --panel:#fff; --ink:#1f2423; --muted:#68706d; --line:#d8ddd9; --blue:#006dff; --orange:#f27b25; }
* { box-sizing:border-box; }
body { margin:0; overflow:hidden; background:var(--bg); color:var(--ink); font:14px/1.45 "Segoe UI","Microsoft YaHei",Arial,sans-serif; }
.app { display:grid; grid-template-columns:minmax(0,1fr) 340px; height:100vh; }
.stage { position:relative; min-width:0; }
canvas { display:block; width:100%; height:100%; background:#fff; cursor:crosshair; }
.panel { border-left:1px solid var(--line); background:var(--panel); padding:14px; overflow:auto; }
h1 { margin:0 0 12px; font-size:18px; }
h2 { margin:16px 0 8px; padding-top:12px; border-top:1px solid var(--line); color:var(--muted); font-size:13px; }
.row { display:grid; grid-template-columns:1fr auto; gap:8px; padding:6px 0; border-bottom:1px solid #eef1ef; }
.row span:first-child { color:var(--muted); }
button, select, input[type=file] { width:100%; min-height:34px; border:1px solid #cbd2ce; border-radius:6px; background:#fafbf9; color:var(--ink); font:inherit; }
button { cursor:pointer; }
button:hover { border-color:#8fa09a; background:#fff; }
button.primary { border-color:#7aa2e8; background:#eaf2ff; color:#0b4cad; }
button.good { border-color:#88bd9d; background:#effaf4; color:#126d47; }
button.warn { border-color:#e1b179; background:#fff4e8; color:#8a4a0c; }
button.bad { border-color:#df9b9b; background:#fff0f0; color:#a52424; }
button:disabled { opacity:.45; cursor:default; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
.stack { display:grid; gap:8px; }
.box { border:1px solid var(--line); border-radius:6px; padding:8px; background:#fbfcfb; color:var(--muted); word-break:break-all; }
.item { border:1px solid var(--line); border-radius:6px; padding:8px; background:#fbfcfb; cursor:pointer; }
.item.active { border-color:var(--orange); box-shadow:0 0 0 2px rgba(242,123,37,.14); }
.item strong { display:block; }
.item small { color:var(--muted); }
.itemHead { display:grid; grid-template-columns:1fr auto; gap:8px; align-items:start; }
.miniBad { width:auto; min-height:26px; padding:0 8px; border-color:#df9b9b; background:#fff0f0; color:#a52424; font-size:12px; }
.floating { position:absolute; left:12px; top:12px; background:rgba(255,255,255,.92); border:1px solid var(--line); border-radius:6px; padding:8px 10px; color:var(--muted); pointer-events:none; }
.hidden { display:none; }
a { color:#075fd7; }
</style>
</head>
<body>
<div class="app">
  <div class="stage">
    <canvas id="canvas"></canvas>
    <div class="floating" id="status">请先上传图片</div>
  </div>
  <aside class="panel">
    <h1>AutoAlias 骨架分段导出</h1>

    <div class="stack" id="uploadBox">
      <input id="fileInput" type="file" accept="image/*" />
      <button class="primary" id="btnUpload">上传并提取骨架</button>
    </div>

    <div id="workBox" class="hidden">
      <div class="row"><span>骨架线段</span><strong id="edgeCount">0</strong></div>
      <div class="row"><span>已保存曲线</span><strong id="curveCount">0</strong></div>
      <div class="row"><span>当前点数</span><strong id="pointCount">0</strong></div>

      <h2>当前分段</h2>
      <div class="stack">
        <select id="semantic">
          <option value="outer_profile">外轮廓</option>
          <option value="door_opening">门洞/车窗</option>
          <option value="wheel_arch">轮拱</option>
          <option value="beltline">腰线/特征线</option>
          <option value="detail_line" selected>细节线</option>
        </select>
        <div class="grid2">
          <button class="good" id="btnSaveNext">保存并下一条</button>
          <button class="primary" id="btnSave">保存当前</button>
          <button id="btnUndo">撤回一点</button>
          <button id="btnDelete">删除选中点</button>
          <button id="btnClose">闭合曲线</button>
          <button id="btnClear">清空当前</button>
        </div>
      </div>
      <div class="box" id="routeBox">在图上点击两个或多个点，蓝线会沿骨架自动生成。</div>
      <div class="grid2" style="margin-top:8px">
        <select id="snapRadius">
          <option value="10">吸附半径 10px</option>
          <option value="24" selected>吸附半径 24px</option>
          <option value="48">吸附半径 48px</option>
          <option value="9999">全局最近骨架</option>
        </select>
        <button class="primary" id="btnAliasPreview">隐藏拟合预览</button>
      </div>
      <div class="box" id="branchBox" style="margin-top:8px">暂无分支候选</div>
      <div class="box" id="qualityBox" style="margin-top:8px">暂无拟合质量</div>

      <h2>显示</h2>
      <div class="grid2">
        <button class="primary" id="btnFullSkeleton">隐藏完整骨架</button>
        <button class="primary" id="btnSkeleton">隐藏切段骨架</button>
        <button id="btnReset">重置视图</button>
      </div>

      <h2>导出 Alias</h2>
      <div class="stack">
        <select id="degree">
          <option value="auto" selected>degree 自动</option>
          <option value="3">degree 3</option>
          <option value="5">degree 5</option>
          <option value="7">degree 7</option>
        </select>
        <button class="warn" id="btnExport">按手动分段导出 IGES</button>
        <div class="box" id="exportBox">尚未导出</div>
      </div>

      <h2>曲线列表</h2>
      <div class="stack" id="curveList"></div>

      <h2>文件</h2>
      <div class="box" id="pathBox"></div>
    </div>
  </aside>
</div>

<script>
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const img = new Image();
let sid = "";
let state = null;
let graph = null;
let designCurves = [];
let cutPoints = [];
let selectedCutIndex = null;
let routePreview = null;
let routeRequestId = 0;
let routeStatus = "";
let fitPreview = null;
let fitPreviewRequestId = 0;
let branchChoices = [];
let closedCurve = false;
let activeCurve = null;
let showSkeleton = true;
let showFullSkeleton = true;
let showAliasPreview = true;
let transform = { scale: 1, x: 0, y: 0 };
let dragging = false;
let lastMouse = null;
let pointDraggingIndex = null;
let pointDragMoved = false;
let suppressNextClick = false;
let snapRequestId = 0;

function api(path) { return path + (path.includes("?") ? "&" : "?") + "sid=" + encodeURIComponent(sid); }

async function uploadImage() {
  const file = document.getElementById("fileInput").files[0];
  if (!file) return;
  setStatus("正在上传并提取骨架...");
  const res = await fetch("/api/upload", {
    method: "POST",
    headers: {
      "Content-Type": "application/octet-stream",
      "X-Filename": encodeURIComponent(file.name)
    },
    body: file
  });
  const data = await res.json();
  if (!data.ok) {
    setStatus("上传失败：" + (data.error || "unknown"));
    return;
  }
  sid = data.sid;
  loadState(data);
}

function loadState(data) {
  state = data;
  graph = state.graph;
  designCurves = state.design_curves || [];
  document.getElementById("uploadBox").classList.add("hidden");
  document.getElementById("workBox").classList.remove("hidden");
  document.getElementById("edgeCount").textContent = graph.edges.length;
  document.getElementById("pathBox").textContent = state.corrections_path || "";
  img.onload = () => { resize(); resetView(); render(); };
  img.src = api("/image") + "&t=" + Date.now();
  updatePanel();
}

function resize() {
  const rect = canvas.parentElement.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(320, Math.floor(rect.width * ratio));
  canvas.height = Math.max(320, Math.floor(rect.height * ratio));
  canvas.style.width = rect.width + "px";
  canvas.style.height = rect.height + "px";
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function resetView() {
  if (!img.width) return;
  const rect = canvas.getBoundingClientRect();
  const s = Math.min(rect.width / img.width, rect.height / img.height) * 0.96;
  transform.scale = s;
  transform.x = (rect.width - img.width * s) * 0.5;
  transform.y = (rect.height - img.height * s) * 0.5;
  render();
}

function worldToScreen(p) { return [p[0] * transform.scale + transform.x, p[1] * transform.scale + transform.y]; }
function screenToWorld(x, y) { return [(x - transform.x) / transform.scale, (y - transform.y) / transform.scale]; }

function render() {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, rect.width, rect.height);
  if (!graph || !img.complete) return;
  ctx.save();
  ctx.translate(transform.x, transform.y);
  ctx.scale(transform.scale, transform.scale);
  ctx.drawImage(img, 0, 0);
  if (showFullSkeleton) drawFullSkeleton();
  if (showSkeleton) drawSkeleton();
  drawBranchAlternatives();
  for (const curve of designCurves) drawPolyline(curve.routed_points || [], "#0b6dff", 2.4, 0.95);
  if (routePreview && routePreview.points) drawPolyline(routePreview.points, "#006dff", 3.2, 1);
  if (showAliasPreview) drawFitPreview();
  drawCutPoints();
  ctx.restore();
}

function drawFullSkeleton() {
  const pts = graph.full_skeleton_points || [];
  if (!pts.length) return;
  ctx.save();
  ctx.fillStyle = "rgba(220,0,0,.70)";
  const r = Math.max(1.2 / transform.scale, 0.62);
  for (const p of pts) {
    ctx.beginPath();
    ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawSkeleton() {
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(0,140,115,.45)";
  ctx.lineWidth = Math.max(1.7 / transform.scale, 0.85);
  for (const edge of graph.edges) {
    const pts = edge.points || [];
    if (pts.length < 2) continue;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();
  }
  ctx.restore();
}

function drawPolyline(points, color, width, alpha) {
  if (!points || points.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = Math.max(width / transform.scale, 0.8);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
  ctx.stroke();
  ctx.restore();
}

function drawBranchAlternatives() {
  if (!routePreview || !routePreview.segments) return;
  const colors = ["rgba(245,128,32,.42)", "rgba(170,80,220,.40)", "rgba(0,150,190,.38)"];
  ctx.save();
  ctx.setLineDash([8 / transform.scale, 7 / transform.scale]);
  for (const segment of routePreview.segments) {
    const selected = segment.selected_candidate || 0;
    const alternatives = segment.alternatives || [];
    for (let i = 0; i < alternatives.length; i++) {
      if (i === selected) continue;
      const alt = alternatives[i];
      drawPolyline(alt.points || [], colors[i % colors.length], 1.6, 1);
    }
  }
  ctx.setLineDash([]);
  ctx.restore();
}

function drawFitPreview() {
  if (!fitPreview || !fitPreview.segments) return;
  for (const seg of fitPreview.segments) {
    if (!seg.ok || !seg.samples) continue;
    drawPolyline(seg.samples, seg.color || "#14a05a", 4.0, 0.95);
  }
}

function drawCutPoints() {
  ctx.save();
  for (let i = 0; i < cutPoints.length; i++) {
    const p = cutPoints[i];
    ctx.beginPath();
    ctx.fillStyle = i === selectedCutIndex ? "#ffe66a" : "#7457ff";
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = Math.max(2 / transform.scale, 0.8);
    ctx.arc(p.x, p.y, Math.max(7 / transform.scale, 3.5), 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#1f2423";
    ctx.font = `${Math.max(13 / transform.scale, 7)}px sans-serif`;
    ctx.fillText(String(i + 1), p.x + Math.max(9 / transform.scale, 5), p.y - Math.max(8 / transform.scale, 4));
  }
  ctx.restore();
}

function setStatus(text) { document.getElementById("status").textContent = text; }

function pickCutPoint(wx, wy) {
  const threshold = Math.max(12 / transform.scale, 4);
  let best = null;
  for (let i = 0; i < cutPoints.length; i++) {
    const p = cutPoints[i];
    const d = Math.hypot(wx - p.x, wy - p.y);
    if (d <= threshold && (!best || d < best.d)) best = { index: i, d };
  }
  return best ? best.index : null;
}

async function addCutPoint(wx, wy) {
  let snapped;
  try {
    snapped = await snapPoint(wx, wy);
  } catch (_err) {
    setStatus("没有找到可吸附的骨架点");
    return;
  }
  cutPoints.push({ x: round3(snapped.x), y: round3(snapped.y), order: cutPoints.length, snap_distance: snapped.distance });
  selectedCutIndex = cutPoints.length - 1;
  refreshRoutePreview();
  updatePanel();
  render();
}

function round3(v) { return Math.round(v * 1000) / 1000; }

function expectedSegmentCount() {
  if (cutPoints.length < 2) return 0;
  return Math.max(0, cutPoints.length - 1) + (closedCurve && cutPoints.length >= 3 ? 1 : 0);
}

function normalizeBranchChoices() {
  const count = expectedSegmentCount();
  while (branchChoices.length < count) branchChoices.push(0);
  if (branchChoices.length > count) branchChoices = branchChoices.slice(0, count);
}

function cleanRouteSegment(segment) {
  return {
    ok: !!segment.ok,
    points: segment.points || [],
    segment_index: segment.segment_index || 0,
    selected_candidate: segment.selected_candidate || 0,
    length: segment.length || 0
  };
}

async function snapPoint(wx, wy) {
  const radiusEl = document.getElementById("snapRadius");
  const maxDistance = radiusEl ? parseFloat(radiusEl.value || "24") : 24;
  const res = await fetch(api("/api/snap"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ point: { x: wx, y: wy }, max_distance: maxDistance })
  });
  const result = await res.json();
  if (!result.ok) throw new Error(result.error || "snap failed");
  return result;
}

async function moveDraggedPoint(index, wx, wy) {
  const requestId = ++snapRequestId;
  let snapped;
  try {
    snapped = await snapPoint(wx, wy);
  } catch (_err) {
    return;
  }
  if (requestId !== snapRequestId || pointDraggingIndex !== index || !cutPoints[index]) return;
  cutPoints[index].x = round3(snapped.x);
  cutPoints[index].y = round3(snapped.y);
  cutPoints[index].snap_distance = snapped.distance;
  selectedCutIndex = index;
  pointDragMoved = true;
  routePreview = null;
  routeStatus = "正在拖动分线点，松开后重新生成蓝线";
  updatePanel();
  render();
}

async function refreshRoutePreview() {
  const requestId = ++routeRequestId;
  if (cutPoints.length < 2) {
    routePreview = null;
    fitPreview = null;
    routeStatus = "";
    updatePanel();
    render();
    return;
  }
  normalizeBranchChoices();
  routeStatus = "正在沿骨架生成蓝线...";
  updatePanel();
  try {
    const res = await fetch(api("/api/route"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        points: cutPoints,
        closed: closedCurve,
        branch_choices: branchChoices,
        candidate_count: 2
      })
    });
    const result = await res.json();
    if (requestId !== routeRequestId) return;
    routePreview = result;
    routeStatus = result.ok
      ? `蓝线已生成：${result.point_count || 0} 个骨架点`
      : "骨架未连通，请在中间多加一个引导点";
    await refreshFitPreview();
  } catch (_err) {
    if (requestId !== routeRequestId) return;
    routePreview = null;
    fitPreview = null;
    routeStatus = "路径生成失败";
  }
  updatePanel();
  render();
}

async function refreshFitPreview() {
  const requestId = ++fitPreviewRequestId;
  if (!routePreview || !routePreview.segments || routePreview.segments.length < 1) {
    fitPreview = null;
    return;
  }
  try {
    const res = await fetch(api("/api/fit-preview"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        route_segments: routePreview.segments.map(cleanRouteSegment),
        degree: document.getElementById("degree").value
      })
    });
    const result = await res.json();
    if (requestId !== fitPreviewRequestId) return;
    fitPreview = result.ok ? result : null;
  } catch (_err) {
    if (requestId !== fitPreviewRequestId) return;
    fitPreview = null;
  }
}

async function saveCurrent(startNext=false) {
  if (cutPoints.length < 2) return;
  if (!routePreview || !routePreview.points || routePreview.points.length < 2) await refreshRoutePreview();
  const item = {
    id: activeCurve || makeId(),
    type: "manual_design_curve",
    semantic: document.getElementById("semantic").value,
    edge_ids: [],
    manual_points: cutPoints.map((p, i) => ({ ...p, order: i })),
    cut_points: cutPoints.map((p, i) => ({ ...p, order: i })),
    closed: closedCurve,
    routed_points: routePreview && routePreview.points ? routePreview.points : [],
    route_segments: routePreview && routePreview.segments ? routePreview.segments.map(cleanRouteSegment) : [],
    branch_choices: branchChoices.slice(),
    route_ok: routePreview ? !!routePreview.ok : false,
    created_at: new Date().toISOString()
  };
  const idx = designCurves.findIndex(c => c.id === item.id);
  if (idx >= 0) designCurves[idx] = item;
  else designCurves.push(item);
  activeCurve = item.id;
  await saveAll();
  if (startNext) clearCurrent();
  updatePanel();
  render();
}

async function saveAll() {
  const res = await fetch(api("/api/corrections"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ corrections: [], design_curves: designCurves })
  });
  const result = await res.json();
  if (result.ok) document.getElementById("pathBox").textContent = result.corrections_path;
}

async function exportIges() {
  if (cutPoints.length >= 2) await saveCurrent(false);
  setExport("正在拟合 single-span NURBS 并导出 IGES...");
  const res = await fetch(api("/api/export-iges"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      corrections: [],
      design_curves: designCurves,
      degree: document.getElementById("degree").value
    })
  });
  const result = await res.json();
  if (!result.ok) {
    setExport("导出失败：" + (result.error || "unknown"));
    return;
  }
  const links = [];
  if (result.exports.iges) links.push(`<a href="${result.exports.iges.url}">下载 IGES</a>`);
  if (result.exports.json) links.push(`<a href="${result.exports.json.url}">下载 JSON</a>`);
  if (result.exports.clean_preview) links.push(`<a href="${result.exports.clean_preview.url}">下载预览 SVG</a>`);
  setExport(`${result.curve_count} 条曲线，${result.passed_count} 条通过。<br>${links.join(" / ")}<br>${result.out}`);
}

function setExport(html) { document.getElementById("exportBox").innerHTML = html; }

function clearCurrent() {
  cutPoints = [];
  selectedCutIndex = null;
  routePreview = null;
  fitPreview = null;
  routeStatus = "";
  branchChoices = [];
  closedCurve = false;
  activeCurve = null;
  updatePanel();
  render();
}

function undoPoint() {
  if (!cutPoints.length) return;
  cutPoints.pop();
  selectedCutIndex = cutPoints.length ? cutPoints.length - 1 : null;
  if (cutPoints.length < 3) closedCurve = false;
  normalizeBranchChoices();
  refreshRoutePreview();
}

function deleteSelected() {
  if (selectedCutIndex == null) return;
  cutPoints.splice(selectedCutIndex, 1);
  selectedCutIndex = cutPoints.length ? Math.min(selectedCutIndex, cutPoints.length - 1) : null;
  if (cutPoints.length < 3) closedCurve = false;
  normalizeBranchChoices();
  refreshRoutePreview();
}

function toggleClosed() {
  if (cutPoints.length < 3) return;
  closedCurve = !closedCurve;
  normalizeBranchChoices();
  refreshRoutePreview();
}

function renderCurveList() {
  const list = document.getElementById("curveList");
  list.innerHTML = "";
  const reversed = designCurves.map((curve, index) => ({ curve, index })).reverse();
  for (const entry of reversed) {
    const curve = entry.curve;
    const item = document.createElement("div");
    item.className = "item" + (activeCurve === curve.id ? " active" : "");
    const head = document.createElement("div");
    head.className = "itemHead";
    const text = document.createElement("div");
    text.innerHTML = `<strong>${curve.semantic || "manual_design_curve"}</strong><small>${(curve.manual_points || []).length} 个分段点，${(curve.routed_points || []).length} 个骨架点${curve.closed ? "，闭合" : ""}</small>`;
    const del = document.createElement("button");
    del.className = "miniBad";
    del.textContent = "删除";
    del.onclick = async (event) => {
      event.stopPropagation();
      await deleteSavedCurve(curve.id);
    };
    head.append(text, del);
    item.append(head);
    item.onclick = () => {
      activeCurve = curve.id;
      cutPoints = (curve.manual_points || curve.cut_points || []).map(p => ({ ...p }));
      closedCurve = !!curve.closed;
      branchChoices = (curve.branch_choices || []).slice();
      normalizeBranchChoices();
      routePreview = { ok: !!curve.route_ok, points: curve.routed_points || [], segments: curve.route_segments || [] };
      routeStatus = routePreview.points.length ? `已加载蓝线：${routePreview.points.length} 个骨架点` : "";
      document.getElementById("semantic").value = curve.semantic || "detail_line";
      selectedCutIndex = null;
      updatePanel();
      render();
      refreshRoutePreview();
    };
    list.appendChild(item);
  }
}

async function deleteSavedCurve(curveId) {
  const curve = designCurves.find(c => c.id === curveId);
  if (!curve) return;
  const label = `${curve.semantic || "manual_design_curve"} / ${(curve.manual_points || []).length} 个分段点`;
  if (!confirm("删除这条已保存曲线？\n" + label)) return;
  designCurves = designCurves.filter(c => c.id !== curveId);
  if (activeCurve === curveId) clearCurrent();
  await saveAll();
  updatePanel();
  render();
}

function updatePanel() {
  document.getElementById("curveCount").textContent = designCurves.length;
  document.getElementById("pointCount").textContent = cutPoints.length;
  document.getElementById("routeBox").textContent = routeStatus || "在图上点击两个或多个点，蓝线会沿骨架自动生成。";
  renderBranchControls();
  renderQualityPreview();
  document.getElementById("btnClose").classList.toggle("primary", closedCurve);
  document.getElementById("btnClose").textContent = closedCurve ? "闭合中" : "闭合曲线";
  document.getElementById("btnSkeleton").classList.toggle("primary", showSkeleton);
  document.getElementById("btnSkeleton").textContent = showSkeleton ? "隐藏切段骨架" : "显示切段骨架";
  document.getElementById("btnFullSkeleton").classList.toggle("primary", showFullSkeleton);
  document.getElementById("btnFullSkeleton").textContent = showFullSkeleton ? "隐藏完整骨架" : "显示完整骨架";
  document.getElementById("btnAliasPreview").classList.toggle("primary", showAliasPreview);
  document.getElementById("btnAliasPreview").textContent = showAliasPreview ? "隐藏拟合预览" : "显示拟合预览";
  document.getElementById("btnSave").disabled = cutPoints.length < 2;
  document.getElementById("btnSaveNext").disabled = cutPoints.length < 2;
  document.getElementById("btnUndo").disabled = cutPoints.length < 1;
  document.getElementById("btnDelete").disabled = selectedCutIndex == null;
  document.getElementById("btnClose").disabled = cutPoints.length < 3;
  renderCurveList();
}

function renderBranchControls() {
  const box = document.getElementById("branchBox");
  if (!routePreview || !routePreview.segments || routePreview.segments.length === 0) {
    box.innerHTML = "暂无分支候选";
    return;
  }
  const rows = [];
  for (const segment of routePreview.segments) {
    const alternatives = segment.alternatives || [];
    if (alternatives.length <= 1) continue;
    const idx = segment.segment_index || 0;
    const selected = segment.selected_candidate || 0;
    const buttons = alternatives.map((alt, altIndex) => {
      const cls = altIndex === selected ? "primary" : "";
      const text = `方案 ${altIndex + 1} / ${Math.round(alt.length || 0)}px`;
      return `<button class="${cls}" data-seg="${idx}" data-alt="${altIndex}">${text}</button>`;
    }).join("");
    rows.push(`<div style="margin-bottom:8px"><strong>第 ${idx + 1} 段分支</strong><div class="grid2" style="margin-top:5px">${buttons}</div></div>`);
  }
  box.innerHTML = rows.length ? rows.join("") : "当前路径没有明显分支候选";
  for (const btn of box.querySelectorAll("button[data-seg]")) {
    btn.onclick = () => {
      const seg = parseInt(btn.getAttribute("data-seg") || "0", 10);
      const alt = parseInt(btn.getAttribute("data-alt") || "0", 10);
      branchChoices[seg] = alt;
      refreshRoutePreview();
    };
  }
}

function renderQualityPreview() {
  const box = document.getElementById("qualityBox");
  if (!fitPreview || !fitPreview.segments || fitPreview.segments.length === 0) {
    box.innerHTML = "暂无拟合质量";
    return;
  }
  const rows = fitPreview.segments.map((seg, i) => {
    const color = seg.color || "#888";
    const status = seg.passed ? "通过" : "警告";
    const degree = seg.degree ? `d${seg.degree}` : "未拟合";
    const warn = seg.warnings && seg.warnings.length ? seg.warnings.slice(0, 2).join("；") : "曲率/CV 检查正常";
    return `<div style="border-left:6px solid ${color};padding:4px 0 5px 8px;margin:4px 0"><strong>第 ${i + 1} 段 ${status} ${degree}</strong><br><small>${warn}</small></div>`;
  });
  box.innerHTML = rows.join("");
}

function makeId() { return "curve_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 7); }

canvas.addEventListener("mousedown", e => {
  if (graph && e.button === 0 && !e.altKey) {
    const rect = canvas.getBoundingClientRect();
    const [wx, wy] = screenToWorld(e.clientX - rect.left, e.clientY - rect.top);
    const hit = pickCutPoint(wx, wy);
    if (hit != null) {
      pointDraggingIndex = hit;
      pointDragMoved = false;
      selectedCutIndex = hit;
      updatePanel();
      render();
      e.preventDefault();
      return;
    }
  }
  if (e.button === 1 || e.altKey) {
    dragging = true;
    lastMouse = [e.clientX, e.clientY];
    e.preventDefault();
  }
});
canvas.addEventListener("mousemove", e => {
  if (pointDraggingIndex != null) {
    const rect = canvas.getBoundingClientRect();
    const [wx, wy] = screenToWorld(e.clientX - rect.left, e.clientY - rect.top);
    suppressNextClick = true;
    moveDraggedPoint(pointDraggingIndex, wx, wy);
    e.preventDefault();
    return;
  }
  if (!dragging || !lastMouse) return;
  transform.x += e.clientX - lastMouse[0];
  transform.y += e.clientY - lastMouse[1];
  lastMouse = [e.clientX, e.clientY];
  render();
});
window.addEventListener("mouseup", () => {
  if (pointDraggingIndex != null) {
    const needsRoute = pointDragMoved;
    pointDraggingIndex = null;
    pointDragMoved = false;
    if (needsRoute) {
      suppressNextClick = true;
      refreshRoutePreview();
    }
  }
  dragging = false;
  lastMouse = null;
});
canvas.addEventListener("click", e => {
  if (suppressNextClick) {
    suppressNextClick = false;
    return;
  }
  if (!graph || e.altKey) return;
  const rect = canvas.getBoundingClientRect();
  const [wx, wy] = screenToWorld(e.clientX - rect.left, e.clientY - rect.top);
  const hit = pickCutPoint(wx, wy);
  if (hit != null) {
    selectedCutIndex = hit;
    updatePanel();
    render();
    return;
  }
  addCutPoint(wx, wy);
});
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
  const before = screenToWorld(sx, sy);
  transform.scale = Math.max(0.05, Math.min(30, transform.scale * (e.deltaY < 0 ? 1.12 : 0.89)));
  transform.x = sx - before[0] * transform.scale;
  transform.y = sy - before[1] * transform.scale;
  render();
}, { passive:false });
window.addEventListener("resize", () => { resize(); render(); });
window.addEventListener("keydown", e => {
  if (e.ctrlKey && e.key.toLowerCase() === "z") { e.preventDefault(); undoPoint(); }
  if (e.key === "Delete" || e.key === "Backspace") { if (selectedCutIndex != null) { e.preventDefault(); deleteSelected(); } }
});

document.getElementById("btnUpload").onclick = uploadImage;
document.getElementById("btnSave").onclick = () => saveCurrent(false);
document.getElementById("btnSaveNext").onclick = () => saveCurrent(true);
document.getElementById("btnUndo").onclick = undoPoint;
document.getElementById("btnDelete").onclick = deleteSelected;
document.getElementById("btnClose").onclick = toggleClosed;
document.getElementById("btnClear").onclick = clearCurrent;
document.getElementById("btnSkeleton").onclick = () => { showSkeleton = !showSkeleton; updatePanel(); render(); };
document.getElementById("btnFullSkeleton").onclick = () => { showFullSkeleton = !showFullSkeleton; updatePanel(); render(); };
document.getElementById("btnAliasPreview").onclick = () => { showAliasPreview = !showAliasPreview; updatePanel(); render(); };
document.getElementById("degree").onchange = () => { refreshFitPreview().then(() => { updatePanel(); render(); }); };
document.getElementById("btnReset").onclick = resetView;
document.getElementById("btnExport").onclick = exportIges;
resize();
render();
</script>
</body>
</html>"""
