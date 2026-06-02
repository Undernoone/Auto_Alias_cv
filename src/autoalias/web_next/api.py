from __future__ import annotations

import base64
import json
import mimetypes
import os
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from autoalias.review.fit_reviewed import fit_reviewed_annotations
from autoalias.review.graph import ReviewGraphOptions
from autoalias.review.server import ReviewSession
from autoalias.review.workflow_server import (
    _auto_segment_curves,
    _clean_bool,
    _clean_extraction_mode,
    _clean_input_preprocess,
    _clean_max_fit_points,
    _clean_parallel_collapse,
    _client_state,
    _edit_session_skeleton,
    _fit_preview_segments,
    _local_ipv4_addresses,
    _make_session_id,
    _route_points_with_choices,
    _safe_filename,
    _unique_path,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "lan_reviews_next"
OUTPUT_DIR = Path(os.environ.get("AUTOALIAS_WEB_OUT", DEFAULT_OUT)).resolve()
FRONTEND_DIST = ROOT / "webapp" / "dist"


app = FastAPI(
    title="AutoAlias Next Web API",
    version="0.2.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8790",
        "http://localhost:8790",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "uploads").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "alias_exports").mkdir(parents=True, exist_ok=True)

sessions: dict[str, ReviewSession] = {}
session_meta: dict[str, dict[str, Any]] = {}
exports: dict[str, dict[str, Path]] = {}
jobs: dict[str, dict[str, Any]] = {}
lock = threading.Lock()


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "workspace": str(OUTPUT_DIR),
        "frontend_dist": str(FRONTEND_DIST),
        "frontend_built": FRONTEND_DIST.exists(),
        "lan_urls": [f"http://{ip}:8790/" for ip in _local_ipv4_addresses()],
    }


@app.post("/api/upload")
async def upload_image(
    file: UploadFile = File(...),
    x_extraction_mode: str = Header("auto", alias="X-Extraction-Mode"),
    x_input_preprocess: str = Header("none", alias="X-Input-Preprocess"),
    x_weak_line_threshold: str = Header("32", alias="X-Weak-Line-Threshold"),
    x_parallel_collapse: str = Header("off", alias="X-Parallel-Collapse"),
) -> dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")

    target = _unique_path(OUTPUT_DIR / "uploads" / _safe_filename(file.filename or "uploaded.png"))
    target.write_bytes(raw)
    try:
        options = _graph_options(
            extraction_mode=x_extraction_mode,
            input_preprocess=x_input_preprocess,
            parallel_collapse=x_parallel_collapse,
            weak_line_threshold=x_weak_line_threshold,
        )
        session = ReviewSession.create(
            target,
            OUTPUT_DIR,
            options,
        )
    except Exception as exc:
        try:
            target.unlink()
        except OSError:
            pass
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    sid = _make_session_id(target)
    sessions[sid] = session
    session_meta[sid] = {
        "source_image": str(target),
        "graph_options": _graph_options_payload(options),
        "skeleton_edits": [],
        "last_skeleton_edit_index": None,
    }
    payload = _session_payload(sid, session)
    payload["ok"] = True
    return payload


@app.get("/api/sessions/{sid}/state")
def session_state(sid: str) -> dict[str, Any]:
    return _session_payload(sid, _require_session(sid))


@app.get("/api/sessions/{sid}/image")
def session_image(sid: str) -> FileResponse:
    session = _require_session(sid)
    return FileResponse(session.image_path)


@app.post("/api/sessions/{sid}/reextract")
async def reextract_image(sid: str, payload: dict[str, Any]) -> dict[str, Any]:
    old_session = _require_session(sid)
    meta = session_meta.setdefault(sid, _default_session_meta(old_session))
    source = Path(str(meta.get("source_image") or old_session.image_path)).resolve()
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"原始图片不存在：{source}")
    options = _graph_options(
        extraction_mode=payload.get("extraction_mode", "auto"),
        input_preprocess=payload.get("input_preprocess", "none"),
        parallel_collapse=payload.get("parallel_collapse", "off"),
        weak_line_threshold=payload.get("weak_line_threshold", 32),
    )
    try:
        new_session = ReviewSession.create(source, OUTPUT_DIR, options)
        new_session.corrections_path = old_session.corrections_path
        new_session.corrections = list(old_session.corrections)
        new_session.design_curves = list(old_session.design_curves)
        _replay_skeleton_edits(new_session, list(meta.get("skeleton_edits") or []))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_friendly_session_error(str(exc))) from exc
    sessions[sid] = new_session
    meta["graph_options"] = _graph_options_payload(options)
    meta["last_skeleton_edit_index"] = None
    _save_session(sid, new_session)
    result = _session_payload(sid, new_session)
    result["ok"] = True
    result["message"] = "已按当前选项重新提取，原有曲线和骨架修补记录已保留"
    return result


@app.post("/api/sessions/{sid}/route")
async def route_points(sid: str, payload: dict[str, Any]) -> dict[str, Any]:
    session = _require_session(sid)
    return _route_points_with_choices(
        session,
        payload.get("points", []),
        closed=bool(payload.get("closed", False)),
        branch_choices=payload.get("branch_choices", []),
        candidate_count=int(payload.get("candidate_count", 3)),
    )


@app.post("/api/sessions/{sid}/fit-preview")
async def fit_preview(sid: str, payload: dict[str, Any]) -> dict[str, Any]:
    _require_session(sid)
    return _fit_preview_segments(
        payload.get("route_segments", []),
        degree=payload.get("degree", "auto"),
        closed=bool(payload.get("closed", False)),
        high_quality=payload.get("quality") == "export",
    )


@app.post("/api/sessions/{sid}/snap")
async def snap_point(sid: str, payload: dict[str, Any]) -> dict[str, Any]:
    session = _require_session(sid)
    point = payload.get("point", payload)
    if not isinstance(point, dict) or "x" not in point or "y" not in point:
        raise HTTPException(status_code=400, detail="point must contain x and y")
    max_distance = float(payload.get("max_distance", 24.0))
    index, distance = session.router.nearest_index((float(point["x"]), float(point["y"])))
    if distance > max_distance:
        return {
            "ok": False,
            "reason": "nearest skeleton point is outside snap radius",
            "distance": round(float(distance), 3),
            "max_distance": round(float(max_distance), 3),
        }
    snapped = session.router.coords[index]
    return {
        "ok": True,
        "x": round(float(snapped[0]), 3),
        "y": round(float(snapped[1]), 3),
        "distance": round(float(distance), 3),
    }


@app.post("/api/sessions/{sid}/skeleton-edit")
async def skeleton_edit(sid: str, payload: dict[str, Any]) -> dict[str, Any]:
    session = _require_session(sid)
    meta = session_meta.setdefault(sid, _default_session_meta(session))
    clean = dict(payload)
    action = str(clean.get("action") or "").strip().lower()
    if action == "add" and clean.get("link_index") in (None, ""):
        clean["link_index"] = meta.get("last_skeleton_edit_index")
    result = _edit_session_skeleton(session, clean)
    if result.get("ok"):
        edits = meta.setdefault("skeleton_edits", [])
        edits.append(clean)
        meta["last_skeleton_edit_index"] = (
            int(result.get("index", -1)) if action == "add" else None
        )
        _save_session(sid, session)
    result["last_skeleton_edit_index"] = meta.get("last_skeleton_edit_index")
    result["skeleton_edit_count"] = len(meta.get("skeleton_edits") or [])
    return result


@app.post("/api/sessions/{sid}/skeleton-edit/break")
async def break_skeleton_edit_chain(sid: str) -> dict[str, Any]:
    session = _require_session(sid)
    meta = session_meta.setdefault(sid, _default_session_meta(session))
    meta["last_skeleton_edit_index"] = None
    return {"ok": True, "message": "已断开连续骨架加点"}


@app.post("/api/sessions/{sid}/auto-segment")
async def auto_segment(sid: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return _auto_segment_curves(_require_session(sid), payload or {})


@app.post("/api/sessions/{sid}/corrections")
async def save_corrections(sid: str, payload: dict[str, Any]) -> dict[str, Any]:
    session = _require_session(sid)
    corrections = payload.get("corrections", [])
    design_curves = payload.get("design_curves", [])
    if not isinstance(corrections, list) or not isinstance(design_curves, list):
        raise HTTPException(status_code=400, detail="corrections and design_curves must be lists")
    session.corrections = corrections
    session.design_curves = design_curves
    _save_session(sid, session)
    return {
        "ok": True,
        "corrections_path": str(session.corrections_path),
        "design_curve_count": len(session.design_curves),
    }


@app.post("/api/sessions/{sid}/project")
async def download_project(sid: str, payload: dict[str, Any] | None = None) -> Response:
    session = _require_session(sid)
    body = payload or {}
    corrections = body.get("corrections", session.corrections)
    design_curves = body.get("design_curves", session.design_curves)
    if not isinstance(corrections, list) or not isinstance(design_curves, list):
        raise HTTPException(status_code=400, detail="corrections and design_curves must be lists")
    session.corrections = corrections
    session.design_curves = design_curves
    _save_session(sid, session)
    project = _project_payload(sid, session, editor_state=body.get("editor_state", {}))
    filename = f"{Path(str(project['source_image']['filename'])).stem}.autoalias_project.json"
    return Response(
        content=json.dumps(project, ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/projects/open")
async def open_project(project: UploadFile = File(...)) -> dict[str, Any]:
    try:
        data = json.loads((await project.read()).decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="工程 JSON 无法读取") from exc
    source = data.get("source_image") or {}
    encoded = str(source.get("data_base64") or "")
    if not encoded:
        raise HTTPException(status_code=400, detail="工程 JSON 中没有内嵌原图")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="工程 JSON 中的原图数据损坏") from exc
    filename = _safe_filename(str(source.get("filename") or "project_image.png"))
    target = _unique_path(OUTPUT_DIR / "uploads" / filename)
    target.write_bytes(raw)
    options = _graph_options_from_payload(data.get("graph_options") or {})
    try:
        session = ReviewSession.create(target, OUTPUT_DIR, options)
        session.corrections = list(data.get("corrections") or [])
        session.design_curves = list(data.get("design_curves") or [])
        edits = list(data.get("skeleton_edits") or [])
        _replay_skeleton_edits(session, edits)
    except Exception as exc:
        try:
            target.unlink()
        except OSError:
            pass
        raise HTTPException(status_code=400, detail=_friendly_session_error(str(exc))) from exc
    sid = _make_session_id(target)
    sessions[sid] = session
    session_meta[sid] = {
        "source_image": str(target),
        "graph_options": _graph_options_payload(options),
        "skeleton_edits": edits,
        "last_skeleton_edit_index": None,
    }
    _save_session(sid, session)
    result = _session_payload(sid, session)
    result["ok"] = True
    result["editor_state"] = data.get("editor_state") or {}
    return result


@app.post("/api/sessions/{sid}/export")
async def export_alias(sid: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    session = _require_session(sid)
    job_id = f"export_{int(time.time() * 1000):x}"
    _set_job(job_id, status="queued", progress=1, message="导出任务已创建")
    thread = threading.Thread(
        target=_run_export_job,
        args=(job_id, sid, session, payload or {}),
        name=f"autoalias-export-{job_id[-8:]}",
        daemon=True,
    )
    thread.start()
    return {"ok": True, "job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    with lock:
        job = dict(jobs.get(job_id, {}))
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/api/sessions/{sid}/download/{kind}")
def download_export(sid: str, kind: str) -> FileResponse:
    path = exports.get(sid, {}).get(kind)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="export not found")
    return FileResponse(path, filename=path.name)


def _run_export_job(job_id: str, sid: str, session: ReviewSession, payload: dict[str, Any]) -> None:
    try:
        _set_job(job_id, status="running", progress=8, message="保存分段数据")
        session.corrections = payload.get("corrections", session.corrections)
        session.design_curves = payload.get("design_curves", session.design_curves)
        _save_session(sid, session)
        _set_job(job_id, status="running", progress=35, message="拟合 Class-A 曲线并生成 Alias 文件")
        export_dir = OUTPUT_DIR / "alias_exports" / f"{session.image_path.stem}_next"
        result = fit_reviewed_annotations(
            [session.corrections_path],
            export_dir,
            degree=payload.get("degree", "auto"),
            min_points=8,
            max_fit_points=_clean_max_fit_points(payload.get("max_fit_points", None)),
            diagnostic_preview=bool(payload.get("diagnostic_preview", False)),
            fast_mode=_clean_bool(payload.get("fast_mode", False)),
            fit_mode=str(payload.get("fit_mode") or "manual_class_a_g2"),
            wire_export=_clean_bool(payload.get("wire_export", True)),
            iges_to_al=payload.get("iges_to_al") or None,
        )
        paths = {
            "iges": export_dir / "reviewed_curves.igs",
            "wire": export_dir / "reviewed_curves.wire",
            "wire_status": export_dir / "reviewed_curves.wire_status.json",
            "json": export_dir / "reviewed_curves.json",
            "preview": export_dir / "reviewed_preview.svg",
            "clean_preview": export_dir / "reviewed_clean_preview.svg",
        }
        exports[sid] = paths
        passed = sum(1 for report in result.reports if report.passed)
        _set_job(
            job_id,
            status="done",
            progress=100,
            message="导出完成",
            result={
                "ok": True,
                "curve_count": len(result.curves),
                "passed_count": passed,
                "skipped_count": result.skipped_count,
                "out": str(export_dir),
                "exports": _next_export_payload(paths, sid),
                "warnings": [
                    {"label": report.label, "warnings": report.warnings}
                    for report in result.reports
                    if report.warnings
                ],
            },
        )
    except Exception as exc:
        _set_job(job_id, status="failed", progress=100, message="导出失败", error=str(exc))


def _session_payload(sid: str, session: ReviewSession) -> dict[str, Any]:
    payload = _client_state(session)
    payload["sid"] = sid
    payload["exports"] = _next_export_payload(exports.get(sid, {}), sid)
    meta = session_meta.setdefault(sid, _default_session_meta(session))
    payload["skeleton_edit_count"] = len(meta.get("skeleton_edits") or [])
    return payload


def _require_session(sid: str) -> ReviewSession:
    session = sessions.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="missing or expired image session")
    return session


def _set_job(job_id: str, **patch: Any) -> None:
    with lock:
        job = jobs.setdefault(
            job_id,
            {"ok": True, "job_id": job_id, "created_at": time.time()},
        )
        job.update(patch)
        job["updated_at"] = time.time()


def _next_export_payload(paths: dict[str, Path], sid: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for kind, path in paths.items():
        if path.exists():
            payload[kind] = {
                "path": str(path),
                "url": f"/api/sessions/{sid}/download/{kind}",
            }
    return payload


def _graph_options(
    *,
    extraction_mode: Any,
    input_preprocess: Any,
    parallel_collapse: Any,
    weak_line_threshold: Any,
) -> ReviewGraphOptions:
    return ReviewGraphOptions(
        extraction_mode=_clean_extraction_mode(str(extraction_mode or "auto")),
        input_preprocess=_clean_input_preprocess(str(input_preprocess or "none")),
        parallel_collapse=_clean_parallel_collapse(str(parallel_collapse or "off")),
        weak_line_threshold=float(weak_line_threshold or 32),
        max_points_per_edge=480,
    )


def _graph_options_payload(options: ReviewGraphOptions) -> dict[str, Any]:
    return {
        "extraction_mode": options.extraction_mode,
        "input_preprocess": options.input_preprocess,
        "parallel_collapse": options.parallel_collapse,
        "weak_line_threshold": options.weak_line_threshold,
        "max_points_per_edge": options.max_points_per_edge,
    }


def _graph_options_from_payload(payload: dict[str, Any]) -> ReviewGraphOptions:
    return _graph_options(
        extraction_mode=payload.get("extraction_mode", "auto"),
        input_preprocess=payload.get("input_preprocess", "none"),
        parallel_collapse=payload.get("parallel_collapse", "off"),
        weak_line_threshold=payload.get("weak_line_threshold", 32),
    )


def _default_session_meta(session: ReviewSession) -> dict[str, Any]:
    return {
        "source_image": str(session.graph.get("source_image") or session.image_path),
        "graph_options": {
            "extraction_mode": session.graph.get("extraction_mode", "auto"),
            "input_preprocess": session.graph.get("input_preprocess", "none"),
            "parallel_collapse": session.graph.get("parallel_collapse", "off"),
            "weak_line_threshold": session.graph.get("weak_line_threshold", 32),
            "max_points_per_edge": 480,
        },
        "skeleton_edits": [],
        "last_skeleton_edit_index": None,
    }


def _save_session(sid: str, session: ReviewSession) -> None:
    session.save(session.corrections, session.design_curves)
    meta = session_meta.setdefault(sid, _default_session_meta(session))
    try:
        data = json.loads(session.corrections_path.read_text(encoding="utf-8"))
        data["skeleton_edits"] = list(meta.get("skeleton_edits") or [])
        data["graph_options"] = dict(meta.get("graph_options") or {})
        session.corrections_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _replay_skeleton_edits(session: ReviewSession, edits: list[dict[str, Any]]) -> None:
    for edit in edits:
        try:
            _edit_session_skeleton(session, edit)
        except Exception:
            continue


def _project_payload(sid: str, session: ReviewSession, *, editor_state: Any) -> dict[str, Any]:
    meta = session_meta.setdefault(sid, _default_session_meta(session))
    source = Path(str(meta.get("source_image") or session.image_path)).resolve()
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"原始图片不存在：{source}")
    mime = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
    return {
        "version": 2,
        "task": "autoalias_web_project",
        "created_or_updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_image": {
            "filename": source.name,
            "media_type": mime,
            "data_base64": base64.b64encode(source.read_bytes()).decode("ascii"),
        },
        "graph_options": dict(meta.get("graph_options") or {}),
        "skeleton_edits": list(meta.get("skeleton_edits") or []),
        "corrections": list(session.corrections),
        "design_curves": list(session.design_curves),
        "editor_state": editor_state if isinstance(editor_state, dict) else {},
    }


def _friendly_session_error(message: str) -> str:
    if "raw feature-line preprocessing found no usable line pixels" in message:
        return (
            "原图预处理没有找到可用线条像素。输入可能已经是线稿，或线条较淡。"
            "请取消原图预处理；黑底白线图请选择黑底白线；淡铅笔线请选择铅笔弱线并降低阈值。"
        )
    return message


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    if FRONTEND_DIST.exists():
        index_path = FRONTEND_DIST / "index.html"
        if index_path.exists():
            return index_path.read_text(encoding="utf-8")
    return """
    <!doctype html>
    <html lang="zh-CN">
      <head><meta charset="utf-8"><title>AutoAlias Next</title></head>
      <body style="font-family:Segoe UI,Microsoft YaHei,sans-serif;padding:40px">
        <h1>AutoAlias Next API 已启动</h1>
        <p>前端尚未构建。请进入 F:\\430AutoAlias\\webapp 运行 npm install 和 npm run dev。</p>
        <p>API 文档：<a href="/api/docs">/api/docs</a></p>
      </body>
    </html>
    """


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
