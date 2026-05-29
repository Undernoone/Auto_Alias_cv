from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
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
        session = ReviewSession.create(
            target,
            OUTPUT_DIR,
            ReviewGraphOptions(
                extraction_mode=_clean_extraction_mode(x_extraction_mode),
                input_preprocess=_clean_input_preprocess(x_input_preprocess),
                parallel_collapse=_clean_parallel_collapse(x_parallel_collapse),
                weak_line_threshold=float(x_weak_line_threshold or 32),
            ),
        )
    except Exception as exc:
        try:
            target.unlink()
        except OSError:
            pass
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    sid = _make_session_id(target)
    sessions[sid] = session
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
    return _edit_session_skeleton(_require_session(sid), payload)


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
    session.save(corrections, design_curves)
    return {
        "ok": True,
        "corrections_path": str(session.corrections_path),
        "design_curve_count": len(session.design_curves),
    }


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
        session.save(
            payload.get("corrections", session.corrections),
            payload.get("design_curves", session.design_curves),
        )
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
