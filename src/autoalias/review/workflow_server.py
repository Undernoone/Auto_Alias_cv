from __future__ import annotations

import json
import mimetypes
import socket
import threading
import time
import uuid
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
    ai_jobs: dict[str, dict[str, Any]] = {}
    ai_job_lock = threading.Lock()
    actual_port = _find_available_port(host, port)
    handler = _make_handler(out, sessions, exports, ai_jobs, ai_job_lock)
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
    ai_jobs: dict[str, dict[str, Any]],
    ai_job_lock: threading.Lock,
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
            if parsed.path == "/api/ai-suggest-status":
                session = self._require_session(parsed)
                if session is None:
                    return
                self._handle_ai_status(parsed)
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
            if parsed.path == "/api/ai-suggest-start":
                session = self._require_session(parsed)
                if session is None:
                    return
                self._handle_ai_start(session)
                return
            if parsed.path == "/api/ai-suggest":
                session = self._require_session(parsed)
                if session is None:
                    return
                self._handle_ai_suggest(session)
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
                self._send_json(
                    _fit_preview_segments(
                        route_segments,
                        degree=degree,
                        closed=bool(payload.get("closed", False)),
                        high_quality=payload.get("quality") == "export",
                    )
                )
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

        def _handle_ai_start(self, session: ReviewSession) -> None:
            try:
                payload = self._read_json(required=False)
                job_id = uuid.uuid4().hex
                sid = _sid(urlparse(self.path))
                now = time.time()
                with ai_job_lock:
                    ai_jobs[job_id] = {
                        "ok": True,
                        "job_id": job_id,
                        "sid": sid,
                        "status": "queued",
                        "progress": 1,
                        "message": "AI 任务已创建，等待后台线程启动",
                        "created_at": now,
                        "updated_at": now,
                    }

                def update(**patch: Any) -> None:
                    with ai_job_lock:
                        job = ai_jobs.get(job_id)
                        if job is not None:
                            job.update(patch)
                            job["updated_at"] = time.time()

                def progress(percent: int, message: str) -> None:
                    update(
                        status="running",
                        progress=max(1, min(int(percent), 99)),
                        message=message,
                    )

                def worker() -> None:
                    try:
                        progress(3, "后台任务已启动")
                        result = _ai_suggest_curves(
                            session,
                            payload,
                            output_dir,
                            progress=progress,
                        )
                        update(
                            status="done",
                            progress=100,
                            message=f"AI 已生成 {result.get('curve_count', 0)} 条候选曲线",
                            result=result,
                        )
                    except Exception as exc:
                        update(
                            status="failed",
                            progress=100,
                            message="AI 建议失败",
                            error=str(exc),
                        )
                    finally:
                        _prune_ai_jobs(ai_jobs, ai_job_lock)

                threading.Thread(target=worker, name=f"autoalias-ai-{job_id[:8]}", daemon=True).start()
                self._send_json({"ok": True, "job_id": job_id})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)

        def _handle_ai_status(self, parsed) -> None:
            job_id = parse_qs(parsed.query).get("job", [""])[0]
            if not job_id:
                self._send_json({"ok": False, "error": "missing job id"}, status=400)
                return
            with ai_job_lock:
                job = dict(ai_jobs.get(job_id, {}))
            if not job:
                self._send_json({"ok": False, "error": "AI job not found"}, status=404)
                return
            self._send_json(job)

        def _handle_ai_suggest(self, session: ReviewSession) -> None:
            try:
                payload = self._read_json(required=False)
                result = _ai_suggest_curves(session, payload, output_dir)
                self._send_json(result)
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


def _ai_suggest_curves(
    session: ReviewSession,
    payload: dict[str, Any],
    output_dir: Path,
    progress: Any = None,
) -> dict[str, Any]:
    from autoalias.review.ai_suggest import DEFAULT_MODEL, VlmSuggestOptions, suggest_design_curves

    options = VlmSuggestOptions(
        model=str(payload.get("model") or DEFAULT_MODEL),
        device=str(payload.get("device") or "auto"),
        local_files_only=bool(payload.get("local_files_only", False)),
        max_curves=max(1, min(int(payload.get("max_curves", 12)), 24)),
        max_points_per_curve=max(2, min(int(payload.get("max_points_per_curve", 12)), 24)),
        max_new_tokens=max(256, min(int(payload.get("max_new_tokens", 1800)), 4096)),
        snap_max_distance=float(payload.get("snap_max_distance", 96.0)),
    )
    result = suggest_design_curves(
        session.image_path,
        router=session.router,
        output_dir=output_dir,
        options=options,
        progress=progress,
    )
    design_curves: list[dict[str, Any]] = []
    for index, item in enumerate(result.get("curves", [])):
        manual_points = item.get("manual_points", [])
        closed = bool(item.get("closed", False))
        route = _route_points_with_choices(
            session,
            manual_points,
            closed=closed,
            branch_choices=[],
            candidate_count=2,
        )
        design_curves.append(
            {
                "id": f"ai_curve_{int(time.time() * 1000):x}_{index:03d}",
                "type": "manual_design_curve",
                "semantic": item.get("semantic", "detail_line"),
                "edge_ids": [],
                "manual_points": manual_points,
                "cut_points": manual_points,
                "closed": closed,
                "routed_points": route.get("points", []),
                "route_segments": [
                    _clean_server_route_segment(segment)
                    for segment in route.get("segments", [])
                ],
                "branch_choices": [0 for _ in route.get("segments", [])],
                "route_ok": bool(route.get("ok")),
                "source": "qwen_vl_suggestion",
                "confidence": item.get("confidence", 0.5),
                "reason": item.get("reason", ""),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        )
    return {
        "ok": True,
        "model": result.get("model"),
        "curve_count": len(design_curves),
        "curves": design_curves,
        "context_image": result.get("context_image"),
    }


def _prune_ai_jobs(
    ai_jobs: dict[str, dict[str, Any]],
    ai_job_lock: threading.Lock,
    *,
    max_age_seconds: float = 6 * 60 * 60,
    max_jobs: int = 30,
) -> None:
    now = time.time()
    with ai_job_lock:
        old = [
            job_id
            for job_id, job in ai_jobs.items()
            if now - float(job.get("updated_at", now)) > max_age_seconds
        ]
        for job_id in old:
            ai_jobs.pop(job_id, None)
        if len(ai_jobs) > max_jobs:
            ordered = sorted(
                ai_jobs.items(),
                key=lambda item: float(item[1].get("updated_at", 0.0)),
            )
            for job_id, _job in ordered[: max(0, len(ai_jobs) - max_jobs)]:
                ai_jobs.pop(job_id, None)


def _clean_server_route_segment(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(segment.get("ok")),
        "points": segment.get("points", []),
        "segment_index": int(segment.get("segment_index", 0)),
        "selected_candidate": int(segment.get("selected_candidate", 0)),
        "length": float(segment.get("length", 0.0) or 0.0),
    }


def _fit_preview_segments(
    route_segments: list[dict[str, Any]],
    degree: int | str = "auto",
    *,
    closed: bool = False,
    high_quality: bool = False,
) -> dict[str, Any]:
    parsed_degree = _parse_preview_degree(degree)
    validator = ClassAValidator()
    previews: list[dict[str, Any]] = []
    if high_quality and len(route_segments) >= 1:
        chain_preview = _fit_preview_g2_chain(route_segments, parsed_degree, closed=closed, validator=validator)
        if chain_preview is not None:
            return chain_preview
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
            warnings = list(report.warnings)
            warnings.append("快速预览：最终导出会再做高质量 G2/CV 优化")
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
                    "warnings": warnings,
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


def _fit_preview_g2_chain(
    route_segments: list[dict[str, Any]],
    degree: int | str,
    *,
    closed: bool,
    validator: ClassAValidator,
) -> dict[str, Any] | None:
    try:
        from autoalias.review.fit_reviewed import (
            _fit_design_curve_chain,
        )

        parsed_segments: list[dict[str, Any]] = []
        segment_count = len(route_segments)
        for index, segment in enumerate(route_segments):
            points = remove_duplicate_points(_as_points3(segment.get("points") or []), eps=0.5)
            points = _ensure_preview_points(points)
            if len(points) < 4:
                return None
            parsed_segments.append(
                {
                    "points": points,
                    "start_order": index,
                    "end_order": 0 if bool(closed) and index == segment_count - 1 else index + 1,
                    "segment_count": segment_count,
                }
            )
        fitted = _fit_design_curve_chain(
            {
                "id": "fit_preview",
                "semantic": "preview",
                "closed": bool(closed),
                "manual_points": [{} for _ in range(max(2, segment_count + (0 if closed else 1)))],
            },
            Path("fit_preview.topology_corrections.json"),
            1,
            parsed_segments,
            degree,
            validator,
        )
        previews: list[dict[str, Any]] = []
        for index, (curve, _candidate, report) in enumerate(fitted):
            samples = evaluate_bezier(curve.cvs, np.linspace(0.0, 1.0, 120), curve.weights)
            warnings = list(report.warnings)
            warnings.append("Export-matched CV preview")
            merge_count = curve.metadata.get("chain_merged_segment_count")
            original_count = curve.metadata.get("chain_original_segment_count")
            if merge_count is not None and original_count is not None and merge_count != original_count:
                warnings.append(f"preview uses export merge: {original_count} -> {merge_count}")
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
                    "warnings": warnings,
                    "metrics": report.metrics,
                }
            )
        return {
            "ok": True,
            "segments": previews,
            "segment_count": len(previews),
            "passed_count": sum(1 for item in previews if item.get("passed")),
            "continuity": "export-matched",
        }
    except Exception:
        return None


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
.progress { height:10px; border-radius:999px; overflow:hidden; background:#e7ebe8; border:1px solid #d3dad5; }
.progress > div { height:100%; width:0%; background:linear-gradient(90deg,#0b6dff,#28b875); transition:width .35s ease; }
.subtle { color:var(--muted); font-size:12px; }
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
          <option value="roofline">车顶线</option>
          <option value="lamp">灯具轮廓</option>
          <option value="bumper">保险杠/裙边</option>
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

      <h2>AI 辅助分线</h2>
      <div class="stack">
        <button class="warn" id="btnAiSuggest">AI 建议分线点</button>
        <div class="box">
          <div id="aiBox">AI 会看原图和红色完整骨架，自动生成可编辑的分线点；生成后你可以继续拖拽、删除、保存和导出。</div>
          <div id="aiProgressWrap" class="hidden" style="margin-top:8px">
            <div class="progress"><div id="aiProgressFill"></div></div>
            <div class="subtle" id="aiProgressText" style="margin-top:6px">0%</div>
          </div>
        </div>
      </div>

      <h2>显示</h2>
      <div class="grid2">
        <button class="primary" id="btnFullSkeleton">隐藏完整骨架</button>
        <button class="primary" id="btnSkeleton">隐藏切段骨架</button>
        <button class="primary" id="btnCvPreview">隐藏CV预览</button>
        <button id="btnG2Edit">G2 Edit OFF</button>
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
let editableFitPreview = null;
let fitPreviewRequestId = 0;
let branchChoices = [];
let closedCurve = false;
let activeCurve = null;
let showSkeleton = true;
let showFullSkeleton = true;
let showAliasPreview = true;
let showCvPreview = true;
let g2EditMode = false;
const G2_CONSTRAINTS_ENABLED = false;
let selectedCv = null;
let cvDragging = null;
let cvDragMoved = false;
let aliasOverrideDirty = false;
let transform = { scale: 1, x: 0, y: 0 };
let dragging = false;
let lastMouse = null;
let pointDraggingIndex = null;
let pointDragMoved = false;
let suppressNextClick = false;
let snapRequestId = 0;
let aiSuggestRequestId = 0;
let aiPollTimer = null;

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

function v2(p) { return [Number(p[0] || 0), Number(p[1] || 0), Number(p[2] || 0)]; }
function vAdd(a, b) { return [a[0] + b[0], a[1] + b[1], (a[2] || 0) + (b[2] || 0)]; }
function vSub(a, b) { return [a[0] - b[0], a[1] - b[1], (a[2] || 0) - (b[2] || 0)]; }
function vScale(a, s) { return [a[0] * s, a[1] * s, (a[2] || 0) * s]; }
function clonePoint(p) { return [round3(Number(p[0] || 0)), round3(Number(p[1] || 0)), round3(Number(p[2] || 0))]; }

function cloneFitPreview(src) {
  if (!src || !src.segments) return null;
  return JSON.parse(JSON.stringify(src));
}

function activeFitPreview() {
  return editableFitPreview || fitPreview;
}

function clearEditableFitPreview() {
  editableFitPreview = null;
  selectedCv = null;
  cvDragging = null;
  cvDragMoved = false;
}

function ensureEditableFitPreview() {
  if (!editableFitPreview) {
    if (!fitPreview || !fitPreview.segments) return false;
    editableFitPreview = cloneFitPreview(fitPreview);
    refreshEditableSamples();
  }
  return true;
}

function sampleBezier(cvs, steps=120) {
  if (!cvs || cvs.length < 2) return [];
  const degree = cvs.length - 1;
  const samples = [];
  for (let s = 0; s < steps; s++) {
    const u = steps <= 1 ? 0 : s / (steps - 1);
    const work = cvs.map(v2);
    for (let r = 1; r <= degree; r++) {
      for (let i = 0; i <= degree - r; i++) {
        work[i] = [
          work[i][0] * (1 - u) + work[i + 1][0] * u,
          work[i][1] * (1 - u) + work[i + 1][1] * u,
          work[i][2] * (1 - u) + work[i + 1][2] * u
        ];
      }
    }
    samples.push([round3(work[0][0]), round3(work[0][1])]);
  }
  return samples;
}

function refreshEditableSamples() {
  if (!editableFitPreview || !editableFitPreview.segments) return;
  for (const seg of editableFitPreview.segments) {
    if (!seg || !seg.cvs || seg.cvs.length < 2) continue;
    seg.samples = sampleBezier(seg.cvs, 120);
    seg.span = 1;
    seg.ok = true;
  }
}

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
  const preview = activeFitPreview();
  if (!preview || !preview.segments) return;
  for (let segIndex = 0; segIndex < preview.segments.length; segIndex++) {
    const seg = preview.segments[segIndex];
    if (!seg || !seg.ok) continue;
    const samples = editableFitPreview && seg.cvs ? sampleBezier(seg.cvs, 120) : seg.samples;
    if (!samples || !samples.length) continue;
    drawPolyline(samples, seg.color || "#14a05a", 4.0, 0.95);
    if (showCvPreview) drawCvPreview(seg, segIndex);
  }
}

function drawCvPreview(seg, segIndex) {
  const cvs = seg.cvs || [];
  if (!cvs || cvs.length < 2) return;
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.setLineDash([7 / transform.scale, 5 / transform.scale]);
  ctx.strokeStyle = "rgba(255,135,0,.92)";
  ctx.lineWidth = Math.max(1.55 / transform.scale, 0.75);
  ctx.beginPath();
  ctx.moveTo(cvs[0][0], cvs[0][1]);
  for (let i = 1; i < cvs.length; i++) ctx.lineTo(cvs[i][0], cvs[i][1]);
  ctx.stroke();
  ctx.setLineDash([]);

  const radius = Math.max(5.2 / transform.scale, 2.8);
  const fontSize = Math.max(10.5 / transform.scale, 5.8);
  for (let i = 0; i < cvs.length; i++) {
    const p = cvs[i];
    const isSelected = selectedCv && selectedCv.segIndex === segIndex && selectedCv.cvIndex === i;
    ctx.beginPath();
    ctx.fillStyle = isSelected ? "#2b84ff" : (i === 0 || i === cvs.length - 1 ? "#ffcf33" : "#fff35c");
    ctx.strokeStyle = isSelected ? "#ffffff" : "#6b4a00";
    ctx.lineWidth = Math.max((isSelected ? 2.6 : 1.45) / transform.scale, 0.7);
    ctx.arc(p[0], p[1], isSelected ? radius * 1.25 : radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#332400";
    ctx.font = `${fontSize}px sans-serif`;
    ctx.fillText(String(i), p[0] + radius * 1.25, p[1] - radius * 0.9);
  }

  if (seg.degree != null) {
    const p = cvs[Math.floor(cvs.length / 2)];
    ctx.fillStyle = "rgba(20,20,20,.72)";
    ctx.font = `${Math.max(12 / transform.scale, 6.5)}px sans-serif`;
    ctx.fillText(`d${seg.degree} · ${cvs.length}CV`, p[0] + radius * 1.4, p[1] + radius * 1.7);
  }
  ctx.restore();
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

function pickCv(wx, wy) {
  const preview = activeFitPreview();
  if (!preview || !preview.segments) return null;
  const threshold = Math.max(13 / transform.scale, 4.5);
  let best = null;
  for (let segIndex = 0; segIndex < preview.segments.length; segIndex++) {
    const cvs = preview.segments[segIndex].cvs || [];
    for (let cvIndex = 0; cvIndex < cvs.length; cvIndex++) {
      const p = cvs[cvIndex];
      const d = Math.hypot(wx - p[0], wy - p[1]);
      if (d <= threshold && (!best || d < best.d)) best = { segIndex, cvIndex, d };
    }
  }
  return best;
}

function nextSegmentIndex(segIndex) {
  const preview = activeFitPreview();
  if (!preview || !preview.segments || !preview.segments.length) return null;
  if (segIndex + 1 < preview.segments.length) return segIndex + 1;
  return closedCurve && preview.segments.length > 1 ? 0 : null;
}

function prevSegmentIndex(segIndex) {
  const preview = activeFitPreview();
  if (!preview || !preview.segments || !preview.segments.length) return null;
  if (segIndex > 0) return segIndex - 1;
  return closedCurve && preview.segments.length > 1 ? preview.segments.length - 1 : null;
}

function enforceJoinFromLeft(leftIndex) {
  if (!editableFitPreview || !editableFitPreview.segments) return;
  const rightIndex = nextSegmentIndex(leftIndex);
  if (rightIndex == null) return;
  const left = editableFitPreview.segments[leftIndex];
  const right = editableFitPreview.segments[rightIndex];
  const lc = left && left.cvs ? left.cvs : [];
  const rc = right && right.cvs ? right.cvs : [];
  if (lc.length < 4 || rc.length < 4) return;
  const pL = Math.max(1, Number(left.degree || lc.length - 1));
  const pR = Math.max(1, Number(right.degree || rc.length - 1));
  const n = lc.length - 1;
  const end = v2(lc[n]);
  const d1 = vScale(vSub(v2(lc[n]), v2(lc[n - 1])), pL);
  const d2 = vScale(vAdd(vSub(v2(lc[n]), vScale(v2(lc[n - 1]), 2)), v2(lc[n - 2])), pL * (pL - 1));
  rc[0] = clonePoint(end);
  rc[1] = clonePoint(vAdd(end, vScale(d1, 1 / pR)));
  rc[2] = clonePoint(vAdd(vSub(vScale(v2(rc[1]), 2), v2(rc[0])), vScale(d2, 1 / (pR * (pR - 1)))));
}

function enforceJoinFromRight(rightIndex) {
  if (!editableFitPreview || !editableFitPreview.segments) return;
  const leftIndex = prevSegmentIndex(rightIndex);
  if (leftIndex == null) return;
  const left = editableFitPreview.segments[leftIndex];
  const right = editableFitPreview.segments[rightIndex];
  const lc = left && left.cvs ? left.cvs : [];
  const rc = right && right.cvs ? right.cvs : [];
  if (lc.length < 4 || rc.length < 4) return;
  const pL = Math.max(1, Number(left.degree || lc.length - 1));
  const pR = Math.max(1, Number(right.degree || rc.length - 1));
  const n = lc.length - 1;
  const start = v2(rc[0]);
  const d1 = vScale(vSub(v2(rc[1]), v2(rc[0])), pR);
  const d2 = vScale(vAdd(vSub(v2(rc[2]), vScale(v2(rc[1]), 2)), v2(rc[0])), pR * (pR - 1));
  lc[n] = clonePoint(start);
  lc[n - 1] = clonePoint(vSub(start, vScale(d1, 1 / pL)));
  lc[n - 2] = clonePoint(vAdd(vSub(vScale(v2(lc[n - 1]), 2), v2(lc[n])), vScale(d2, 1 / (pL * (pL - 1)))));
}

function applyG2ConstraintFromCv(segIndex, cvIndex) {
  const preview = editableFitPreview;
  if (!preview || !preview.segments || !preview.segments[segIndex]) return;
  const cvs = preview.segments[segIndex].cvs || [];
  const n = cvs.length - 1;
  if (n < 3) return;
  let startZone = cvIndex <= 2;
  let endZone = n - cvIndex <= 2;
  if (startZone && endZone) {
    startZone = cvIndex <= n / 2;
    endZone = !startZone;
  }
  if (startZone) enforceJoinFromRight(segIndex);
  if (endZone) enforceJoinFromLeft(segIndex);
  refreshEditableSamples();
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
  aliasOverrideDirty = true;
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

function buildAliasOverrides() {
  if (!G2_CONSTRAINTS_ENABLED) return [];
  if (!g2EditMode || !editableFitPreview || !editableFitPreview.segments) return [];
  refreshEditableSamples();
  const out = [];
  for (let i = 0; i < editableFitPreview.segments.length; i++) {
    const seg = editableFitPreview.segments[i];
    const cvs = seg && seg.cvs ? seg.cvs : [];
    if (cvs.length < 4) continue;
    const degree = Number(seg.degree || cvs.length - 1);
    if (degree < 3 || degree > 7 || cvs.length !== degree + 1) continue;
    out.push({
      segment_index: i,
      degree,
      span: 1,
      cvs: cvs.map(p => [round3(Number(p[0] || 0)), round3(Number(p[1] || 0)), round3(Number(p[2] || 0))]),
      source: "dynamic_g2_cv_editor"
    });
  }
  return out;
}

function overridesToFitPreview(curve) {
  const overrides = curve && curve.alias_curve_overrides ? curve.alias_curve_overrides : [];
  if (!overrides.length) return null;
  const segments = [];
  for (let i = 0; i < overrides.length; i++) {
    const ov = overrides[i] || {};
    const cvs = (ov.cvs || ov.cv || []).map(clonePoint);
    const degree = Number(ov.degree || cvs.length - 1);
    if (degree < 3 || degree > 7 || cvs.length !== degree + 1) continue;
    segments.push({
      ok: true,
      segment_index: i,
      degree,
      span: 1,
      samples: sampleBezier(cvs, 120),
      cvs,
      passed: true,
      color: "#14a05a",
      warnings: ["Saved G2 constrained CV override"],
      metrics: {}
    });
  }
  if (!segments.length) return null;
  return {
    ok: true,
    segments,
    segment_count: segments.length,
    passed_count: segments.length,
    continuity: "dynamic-g2-cv-editor"
  };
}

function restoreAliasOverrides(curve) {
  if (!G2_CONSTRAINTS_ENABLED) return false;
  const preview = overridesToFitPreview(curve);
  if (!preview) return false;
  fitPreview = preview;
  editableFitPreview = cloneFitPreview(preview);
  g2EditMode = true;
  selectedCv = null;
  aliasOverrideDirty = false;
  return true;
}

async function toggleG2Edit() {
  if (!G2_CONSTRAINTS_ENABLED) {
    g2EditMode = false;
    clearEditableFitPreview();
    setStatus("G2 constraint is disabled. 当前只调试基础拟合。");
    updatePanel();
    render();
    return;
  }
  if (g2EditMode) {
    g2EditMode = false;
    clearEditableFitPreview();
    setStatus("G2 Edit OFF");
    updatePanel();
    render();
    return;
  }
  if (!fitPreview && routePreview && routePreview.segments && routePreview.segments.length) {
    await refreshFitPreview();
  }
  if (!ensureEditableFitPreview()) {
    setStatus("请先生成蓝色分段线，再打开 G2 Edit");
    return;
  }
  g2EditMode = true;
  showCvPreview = true;
  setStatus("G2 Edit ON：拖动连接端附近的 CV，会联动相邻曲线的 G0/G1/G2 CV");
  updatePanel();
  render();
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
  aliasOverrideDirty = true;
  routePreview = null;
  fitPreview = null;
  clearEditableFitPreview();
  routeStatus = "正在拖动分线点，松开后重新生成蓝线";
  updatePanel();
  render();
}

async function refreshRoutePreview() {
  const requestId = ++routeRequestId;
  clearEditableFitPreview();
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
    clearEditableFitPreview();
    routeStatus = "路径生成失败";
  }
  updatePanel();
  render();
}

async function refreshFitPreview() {
  const requestId = ++fitPreviewRequestId;
  if (!routePreview || !routePreview.segments || routePreview.segments.length < 1) {
    fitPreview = null;
    clearEditableFitPreview();
    return;
  }
  try {
    const res = await fetch(api("/api/fit-preview"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        route_segments: routePreview.segments.map(cleanRouteSegment),
        closed: closedCurve,
        degree: document.getElementById("degree").value,
        quality: "export"
      })
    });
    const result = await res.json();
    if (requestId !== fitPreviewRequestId) return;
    fitPreview = result.ok ? result : null;
    if (g2EditMode && fitPreview) {
      editableFitPreview = cloneFitPreview(fitPreview);
      refreshEditableSamples();
    } else if (!g2EditMode) {
      clearEditableFitPreview();
    }
  } catch (_err) {
    if (requestId !== fitPreviewRequestId) return;
    fitPreview = null;
    clearEditableFitPreview();
  }
}

async function runAiSuggest() {
  if (!sid || !graph) return;
  const requestId = ++aiSuggestRequestId;
  const btn = document.getElementById("btnAiSuggest");
  btn.disabled = true;
  if (aiPollTimer) clearInterval(aiPollTimer);
  const startedAt = Date.now();
  setAiProgress(1, "正在创建 AI 后台任务", startedAt);
  try {
    const res = await fetch(api("/api/ai-suggest-start"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        max_curves: 12,
        max_points_per_curve: 12,
        snap_max_distance: 96
      })
    });
    const result = await res.json();
    if (requestId !== aiSuggestRequestId) return;
    if (!result.ok) {
      setAiProgress(100, "AI 建议失败：" + (result.error || "unknown"), startedAt);
      btn.disabled = false;
      return;
    }
    setAiProgress(3, "AI 后台任务已启动，正在等待进度", startedAt);
    aiPollTimer = setInterval(() => {
      pollAiSuggestJob(result.job_id, requestId, startedAt);
    }, 1000);
    await pollAiSuggestJob(result.job_id, requestId, startedAt);
  } catch (err) {
    if (requestId !== aiSuggestRequestId) return;
    setAiProgress(100, "AI 建议失败：" + (err && err.message ? err.message : String(err)), startedAt);
    btn.disabled = false;
  }
}

async function pollAiSuggestJob(jobId, requestId, startedAt) {
  const btn = document.getElementById("btnAiSuggest");
  try {
    const res = await fetch(api("/api/ai-suggest-status") + "&job=" + encodeURIComponent(jobId));
    const result = await res.json();
    if (requestId !== aiSuggestRequestId) return;
    if (!result.ok) {
      setAiProgress(100, "AI 状态读取失败：" + (result.error || "unknown"), startedAt);
      if (aiPollTimer) clearInterval(aiPollTimer);
      btn.disabled = false;
      return;
    }
    const rawProgress = Number(result.progress || 0);
    const elapsed = Math.max(0, (Date.now() - startedAt) / 1000);
    let displayProgress = rawProgress;
    if (result.status === "running" || result.status === "queued") {
      const softCeiling = rawProgress < 62 ? 61 : 88;
      displayProgress = Math.max(rawProgress, Math.min(softCeiling, rawProgress + elapsed * 0.6));
    }
    setAiProgress(displayProgress, result.message || "AI 正在处理", startedAt);
    if (result.status === "failed") {
      if (aiPollTimer) clearInterval(aiPollTimer);
      setAiProgress(100, "AI 建议失败：" + (result.error || result.message || "unknown"), startedAt);
      btn.disabled = false;
      return;
    }
    if (result.status !== "done") return;
    if (aiPollTimer) clearInterval(aiPollTimer);
    const curves = result.result && result.result.curves ? result.result.curves : [];
    if (!curves.length) {
      setAiProgress(100, "AI 没有返回可用曲线。可以换更清晰图片，或者先打开完整骨架确认线条是否被提取。", startedAt);
      btn.disabled = false;
      return;
    }
    for (const curve of curves) designCurves.push(curve);
    await saveAll();
    setAiProgress(100, `AI 已生成 ${curves.length} 条候选曲线，已经加入下面的曲线列表。`, startedAt);
    const box = document.getElementById("aiBox");
    box.innerHTML += `<br>${result.result.model || ""}<br>${result.result.context_image || ""}`;
    updatePanel();
    render();
    btn.disabled = false;
  } catch (err) {
    if (requestId !== aiSuggestRequestId) return;
    setAiProgress(100, "AI 状态读取失败：" + (err && err.message ? err.message : String(err)), startedAt);
    if (aiPollTimer) clearInterval(aiPollTimer);
    btn.disabled = false;
  }
}

function setAiProgress(percent, message, startedAt) {
  const wrap = document.getElementById("aiProgressWrap");
  const fill = document.getElementById("aiProgressFill");
  const text = document.getElementById("aiProgressText");
  const box = document.getElementById("aiBox");
  const p = Math.max(0, Math.min(100, Math.round(percent)));
  wrap.classList.remove("hidden");
  fill.style.width = p + "%";
  const elapsed = startedAt ? Math.round((Date.now() - startedAt) / 1000) : 0;
  box.textContent = message;
  text.textContent = `${p}% · 已等待 ${elapsed}s`;
}

async function saveCurrent(startNext=false) {
  if (cutPoints.length < 2) return;
  if (!routePreview || !routePreview.points || routePreview.points.length < 2) await refreshRoutePreview();
  const previous = designCurves.find(c => c.id === activeCurve) || null;
  const aliasOverrides = buildAliasOverrides();
  const preservedAliasOverrides = previous && !aliasOverrideDirty ? (previous.alias_curve_overrides || []) : [];
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
    alias_curve_overrides: G2_CONSTRAINTS_ENABLED ? (aliasOverrides.length ? aliasOverrides : preservedAliasOverrides) : [],
    alias_constraint_mode: G2_CONSTRAINTS_ENABLED ? (aliasOverrides.length ? "dynamic_g2_cv_editor" : (preservedAliasOverrides.length ? ((previous && previous.alias_constraint_mode) || "dynamic_g2_cv_editor") : "")) : "",
    created_at: new Date().toISOString()
  };
  const idx = designCurves.findIndex(c => c.id === item.id);
  if (idx >= 0) designCurves[idx] = item;
  else designCurves.push(item);
  activeCurve = item.id;
  await saveAll();
  aliasOverrideDirty = false;
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
  clearEditableFitPreview();
  g2EditMode = false;
  routeStatus = "";
  branchChoices = [];
  closedCurve = false;
  activeCurve = null;
  aliasOverrideDirty = false;
  updatePanel();
  render();
}

function undoPoint() {
  if (!cutPoints.length) return;
  cutPoints.pop();
  aliasOverrideDirty = true;
  selectedCutIndex = cutPoints.length ? cutPoints.length - 1 : null;
  if (cutPoints.length < 3) closedCurve = false;
  normalizeBranchChoices();
  refreshRoutePreview();
}

function deleteSelected() {
  if (selectedCutIndex == null) return;
  cutPoints.splice(selectedCutIndex, 1);
  aliasOverrideDirty = true;
  selectedCutIndex = cutPoints.length ? Math.min(selectedCutIndex, cutPoints.length - 1) : null;
  if (cutPoints.length < 3) closedCurve = false;
  normalizeBranchChoices();
  refreshRoutePreview();
}

function toggleClosed() {
  if (cutPoints.length < 3) return;
  closedCurve = !closedCurve;
  aliasOverrideDirty = true;
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
      aliasOverrideDirty = false;
      routePreview = { ok: !!curve.route_ok, points: curve.routed_points || [], segments: curve.route_segments || [] };
      routeStatus = routePreview.points.length ? `已加载蓝线：${routePreview.points.length} 个骨架点` : "";
      document.getElementById("semantic").value = curve.semantic || "detail_line";
      selectedCutIndex = null;
      const restoredCv = restoreAliasOverrides(curve);
      if (!restoredCv) clearEditableFitPreview();
      updatePanel();
      render();
      if (!routePreview.points.length) {
        refreshRoutePreview();
      } else if (!restoredCv) {
        refreshFitPreview().then(() => { updatePanel(); render(); });
      }
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
  document.getElementById("btnCvPreview").classList.toggle("primary", showCvPreview);
  document.getElementById("btnCvPreview").textContent = showCvPreview ? "隐藏CV预览" : "显示CV预览";
  document.getElementById("btnG2Edit").classList.toggle("primary", G2_CONSTRAINTS_ENABLED && g2EditMode);
  document.getElementById("btnG2Edit").disabled = !G2_CONSTRAINTS_ENABLED;
  document.getElementById("btnG2Edit").textContent = G2_CONSTRAINTS_ENABLED ? (g2EditMode ? "G2 Edit ON" : "G2 Edit OFF") : "G2 Disabled";
  document.getElementById("btnSave").disabled = cutPoints.length < 2;
  document.getElementById("btnSaveNext").disabled = cutPoints.length < 2;
  document.getElementById("btnUndo").disabled = cutPoints.length < 1;
  document.getElementById("btnDelete").disabled = selectedCutIndex == null;
  document.getElementById("btnClose").disabled = cutPoints.length < 3;
  document.getElementById("btnAiSuggest").disabled = !graph;
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
      aliasOverrideDirty = true;
      refreshRoutePreview();
    };
  }
}

function renderQualityPreview() {
  const box = document.getElementById("qualityBox");
  const preview = activeFitPreview();
  if (!preview || !preview.segments || preview.segments.length === 0) {
    box.innerHTML = "暂无拟合质量";
    return;
  }
  const rows = preview.segments.map((seg, i) => {
    const color = seg.color || "#888";
    const status = seg.passed ? "通过" : "警告";
    const degree = seg.degree ? `d${seg.degree}` : "未拟合";
    const warn = seg.warnings && seg.warnings.length ? seg.warnings.slice(0, 2).join("；") : "曲率/CV 检查正常";
    return `<div style="border-left:6px solid ${color};padding:4px 0 5px 8px;margin:4px 0"><strong>第 ${i + 1} 段 ${status} ${degree}</strong><br><small>${warn}</small></div>`;
  });
  if (g2EditMode) rows.unshift(`<div style="border-left:6px solid #2b84ff;padding:4px 0 5px 8px;margin:4px 0"><strong>G2 Edit ON</strong><br><small>导出会优先使用当前受约束 CV，不再重新拟合覆盖。</small></div>`);
  box.innerHTML = rows.join("");
}

function makeId() { return "curve_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 7); }

canvas.addEventListener("mousedown", e => {
  if (graph && e.button === 0 && !e.altKey) {
    const rect = canvas.getBoundingClientRect();
    const [wx, wy] = screenToWorld(e.clientX - rect.left, e.clientY - rect.top);
    if (g2EditMode && ensureEditableFitPreview()) {
      const cvHit = pickCv(wx, wy);
      if (cvHit) {
        cvDragging = cvHit;
        selectedCv = cvHit;
        cvDragMoved = false;
        updatePanel();
        render();
        e.preventDefault();
        return;
      }
    }
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
  if (cvDragging != null && editableFitPreview && editableFitPreview.segments) {
    const rect = canvas.getBoundingClientRect();
    const [wx, wy] = screenToWorld(e.clientX - rect.left, e.clientY - rect.top);
    const seg = editableFitPreview.segments[cvDragging.segIndex];
    const cvs = seg && seg.cvs ? seg.cvs : null;
    if (cvs && cvs[cvDragging.cvIndex]) {
      cvs[cvDragging.cvIndex] = [round3(wx), round3(wy), Number(cvs[cvDragging.cvIndex][2] || 0)];
      selectedCv = { segIndex: cvDragging.segIndex, cvIndex: cvDragging.cvIndex };
      cvDragMoved = true;
      suppressNextClick = true;
      applyG2ConstraintFromCv(cvDragging.segIndex, cvDragging.cvIndex);
      render();
    }
    e.preventDefault();
    return;
  }
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
  if (cvDragging != null) {
    if (cvDragMoved) {
      suppressNextClick = true;
      setStatus("G2 constrained CV updated. 点击保存当前后会写入 JSON/IGES。");
      updatePanel();
    }
    cvDragging = null;
    cvDragMoved = false;
  }
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
  if (g2EditMode && ensureEditableFitPreview()) {
    const cvHit = pickCv(wx, wy);
    if (cvHit) {
      selectedCv = cvHit;
      updatePanel();
      render();
      return;
    }
  }
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
document.getElementById("btnCvPreview").onclick = () => { showCvPreview = !showCvPreview; updatePanel(); render(); };
document.getElementById("btnG2Edit").onclick = toggleG2Edit;
document.getElementById("degree").onchange = () => { refreshFitPreview().then(() => { updatePanel(); render(); }); };
document.getElementById("btnAiSuggest").onclick = runAiSuggest;
document.getElementById("btnReset").onclick = resetView;
document.getElementById("btnExport").onclick = exportIges;
resize();
render();
</script>
</body>
</html>"""
