from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from PySide6.QtCore import QObject, QPointF, Qt, QThread, QUrl, Signal
    from PySide6.QtGui import (
        QAction,
        QBrush,
        QColor,
        QDesktopServices,
        QImage,
        QKeySequence,
        QPainter,
        QPainterPath,
        QPen,
        QPixmap,
    )
    from PySide6.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QGraphicsEllipseItem,
        QGraphicsItem,
        QGraphicsPathItem,
        QGraphicsPixmapItem,
        QGraphicsScene,
        QGraphicsSimpleTextItem,
        QGraphicsView,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QSplitter,
        QStatusBar,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - only hit on missing GUI deps.
    raise SystemExit(
        "PySide6 is not installed. Install GUI dependencies first:\n"
        "  F:\\ComfyUI\\.venv\\Scripts\\python.exe -m pip install PySide6\n"
        "Then run:\n"
        "  F:\\430AutoAlias\\scripts\\autoalias_gui.cmd"
    ) from exc

from autoalias.review.fit_reviewed import ReviewedFitResult, fit_reviewed_annotations
from autoalias.review.auto_segment import suggest_geometry_segments
from autoalias.review.graph import ReviewGraphOptions
from autoalias.review.server import ReviewSession
from autoalias.review.workflow_server import _edit_session_skeleton


DEFAULT_GUI_SEMANTIC = "detail_line"


def _round3(value: float) -> float:
    return round(float(value), 3)


def _curve_id() -> str:
    return "gui_curve_" + time.strftime("%Y%m%d_%H%M%S_") + f"{int(time.time() * 1000) % 1000:03d}"


def _point_dict(
    point: QPointF,
    order: int,
    snap_distance: float | None = None,
    *,
    snap_source: str = "gui_skeleton",
    anchor_curve_id: str = "",
    anchor_point_order: int | None = None,
    anchor_semantic: str = "",
) -> dict[str, Any]:
    return {
        "x": _round3(point.x()),
        "y": _round3(point.y()),
        "order": int(order),
        "snap_distance": None if snap_distance is None else _round3(snap_distance),
        "snap_source": snap_source,
        "anchor_curve_id": anchor_curve_id,
        "anchor_point_order": anchor_point_order,
        "anchor_semantic": anchor_semantic,
    }


def _path_from_points(points: list[Any]) -> QPainterPath:
    path = QPainterPath()
    if not points:
        return path
    first = points[0]
    path.moveTo(float(first[0]), float(first[1]))
    for point in points[1:]:
        path.lineTo(float(point[0]), float(point[1]))
    return path


def _clean_route_segment(segment: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "ok": bool(segment.get("ok")),
        "points": segment.get("points", []),
        "segment_index": int(segment.get("segment_index", index)),
        "selected_candidate": int(segment.get("selected_candidate", 0)),
        "length": float(segment.get("length", 0.0) or 0.0),
        "alternatives": segment.get("alternatives", []),
    }


def _safe_choice(choices: list[int], index: int, candidate_count: int) -> int:
    if candidate_count <= 0:
        return 0
    try:
        value = int(choices[index])
    except Exception:
        value = 0
    return max(0, min(value, candidate_count - 1))


@dataclass(slots=True)
class LoadedSession:
    session: ReviewSession
    image: QImage
    skeleton_edits: list[dict[str, Any]]


def _read_skeleton_edits(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    edits = data.get("skeleton_edits", [])
    return list(edits) if isinstance(edits, list) else []


def _replay_skeleton_edits(session: ReviewSession, edits: list[dict[str, Any]]) -> None:
    for edit in edits:
        try:
            _edit_session_skeleton(session, edit)
        except Exception:
            continue


class SessionWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        image_path: Path,
        output_dir: Path,
        extraction_mode: str,
        parallel_collapse: str,
    ) -> None:
        super().__init__()
        self.image_path = image_path
        self.output_dir = output_dir
        self.extraction_mode = extraction_mode
        self.parallel_collapse = parallel_collapse

    def run(self) -> None:
        try:
            options = ReviewGraphOptions(
                extraction_mode=self.extraction_mode,
                parallel_collapse=self.parallel_collapse,
                max_points_per_edge=480,
            )
            session = ReviewSession.create(self.image_path, self.output_dir, options)
            skeleton_edits = _read_skeleton_edits(session.corrections_path)
            _replay_skeleton_edits(session, skeleton_edits)
            image = QImage(str(self.image_path))
            if image.isNull():
                raise FileNotFoundError(f"cannot load image: {self.image_path}")
            self.finished.emit(
                LoadedSession(session=session, image=image, skeleton_edits=skeleton_edits)
            )
        except Exception as exc:  # pragma: no cover - GUI worker path.
            self.failed.emit(str(exc))


class ExportWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        annotation_path: Path,
        output_dir: Path,
        degree: int | str,
    ) -> None:
        super().__init__()
        self.annotation_path = annotation_path
        self.output_dir = output_dir
        self.degree = degree

    def run(self) -> None:
        try:
            result = fit_reviewed_annotations(
                [self.annotation_path],
                self.output_dir,
                degree=self.degree,
                min_points=4,
                max_fit_points=180,
                diagnostic_preview=False,
                fast_mode=True,
            )
            self.finished.emit(result)
        except Exception as exc:  # pragma: no cover - GUI worker path.
            self.failed.emit(str(exc))


class CutPointItem(QGraphicsEllipseItem):
    def __init__(self, window: "DesktopEditor", index: int, point: QPointF) -> None:
        radius = 6.0
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.window = window
        self.index = index
        self.setPos(point)
        self.setZValue(50)
        self.setBrush(QBrush(QColor(116, 87, 255)))
        self.setPen(QPen(QColor("white"), 1.6))
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.label = QGraphicsSimpleTextItem(str(index + 1), self)
        self.label.setBrush(QBrush(QColor(30, 36, 35)))
        self.label.setPos(8, -18)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.window.select_cut_point(self.index)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().mouseMoveEvent(event)
        self.window.move_cut_point_free(self.index, self.pos())

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().mouseReleaseEvent(event)
        self.window.finish_cut_point_drag(self.index, self.pos())

    def set_index(self, index: int) -> None:
        self.index = index
        self.label.setText(str(index + 1))

    def set_selected_style(self, selected: bool) -> None:
        self.setBrush(QBrush(QColor(255, 230, 106) if selected else QColor(116, 87, 255)))


class EditorView(QGraphicsView):
    def __init__(self, window: "DesktopEditor", scene: QGraphicsScene) -> None:
        super().__init__(scene)
        self.window = window
        self._panning = False
        self._last_pan = QPointF()
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.AltModifier
        ):
            self._panning = True
            self._last_pan = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self.window.skeleton_edit_mode.isChecked():
                scene_pos = self.mapToScene(event.position().toPoint())
                self.window.edit_skeleton(scene_pos)
                event.accept()
                return
            item = self.itemAt(event.position().toPoint())
            if item is None or not isinstance(item, CutPointItem):
                scene_pos = self.mapToScene(event.position().toPoint())
                self.window.add_cut_point(scene_pos)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._panning:
            delta = event.position() - self._last_pan
            self._last_pan = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class DesktopEditor(QMainWindow):
    def __init__(self, output_dir: Path) -> None:
        super().__init__()
        self.setWindowTitle("AutoAlias Desktop Editor")
        self.resize(1480, 900)
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session: ReviewSession | None = None
        self.current_image: QImage | None = None
        self.image_item: QGraphicsPixmapItem | None = None
        self.full_skeleton_item: QGraphicsPixmapItem | None = None
        self.edge_skeleton_item: QGraphicsPixmapItem | None = None
        self.current_route_item: QGraphicsPathItem | None = None
        self.alt_route_items: list[QGraphicsPathItem] = []
        self.saved_route_items: list[QGraphicsPathItem] = []
        self.connection_items: list[QGraphicsItem] = []
        self.point_items: list[CutPointItem] = []
        self.cut_points: list[dict[str, Any]] = []
        self.design_curves: list[dict[str, Any]] = []
        self.route_preview: dict[str, Any] | None = None
        self.branch_choices: list[int] = []
        self.closed_curve = False
        self.active_curve_id: str | None = None
        self.selected_cut_index: int | None = None
        self._worker_thread: QThread | None = None
        self._worker: SessionWorker | None = None
        self._export_thread: QThread | None = None
        self._export_worker: ExportWorker | None = None
        self.last_export_dir: Path | None = None
        self.last_skeleton_edit_index: int | None = None
        self.skeleton_edits: list[dict[str, Any]] = []

        self.scene = QGraphicsScene(self)
        self.view = EditorView(self, self.scene)
        self.curve_list = QListWidget()
        self.curve_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.curve_list.currentItemChanged.connect(self._load_curve_from_item)
        self.branch_list = QListWidget()
        self.branch_list.currentRowChanged.connect(self._branch_selection_changed)

        self.degree = QComboBox()
        self.degree.addItems(["auto", "3", "5", "7"])

        self.extraction_mode = QComboBox()
        self.extraction_mode.addItems(
            ["auto", "white_on_black_sketch", "black_on_white_line_art", "canny_edges"]
        )

        self.parallel_collapse = QComboBox()
        self.parallel_collapse.addItems(["off", "soft", "medium", "strong"])

        self.auto_segment_mode = QComboBox()
        self.auto_segment_mode.addItem("主线模式", "main")
        self.auto_segment_mode.addItem("连续覆盖模式", "coverage")
        self.auto_segment_mode.addItem("局部细节模式", "detail")
        self.auto_segment_mode.addItem("全量骨架模式", "full")
        self.auto_segment_mode.setToolTip(
            "主线模式更保守；连续覆盖会跨小断口追长线；局部细节会提取更多短线；全量骨架会尽量把所有可追踪骨架都变成候选曲线。"
        )

        self.snap_radius = QSpinBox()
        self.snap_radius.setRange(1, 9999)
        self.snap_radius.setValue(36)
        self.snap_radius.setSuffix(" px")

        self.show_image = QCheckBox("原图")
        self.show_image.setChecked(True)
        self.show_image.toggled.connect(self._update_visibility)
        self.show_full_skeleton = QCheckBox("完整骨架红点")
        self.show_full_skeleton.setChecked(True)
        self.show_full_skeleton.toggled.connect(self._update_visibility)
        self.show_edge_skeleton = QCheckBox("切段骨架绿线")
        self.show_edge_skeleton.setChecked(True)
        self.show_edge_skeleton.toggled.connect(self._update_visibility)
        self.show_saved = QCheckBox("已保存蓝线")
        self.show_saved.setChecked(True)
        self.show_saved.toggled.connect(self._update_visibility)

        self.skeleton_edit_mode = QCheckBox("骨架修补")
        self.skeleton_edit_tool = QComboBox()
        self.skeleton_edit_tool.addItems(["add", "delete"])
        self.skeleton_edit_radius = QSpinBox()
        self.skeleton_edit_radius.setRange(4, 240)
        self.skeleton_edit_radius.setValue(24)
        self.skeleton_edit_radius.setSuffix(" px")

        self.last_export_label = QLabel("最近导出：无")

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._build_ui()
        self._install_shortcuts()
        self._set_status("打开一张图片开始。左键添加分段点，拖动点后松手自动吸附骨架。")

    def _build_ui(self) -> None:
        toolbar = QToolBar("AutoAlias")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("打开图片", self)
        open_action.triggered.connect(self.open_image_dialog)
        toolbar.addAction(open_action)
        open_project_action = QAction("打开工程", self)
        open_project_action.triggered.connect(self.open_project_dialog)
        toolbar.addAction(open_project_action)
        save_project_action = QAction("保存工程", self)
        save_project_action.triggered.connect(self.save_project)
        toolbar.addAction(save_project_action)
        save_project_as_action = QAction("工程另存为", self)
        save_project_as_action.triggered.connect(self.save_project_as)
        toolbar.addAction(save_project_as_action)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("提取 "))
        toolbar.addWidget(self.extraction_mode)
        toolbar.addWidget(QLabel(" 并线 "))
        toolbar.addWidget(self.parallel_collapse)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("吸附 "))
        toolbar.addWidget(self.snap_radius)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.addWidget(QLabel("导出 Degree"))
        side_layout.addWidget(self.degree)
        side_layout.addWidget(self.show_image)
        side_layout.addWidget(self.show_full_skeleton)
        side_layout.addWidget(self.show_edge_skeleton)
        side_layout.addWidget(self.show_saved)

        skeleton_row = QHBoxLayout()
        skeleton_row.addWidget(self.skeleton_edit_mode)
        skeleton_row.addWidget(self.skeleton_edit_tool)
        skeleton_row.addWidget(self.skeleton_edit_radius)
        side_layout.addLayout(skeleton_row)
        skeleton_break_btn = QPushButton("断开连续加点")
        skeleton_break_btn.clicked.connect(self.break_skeleton_add_chain)
        side_layout.addWidget(skeleton_break_btn)

        row1 = QHBoxLayout()
        save_btn = QPushButton("保存当前")
        save_btn.clicked.connect(lambda: self.save_current(start_next=False))
        save_next_btn = QPushButton("保存并下一条")
        save_next_btn.clicked.connect(lambda: self.save_current(start_next=True))
        row1.addWidget(save_btn)
        row1.addWidget(save_next_btn)
        side_layout.addLayout(row1)

        row2 = QHBoxLayout()
        undo_btn = QPushButton("撤回点")
        undo_btn.clicked.connect(self.undo_point)
        delete_btn = QPushButton("删除点")
        delete_btn.clicked.connect(self.delete_selected_point)
        row2.addWidget(undo_btn)
        row2.addWidget(delete_btn)
        side_layout.addLayout(row2)

        row3 = QHBoxLayout()
        close_btn = QPushButton("闭合开关")
        close_btn.clicked.connect(self.toggle_closed)
        clear_btn = QPushButton("清空当前")
        clear_btn.clicked.connect(self.clear_current)
        row3.addWidget(close_btn)
        row3.addWidget(clear_btn)
        side_layout.addLayout(row3)

        side_layout.addWidget(QLabel("几何自动分段模式"))
        side_layout.addWidget(self.auto_segment_mode)
        auto_btn = QPushButton("几何自动分段")
        auto_btn.clicked.connect(self.run_geometry_auto_segment)
        side_layout.addWidget(auto_btn)

        side_layout.addWidget(QLabel("分支/多路径候选"))
        side_layout.addWidget(self.branch_list)
        branch_row = QHBoxLayout()
        prev_branch_btn = QPushButton("上一候选")
        prev_branch_btn.clicked.connect(lambda: self.shift_branch_choice(-1))
        next_branch_btn = QPushButton("下一候选")
        next_branch_btn.clicked.connect(lambda: self.shift_branch_choice(1))
        branch_row.addWidget(prev_branch_btn)
        branch_row.addWidget(next_branch_btn)
        side_layout.addLayout(branch_row)

        self.export_btn = QPushButton("导出 IGES")
        self.export_btn.clicked.connect(self.export_iges)
        side_layout.addWidget(self.export_btn)
        open_export_btn = QPushButton("打开最近导出文件夹")
        open_export_btn.clicked.connect(self.open_last_export_dir)
        side_layout.addWidget(open_export_btn)
        side_layout.addWidget(self.last_export_label)

        side_layout.addWidget(QLabel("曲线列表"))
        curve_action_row = QHBoxLayout()
        select_all_btn = QPushButton("全选曲线")
        select_all_btn.clicked.connect(self.select_all_curves)
        delete_curve_btn = QPushButton("批量删除选中")
        delete_curve_btn.clicked.connect(self.delete_selected_curve)
        curve_action_row.addWidget(select_all_btn)
        curve_action_row.addWidget(delete_curve_btn)
        side_layout.addLayout(curve_action_row)
        side_layout.addWidget(self.curve_list, stretch=1)

        splitter = QSplitter()
        splitter.addWidget(self.view)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

    def _install_shortcuts(self) -> None:
        undo = QAction(self)
        undo.setShortcut(QKeySequence.StandardKey.Undo)
        undo.triggered.connect(self.undo_point)
        self.addAction(undo)
        delete = QAction(self)
        delete.setShortcut(QKeySequence.StandardKey.Delete)
        delete.triggered.connect(self.delete_selected_point)
        self.addAction(delete)

    def open_image_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择汽车线稿/图片",
            str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*.*)",
        )
        if path:
            self.load_image(Path(path))

    def open_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开 AutoAlias 工程 JSON",
            str(self.output_dir),
            "AutoAlias JSON (*.json);;All Files (*.*)",
        )
        if path:
            self.open_project(Path(path))

    def open_project(self, project_path: Path) -> None:
        try:
            data = json.loads(project_path.read_text(encoding="utf-8"))
            image_path = Path(str(data.get("graph", {}).get("image") or ""))
            if not image_path.exists():
                raise FileNotFoundError(f"工程里的图片路径不存在：{image_path}")
            session = ReviewSession.create(
                image_path,
                project_path.parent,
                ReviewGraphOptions(
                    extraction_mode=self.extraction_mode.currentText(),
                    parallel_collapse=self.parallel_collapse.currentText(),
                    max_points_per_edge=480,
                ),
            )
            session.corrections_path = project_path.resolve()
            session.corrections = list(data.get("corrections", []))
            session.design_curves = list(data.get("design_curves", []))
            skeleton_edits = data.get("skeleton_edits", [])
            skeleton_edits = list(skeleton_edits) if isinstance(skeleton_edits, list) else []
            _replay_skeleton_edits(session, skeleton_edits)
            image = QImage(str(image_path))
            if image.isNull():
                raise FileNotFoundError(f"cannot load image: {image_path}")
            self._session_loaded(
                LoadedSession(session=session, image=image, skeleton_edits=skeleton_edits)
            )
            self._set_status(f"已打开工程：{project_path}")
        except Exception as exc:
            QMessageBox.critical(self, "打开工程失败", str(exc))

    def save_project(self) -> None:
        if self.session is None:
            return
        if len(self.cut_points) >= 2:
            self.save_current(start_next=False)
        self._save_all()
        self._set_status(f"工程已保存：{self.session.corrections_path}")

    def save_project_as(self) -> None:
        if self.session is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "工程另存为",
            str(self.output_dir / f"{self.session.image_path.stem}.topology_corrections.json"),
            "AutoAlias JSON (*.json);;All Files (*.*)",
        )
        if not path:
            return
        if len(self.cut_points) >= 2:
            self.save_current(start_next=False)
        self.session.corrections_path = Path(path).resolve()
        self._save_all()
        self._set_status(f"工程已另存为：{self.session.corrections_path}")

    def load_image(self, image_path: Path) -> None:
        if self._worker_thread is not None:
            QMessageBox.information(self, "AutoAlias", "正在提取上一张图片，请稍等。")
            return
        self._set_status("正在提取骨架，GUI 不会阻塞。")
        worker = SessionWorker(
            image_path=image_path,
            output_dir=self.output_dir,
            extraction_mode=self.extraction_mode.currentText(),
            parallel_collapse=self.parallel_collapse.currentText(),
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._session_loaded)
        worker.failed.connect(self._session_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_worker)
        self._worker = worker
        self._worker_thread = thread
        thread.start()

    def _clear_worker(self) -> None:
        self._worker = None
        self._worker_thread = None

    def _session_failed(self, message: str) -> None:
        QMessageBox.critical(self, "AutoAlias", message)
        self._set_status("图片加载失败。")

    def _session_loaded(self, loaded: LoadedSession) -> None:
        self.session = loaded.session
        self.current_image = loaded.image
        self.skeleton_edits = list(loaded.skeleton_edits)
        self.design_curves = list(loaded.session.design_curves)
        self.cut_points = []
        self.route_preview = None
        self.branch_choices = []
        self.closed_curve = False
        self.active_curve_id = None
        self.selected_cut_index = None

        self.scene.clear()
        self.point_items.clear()
        self.alt_route_items.clear()
        self.saved_route_items.clear()
        self.connection_items.clear()
        self.current_route_item = None
        self.image_item = self.scene.addPixmap(QPixmap.fromImage(loaded.image))
        self.image_item.setZValue(0)
        self.full_skeleton_item = self.scene.addPixmap(
            self._make_full_skeleton_pixmap(loaded.image, loaded.session)
        )
        self.full_skeleton_item.setZValue(5)
        self.edge_skeleton_item = self.scene.addPixmap(
            self._make_edge_skeleton_pixmap(loaded.image, loaded.session)
        )
        self.edge_skeleton_item.setZValue(6)
        self.scene.setSceneRect(0, 0, loaded.image.width(), loaded.image.height())
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._rebuild_saved_routes()
        self._refresh_curve_list()
        self._update_visibility()
        self._set_status(
            f"已加载：{loaded.session.image_path.name}，骨架点 {len(loaded.session.router.coords)}。"
        )

    def _transparent_image_like(self, image_source: QImage) -> QImage:
        image = QImage(
            image_source.width(),
            image_source.height(),
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.fill(Qt.GlobalColor.transparent)
        return image

    def _make_full_skeleton_pixmap(self, image_source: QImage, session: ReviewSession) -> QPixmap:
        image = self._transparent_image_like(image_source)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(QPen(QColor(220, 0, 0, 135), 1))
        for x, y in session.router.coords:
            painter.drawPoint(int(x), int(y))
        painter.end()
        return QPixmap.fromImage(image)

    def _make_edge_skeleton_pixmap(self, image_source: QImage, session: ReviewSession) -> QPixmap:
        image = self._transparent_image_like(image_source)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(QPen(QColor(0, 140, 115, 130), 1.2))
        for edge in session.graph.get("edges", []):
            points = edge.get("points") or []
            if len(points) < 2:
                continue
            painter.drawPath(_path_from_points(points))
        painter.end()
        return QPixmap.fromImage(image)

    def refresh_skeleton_layer(self) -> None:
        if self.session is None or self.current_image is None:
            return
        full_pixmap = self._make_full_skeleton_pixmap(self.current_image, self.session)
        edge_pixmap = self._make_edge_skeleton_pixmap(self.current_image, self.session)
        if self.full_skeleton_item is None:
            self.full_skeleton_item = self.scene.addPixmap(full_pixmap)
            self.full_skeleton_item.setZValue(5)
        else:
            self.full_skeleton_item.setPixmap(full_pixmap)
        if self.edge_skeleton_item is None:
            self.edge_skeleton_item = self.scene.addPixmap(edge_pixmap)
            self.edge_skeleton_item.setZValue(6)
        else:
            self.edge_skeleton_item.setPixmap(edge_pixmap)
        self._update_visibility()

    def break_skeleton_add_chain(self) -> None:
        self.last_skeleton_edit_index = None
        self._set_status("已断开连续骨架加点，下次添加会重新寻找附近骨架连接。")

    def edit_skeleton(self, scene_pos: QPointF) -> None:
        if self.session is None:
            return
        action = self.skeleton_edit_tool.currentText()
        radius = float(self.skeleton_edit_radius.value())
        payload = {
            "action": action,
            "point": {"x": scene_pos.x(), "y": scene_pos.y()},
            "connect_radius": radius,
            "delete_radius": radius,
            "link_radius": max(radius * 2.0, 96.0),
            "link_index": self.last_skeleton_edit_index,
        }
        try:
            result = _edit_session_skeleton(self.session, payload)
        except Exception as exc:
            QMessageBox.critical(self, "骨架修补失败", str(exc))
            return
        if not result.get("ok"):
            self._set_status(str(result.get("reason") or "骨架修补失败"))
            return
        self.skeleton_edits.append(payload)
        if action == "add":
            self.last_skeleton_edit_index = int(result.get("index", -1))
            self._set_status(f"已添加骨架点 {self.last_skeleton_edit_index}。")
        else:
            self.last_skeleton_edit_index = None
            self._set_status(f"已删除骨架点 {result.get('deleted_index')}。")
        self.refresh_skeleton_layer()
        if len(self.cut_points) >= 2:
            self._update_route_preview()

    def add_cut_point(self, scene_pos: QPointF) -> None:
        if self.session is None:
            return
        snapped = self._snap_point(scene_pos, require_radius=True)
        if snapped is None:
            self._set_status("附近没有骨架点。可以调大吸附半径。")
            return
        index = len(self.cut_points)
        self.cut_points.append(self._point_from_snap(snapped, index))
        self._normalize_branch_choices()
        self.selected_cut_index = index
        item = CutPointItem(self, index, snapped["point"])
        self.point_items.append(item)
        self.scene.addItem(item)
        self._update_point_styles()
        self._update_route_preview()
        if snapped.get("source") == "saved_curve_anchor":
            self._set_status(
                f"已吸附到已保存曲线点 {int(snapped.get('anchor_point_order', 0)) + 1}。"
            )

    def move_cut_point_free(self, index: int, scene_pos: QPointF) -> None:
        if index < 0 or index >= len(self.cut_points):
            return
        self.cut_points[index]["x"] = _round3(scene_pos.x())
        self.cut_points[index]["y"] = _round3(scene_pos.y())
        self.cut_points[index]["snap_distance"] = None
        self.cut_points[index]["snap_source"] = "free_drag_pending"
        self.cut_points[index]["anchor_curve_id"] = ""
        self.cut_points[index]["anchor_point_order"] = None
        self.cut_points[index]["anchor_semantic"] = ""
        self.selected_cut_index = index
        self._normalize_branch_choices()
        self._rebuild_connection_markers()
        self._update_point_styles()

    def finish_cut_point_drag(self, index: int, scene_pos: QPointF) -> None:
        if index < 0 or index >= len(self.cut_points):
            return
        snapped = self._snap_point(scene_pos, require_radius=False)
        if snapped is not None:
            self.cut_points[index] = self._point_from_snap(snapped, index)
            self.point_items[index].setPos(snapped["point"])
            if snapped.get("source") == "saved_curve_anchor":
                self._set_status(
                    f"已吸附到已保存曲线点 {int(snapped.get('anchor_point_order', 0)) + 1}。"
                )
        else:
            self.cut_points[index] = _point_dict(scene_pos, index, None)
        self._maybe_merge_dragged_point(index)
        self._normalize_branch_choices()
        self._rebuild_connection_markers()
        self._update_route_preview()

    def _snap_point(
        self,
        point: QPointF,
        *,
        require_radius: bool,
    ) -> dict[str, Any] | None:
        saved_anchor = self._snap_to_saved_anchor(point)
        if saved_anchor is not None:
            return saved_anchor
        if self.session is None or len(self.session.router.coords) == 0:
            return None
        idx, distance = self.session.router.nearest_index((point.x(), point.y()))
        max_distance = float(self.snap_radius.value())
        if require_radius and distance > max_distance:
            return None
        if distance > max_distance and max_distance < 9999:
            return None
        x, y = self.session.router.coords[idx]
        return {
            "point": QPointF(float(x), float(y)),
            "distance": float(distance),
            "source": "gui_skeleton",
        }

    def _point_from_snap(self, snapped: dict[str, Any], order: int) -> dict[str, Any]:
        return _point_dict(
            snapped["point"],
            order,
            float(snapped.get("distance", 0.0)),
            snap_source=str(snapped.get("source") or "gui_skeleton"),
            anchor_curve_id=str(snapped.get("anchor_curve_id") or ""),
            anchor_point_order=snapped.get("anchor_point_order"),
            anchor_semantic=str(snapped.get("anchor_semantic") or ""),
        )

    def _saved_anchor_radius(self) -> float:
        return max(4.0, float(self.snap_radius.value()))

    def _snap_to_saved_anchor(self, point: QPointF) -> dict[str, Any] | None:
        radius = self._saved_anchor_radius()
        best: dict[str, Any] | None = None
        for curve in self.design_curves:
            curve_id = str(curve.get("id") or "")
            if self.active_curve_id and curve_id == self.active_curve_id:
                continue
            points = curve.get("manual_points") or curve.get("cut_points") or []
            for order, saved_point in enumerate(points):
                try:
                    x = float(saved_point.get("x"))
                    y = float(saved_point.get("y"))
                except Exception:
                    continue
                distance = ((point.x() - x) ** 2 + (point.y() - y) ** 2) ** 0.5
                if distance > radius:
                    continue
                if best is None or distance < float(best["distance"]):
                    best = {
                        "point": QPointF(x, y),
                        "distance": float(distance),
                        "source": "saved_curve_anchor",
                        "anchor_curve_id": curve_id,
                        "anchor_point_order": int(saved_point.get("order", order) or order),
                        "anchor_semantic": str(curve.get("semantic") or DEFAULT_GUI_SEMANTIC),
                    }
        return best

    def _maybe_merge_dragged_point(self, index: int) -> None:
        if index < 0 or index >= len(self.cut_points):
            return
        p = self.cut_points[index]
        threshold = 10.0
        target: int | None = None
        best = float("inf")
        for i, q in enumerate(self.cut_points):
            if i == index:
                continue
            d = ((float(p["x"]) - float(q["x"])) ** 2 + (float(p["y"]) - float(q["y"])) ** 2) ** 0.5
            if d < threshold and d < best:
                target = i
                best = d
        if target is None:
            return
        answer = QMessageBox.question(
            self,
            "合并分段点",
            f"分段点 {index + 1} 已靠近分段点 {target + 1}，是否合并？",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.cut_points.pop(index)
        self._normalize_branch_choices()
        self._rebuild_point_items()

    def select_cut_point(self, index: int) -> None:
        self.selected_cut_index = index
        self._update_point_styles()

    def _update_point_styles(self) -> None:
        for i, item in enumerate(self.point_items):
            item.set_selected_style(i == self.selected_cut_index)

    def _rebuild_connection_markers(self) -> None:
        for item in self.connection_items:
            self.scene.removeItem(item)
        self.connection_items.clear()
        for point in self.cut_points:
            if point.get("snap_source") != "saved_curve_anchor":
                continue
            x = float(point.get("x", 0.0))
            y = float(point.get("y", 0.0))
            ring = QGraphicsEllipseItem(x - 11.0, y - 11.0, 22.0, 22.0)
            ring.setPen(QPen(QColor(0, 190, 210), 2.2))
            ring.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            ring.setZValue(48)
            ring.setToolTip(
                f"连接到曲线 {point.get('anchor_curve_id', '')} "
                f"点 {point.get('anchor_point_order', '')}"
            )
            self.scene.addItem(ring)
            self.connection_items.append(ring)
            label = QGraphicsSimpleTextItem("link")
            label.setBrush(QBrush(QColor(0, 118, 130)))
            label.setPos(x + 12.0, y + 8.0)
            label.setZValue(49)
            self.scene.addItem(label)
            self.connection_items.append(label)

    def _rebuild_point_items(self) -> None:
        for item in self.point_items:
            self.scene.removeItem(item)
        self.point_items.clear()
        for i, point in enumerate(self.cut_points):
            point["order"] = i
            item = CutPointItem(self, i, QPointF(float(point["x"]), float(point["y"])))
            self.point_items.append(item)
            self.scene.addItem(item)
        if self.selected_cut_index is not None and self.selected_cut_index >= len(self.cut_points):
            self.selected_cut_index = len(self.cut_points) - 1 if self.cut_points else None
        self._update_point_styles()
        self._rebuild_connection_markers()

    def _expected_segment_count(self) -> int:
        if len(self.cut_points) < 2:
            return 0
        count = len(self.cut_points) - 1
        if self.closed_curve and len(self.cut_points) >= 3:
            count += 1
        return count

    def _normalize_branch_choices(self) -> None:
        expected = self._expected_segment_count()
        while len(self.branch_choices) < expected:
            self.branch_choices.append(0)
        if len(self.branch_choices) > expected:
            self.branch_choices = self.branch_choices[:expected]

    def _route_points_with_choices(self) -> dict[str, Any]:
        if self.session is None:
            return {"ok": False, "reason": "no session", "segments": [], "points": []}
        clean = [(float(p["x"]), float(p["y"])) for p in self.cut_points]
        if len(clean) < 2:
            return {"ok": False, "reason": "need at least two points", "segments": [], "points": []}
        self._normalize_branch_choices()
        pairs = list(zip(clean, clean[1:]))
        is_closed = bool(self.closed_curve and len(clean) >= 3)
        if is_closed:
            pairs.append((clean[-1], clean[0]))
        segments: list[dict[str, Any]] = []
        combined: list[list[float]] = []
        all_ok = True
        for index, (start, end) in enumerate(pairs):
            candidates = self.session.router.route_candidates(start, end, count=4)
            choice = _safe_choice(self.branch_choices, index, len(candidates))
            chosen = candidates[choice] if candidates else {
                "ok": False,
                "points": [[start[0], start[1]], [end[0], end[1]]],
            }
            segment_points = chosen.get("points") or [[start[0], start[1]], [end[0], end[1]]]
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

    def _update_route_preview(self) -> None:
        if self.session is None:
            return
        if self.current_route_item is not None:
            self.scene.removeItem(self.current_route_item)
            self.current_route_item = None
        for item in self.alt_route_items:
            self.scene.removeItem(item)
        self.alt_route_items.clear()
        if len(self.cut_points) < 2:
            self.route_preview = None
            self._refresh_branch_list()
            return
        self.route_preview = self._route_points_with_choices()
        points = self.route_preview.get("points") or []
        if points:
            item = QGraphicsPathItem(_path_from_points(points))
            item.setPen(QPen(QColor(0, 109, 255), 2.4))
            item.setZValue(30)
            self.scene.addItem(item)
            self.current_route_item = item
        self._draw_branch_alternatives()
        self._refresh_branch_list()
        ok = bool(self.route_preview.get("ok"))
        self._set_status(
            f"蓝线路径：{len(points)} 点，{'连通' if ok else '未完全连通'}。"
        )

    def _draw_branch_alternatives(self) -> None:
        if not self.route_preview:
            return
        segment_index = self.branch_list.currentRow()
        segments = self.route_preview.get("segments") or []
        if segment_index < 0 or segment_index >= len(segments):
            return
        segment = segments[segment_index]
        selected = int(segment.get("selected_candidate", 0))
        colors = [QColor(245, 128, 32, 125), QColor(170, 80, 220, 115), QColor(0, 150, 190, 110)]
        for alt_index, alternative in enumerate(segment.get("alternatives") or []):
            if alt_index == selected:
                continue
            points = alternative.get("points") or []
            if len(points) < 2:
                continue
            item = QGraphicsPathItem(_path_from_points(points))
            pen = QPen(colors[alt_index % len(colors)], 1.2)
            pen.setStyle(Qt.PenStyle.DashLine)
            item.setPen(pen)
            item.setZValue(25)
            self.scene.addItem(item)
            self.alt_route_items.append(item)

    def _refresh_branch_list(self) -> None:
        current_row = self.branch_list.currentRow()
        self.branch_list.blockSignals(True)
        self.branch_list.clear()
        segments = self.route_preview.get("segments") if self.route_preview else []
        for index, segment in enumerate(segments or []):
            alternatives = segment.get("alternatives") or []
            selected = int(segment.get("selected_candidate", 0)) + 1
            count = len(alternatives) if alternatives else 1
            length = float(segment.get("length", 0.0) or 0.0)
            status = "OK" if segment.get("ok") else "断"
            item = QListWidgetItem(f"段 {index + 1}: 候选 {selected}/{count} / {length:.0f}px / {status}")
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.branch_list.addItem(item)
        if self.branch_list.count():
            row = current_row if 0 <= current_row < self.branch_list.count() else 0
            self.branch_list.setCurrentRow(row)
        self.branch_list.blockSignals(False)

    def _branch_selection_changed(self, _row: int) -> None:
        if self.route_preview is None:
            return
        for item in self.alt_route_items:
            self.scene.removeItem(item)
        self.alt_route_items.clear()
        self._draw_branch_alternatives()

    def shift_branch_choice(self, delta: int) -> None:
        if self.route_preview is None:
            return
        index = self.branch_list.currentRow()
        segments = self.route_preview.get("segments") or []
        if index < 0 or index >= len(segments):
            return
        alternatives = segments[index].get("alternatives") or []
        if len(alternatives) <= 1:
            self._set_status(f"段 {index + 1} 没有其他候选路径。")
            return
        self._normalize_branch_choices()
        self.branch_choices[index] = (self.branch_choices[index] + delta) % len(alternatives)
        self._update_route_preview()

    def save_current(self, *, start_next: bool) -> None:
        if self.session is None or len(self.cut_points) < 2:
            return
        if self.route_preview is None:
            self._update_route_preview()
        route = self.route_preview or {}
        route_segments = [
            _clean_route_segment(segment, i)
            for i, segment in enumerate(route.get("segments") or [])
        ]
        item = {
            "id": self.active_curve_id or _curve_id(),
            "type": "manual_design_curve",
            "semantic": DEFAULT_GUI_SEMANTIC,
            "edge_ids": [],
            "manual_points": [dict(p, order=i) for i, p in enumerate(self.cut_points)],
            "cut_points": [dict(p, order=i) for i, p in enumerate(self.cut_points)],
            "closed": bool(self.closed_curve),
            "routed_points": route.get("points") or [],
            "route_segments": route_segments,
            "branch_choices": self.branch_choices[:],
            "route_ok": bool(route.get("ok")),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source": "autoalias_desktop_gui",
        }
        idx = next((i for i, curve in enumerate(self.design_curves) if curve.get("id") == item["id"]), -1)
        if idx >= 0:
            self.design_curves[idx] = item
        else:
            self.design_curves.append(item)
        self.active_curve_id = item["id"]
        self._save_all()
        self._rebuild_saved_routes()
        self._refresh_curve_list()
        self._set_status(f"已保存曲线：{item['id']}。")
        if start_next:
            self.clear_current()

    def _save_all(self) -> None:
        if self.session is None:
            return
        self.session.save([], self.design_curves)
        try:
            data = json.loads(self.session.corrections_path.read_text(encoding="utf-8"))
            data["skeleton_edits"] = self.skeleton_edits
            self.session.corrections_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _geometry_auto_segment_params(self) -> tuple[str, str, dict[str, Any]]:
        if self.session is None:
            return "main", "主线模式", {}
        mode = str(self.auto_segment_mode.currentData() or "main")
        image_size = self.session.graph.get("image_size", {}) or {}
        width = float(image_size.get("width", 0.0) or 0.0)
        height = float(image_size.get("height", 0.0) or 0.0)
        diag = max(1.0, (width * width + height * height) ** 0.5)

        if mode == "full":
            return (
                mode,
                "全量骨架模式",
                {
                    "max_curves": 420,
                    "min_length": max(2.0, diag * 0.0025),
                    "max_turn_deg": 68.0,
                    "max_junction_turn_deg": 44.0,
                    "max_chain_edges": 28,
                    "max_gap": max(6.0, diag * 0.006),
                    "max_gap_turn_deg": 58.0,
                },
            )
        if mode == "coverage":
            return (
                mode,
                "连续覆盖模式",
                {
                    "max_curves": 220,
                    "min_length": max(4.0, diag * 0.005),
                    "max_turn_deg": 54.0,
                    "max_junction_turn_deg": 36.0,
                    "max_chain_edges": 36,
                    "max_gap": max(8.0, diag * 0.009),
                    "max_gap_turn_deg": 50.0,
                },
            )
        if mode == "detail":
            return (
                mode,
                "局部细节模式",
                {
                    "max_curves": 160,
                    "min_length": max(5.0, diag * 0.006),
                    "max_turn_deg": 46.0,
                    "max_junction_turn_deg": 32.0,
                    "max_chain_edges": 14,
                    "max_gap": max(4.0, diag * 0.004),
                    "max_gap_turn_deg": 42.0,
                },
            )
        return (
            "main",
            "主线模式",
            {
                "max_curves": 32,
                "max_turn_deg": 28.0,
                "max_junction_turn_deg": 18.0,
                "max_chain_edges": 8,
            },
        )

    def run_geometry_auto_segment(self) -> None:
        if self.session is None:
            return
        mode, mode_label, params = self._geometry_auto_segment_params()
        answer = QMessageBox.question(
            self,
            "几何自动分段",
            f"将使用“{mode_label}”追加生成曲线，不会删除你已经保存的曲线。是否继续？",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            curves = suggest_geometry_segments(self.session.graph, **params)
        except Exception as exc:
            QMessageBox.critical(self, "几何自动分段失败", str(exc))
            return
        if not curves:
            QMessageBox.information(self, "几何自动分段", "没有生成可用曲线。")
            return
        existing_ids = {str(curve.get("id")) for curve in self.design_curves}
        added = 0
        for curve in curves:
            curve = dict(curve)
            while str(curve.get("id")) in existing_ids:
                curve["id"] = _curve_id()
            curve["semantic"] = DEFAULT_GUI_SEMANTIC
            curve["source"] = "geometry_auto_segment_desktop_gui"
            curve["auto_segment_mode"] = mode
            existing_ids.add(str(curve.get("id")))
            self.design_curves.append(curve)
            added += 1
        self._save_all()
        self._rebuild_saved_routes()
        self._refresh_curve_list()
        self._set_status(f"{mode_label}已添加 {added} 条曲线。选择曲线后可继续拖点和切换候选路径。")

    def _rebuild_saved_routes(self) -> None:
        for item in self.saved_route_items:
            self.scene.removeItem(item)
        self.saved_route_items.clear()
        for curve in self.design_curves:
            points = curve.get("routed_points") or []
            if len(points) < 2:
                continue
            item = QGraphicsPathItem(_path_from_points(points))
            item.setPen(QPen(QColor(11, 109, 255, 210), 1.8))
            item.setZValue(20)
            item.setVisible(self.show_saved.isChecked())
            self.scene.addItem(item)
            self.saved_route_items.append(item)

    def _refresh_curve_list(self) -> None:
        self.curve_list.blockSignals(True)
        self.curve_list.clear()
        for curve in self.design_curves:
            points = curve.get("manual_points") or []
            label = f"{len(points)} 点 / {curve.get('id', '')}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, curve.get("id"))
            self.curve_list.addItem(item)
            if curve.get("id") == self.active_curve_id:
                self.curve_list.setCurrentItem(item)
        self.curve_list.blockSignals(False)

    def _load_curve_from_item(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        curve_id = current.data(Qt.ItemDataRole.UserRole)
        curve = next((item for item in self.design_curves if item.get("id") == curve_id), None)
        if curve is None:
            return
        self.active_curve_id = str(curve_id)
        self.cut_points = [dict(point) for point in (curve.get("manual_points") or curve.get("cut_points") or [])]
        self.closed_curve = bool(curve.get("closed"))
        self.branch_choices = [int(value) for value in (curve.get("branch_choices") or [])]
        self._normalize_branch_choices()
        self.route_preview = {
            "ok": bool(curve.get("route_ok")),
            "points": curve.get("routed_points") or [],
            "segments": curve.get("route_segments") or [],
        }
        self._rebuild_point_items()
        self._update_route_preview()

    def select_all_curves(self) -> None:
        self.curve_list.selectAll()

    def _selected_curve_ids(self) -> set[str]:
        selected = self.curve_list.selectedItems()
        if not selected and self.curve_list.currentItem() is not None:
            selected = [self.curve_list.currentItem()]
        ids: set[str] = set()
        for item in selected:
            curve_id = item.data(Qt.ItemDataRole.UserRole)
            if curve_id is not None:
                ids.add(str(curve_id))
        return ids

    def delete_selected_curve(self) -> None:
        curve_ids = self._selected_curve_ids()
        if not curve_ids:
            return
        answer = QMessageBox.question(
            self,
            "批量删除曲线",
            f"确认删除选中的 {len(curve_ids)} 条已保存曲线？",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        active_deleted = self.active_curve_id in curve_ids if self.active_curve_id else False
        self.design_curves = [
            curve for curve in self.design_curves if str(curve.get("id")) not in curve_ids
        ]
        if active_deleted:
            self.cut_points = []
            self.route_preview = None
            self.branch_choices = []
            self.closed_curve = False
            self.active_curve_id = None
            self.selected_cut_index = None
            if self.current_route_item is not None:
                self.scene.removeItem(self.current_route_item)
                self.current_route_item = None
            for item in self.alt_route_items:
                self.scene.removeItem(item)
            self.alt_route_items.clear()
            self._rebuild_point_items()
            self._refresh_branch_list()
        self._save_all()
        self._rebuild_saved_routes()
        self._refresh_curve_list()
        self._set_status(f"已删除 {len(curve_ids)} 条曲线。")

    def undo_point(self) -> None:
        if not self.cut_points:
            return
        self.cut_points.pop()
        self._normalize_branch_choices()
        self._rebuild_point_items()
        self._update_route_preview()

    def delete_selected_point(self) -> None:
        if self.selected_cut_index is None:
            return
        if 0 <= self.selected_cut_index < len(self.cut_points):
            self.cut_points.pop(self.selected_cut_index)
        self._normalize_branch_choices()
        self._rebuild_point_items()
        self._update_route_preview()

    def clear_current(self) -> None:
        self.cut_points = []
        self.route_preview = None
        self.closed_curve = False
        self.active_curve_id = None
        self.selected_cut_index = None
        if self.current_route_item is not None:
            self.scene.removeItem(self.current_route_item)
            self.current_route_item = None
        for item in self.alt_route_items:
            self.scene.removeItem(item)
        self.alt_route_items.clear()
        for item in self.connection_items:
            self.scene.removeItem(item)
        self.connection_items.clear()
        self._rebuild_point_items()
        self._refresh_branch_list()
        self._refresh_curve_list()
        self._set_status("当前曲线已清空。")

    def toggle_closed(self) -> None:
        self.closed_curve = not self.closed_curve
        self._normalize_branch_choices()
        self._update_route_preview()
        self._set_status("闭合已开启。" if self.closed_curve else "闭合已关闭。")

    def export_iges(self) -> None:
        if self.session is None:
            return
        if self._export_thread is not None:
            QMessageBox.information(self, "AutoAlias", "正在导出上一批 IGES，请稍等。")
            return
        if len(self.cut_points) >= 2:
            self.save_current(start_next=False)
        if not self.design_curves:
            QMessageBox.information(self, "AutoAlias", "还没有保存曲线。")
            return
        degree_text = self.degree.currentText()
        degree: int | str = degree_text if degree_text == "auto" else int(degree_text)
        out = self.output_dir / "alias_exports" / (
            self.session.image_path.stem + "_desktop_" + time.strftime("%Y%m%d_%H%M%S")
        )
        worker = ExportWorker(self.session.corrections_path, out, degree)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._export_finished)
        worker.failed.connect(self._export_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_export_worker)
        self._export_worker = worker
        self._export_thread = thread
        self.export_btn.setEnabled(False)
        self._set_status(f"正在后台导出 IGES：{out}")
        thread.start()

    def _export_finished(self, result: ReviewedFitResult) -> None:
        self.last_export_dir = result.out
        self.last_export_label.setText(f"最近导出：{result.out}")
        self._set_status(f"IGES 导出完成：{result.out}")
        QMessageBox.information(
            self,
            "导出完成",
            f"曲线：{len(result.curves)}\n"
            f"通过：{sum(1 for report in result.reports if report.passed)}/{len(result.reports)}\n"
            f"IGES：{result.out / 'reviewed_curves.igs'}\n"
            f"SVG：{result.out / 'reviewed_clean_preview.svg'}",
        )

    def _export_failed(self, message: str) -> None:
        QMessageBox.critical(self, "导出失败", message)
        self._set_status("IGES 导出失败。")

    def _clear_export_worker(self) -> None:
        self._export_worker = None
        self._export_thread = None
        self.export_btn.setEnabled(True)

    def open_last_export_dir(self) -> None:
        if self.last_export_dir is None:
            QMessageBox.information(self, "AutoAlias", "还没有导出版本。")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_export_dir)))

    def _update_visibility(self) -> None:
        if self.image_item is not None:
            self.image_item.setVisible(self.show_image.isChecked())
        if self.full_skeleton_item is not None:
            self.full_skeleton_item.setVisible(self.show_full_skeleton.isChecked())
        if self.edge_skeleton_item is not None:
            self.edge_skeleton_item.setVisible(self.show_edge_skeleton.isChecked())
        for item in self.saved_route_items:
            item.setVisible(self.show_saved.isChecked())

    def _set_status(self, message: str) -> None:
        self.status.showMessage(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoAlias desktop skeleton segmentation GUI.")
    parser.add_argument("image", nargs="?", type=Path, help="optional image to open on startup")
    parser.add_argument("--out", type=Path, default=Path("lan_reviews"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = QApplication(sys.argv[:1])
    window = DesktopEditor(args.out)
    window.show()
    if args.image:
        window.load_image(args.image)
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
