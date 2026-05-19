from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from PySide6.QtCore import QObject, QPointF, Qt, QThread, Signal
    from PySide6.QtGui import (
        QAction,
        QBrush,
        QColor,
        QImage,
        QKeySequence,
        QPainter,
        QPainterPath,
        QPen,
        QPixmap,
    )
    from PySide6.QtWidgets import (
        QApplication,
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

from autoalias.review.fit_reviewed import fit_reviewed_annotations
from autoalias.review.graph import ReviewGraphOptions
from autoalias.review.server import ReviewSession


def _round3(value: float) -> float:
    return round(float(value), 3)


def _curve_id() -> str:
    return "gui_curve_" + time.strftime("%Y%m%d_%H%M%S_") + f"{int(time.time() * 1000) % 1000:03d}"


def _point_dict(point: QPointF, order: int, snap_distance: float | None = None) -> dict[str, Any]:
    return {
        "x": _round3(point.x()),
        "y": _round3(point.y()),
        "order": int(order),
        "snap_distance": None if snap_distance is None else _round3(snap_distance),
        "snap_source": "gui_skeleton",
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
    }


@dataclass(slots=True)
class LoadedSession:
    session: ReviewSession
    image: QImage


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
            image = QImage(str(self.image_path))
            if image.isNull():
                raise FileNotFoundError(f"cannot load image: {self.image_path}")
            self.finished.emit(LoadedSession(session=session, image=image))
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
        self.image_item: QGraphicsPixmapItem | None = None
        self.skeleton_item: QGraphicsPixmapItem | None = None
        self.current_route_item: QGraphicsPathItem | None = None
        self.saved_route_items: list[QGraphicsPathItem] = []
        self.point_items: list[CutPointItem] = []
        self.cut_points: list[dict[str, Any]] = []
        self.design_curves: list[dict[str, Any]] = []
        self.route_preview: dict[str, Any] | None = None
        self.closed_curve = False
        self.active_curve_id: str | None = None
        self.selected_cut_index: int | None = None
        self._worker_thread: QThread | None = None
        self._worker: SessionWorker | None = None

        self.scene = QGraphicsScene(self)
        self.view = EditorView(self, self.scene)
        self.curve_list = QListWidget()
        self.curve_list.currentItemChanged.connect(self._load_curve_from_item)

        self.semantic = QComboBox()
        self.semantic.addItems(
            [
                "outer_profile",
                "door_opening",
                "wheel_arch",
                "beltline",
                "roofline",
                "lamp",
                "bumper",
                "detail_line",
            ]
        )
        self.semantic.setCurrentText("detail_line")

        self.degree = QComboBox()
        self.degree.addItems(["auto", "3", "5", "7"])

        self.extraction_mode = QComboBox()
        self.extraction_mode.addItems(
            ["auto", "white_on_black_sketch", "black_on_white_line_art", "canny_edges"]
        )

        self.parallel_collapse = QComboBox()
        self.parallel_collapse.addItems(["off", "soft", "medium", "strong"])

        self.snap_radius = QSpinBox()
        self.snap_radius.setRange(1, 9999)
        self.snap_radius.setValue(36)
        self.snap_radius.setSuffix(" px")

        self.show_image = QCheckBox("原图")
        self.show_image.setChecked(True)
        self.show_image.toggled.connect(self._update_visibility)
        self.show_skeleton = QCheckBox("骨架")
        self.show_skeleton.setChecked(True)
        self.show_skeleton.toggled.connect(self._update_visibility)
        self.show_saved = QCheckBox("已保存蓝线")
        self.show_saved.setChecked(True)
        self.show_saved.toggled.connect(self._update_visibility)

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
        side_layout.addWidget(QLabel("当前分段语义"))
        side_layout.addWidget(self.semantic)
        side_layout.addWidget(QLabel("导出 Degree"))
        side_layout.addWidget(self.degree)
        side_layout.addWidget(self.show_image)
        side_layout.addWidget(self.show_skeleton)
        side_layout.addWidget(self.show_saved)

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

        export_btn = QPushButton("导出 IGES")
        export_btn.clicked.connect(self.export_iges)
        side_layout.addWidget(export_btn)

        delete_curve_btn = QPushButton("删除选中曲线")
        delete_curve_btn.clicked.connect(self.delete_selected_curve)
        side_layout.addWidget(delete_curve_btn)

        side_layout.addWidget(QLabel("曲线列表"))
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
        self.design_curves = list(loaded.session.design_curves)
        self.cut_points = []
        self.route_preview = None
        self.closed_curve = False
        self.active_curve_id = None
        self.selected_cut_index = None

        self.scene.clear()
        self.point_items.clear()
        self.saved_route_items.clear()
        self.current_route_item = None
        self.image_item = self.scene.addPixmap(QPixmap.fromImage(loaded.image))
        self.image_item.setZValue(0)
        self.skeleton_item = self.scene.addPixmap(self._make_skeleton_pixmap(loaded))
        self.skeleton_item.setZValue(5)
        self.scene.setSceneRect(0, 0, loaded.image.width(), loaded.image.height())
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._rebuild_saved_routes()
        self._refresh_curve_list()
        self._update_visibility()
        self._set_status(
            f"已加载：{loaded.session.image_path.name}，骨架点 {len(loaded.session.router.coords)}。"
        )

    def _make_skeleton_pixmap(self, loaded: LoadedSession) -> QPixmap:
        image = QImage(
            loaded.image.width(),
            loaded.image.height(),
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(QPen(QColor(220, 0, 0, 135), 1))
        for x, y in loaded.session.router.coords:
            painter.drawPoint(int(x), int(y))
        painter.setPen(QPen(QColor(0, 140, 115, 130), 1.2))
        for edge in loaded.session.graph.get("edges", []):
            points = edge.get("points") or []
            if len(points) < 2:
                continue
            painter.drawPath(_path_from_points(points))
        painter.end()
        return QPixmap.fromImage(image)

    def add_cut_point(self, scene_pos: QPointF) -> None:
        if self.session is None:
            return
        snapped = self._snap_point(scene_pos, require_radius=True)
        if snapped is None:
            self._set_status("附近没有骨架点。可以调大吸附半径。")
            return
        index = len(self.cut_points)
        self.cut_points.append(_point_dict(snapped[0], index, snapped[1]))
        self.selected_cut_index = index
        item = CutPointItem(self, index, snapped[0])
        self.point_items.append(item)
        self.scene.addItem(item)
        self._update_point_styles()
        self._update_route_preview()

    def move_cut_point_free(self, index: int, scene_pos: QPointF) -> None:
        if index < 0 or index >= len(self.cut_points):
            return
        self.cut_points[index]["x"] = _round3(scene_pos.x())
        self.cut_points[index]["y"] = _round3(scene_pos.y())
        self.cut_points[index]["snap_distance"] = None
        self.selected_cut_index = index
        self._update_point_styles()

    def finish_cut_point_drag(self, index: int, scene_pos: QPointF) -> None:
        if index < 0 or index >= len(self.cut_points):
            return
        snapped = self._snap_point(scene_pos, require_radius=False)
        if snapped is not None:
            point, distance = snapped
            self.cut_points[index] = _point_dict(point, index, distance)
            self.point_items[index].setPos(point)
        else:
            self.cut_points[index] = _point_dict(scene_pos, index, None)
        self._maybe_merge_dragged_point(index)
        self._update_route_preview()

    def _snap_point(
        self,
        point: QPointF,
        *,
        require_radius: bool,
    ) -> tuple[QPointF, float] | None:
        if self.session is None or len(self.session.router.coords) == 0:
            return None
        idx, distance = self.session.router.nearest_index((point.x(), point.y()))
        max_distance = float(self.snap_radius.value())
        if require_radius and distance > max_distance:
            return None
        if distance > max_distance and max_distance < 9999:
            return None
        x, y = self.session.router.coords[idx]
        return QPointF(float(x), float(y)), float(distance)

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
        self._rebuild_point_items()

    def select_cut_point(self, index: int) -> None:
        self.selected_cut_index = index
        self._update_point_styles()

    def _update_point_styles(self) -> None:
        for i, item in enumerate(self.point_items):
            item.set_selected_style(i == self.selected_cut_index)

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

    def _update_route_preview(self) -> None:
        if self.session is None:
            return
        if self.current_route_item is not None:
            self.scene.removeItem(self.current_route_item)
            self.current_route_item = None
        if len(self.cut_points) < 2:
            self.route_preview = None
            return
        self.route_preview = self.session.route_points(self.cut_points, closed=self.closed_curve)
        points = self.route_preview.get("points") or []
        if points:
            item = QGraphicsPathItem(_path_from_points(points))
            item.setPen(QPen(QColor(0, 109, 255), 2.4))
            item.setZValue(30)
            self.scene.addItem(item)
            self.current_route_item = item
        ok = bool(self.route_preview.get("ok"))
        self._set_status(
            f"蓝线路径：{len(points)} 点，{'连通' if ok else '未完全连通'}。"
        )

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
            "semantic": self.semantic.currentText(),
            "edge_ids": [],
            "manual_points": [dict(p, order=i) for i, p in enumerate(self.cut_points)],
            "cut_points": [dict(p, order=i) for i, p in enumerate(self.cut_points)],
            "closed": bool(self.closed_curve),
            "routed_points": route.get("points") or [],
            "route_segments": route_segments,
            "branch_choices": [],
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
            label = f"{curve.get('semantic', 'curve')} / {len(points)} 点 / {curve.get('id', '')}"
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
        self.semantic.setCurrentText(curve.get("semantic") or "detail_line")
        self.route_preview = {
            "ok": bool(curve.get("route_ok")),
            "points": curve.get("routed_points") or [],
            "segments": curve.get("route_segments") or [],
        }
        self._rebuild_point_items()
        self._update_route_preview()

    def delete_selected_curve(self) -> None:
        item = self.curve_list.currentItem()
        if item is None:
            return
        curve_id = item.data(Qt.ItemDataRole.UserRole)
        answer = QMessageBox.question(self, "删除曲线", "确认删除选中的已保存曲线？")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.design_curves = [curve for curve in self.design_curves if curve.get("id") != curve_id]
        if self.active_curve_id == curve_id:
            self.clear_current()
        self._save_all()
        self._rebuild_saved_routes()
        self._refresh_curve_list()

    def undo_point(self) -> None:
        if not self.cut_points:
            return
        self.cut_points.pop()
        self._rebuild_point_items()
        self._update_route_preview()

    def delete_selected_point(self) -> None:
        if self.selected_cut_index is None:
            return
        if 0 <= self.selected_cut_index < len(self.cut_points):
            self.cut_points.pop(self.selected_cut_index)
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
        self._rebuild_point_items()
        self._refresh_curve_list()
        self._set_status("当前曲线已清空。")

    def toggle_closed(self) -> None:
        self.closed_curve = not self.closed_curve
        self._update_route_preview()
        self._set_status("闭合已开启。" if self.closed_curve else "闭合已关闭。")

    def export_iges(self) -> None:
        if self.session is None:
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
        try:
            result = fit_reviewed_annotations(
                [self.session.corrections_path],
                out,
                degree=degree,
                min_points=4,
            )
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        QMessageBox.information(
            self,
            "导出完成",
            f"曲线：{len(result.curves)}\n"
            f"通过：{sum(1 for report in result.reports if report.passed)}/{len(result.reports)}\n"
            f"IGES：{result.out / 'reviewed_curves.igs'}\n"
            f"SVG：{result.out / 'reviewed_clean_preview.svg'}",
        )

    def _update_visibility(self) -> None:
        if self.image_item is not None:
            self.image_item.setVisible(self.show_image.isChecked())
        if self.skeleton_item is not None:
            self.skeleton_item.setVisible(self.show_skeleton.isChecked())
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

