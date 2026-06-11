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
        QFont,
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
        QGraphicsOpacityEffect,
        QGraphicsPathItem,
        QGraphicsPixmapItem,
        QGraphicsScene,
        QGraphicsSimpleTextItem,
        QGraphicsView,
        QFrame,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
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
        "segment_rule": str(segment.get("segment_rule") or "auto"),
        "end_join_continuity": str(segment.get("end_join_continuity") or "auto"),
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
    reference_image: QImage
    source_image_path: Path
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
        input_preprocess: str,
        parallel_collapse: str,
        weak_line_threshold: float,
    ) -> None:
        super().__init__()
        self.image_path = image_path
        self.output_dir = output_dir
        self.extraction_mode = extraction_mode
        self.input_preprocess = input_preprocess
        self.parallel_collapse = parallel_collapse
        self.weak_line_threshold = weak_line_threshold

    def run(self) -> None:
        try:
            options = ReviewGraphOptions(
                extraction_mode=self.extraction_mode,
                input_preprocess=self.input_preprocess,
                parallel_collapse=self.parallel_collapse,
                weak_line_threshold=self.weak_line_threshold,
                max_points_per_edge=480,
            )
            session = ReviewSession.create(self.image_path, self.output_dir, options)
            skeleton_edits = _read_skeleton_edits(session.corrections_path)
            _replay_skeleton_edits(session, skeleton_edits)
            image = QImage(str(session.image_path))
            if image.isNull():
                raise FileNotFoundError(f"cannot load image: {session.image_path}")
            reference_path = Path(str(session.graph.get("source_image") or self.image_path)).resolve()
            reference_image = QImage(str(reference_path))
            if reference_image.isNull():
                reference_image = image
            self.finished.emit(
                LoadedSession(
                    session=session,
                    image=image,
                    reference_image=reference_image,
                    source_image_path=reference_path,
                    skeleton_edits=skeleton_edits,
                )
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
        fit_mode: str,
    ) -> None:
        super().__init__()
        self.annotation_path = annotation_path
        self.output_dir = output_dir
        self.degree = degree
        self.fit_mode = fit_mode

    def run(self) -> None:
        try:
            result = fit_reviewed_annotations(
                [self.annotation_path],
                self.output_dir,
                degree=self.degree,
                min_points=4,
                max_fit_points=None,
                diagnostic_preview=False,
                fast_mode=False,
                fit_mode=self.fit_mode,
                wire_export=True,
            )
            self.finished.emit(result)
        except Exception as exc:  # pragma: no cover - GUI worker path.
            self.failed.emit(str(exc))


class CutPointItem(QGraphicsEllipseItem):
    def __init__(self, window: "DesktopEditor", index: int, point: QPointF) -> None:
        radius = 4.2
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.window = window
        self.index = index
        self.setPos(point)
        self.setZValue(50)
        self.setBrush(QBrush(QColor(116, 87, 255)))
        self.setPen(QPen(QColor("white"), 1.2))
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.label = QGraphicsSimpleTextItem(str(index + 1), self)
        self.label.setBrush(QBrush(QColor(30, 36, 35)))
        self.label.setPos(6, -16)

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

    def set_selected_style(self, selected: bool, join_rule: str = "auto") -> None:
        if selected:
            color = QColor(255, 230, 106)
        elif join_rule == "g2":
            color = QColor(220, 38, 38)
        elif join_rule == "hard":
            color = QColor(235, 86, 86)
        else:
            color = QColor(116, 87, 255)
        self.setBrush(QBrush(color))


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
        self.window.update_reference_preview_position()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self.window.update_reference_preview_position()
        self.window.update_empty_scene_layout()

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
            self.window.update_reference_preview_position()
            event.accept()
            return
        self.window.update_link_hover(self.mapToScene(event.position().toPoint()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.window.clear_link_hover()
        super().leaveEvent(event)


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.hasFocus():
            super().wheelEvent(event)
            return
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.hasFocus():
            super().wheelEvent(event)
            return
        event.ignore()


class DesktopEditor(QMainWindow):
    def __init__(self, output_dir: Path) -> None:
        super().__init__()
        self.setWindowTitle("AutoAlias Desktop Editor")
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session: ReviewSession | None = None
        self.current_image: QImage | None = None
        self.source_image_path: Path | None = None
        self.image_item: QGraphicsPixmapItem | None = None
        self.reference_preview_pixmap: QPixmap | None = None
        self.full_skeleton_item: QGraphicsPixmapItem | None = None
        self.edge_skeleton_item: QGraphicsPixmapItem | None = None
        self.design_stroke_item: QGraphicsPixmapItem | None = None
        self.current_route_item: QGraphicsPathItem | None = None
        self.alt_route_items: list[QGraphicsPathItem] = []
        self.saved_route_items: list[QGraphicsPathItem] = []
        self.connection_items: list[QGraphicsItem] = []
        self.link_hover_items: list[QGraphicsItem] = []
        self._last_link_hover_key: str | None = None
        self.point_items: list[CutPointItem] = []
        self.cut_points: list[dict[str, Any]] = []
        self.design_curves: list[dict[str, Any]] = []
        self.route_preview: dict[str, Any] | None = None
        self.branch_choices: list[int] = []
        self.segment_rules: list[str] = []
        self.join_continuities: list[str] = []
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
        self.view.setMinimumSize(520, 420)
        self.view.setBackgroundBrush(QBrush(QColor("#f7f9fb")))
        self.reference_preview = QLabel(self.view.viewport())
        self.reference_preview.setObjectName("referencePreview")
        self.reference_preview.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.reference_preview.setScaledContents(False)
        self.reference_preview.hide()
        self.reference_preview_opacity_effect = QGraphicsOpacityEffect(self.reference_preview)
        self.reference_preview.setGraphicsEffect(self.reference_preview_opacity_effect)
        self.empty_hint_items: list[QGraphicsItem] = []
        self.curve_list = QListWidget()
        self.curve_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.curve_list.currentItemChanged.connect(self._load_curve_from_item)
        self.branch_list = QListWidget()
        self.branch_list.currentRowChanged.connect(self._branch_selection_changed)

        self.segment_rule_combo = NoWheelComboBox()
        self.segment_rule_combo.addItem("当前段：自动判断", "auto")
        self.segment_rule_combo.addItem("当前段：曲线，可参与 G2", "curve")
        self.segment_rule_combo.addItem("当前段：直线保护", "line")
        self.segment_rule_combo.setToolTip("给分支列表中选中的那一段设置规则；直线保护会导出精确共线 CV，两端不做 G2。")

        self.join_rule_combo = NoWheelComboBox()
        self.join_rule_combo.addItem("连接点：自动判断", "auto")
        self.join_rule_combo.addItem("连接点：硬连接/G0，不做 G2", "hard")
        self.join_rule_combo.addItem("连接点：曲率连续/G2", "g2")
        self.join_rule_combo.setToolTip("给当前选中的紫色分段点设置连接规则；首尾开放端点没有连接关系。")

        self.degree = NoWheelComboBox()
        self.degree.addItems(["auto", "3", "5", "7"])
        self.precision_fit = QCheckBox("精度优先")
        self.precision_fit.setToolTip(
            "导出时忽略 CV 美学和 G2 关系，尽量贴合手动路由的目标线；适合 Logo、粗笔画、复杂装饰细节。"
        )

        self.raw_feature_preprocess = QCheckBox("原图预处理")
        self.raw_feature_preprocess.setToolTip(
            "勾选后先把未处理照片/渲染图转换成黑线白底特征线，再提取骨架；已是线稿/ControlNet 结果时不要勾选。"
        )
        self.thick_stroke_preprocess = QCheckBox("粗笔画轮廓")
        self.thick_stroke_preprocess.setToolTip(
            "用于 Logo、粗马克笔、粗黑实体图形：提取外轮廓和内孔轮廓，不提取粗笔画中心线。"
        )
        self.raw_feature_preprocess.toggled.connect(
            lambda checked: self.thick_stroke_preprocess.setChecked(False) if checked else None
        )
        self.thick_stroke_preprocess.toggled.connect(
            lambda checked: self.raw_feature_preprocess.setChecked(False) if checked else None
        )

        self.extraction_mode = NoWheelComboBox()
        self.extraction_mode.addItem("自动识别", "auto")
        self.extraction_mode.addItem("铅笔弱线增强", "pencil_weak_line_art")
        self.extraction_mode.addItem("黑底白线草图", "white_on_black_sketch")
        self.extraction_mode.addItem("白底黑线线稿", "black_on_white_line_art")
        self.extraction_mode.addItem("照片/渲染边缘", "canny_edges")

        self.weak_line_threshold = NoWheelSpinBox()
        self.weak_line_threshold.setRange(5, 95)
        self.weak_line_threshold.setValue(32)
        self.weak_line_threshold.setSuffix(" 阈值")
        self.weak_line_threshold.setToolTip("铅笔弱线增强阈值：数值越低越容易保留淡线，数值越高越抑制噪点。")

        self.parallel_collapse = NoWheelComboBox()
        self.parallel_collapse.addItem("关闭", "off")
        self.parallel_collapse.addItem("轻度并线", "soft")
        self.parallel_collapse.addItem("中度并线", "medium")
        self.parallel_collapse.addItem("强并线", "strong")
        self.parallel_collapse.setToolTip("用于把光影造成的平行多条线合成一条设计笔画。强度越高，越容易合并近距离平行线。")

        self.auto_segment_mode = NoWheelComboBox()
        self.auto_segment_mode.addItem("主线模式", "main")
        self.auto_segment_mode.addItem("连续覆盖模式", "coverage")
        self.auto_segment_mode.addItem("局部细节模式", "detail")
        self.auto_segment_mode.addItem("全量骨架模式", "full")
        self._set_combo_data(self.auto_segment_mode, "coverage")
        self.auto_segment_mode.setToolTip(
            "主线模式更保守；连续覆盖会跨小断口追长线；局部细节会提取更多短线；全量骨架会尽量把所有可追踪骨架都变成候选曲线。"
        )

        self.snap_radius = NoWheelSpinBox()
        self.snap_radius.setRange(1, 9999)
        self.snap_radius.setValue(10)
        self.snap_radius.setSuffix(" px")
        self.snap_radius.setToolTip("鼠标点、拖动分段点时吸附到骨架或已保存端点的搜索半径。")

        self.show_image = QCheckBox("原图")
        self.show_image.setChecked(True)
        self.show_image.toggled.connect(self._update_visibility)
        self.show_full_skeleton = QCheckBox("完整骨架红点")
        self.show_full_skeleton.setChecked(True)
        self.show_full_skeleton.toggled.connect(self._update_visibility)
        self.show_edge_skeleton = QCheckBox("切段骨架绿线")
        self.show_edge_skeleton.setChecked(True)
        self.show_edge_skeleton.toggled.connect(self._update_visibility)
        self.show_design_strokes = QCheckBox("设计笔画绿线")
        self.show_design_strokes.setChecked(True)
        self.show_design_strokes.toggled.connect(self._update_visibility)
        self.show_saved = QCheckBox("已保存蓝线")
        self.show_saved.setChecked(True)
        self.show_saved.toggled.connect(self._update_visibility)
        self.show_reference_preview = QCheckBox("右上角原图参考")
        self.show_reference_preview.setChecked(True)
        self.show_reference_preview.toggled.connect(self._update_visibility)
        self.reference_preview_opacity = NoWheelSpinBox()
        self.reference_preview_opacity.setRange(20, 100)
        self.reference_preview_opacity.setValue(78)
        self.reference_preview_opacity.setSuffix(" %")
        self.reference_preview_opacity.valueChanged.connect(self._update_reference_preview)

        self.skeleton_edit_mode = QCheckBox("骨架修补")
        self.skeleton_edit_tool = NoWheelComboBox()
        self.skeleton_edit_tool.addItems(["add", "delete"])
        self.skeleton_edit_radius = NoWheelSpinBox()
        self.skeleton_edit_radius.setRange(4, 240)
        self.skeleton_edit_radius.setValue(24)
        self.skeleton_edit_radius.setSuffix(" px")

        self.last_export_label = QLabel("最近导出：无")
        self.last_export_label.setWordWrap(True)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._build_ui()
        self._install_shortcuts()
        self._set_initial_window_geometry()
        self._set_status("打开一张图片开始。左键添加分段点，拖动点后松手自动吸附骨架。")

    def _current_extraction_mode(self) -> str:
        return str(self.extraction_mode.currentData() or self.extraction_mode.currentText())

    def _current_input_preprocess(self) -> str:
        if self.thick_stroke_preprocess.isChecked():
            return "thick_stroke_contours"
        return "raw_feature_lines" if self.raw_feature_preprocess.isChecked() else "none"

    def _current_parallel_collapse(self) -> str:
        return str(self.parallel_collapse.currentData() or self.parallel_collapse.currentText())

    def _set_combo_data(self, combo: QComboBox, value: str) -> None:
        for index in range(combo.count()):
            if str(combo.itemData(index)) == value:
                combo.setCurrentIndex(index)
                return
        combo.setCurrentIndex(0)

    def _make_group(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("sectionCard")
        card.setFrameShape(QFrame.Shape.NoFrame)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(9)
        header = QLabel(title)
        header.setProperty("sectionTitle", True)
        header.setWordWrap(True)
        layout.addWidget(header)
        return card, layout

    def _add_field(self, layout: QVBoxLayout, label: str, widget: QWidget) -> None:
        text = QLabel(label)
        text.setProperty("fieldLabel", True)
        text.setWordWrap(True)
        layout.addWidget(text)
        layout.addWidget(widget)

    def _make_primary_button(self, text: str) -> QPushButton:
        button = QPushButton(text)
        button.setProperty("primary", True)
        button.setMinimumHeight(38)
        return button

    def _set_initial_window_geometry(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 820)
            return
        available = screen.availableGeometry()
        width = min(1320, max(1120, int(available.width() * 0.82)))
        height = min(860, max(720, int(available.height() * 0.82)))
        width = min(width, max(900, available.width() - 80))
        height = min(height, max(640, available.height() - 80))
        x = available.x() + max(0, (available.width() - width) // 2)
        y = available.y() + max(0, (available.height() - height) // 2)
        self.setGeometry(x, y, width, height)

    def reset_default_settings(self) -> None:
        self.degree.setCurrentText("auto")
        self.precision_fit.setChecked(False)
        self.raw_feature_preprocess.setChecked(False)
        self.thick_stroke_preprocess.setChecked(False)
        self._set_combo_data(self.extraction_mode, "auto")
        self.weak_line_threshold.setValue(32)
        self._set_combo_data(self.parallel_collapse, "off")
        self._set_combo_data(self.auto_segment_mode, "coverage")
        self.snap_radius.setValue(10)
        self.show_image.setChecked(True)
        self.show_full_skeleton.setChecked(True)
        self.show_edge_skeleton.setChecked(True)
        self.show_design_strokes.setChecked(True)
        self.show_saved.setChecked(True)
        self.show_reference_preview.setChecked(True)
        self.reference_preview_opacity.setValue(78)
        self.skeleton_edit_mode.setChecked(False)
        self.skeleton_edit_tool.setCurrentIndex(0)
        self.skeleton_edit_radius.setValue(24)
        self._update_visibility()
        self._set_status("已恢复默认设置。已保存曲线和当前工程不会被清空。")

    def open_user_guide(self) -> None:
        guide = Path(__file__).resolve().parents[3] / "docs" / "AutoAlias_Desktop_Editor_User_Guide.pdf"
        if not guide.exists():
            QMessageBox.information(
                self,
                "AutoAlias",
                f"没有找到使用教程 PDF：\n{guide}\n\n"
                "请先运行：\n"
                r"F:\ComfyUI\.venv\Scripts\python.exe F:\430AutoAlias\scripts\generate_gui_user_guide_pdf.py",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(guide)))

    def _build_ui(self) -> None:
        toolbar = QToolBar("AutoAlias")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("上传图片", self)
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
        toolbar.setVisible(False)

        asset_dir = Path(__file__).with_name("assets")
        chevron_down = (asset_dir / "chevron_down.svg").as_posix()
        chevron_up = (asset_dir / "chevron_up.svg").as_posix()
        style = (
            """
            QMainWindow {
                background: #f3f6f8;
                color: #17202a;
                font-family: "Times New Roman", "SimSun", "宋体";
                font-size: 13px;
            }
            QToolBar { background: #f7f9fb; border: 0; border-bottom: 1px solid #cfd7df; spacing: 8px; }
            QWidget#sidePanel { background: #e7edf2; }
            QFrame#sectionCard {
                background: #fbfcfd;
                border: 1px solid #cfd8e2;
                border-radius: 7px;
            }
            QLabel[sectionTitle="true"] {
                color: #17202a;
                font-size: 15px;
                font-weight: 700;
                padding-bottom: 2px;
                border: 0;
            }
            QLabel[fieldLabel="true"] { color: #5e6b78; font-size: 13px; font-weight: 500; }
            QLabel { color: #17202a; font-size: 13px; }
            QCheckBox { spacing: 7px; color: #24313d; font-size: 13px; }
            QCheckBox::indicator { width: 15px; height: 15px; }
            QCheckBox::indicator:unchecked {
                border: 1px solid #8c99a6;
                background: #ffffff;
                border-radius: 2px;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #1668dc;
                background: #1668dc;
                border-radius: 2px;
            }
            QPushButton {
                min-height: 33px;
                padding: 4px 10px;
                border: 1px solid #b9c4cf;
                border-radius: 5px;
                background: #f5f7fa;
                color: #17202a;
                font-size: 13px;
            }
            QPushButton:hover { background: #edf4ff; border-color: #8bb7f0; }
            QPushButton:pressed { background: #dcecff; }
            QPushButton[primary="true"] {
                background: #1668dc;
                color: #ffffff;
                border: 1px solid #1668dc;
                border-radius: 5px;
                font-weight: 600;
            }
            QPushButton[primary="true"]:hover { background: #0f5fc8; border-color: #0f5fc8; }
            QPushButton[primary="true"]:disabled { background: #9ebff0; border-color: #9ebff0; }
            QListWidget {
                background: #ffffff;
                color: #17202a;
                border: 1px solid #cfd8e2;
                border-radius: 5px;
                font-size: 13px;
            }
            QListWidget::item:selected { background: #1668dc; color: #ffffff; }
            QLabel#referencePreview {
                background: rgba(255, 255, 255, 232);
                border: 1px solid #93a4b5;
                border-radius: 6px;
                padding: 5px;
            }
            QComboBox, QSpinBox {
                min-height: 33px;
                border: 1px solid #b9c4cf;
                border-radius: 5px;
                background: #ffffff;
                color: #17202a;
                padding-left: 8px;
                padding-right: 4px;
                font-size: 13px;
            }
            QComboBox:hover, QSpinBox:hover { border-color: #7aa8dc; }
            QComboBox::drop-down {
                width: 28px;
                border: 0;
                border-left: 1px solid #d2dbe5;
                border-top-right-radius: 5px;
                border-bottom-right-radius: 5px;
                background: #dce5ee;
            }
            QComboBox::down-arrow {
                image: url(__CHEVRON_DOWN__);
                width: 12px;
                height: 12px;
                margin-right: 8px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 22px;
                border-left: 1px solid #d2dbe5;
                background: #dce5ee;
            }
            QSpinBox::up-arrow {
                image: url(__CHEVRON_UP__);
                width: 10px;
                height: 10px;
            }
            QSpinBox::down-arrow {
                image: url(__CHEVRON_DOWN__);
                width: 10px;
                height: 10px;
            }
            QScrollArea { background: #e7edf2; border: 0; }
            QScrollBar:vertical {
                background: #dde5ec;
                width: 12px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #9aa8b5;
                min-height: 36px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QSplitter::handle { background: #c2ccd6; }
            QStatusBar {
                background: #eef3f7;
                color: #17202a;
                border-top: 1px solid #cfd7df;
            }
            """
        )
        self.setStyleSheet(
            style.replace("__CHEVRON_DOWN__", chevron_down).replace("__CHEVRON_UP__", chevron_up)
        )

        side = QWidget()
        side.setObjectName("sidePanel")
        side.setMinimumWidth(390)
        side.setMaximumWidth(470)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(12, 12, 12, 14)
        side_layout.setSpacing(12)

        welcome_group, welcome_layout = self._make_group("开始")
        title = QLabel("请先上传图片")
        title.setStyleSheet(
            'font-family: "Times New Roman", "SimSun", "宋体"; '
            "font-size: 22px; font-weight: 700; color: #17202a;"
        )
        title.setWordWrap(True)
        welcome_layout.addWidget(title)
        intro = QLabel("上传后会自动提取骨架。再根据骨架手动分段、检查路径候选，最后导出 IGES / WIRE。")
        intro.setWordWrap(True)
        intro.setStyleSheet(
            'font-family: "Times New Roman", "SimSun", "宋体"; '
            "font-size: 13px; color: #5e6b78; line-height: 140%;"
        )
        welcome_layout.addWidget(intro)
        upload_btn = self._make_primary_button("上传图片并提取骨架")
        upload_btn.clicked.connect(self.open_image_dialog)
        welcome_layout.addWidget(upload_btn)
        project_row = QHBoxLayout()
        open_project_btn = QPushButton("打开工程")
        open_project_btn.clicked.connect(self.open_project_dialog)
        save_project_btn = QPushButton("保存工程")
        save_project_btn.clicked.connect(self.save_project)
        project_row.addWidget(open_project_btn)
        project_row.addWidget(save_project_btn)
        welcome_layout.addLayout(project_row)
        reset_btn = QPushButton("恢复默认设置")
        reset_btn.clicked.connect(self.reset_default_settings)
        welcome_layout.addWidget(reset_btn)
        guide_btn = QPushButton("打开使用教程")
        guide_btn.clicked.connect(self.open_user_guide)
        welcome_layout.addWidget(guide_btn)
        side_layout.addWidget(welcome_group)

        extract_group, extract_layout = self._make_group("1 图片与骨架提取")
        self.image_state_label = QLabel("当前图片：未上传")
        self.image_state_label.setWordWrap(True)
        self.image_state_label.setStyleSheet("color: #5e6b78;")
        extract_layout.addWidget(self.image_state_label)
        preprocess_row = QHBoxLayout()
        preprocess_row.addWidget(self.raw_feature_preprocess)
        preprocess_row.addWidget(self.thick_stroke_preprocess)
        extract_layout.addLayout(preprocess_row)
        self._add_field(extract_layout, "提取模式", self.extraction_mode)
        self._add_field(extract_layout, "铅笔弱线阈值", self.weak_line_threshold)
        self._add_field(extract_layout, "并线强度", self.parallel_collapse)
        reload_btn = QPushButton("按当前选项重新提取")
        reload_btn.clicked.connect(self.reload_current_image)
        extract_layout.addWidget(reload_btn)
        side_layout.addWidget(extract_group)

        view_group, view_layout = self._make_group("2 视图与吸附")
        self._add_field(view_layout, "分段点吸附半径", self.snap_radius)
        view_row1 = QHBoxLayout()
        view_row1.addWidget(self.show_image)
        view_row1.addWidget(self.show_saved)
        view_layout.addLayout(view_row1)
        view_row2 = QHBoxLayout()
        view_row2.addWidget(self.show_full_skeleton)
        view_row2.addWidget(self.show_design_strokes)
        view_layout.addLayout(view_row2)
        view_row3 = QHBoxLayout()
        view_row3.addWidget(self.show_edge_skeleton)
        view_row3.addStretch(1)
        view_layout.addLayout(view_row3)
        view_row4 = QHBoxLayout()
        view_row4.addWidget(self.show_reference_preview)
        view_row4.addStretch(1)
        view_layout.addLayout(view_row4)
        self._add_field(view_layout, "右上角参考图透明度", self.reference_preview_opacity)
        side_layout.addWidget(view_group)

        manual_group, manual_layout = self._make_group("3 手动分段")
        skeleton_row = QHBoxLayout()
        skeleton_row.setSpacing(8)
        self.skeleton_edit_tool.setMinimumWidth(96)
        self.skeleton_edit_radius.setMinimumWidth(86)
        skeleton_row.addWidget(self.skeleton_edit_mode)
        skeleton_row.addWidget(self.skeleton_edit_tool)
        skeleton_row.addWidget(self.skeleton_edit_radius)
        manual_layout.addLayout(skeleton_row)
        skeleton_break_btn = QPushButton("断开连续加点")
        skeleton_break_btn.clicked.connect(self.break_skeleton_add_chain)
        manual_layout.addWidget(skeleton_break_btn)

        row1 = QHBoxLayout()
        save_btn = QPushButton("保存当前")
        save_btn.clicked.connect(lambda: self.save_current(start_next=False))
        save_next_btn = QPushButton("保存并下一条")
        save_next_btn.clicked.connect(lambda: self.save_current(start_next=True))
        row1.addWidget(save_btn)
        row1.addWidget(save_next_btn)
        manual_layout.addLayout(row1)

        row2 = QHBoxLayout()
        undo_btn = QPushButton("撤回点")
        undo_btn.clicked.connect(self.undo_point)
        delete_btn = QPushButton("删除点")
        delete_btn.clicked.connect(self.delete_selected_point)
        row2.addWidget(undo_btn)
        row2.addWidget(delete_btn)
        manual_layout.addLayout(row2)

        row3 = QHBoxLayout()
        close_btn = QPushButton("闭合开关")
        close_btn.clicked.connect(self.toggle_closed)
        clear_btn = QPushButton("清空当前")
        clear_btn.clicked.connect(self.clear_current)
        row3.addWidget(close_btn)
        row3.addWidget(clear_btn)
        manual_layout.addLayout(row3)

        continuity_row1 = QHBoxLayout()
        apply_segment_rule_btn = QPushButton("应用段规则")
        apply_segment_rule_btn.setToolTip("先在下面的分支/路径候选列表中选择一段，再应用段规则。")
        apply_segment_rule_btn.clicked.connect(self.apply_selected_segment_rule)
        continuity_row1.addWidget(self.segment_rule_combo)
        continuity_row1.addWidget(apply_segment_rule_btn)
        manual_layout.addLayout(continuity_row1)

        continuity_row2 = QHBoxLayout()
        apply_join_rule_btn = QPushButton("应用连接规则")
        apply_join_rule_btn.setToolTip("先点击一个手动分段点，再设置该连接点是否 G2。")
        apply_join_rule_btn.clicked.connect(self.apply_selected_join_rule)
        continuity_row2.addWidget(self.join_rule_combo)
        continuity_row2.addWidget(apply_join_rule_btn)
        manual_layout.addLayout(continuity_row2)
        side_layout.addWidget(manual_group)

        auto_group, auto_layout = self._make_group("4 自动分段与路径候选")
        self._add_field(auto_layout, "几何自动分段模式", self.auto_segment_mode)
        auto_btn = QPushButton("几何自动分段")
        auto_btn.clicked.connect(self.run_geometry_auto_segment)
        auto_layout.addWidget(auto_btn)

        self._add_field(auto_layout, "分支 / 多路径候选", self.branch_list)
        self.branch_list.setMinimumHeight(88)
        branch_row = QHBoxLayout()
        prev_branch_btn = QPushButton("上一候选")
        prev_branch_btn.clicked.connect(lambda: self.shift_branch_choice(-1))
        next_branch_btn = QPushButton("下一候选")
        next_branch_btn.clicked.connect(lambda: self.shift_branch_choice(1))
        branch_row.addWidget(prev_branch_btn)
        branch_row.addWidget(next_branch_btn)
        auto_layout.addLayout(branch_row)
        side_layout.addWidget(auto_group)

        export_group, export_layout = self._make_group("5 导出 Alias")
        self._add_field(export_layout, "导出 Degree", self.degree)
        export_layout.addWidget(self.precision_fit)
        self.export_btn = self._make_primary_button("导出 IGES / WIRE")
        self.export_btn.clicked.connect(self.export_iges)
        export_layout.addWidget(self.export_btn)
        open_export_btn = QPushButton("打开最近导出文件夹")
        open_export_btn.clicked.connect(self.open_last_export_dir)
        export_layout.addWidget(open_export_btn)
        export_layout.addWidget(self.last_export_label)
        side_layout.addWidget(export_group)

        list_group, list_layout = self._make_group("曲线列表")
        curve_action_row = QHBoxLayout()
        select_all_btn = QPushButton("全选曲线")
        select_all_btn.clicked.connect(self.select_all_curves)
        delete_curve_btn = QPushButton("批量删除选中")
        delete_curve_btn.clicked.connect(self.delete_selected_curve)
        curve_action_row.addWidget(select_all_btn)
        curve_action_row.addWidget(delete_curve_btn)
        list_layout.addLayout(curve_action_row)
        self.curve_list.setMinimumHeight(450)
        list_layout.addWidget(self.curve_list)
        side_layout.addWidget(list_group)
        side_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumWidth(416)
        scroll.setMaximumWidth(500)

        splitter = QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(self.view)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([450, 1050])
        self.setCentralWidget(splitter)
        self._show_empty_scene()

    def _install_shortcuts(self) -> None:
        undo = QAction(self)
        undo.setShortcut(QKeySequence.StandardKey.Undo)
        undo.triggered.connect(self.undo_point)
        self.addAction(undo)
        delete = QAction(self)
        delete.setShortcut(QKeySequence.StandardKey.Delete)
        delete.triggered.connect(self.delete_selected_point)
        self.addAction(delete)
        force_g2 = QAction(self)
        force_g2.setShortcut(QKeySequence("Ctrl+G"))
        force_g2.triggered.connect(self.force_current_curve_g2)
        self.addAction(force_g2)

    def _show_empty_scene(self) -> None:
        self.scene.clear()
        self.empty_hint_items.clear()
        self.scene.setSceneRect(0, 0, 1100, 700)
        card = self.scene.addRect(0, 0, 520, 220, QPen(QColor("#d7e0e7"), 1.2), QBrush(QColor("#ffffff")))
        card.setZValue(1)
        title = self.scene.addSimpleText("请上传图片")
        title.setBrush(QBrush(QColor("#17202a")))
        title.setScale(1.8)
        title.setZValue(2)
        body = self.scene.addSimpleText("左侧选择预处理、提取模式和阈值，然后上传图片生成骨架。")
        body.setBrush(QBrush(QColor("#5c6873")))
        body.setScale(1.05)
        body.setZValue(2)
        self.empty_hint_items.extend([card, title, body])
        self.update_empty_scene_layout()
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def update_empty_scene_layout(self) -> None:
        empty_hint_items = getattr(self, "empty_hint_items", [])
        if len(empty_hint_items) < 3 or self.session is not None:
            return
        viewport = self.view.viewport().size()
        if viewport.width() > 100 and viewport.height() > 100:
            scene_width = max(900.0, float(viewport.width()))
            scene_height = max(560.0, float(viewport.height()))
            self.scene.setSceneRect(0, 0, scene_width, scene_height)
        scene_rect = self.scene.sceneRect()
        card = empty_hint_items[0]
        title = empty_hint_items[1]
        body = empty_hint_items[2]
        card_width = min(560.0, max(460.0, scene_rect.width() * 0.46))
        card_height = 220.0
        card_x = scene_rect.center().x() - card_width / 2.0
        card_y = scene_rect.center().y() - card_height / 2.0
        if hasattr(card, "setRect"):
            card.setRect(card_x, card_y, card_width, card_height)
        title_rect = title.boundingRect()
        body_rect = body.boundingRect()
        title_scale = float(title.scale())
        body_scale = float(body.scale())
        title.setPos(
            scene_rect.center().x() - title_rect.width() * title_scale / 2.0,
            card_y + 52.0,
        )
        body.setPos(
            scene_rect.center().x() - body_rect.width() * body_scale / 2.0,
            card_y + 132.0,
        )

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
            graph = data.get("graph", {}) if isinstance(data.get("graph"), dict) else {}
            image_path = Path(str(graph.get("image") or ""))
            if not image_path.exists():
                raise FileNotFoundError(f"工程里的图片路径不存在：{image_path}")
            source_image_path = Path(str(graph.get("source_image") or image_path))
            if not source_image_path.exists():
                source_image_path = image_path
            session = ReviewSession.create(
                image_path,
                project_path.parent,
                ReviewGraphOptions(
                    extraction_mode=self._current_extraction_mode(),
                    input_preprocess="none",
                    parallel_collapse=self._current_parallel_collapse(),
                    weak_line_threshold=float(self.weak_line_threshold.value()),
                    max_points_per_edge=480,
                ),
            )
            session.corrections_path = project_path.resolve()
            session.corrections = list(data.get("corrections", []))
            session.design_curves = list(data.get("design_curves", []))
            skeleton_edits = data.get("skeleton_edits", [])
            skeleton_edits = list(skeleton_edits) if isinstance(skeleton_edits, list) else []
            _replay_skeleton_edits(session, skeleton_edits)
            image = QImage(str(session.image_path))
            if image.isNull():
                raise FileNotFoundError(f"cannot load image: {session.image_path}")
            reference_image = QImage(str(source_image_path))
            if reference_image.isNull():
                reference_image = image
            self._session_loaded(
                LoadedSession(
                    session=session,
                    image=image,
                    reference_image=reference_image,
                    source_image_path=source_image_path.resolve(),
                    skeleton_edits=skeleton_edits,
                )
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
        if hasattr(self, "image_state_label"):
            self.image_state_label.setText(f"正在提取：{image_path.name}")
        self._set_status("正在提取骨架，GUI 不会阻塞。")
        worker = SessionWorker(
            image_path=image_path,
            output_dir=self.output_dir,
            extraction_mode=self._current_extraction_mode(),
            input_preprocess=self._current_input_preprocess(),
            parallel_collapse=self._current_parallel_collapse(),
            weak_line_threshold=float(self.weak_line_threshold.value()),
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

    def reload_current_image(self) -> None:
        if self.session is None:
            QMessageBox.information(self, "AutoAlias", "请先上传图片。")
            return
        self.load_image(self.source_image_path or self.session.image_path)

    def _clear_worker(self) -> None:
        self._worker = None
        self._worker_thread = None

    def _friendly_session_error(self, message: str) -> str:
        if (
            "raw feature-line preprocessing found no usable line pixels" in message
            or "原图预处理没有找到可用线条像素" in message
        ):
            return (
                "原图预处理没有找到可用线条像素。\n\n"
                "原因：\n"
                "当前勾选了“原图预处理”。这个功能主要用于未处理过的照片或渲染图，"
                "它会先估计背景和主体区域，再用 Canny 提取主体里的结构边缘。"
                "如果输入本身已经是线稿、ControlNet 线稿、黑底白线图，"
                "或者线条太淡、对比度太低，就可能在主体遮罩和噪声过滤阶段被过滤掉，"
                "最终没有留下可追踪的线条像素。\n\n"
                "建议：\n"
                "1. 取消勾选“原图预处理”；\n"
                "2. 如果是黑底白线图，提取模式改成“黑底白线草图”；\n"
                "3. 如果是淡铅笔线，选择“铅笔弱线增强”，并把阈值调低到 20-28；\n"
                "4. 然后再点击“按当前选项重新提取”。\n\n"
                "当前已加载的骨架不会被清空。"
            )
        return message

    def _session_failed(self, message: str) -> None:
        friendly = self._friendly_session_error(message)
        QMessageBox.critical(self, "AutoAlias", friendly)
        if hasattr(self, "image_state_label"):
            if self.session is None:
                self.image_state_label.setText("当前图片：加载失败")
            else:
                self.image_state_label.setText("重新提取失败：已保留当前骨架")
        self._set_status("图片重新提取失败，已保留当前骨架。")

    def _session_loaded(self, loaded: LoadedSession) -> None:
        self.session = loaded.session
        self.current_image = loaded.image
        self.source_image_path = loaded.source_image_path
        self.skeleton_edits = list(loaded.skeleton_edits)
        self.design_curves = list(loaded.session.design_curves)
        self.cut_points = []
        self.route_preview = None
        self.branch_choices = []
        self.closed_curve = False
        self.active_curve_id = None
        self.selected_cut_index = None

        self.scene.clear()
        self.empty_hint_items.clear()
        self.point_items.clear()
        self.alt_route_items.clear()
        self.saved_route_items.clear()
        self.connection_items.clear()
        self.current_route_item = None
        self.reference_preview_pixmap = self._make_reference_preview_pixmap(loaded.reference_image)
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
        self.design_stroke_item = self.scene.addPixmap(
            self._make_design_stroke_pixmap(loaded.image, loaded.session)
        )
        self.design_stroke_item.setZValue(7)
        self.scene.setSceneRect(0, 0, loaded.image.width(), loaded.image.height())
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._rebuild_saved_routes()
        self._refresh_curve_list()
        self._update_visibility()
        self.update_reference_preview_position()
        if hasattr(self, "image_state_label"):
            mode = self.extraction_mode.currentText()
            preprocess = "无"
            if self.raw_feature_preprocess.isChecked():
                preprocess = "原图预处理"
            elif self.thick_stroke_preprocess.isChecked():
                preprocess = "粗笔画轮廓"
            self.image_state_label.setText(
                f"当前图片：{loaded.session.image_path.name}\n"
                f"提取模式：{mode} / 预处理：{preprocess}\n"
                f"骨架点：{len(loaded.session.router.coords)}"
            )
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

    def _make_reference_preview_pixmap(self, image_source: QImage) -> QPixmap:
        max_width = 260
        max_height = 190
        pixmap = QPixmap.fromImage(image_source)
        if pixmap.isNull():
            return pixmap
        return pixmap.scaled(
            max_width,
            max_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _update_reference_preview(self) -> None:
        if self.reference_preview_pixmap is None or self.reference_preview_pixmap.isNull():
            self.reference_preview.hide()
            return
        if not self.show_reference_preview.isChecked():
            self.reference_preview.hide()
            return
        self.reference_preview.setPixmap(self.reference_preview_pixmap)
        self.reference_preview.adjustSize()
        self.reference_preview_opacity_effect.setOpacity(
            max(0.2, min(1.0, float(self.reference_preview_opacity.value()) / 100.0))
        )
        self.reference_preview.show()
        self.reference_preview.raise_()
        self.update_reference_preview_position()

    def update_reference_preview_position(self) -> None:
        if not self.reference_preview.isVisible():
            return
        margin = 16
        viewport_size = self.view.viewport().size()
        x = max(margin, viewport_size.width() - self.reference_preview.width() - margin)
        y = margin
        self.reference_preview.move(x, y)

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

    def _make_design_stroke_pixmap(self, image_source: QImage, session: ReviewSession) -> QPixmap:
        image = self._transparent_image_like(image_source)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor(0, 95, 55, 210), 1.8))
        strokes = session.graph.get("design_strokes") or session.graph.get("edges", [])
        for stroke in strokes:
            points = stroke.get("points") or []
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
        design_pixmap = self._make_design_stroke_pixmap(self.current_image, self.session)
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
        if self.design_stroke_item is None:
            self.design_stroke_item = self.scene.addPixmap(design_pixmap)
            self.design_stroke_item.setZValue(7)
        else:
            self.design_stroke_item.setPixmap(design_pixmap)
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
        self.update_link_hover(scene_pos)

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
        self.clear_link_hover()
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

    def _iter_saved_anchor_points(self):
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
                yield {
                    "point": QPointF(x, y),
                    "curve_id": curve_id,
                    "order": int(saved_point.get("order", order) or order),
                    "semantic": str(curve.get("semantic") or DEFAULT_GUI_SEMANTIC),
                }

    def _snap_to_saved_anchor(self, point: QPointF, radius: float | None = None) -> dict[str, Any] | None:
        radius = self._saved_anchor_radius() if radius is None else max(0.0, float(radius))
        best: dict[str, Any] | None = None
        for anchor in self._iter_saved_anchor_points():
            anchor_point = anchor["point"]
            distance = ((point.x() - anchor_point.x()) ** 2 + (point.y() - anchor_point.y()) ** 2) ** 0.5
            if distance > radius:
                continue
            if best is None or distance < float(best["distance"]):
                best = {
                    "point": QPointF(float(anchor_point.x()), float(anchor_point.y())),
                    "distance": float(distance),
                    "source": "saved_curve_anchor",
                    "anchor_curve_id": anchor["curve_id"],
                    "anchor_point_order": anchor["order"],
                    "anchor_semantic": anchor["semantic"],
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
        self._sync_rule_controls()

    def _update_point_styles(self) -> None:
        for i, item in enumerate(self.point_items):
            item.set_selected_style(i == self.selected_cut_index, self._join_rule_for_point(i))

    def _join_rule_for_point(self, point_index: int) -> str:
        if not self.join_continuities:
            return "auto"
        if self.closed_curve and len(self.cut_points) >= 3:
            return self.join_continuities[(point_index - 1) % len(self.join_continuities)]
        join_index = point_index - 1
        if 0 <= join_index < len(self.join_continuities):
            return self.join_continuities[join_index]
        return "auto"

    def _rebuild_connection_markers(self) -> None:
        for item in self.connection_items:
            self.scene.removeItem(item)
        self.connection_items.clear()
        if self.show_saved.isChecked():
            for anchor in self._iter_saved_anchor_points():
                point = anchor["point"]
                x = float(point.x())
                y = float(point.y())
                hint = QGraphicsEllipseItem(x - 4.5, y - 4.5, 9.0, 9.0)
                hint.setPen(QPen(QColor(220, 38, 38, 150), 1.2))
                hint.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                hint.setZValue(43)
                hint.setToolTip(
                    f"可连接到已有曲线 {anchor.get('curve_id', '')} "
                    f"点 {int(anchor.get('order', 0)) + 1}"
                )
                self.scene.addItem(hint)
                self.connection_items.append(hint)
        for point in self.cut_points:
            if point.get("snap_source") != "saved_curve_anchor":
                continue
            x = float(point.get("x", 0.0))
            y = float(point.get("y", 0.0))
            ring = QGraphicsEllipseItem(x - 11.0, y - 11.0, 22.0, 22.0)
            ring.setPen(QPen(QColor(220, 38, 38), 2.2))
            ring.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            ring.setZValue(48)
            ring.setToolTip(
                f"连接到曲线 {point.get('anchor_curve_id', '')} "
                f"点 {point.get('anchor_point_order', '')}"
            )
            self.scene.addItem(ring)
            self.connection_items.append(ring)
            label = QGraphicsSimpleTextItem("link")
            label.setBrush(QBrush(QColor(185, 28, 28)))
            label.setPos(x + 12.0, y + 8.0)
            label.setZValue(49)
            self.scene.addItem(label)
            self.connection_items.append(label)

    def clear_link_hover(self) -> None:
        for item in self.link_hover_items:
            self.scene.removeItem(item)
        self.link_hover_items.clear()
        self._last_link_hover_key = None

    def update_link_hover(self, scene_pos: QPointF) -> None:
        if self.session is None or not self.design_curves or self.skeleton_edit_mode.isChecked():
            self.clear_link_hover()
            return
        snap_radius = self._saved_anchor_radius()
        anchor = self._snap_to_saved_anchor(scene_pos, radius=snap_radius)
        active = anchor is not None
        if anchor is None:
            anchor = self._snap_to_saved_anchor(scene_pos, radius=snap_radius * 2.0)
        if anchor is None:
            self.clear_link_hover()
            return
        self._show_link_hover(scene_pos, anchor, active=active)
        key = (
            f"{'active' if active else 'near'}:"
            f"{anchor.get('anchor_curve_id', '')}:"
            f"{anchor.get('anchor_point_order', '')}"
        )
        if key != self._last_link_hover_key:
            self._last_link_hover_key = key
            order = int(anchor.get("anchor_point_order", 0)) + 1
            distance = float(anchor.get("distance", 0.0))
            if active:
                self._set_status(f"可连接：松开/点击后会吸附到已有曲线点 {order}（距离 {distance:.1f}px）。")
            else:
                self._set_status(f"靠近已有曲线点 {order}，继续靠近会出现连接吸附。")

    def _show_link_hover(self, scene_pos: QPointF, anchor: dict[str, Any], *, active: bool) -> None:
        for item in self.link_hover_items:
            self.scene.removeItem(item)
        self.link_hover_items.clear()
        point = anchor["point"]
        x = float(point.x())
        y = float(point.y())
        color = QColor(220, 38, 38) if active else QColor(255, 178, 48)
        fill = QColor(color)
        fill.setAlpha(38 if active else 28)
        outer_radius = 20.0 if active else 15.0
        halo = QGraphicsEllipseItem(x - outer_radius, y - outer_radius, outer_radius * 2.0, outer_radius * 2.0)
        halo.setPen(QPen(color, 1.5, Qt.PenStyle.DashLine if not active else Qt.PenStyle.SolidLine))
        halo.setBrush(QBrush(fill))
        halo.setZValue(72)
        self.scene.addItem(halo)
        self.link_hover_items.append(halo)

        ring_radius = 8.0 if active else 6.0
        ring = QGraphicsEllipseItem(x - ring_radius, y - ring_radius, ring_radius * 2.0, ring_radius * 2.0)
        ring.setPen(QPen(color, 2.4 if active else 1.7))
        ring.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        ring.setZValue(73)
        self.scene.addItem(ring)
        self.link_hover_items.append(ring)

        if ((scene_pos.x() - x) ** 2 + (scene_pos.y() - y) ** 2) ** 0.5 > 1.5:
            path = QPainterPath(scene_pos)
            path.lineTo(point)
            guide = QGraphicsPathItem(path)
            guide.setPen(QPen(color, 1.4, Qt.PenStyle.DashLine))
            guide.setZValue(71)
            self.scene.addItem(guide)
            self.link_hover_items.append(guide)

        label = QGraphicsSimpleTextItem("可连接" if active else "靠近端点")
        label.setBrush(QBrush(color))
        label.setPos(x + 12.0, y - 24.0)
        label.setZValue(74)
        self.scene.addItem(label)
        self.link_hover_items.append(label)

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
        self._normalize_continuity_rules()

    def _normalize_continuity_rules(self) -> None:
        expected_segments = self._expected_segment_count()
        while len(self.segment_rules) < expected_segments:
            self.segment_rules.append("auto")
        if len(self.segment_rules) > expected_segments:
            self.segment_rules = self.segment_rules[:expected_segments]

        expected_joins = expected_segments if self.closed_curve and expected_segments > 1 else max(0, expected_segments - 1)
        while len(self.join_continuities) < expected_joins:
            self.join_continuities.append("auto")
        if len(self.join_continuities) > expected_joins:
            self.join_continuities = self.join_continuities[:expected_joins]

    def _selected_segment_index(self) -> int | None:
        self._normalize_continuity_rules()
        row = self.branch_list.currentRow()
        if 0 <= row < len(self.segment_rules):
            return row
        if self.selected_cut_index is not None:
            index = min(max(int(self.selected_cut_index), 0), max(0, len(self.segment_rules) - 1))
            if 0 <= index < len(self.segment_rules):
                return index
        return 0 if self.segment_rules else None

    def _selected_join_index(self) -> int | None:
        self._normalize_continuity_rules()
        if not self.join_continuities or self.selected_cut_index is None:
            return None
        point_index = int(self.selected_cut_index)
        if self.closed_curve and len(self.cut_points) >= 3:
            return (point_index - 1) % len(self.join_continuities)
        join_index = point_index - 1
        if 0 <= join_index < len(self.join_continuities):
            return join_index
        return None

    def apply_selected_segment_rule(self) -> None:
        index = self._selected_segment_index()
        if index is None:
            self._set_status("请先添加至少两个分段点，再设置段规则。")
            return
        rule = str(self.segment_rule_combo.currentData() or "auto")
        self.segment_rules[index] = rule
        if rule == "line":
            previous_join = index - 1
            if self.closed_curve and self.join_continuities:
                previous_join %= len(self.join_continuities)
            if 0 <= previous_join < len(self.join_continuities):
                self.join_continuities[previous_join] = "hard"
            if index < len(self.join_continuities):
                self.join_continuities[index] = "hard"
        self._update_point_styles()
        self._update_route_preview()
        self._set_status(f"已设置第 {index + 1} 段规则：{self._segment_rule_label(rule)}。")

    def apply_selected_join_rule(self) -> None:
        index = self._selected_join_index()
        if index is None:
            self._set_status("请先选中中间连接点；开放曲线首尾端点没有连接规则。")
            return
        rule = str(self.join_rule_combo.currentData() or "auto")
        self.join_continuities[index] = rule
        self._update_point_styles()
        self._update_route_preview()
        self._set_status(f"已设置连接点 {index + 1}：{self._join_rule_label(rule)}。")

    def force_current_curve_g2(self) -> None:
        self._normalize_continuity_rules()
        if len(self.cut_points) < 3 or not self.join_continuities:
            self._set_status("当前曲线没有可设置 G2 的连接点。")
            return
        self.segment_rules = ["curve" for _ in self.segment_rules]
        self.join_continuities = ["g2" for _ in self.join_continuities]
        self._update_point_styles()
        self._update_route_preview()
        self._sync_rule_controls()
        self.save_current(start_next=False)
        self._set_status("已将当前曲线所有段设为曲线，所有连接点设为 G2（Ctrl+G）。")

    def _segment_rule_label(self, rule: str) -> str:
        return {"auto": "自动", "curve": "曲线", "line": "直线保护"}.get(str(rule), str(rule))

    def _join_rule_label(self, rule: str) -> str:
        return {"auto": "自动", "hard": "硬连接/G0", "g2": "G2"}.get(str(rule), str(rule))

    def _set_combo_by_data(self, combo: QComboBox, value: str) -> None:
        idx = combo.findData(value)
        if idx < 0:
            idx = 0
        combo.blockSignals(True)
        combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _sync_rule_controls(self) -> None:
        segment_index = self._selected_segment_index()
        if segment_index is not None and segment_index < len(self.segment_rules):
            self._set_combo_by_data(self.segment_rule_combo, self.segment_rules[segment_index])
        else:
            self._set_combo_by_data(self.segment_rule_combo, "auto")
        join_index = self._selected_join_index()
        if join_index is not None and join_index < len(self.join_continuities):
            self._set_combo_by_data(self.join_rule_combo, self.join_continuities[join_index])
        else:
            self._set_combo_by_data(self.join_rule_combo, "auto")

    def _route_points_with_choices(self) -> dict[str, Any]:
        if self.session is None:
            return {"ok": False, "reason": "no session", "segments": [], "points": []}
        clean = [(float(p["x"]), float(p["y"])) for p in self.cut_points]
        if len(clean) < 2:
            return {"ok": False, "reason": "need at least two points", "segments": [], "points": []}
        self._normalize_branch_choices()
        self._normalize_continuity_rules()
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
                    "segment_rule": self.segment_rules[index] if index < len(self.segment_rules) else "auto",
                    "end_join_continuity": self.join_continuities[index] if index < len(self.join_continuities) else "auto",
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
        self._sync_rule_controls()
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
            status = "OK" if segment.get("ok") else "BROKEN"
            segment_rule = self._segment_rule_label(str(segment.get("segment_rule") or "auto"))
            join_rule = self._join_rule_label(str(segment.get("end_join_continuity") or "auto"))
            item = QListWidgetItem(
                f"Seg {index + 1}: cand {selected}/{count} / {length:.0f}px / {status} / segment:{segment_rule} / join:{join_rule}"
            )
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.branch_list.addItem(item)
        if self.branch_list.count():
            row = current_row if 0 <= current_row < self.branch_list.count() else 0
            self.branch_list.setCurrentRow(row)
        self.branch_list.blockSignals(False)

    def _branch_selection_changed(self, _row: int) -> None:
        if self.route_preview is None:
            return
        self._sync_rule_controls()
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
        self._normalize_continuity_rules()
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
            "segment_rules": self.segment_rules[:],
            "join_continuities": self.join_continuities[:],
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
        self._rebuild_connection_markers()

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
        route_segments = curve.get("route_segments") or []
        self.segment_rules = [str(value or "auto") for value in (curve.get("segment_rules") or [])]
        if not self.segment_rules and route_segments:
            self.segment_rules = [str(segment.get("segment_rule") or "auto") for segment in route_segments]
        self.join_continuities = [str(value or "auto") for value in (curve.get("join_continuities") or [])]
        if not self.join_continuities and route_segments:
            self.join_continuities = [str(segment.get("end_join_continuity") or "auto") for segment in route_segments]
        self._normalize_branch_choices()
        self.route_preview = {
            "ok": bool(curve.get("route_ok")),
            "points": curve.get("routed_points") or [],
            "segments": route_segments,
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
            self.segment_rules = []
            self.join_continuities = []
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
            self._sync_rule_controls()
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
        self._sync_rule_controls()

    def delete_selected_point(self) -> None:
        if self.selected_cut_index is None:
            return
        if 0 <= self.selected_cut_index < len(self.cut_points):
            self.cut_points.pop(self.selected_cut_index)
        self._normalize_branch_choices()
        self._rebuild_point_items()
        self._update_route_preview()
        self._sync_rule_controls()

    def clear_current(self) -> None:
        self.cut_points = []
        self.route_preview = None
        self.branch_choices = []
        self.segment_rules = []
        self.join_continuities = []
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
        self._sync_rule_controls()
        self._set_status("当前曲线已清空。")

    def toggle_closed(self) -> None:
        self.closed_curve = not self.closed_curve
        self._normalize_branch_choices()
        self._update_route_preview()
        self._sync_rule_controls()
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
        fit_mode = "precision" if self.precision_fit.isChecked() else "manual_class_a_g2"
        worker = ExportWorker(self.session.corrections_path, out, degree, fit_mode)
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
        wire_text = "WIRE：未请求"
        if result.wire_result is not None:
            if result.wire_result.ok:
                wire_text = f"WIRE：{result.wire_result.wire_path}"
            else:
                wire_text = f"WIRE：未生成（见 {result.out / 'reviewed_curves.wire_status.json'}）"
        QMessageBox.information(
            self,
            "导出完成",
            f"曲线：{len(result.curves)}\n"
            f"通过：{sum(1 for report in result.reports if report.passed)}/{len(result.reports)}\n"
            f"IGES：{result.out / 'reviewed_curves.igs'}\n"
            f"{wire_text}\n"
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
        if self.design_stroke_item is not None:
            self.design_stroke_item.setVisible(self.show_design_strokes.isChecked())
        for item in self.saved_route_items:
            item.setVisible(self.show_saved.isChecked())
        for item in self.connection_items:
            item.setVisible(self.show_saved.isChecked())
        if not self.show_saved.isChecked():
            self.clear_link_hover()
        self._update_reference_preview()

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
    app_font = QFont("SimSun", 11)
    try:
        app_font.setFamilies(["Times New Roman", "SimSun", "宋体"])
    except AttributeError:
        pass
    app.setFont(app_font)
    window = DesktopEditor(args.out)
    window.show()
    if args.image:
        window.load_image(args.image)
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
