import { useEffect, useMemo, useRef, useState } from "react";
import { Circle, Group, Image as KonvaImage, Layer, Line, Rect, Shape, Stage, Text } from "react-konva";
import type { KonvaEventObject } from "konva/lib/Node";
import {
  autoSegment,
  breakSkeletonChain,
  downloadProject as downloadProjectFile,
  editSkeleton,
  getJob,
  imageUrl,
  openProject,
  reextractImage,
  routePoints,
  saveDesignCurves,
  snapPoint,
  startExport,
  uploadImage
} from "./api";
import type { DesignCurve, JobState, Point, RouteSegment, SessionState } from "./types";
import "./styles.css";

type Transform = { x: number; y: number; scale: number };
type RouteStats = { pointCount: number; length: number; bbox: unknown } | null;

const semantics = [
  ["outer_profile", "外轮廓"],
  ["door_opening", "门洞/车窗"],
  ["wheel_arch", "轮拱"],
  ["beltline", "腰线/特征线"],
  ["roofline", "车顶线"],
  ["lamp", "灯具轮廓"],
  ["detail_line", "细节线"]
] as const;

const autoSegmentModes = [
  ["main", "主线模式"],
  ["coverage", "连续覆盖模式"],
  ["detail", "局部细节模式"],
  ["full", "全量骨架模式"]
] as const;

function useHtmlImage(src: string | null) {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    if (!src) {
      setImage(null);
      return;
    }
    const img = new Image();
    img.onload = () => setImage(img);
    img.src = src;
  }, [src]);
  return image;
}

function flatten(points: number[][] | undefined): number[] {
  return (points || []).flatMap(([x, y]) => [x, y]);
}

function makeCurveId() {
  return `curve_${Date.now().toString(16)}_${Math.random().toString(16).slice(2, 7)}`;
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [session, setSession] = useState<SessionState | null>(null);
  const [inputPreprocess, setInputPreprocess] = useState("none");
  const [extractionMode, setExtractionMode] = useState("auto");
  const [weakLineThreshold, setWeakLineThreshold] = useState("32");
  const [parallelCollapse, setParallelCollapse] = useState("off");
  const [semantic, setSemantic] = useState("detail_line");
  const [degree, setDegree] = useState("auto");
  const [precisionFit, setPrecisionFit] = useState(false);
  const [snapRadius, setSnapRadius] = useState(24);
  const [autoSegmentMode, setAutoSegmentMode] = useState("coverage");
  const [skeletonEditMode, setSkeletonEditMode] = useState(false);
  const [skeletonEditTool, setSkeletonEditTool] = useState<"add" | "delete">("add");
  const [skeletonEditRadius, setSkeletonEditRadius] = useState(24);
  const [showImage, setShowImage] = useState(true);
  const [showFullSkeleton, setShowFullSkeleton] = useState(true);
  const [showEdgeSkeleton, setShowEdgeSkeleton] = useState(true);
  const [showDesignStrokes, setShowDesignStrokes] = useState(true);
  const [showSavedCurves, setShowSavedCurves] = useState(true);
  const [showReference, setShowReference] = useState(true);
  const [referenceOpacity, setReferenceOpacity] = useState(78);
  const [currentPoints, setCurrentPoints] = useState<Point[]>([]);
  const [selectedPointIndex, setSelectedPointIndex] = useState<number | null>(null);
  const [closed, setClosed] = useState(false);
  const [routeSegments, setRouteSegments] = useState<RouteSegment[]>([]);
  const [routedPoints, setRoutedPoints] = useState<number[][]>([]);
  const [branchChoices, setBranchChoices] = useState<number[]>([]);
  const [designCurves, setDesignCurves] = useState<DesignCurve[]>([]);
  const [activeCurveId, setActiveCurveId] = useState<string | null>(null);
  const [selectedCurveIds, setSelectedCurveIds] = useState<Set<string>>(new Set());
  const [status, setStatus] = useState("请上传图片");
  const [routeStats, setRouteStats] = useState<RouteStats>(null);
  const [exportJob, setExportJob] = useState<JobState | null>(null);
  const [stageSize, setStageSize] = useState({ width: 900, height: 700 });
  const [transform, setTransform] = useState<Transform>({ x: 0, y: 0, scale: 1 });
  const stageWrapRef = useRef<HTMLDivElement | null>(null);
  const projectInputRef = useRef<HTMLInputElement | null>(null);
  const workerSeq = useRef(0);

  const img = useHtmlImage(session ? imageUrl(session.sid) : null);
  const worker = useMemo(() => new Worker(new URL("./workers/geometry.worker.ts", import.meta.url), { type: "module" }), []);
  const savedAnchors = useMemo(
    () =>
      designCurves.flatMap((curve) =>
        (curve.manual_points || curve.cut_points || []).map((point, order) => ({
          x: point.x,
          y: point.y,
          curveId: curve.id,
          order: point.order ?? order,
          semantic: curve.semantic
        }))
      ),
    [designCurves]
  );

  useEffect(() => {
    function handleKeydown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, select, textarea, button")) return;
      if (event.ctrlKey && event.key.toLowerCase() === "z") {
        event.preventDefault();
        undoPoint();
        return;
      }
      if ((event.key === "Delete" || event.key === "Backspace") && selectedPointIndex !== null) {
        event.preventDefault();
        deleteSelectedPoint();
      }
    }
    window.addEventListener("keydown", handleKeydown);
    return () => window.removeEventListener("keydown", handleKeydown);
  });

  useEffect(() => {
    const wrap = stageWrapRef.current;
    if (!wrap) return;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setStageSize({ width: Math.max(360, width), height: Math.max(360, height) });
    });
    observer.observe(wrap);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!session?.graph.image_size || !stageSize.width || !stageSize.height) return;
    const { width, height } = session.graph.image_size;
    const scale = Math.min((stageSize.width - 72) / width, (stageSize.height - 72) / height);
    setTransform({
      scale: Math.max(0.05, scale),
      x: (stageSize.width - width * scale) / 2,
      y: (stageSize.height - height * scale) / 2
    });
  }, [session?.sid, session?.graph.image_size, stageSize.width, stageSize.height]);

  useEffect(() => {
    worker.onmessage = (event: MessageEvent) => {
      if (event.data?.type === "route-measured") {
        setRouteStats({
          pointCount: event.data.pointCount,
          length: event.data.length,
          bbox: event.data.bbox
        });
      }
    };
    return () => worker.terminate();
  }, [worker]);

  useEffect(() => {
    if (!exportJob || !["queued", "running"].includes(exportJob.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const next = await getJob(exportJob.job_id);
        setExportJob(next);
        if (next.status === "done") setStatus("导出完成");
        if (next.status === "failed") setStatus(`导出失败：${next.error || next.message || ""}`);
      } catch (err) {
        setStatus(`查询导出进度失败：${String(err)}`);
      }
    }, 900);
    return () => window.clearInterval(timer);
  }, [exportJob]);

  function loadSession(next: SessionState, curves = next.design_curves || []) {
    setSession(next);
    setDesignCurves(curves);
    setCurrentPoints([]);
    setSelectedPointIndex(null);
    setRouteSegments([]);
    setRoutedPoints([]);
    setBranchChoices([]);
    setActiveCurveId(null);
    setSelectedCurveIds(new Set());
  }

  async function handleUpload() {
    if (!file) {
      setStatus("请先选择图片");
      return;
    }
    const restoreImportedCurves = !session && designCurves.length > 0;
    const importedCurves = designCurves;
    setStatus("正在上传并提取骨架...");
    try {
      const next = await uploadImage({ file, inputPreprocess, extractionMode, weakLineThreshold, parallelCollapse });
      const nextCurves = restoreImportedCurves ? importedCurves : next.design_curves || [];
      loadSession(next, nextCurves);
      if (restoreImportedCurves) {
        await saveDesignCurves(next.sid, nextCurves);
      }
      setStatus(`已提取骨架：${next.graph.edge_count || 0} 条`);
    } catch (err) {
      setStatus(`提取失败：${String(err)}`);
    }
  }

  async function handleReextract() {
    if (!session) return;
    setStatus("正在按当前选项重新提取；失败时会保留当前骨架...");
    try {
      const next = await reextractImage(session.sid, {
        inputPreprocess,
        extractionMode,
        weakLineThreshold,
        parallelCollapse
      });
      loadSession(next, next.design_curves || designCurves);
      setStatus(next.message || `重新提取完成：${next.graph.edge_count || 0} 条`);
    } catch (err) {
      setStatus(`重新提取失败，已保留当前骨架：${String(err)}`);
    }
  }

  async function refreshRoute(points: Point[], nextClosed = closed, choices = branchChoices) {
    if (!session || points.length < 2) {
      setRouteSegments([]);
      setRoutedPoints([]);
      return;
    }
    const route = await routePoints(session.sid, points, nextClosed, choices);
    setRouteSegments(route.segments || []);
    setRoutedPoints(route.points || []);
    const id = ++workerSeq.current;
    worker.postMessage({ type: "measure-route", id, segments: route.segments || [] });
  }

  function clearCurrent() {
    setCurrentPoints([]);
    setSelectedPointIndex(null);
    setRouteSegments([]);
    setRoutedPoints([]);
    setBranchChoices([]);
    setClosed(false);
    setActiveCurveId(null);
  }

  function resetDefaults() {
    setInputPreprocess("none");
    setExtractionMode("auto");
    setWeakLineThreshold("32");
    setParallelCollapse("off");
    setSemantic("detail_line");
    setDegree("auto");
    setPrecisionFit(false);
    setSnapRadius(24);
    setAutoSegmentMode("coverage");
    setSkeletonEditMode(false);
    setSkeletonEditTool("add");
    setSkeletonEditRadius(24);
    setShowImage(true);
    setShowFullSkeleton(true);
    setShowEdgeSkeleton(true);
    setShowDesignStrokes(true);
    setShowSavedCurves(true);
    setShowReference(true);
    setReferenceOpacity(78);
    setClosed(false);
    setBranchChoices([]);
    setStatus("已恢复默认设置");
  }

  function editorState() {
    return {
      options: {
        inputPreprocess,
        extractionMode,
        weakLineThreshold,
        parallelCollapse,
        semantic,
        degree,
        precisionFit,
        snapRadius,
        autoSegmentMode,
        skeletonEditRadius,
        showImage,
        showFullSkeleton,
        showEdgeSkeleton,
        showDesignStrokes,
        showSavedCurves,
        showReference,
        referenceOpacity
      },
      current_points: currentPoints,
      closed,
      branch_choices: branchChoices
    };
  }

  async function downloadProject() {
    if (!session) {
      setStatus("请先上传图片再保存工程");
      return;
    }
    try {
      const blob = await downloadProjectFile(session.sid, designCurves, editorState());
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      link.href = url;
      link.download = `autoalias_project_${stamp}.json`;
      link.click();
      URL.revokeObjectURL(url);
      setStatus("工程 JSON 已保存：包含原图、提取参数、骨架修补和曲线");
    } catch (err) {
      setStatus(`保存工程失败：${String(err)}`);
    }
  }

  async function importProjectFile(projectFile: File | null | undefined) {
    if (!projectFile) return;
    try {
      setStatus("正在打开工程并重建骨架...");
      const next = await openProject(projectFile);
      const state = next.editor_state || {};
      const options = (state.options || {}) as Record<string, unknown>;
      if (typeof options.inputPreprocess === "string") setInputPreprocess(options.inputPreprocess);
      if (typeof options.extractionMode === "string") setExtractionMode(options.extractionMode);
      if (typeof options.weakLineThreshold === "string") setWeakLineThreshold(options.weakLineThreshold);
      if (typeof options.parallelCollapse === "string") setParallelCollapse(options.parallelCollapse);
      if (typeof options.semantic === "string") setSemantic(options.semantic);
      if (typeof options.degree === "string") setDegree(options.degree);
      if (typeof options.precisionFit === "boolean") setPrecisionFit(options.precisionFit);
      if (typeof options.snapRadius === "number") setSnapRadius(options.snapRadius);
      if (typeof options.autoSegmentMode === "string") setAutoSegmentMode(options.autoSegmentMode);
      if (typeof options.skeletonEditRadius === "number") setSkeletonEditRadius(options.skeletonEditRadius);
      if (typeof options.showImage === "boolean") setShowImage(options.showImage);
      if (typeof options.showFullSkeleton === "boolean") setShowFullSkeleton(options.showFullSkeleton);
      if (typeof options.showEdgeSkeleton === "boolean") setShowEdgeSkeleton(options.showEdgeSkeleton);
      if (typeof options.showDesignStrokes === "boolean") setShowDesignStrokes(options.showDesignStrokes);
      if (typeof options.showSavedCurves === "boolean") setShowSavedCurves(options.showSavedCurves);
      if (typeof options.showReference === "boolean") setShowReference(options.showReference);
      if (typeof options.referenceOpacity === "number") setReferenceOpacity(options.referenceOpacity);

      const curves = next.design_curves || [];
      const points = Array.isArray(state.current_points) ? state.current_points as Point[] : [];
      loadSession(next, curves);
      setCurrentPoints(points);
      setClosed(Boolean(state.closed));
      setBranchChoices(Array.isArray(state.branch_choices) ? state.branch_choices as number[] : []);
      setActiveCurveId(null);
      setStatus("工程已完整打开：原图、骨架修补记录和曲线均已恢复");
    } catch (err) {
      setStatus(`读取工程失败：${String(err)}`);
    } finally {
      if (projectInputRef.current) projectInputRef.current.value = "";
    }
  }

  function snapToSavedAnchor(point: Point): Point | null {
    let best: { x: number; y: number; curveId: string; order: number; semantic: string; distance: number } | null = null;
    for (const anchor of savedAnchors) {
      if (anchor.curveId === activeCurveId) continue;
      const distance = Math.hypot(anchor.x - point.x, anchor.y - point.y);
      if (distance <= snapRadius && (!best || distance < best.distance)) {
        best = { ...anchor, distance };
      }
    }
    return best
      ? {
          x: best.x,
          y: best.y,
          snap_source: "saved_curve_anchor",
          anchor_curve_id: best.curveId,
          anchor_point_order: best.order,
          anchor_semantic: best.semantic
        }
      : null;
  }

  async function snapWorkingPoint(point: Point): Promise<Point | null> {
    const saved = snapToSavedAnchor(point);
    if (saved) return saved;
    if (!session) return null;
    const snapped = await snapPoint(session.sid, point, snapRadius).catch(() => null);
    return snapped?.ok ? { x: snapped.x, y: snapped.y, snap_source: "skeleton" } : null;
  }

  async function handleStageClick(event: KonvaEventObject<MouseEvent>) {
    if (!session) return;
    if (event.target !== event.target.getStage()) return;
    const pointer = event.target.getStage()?.getPointerPosition();
    if (!pointer) return;
    const world = {
      x: (pointer.x - transform.x) / transform.scale,
      y: (pointer.y - transform.y) / transform.scale
    };
    if (skeletonEditMode) {
      const result = await editSkeleton(session.sid, skeletonEditTool, world, skeletonEditRadius);
      if (result.ok && result.full_skeleton_points) {
        setSession({
          ...session,
          graph: {
            ...session.graph,
            full_skeleton_points: result.full_skeleton_points
          }
        });
        setStatus(`骨架修补完成：${skeletonEditTool}`);
      } else {
        setStatus(result.reason || "骨架修补失败");
      }
      return;
    }
    const point = await snapWorkingPoint(world);
    if (!point) {
      setStatus("附近没有骨架点，可以调大吸附半径");
      return;
    }
    const next = [...currentPoints, point];
    setCurrentPoints(next);
    setSelectedPointIndex(next.length - 1);
    await refreshRoute(next);
  }

  async function updatePoint(index: number, point: Point) {
    const snapped = await snapWorkingPoint(point);
    const settled = snapped || { ...point, snap_source: "free_drag" };
    let next = currentPoints.map((item, i) => (i === index ? settled : item));
    const mergeIndex = next.findIndex(
      (item, i) => i !== index && Math.hypot(item.x - settled.x, item.y - settled.y) < 10
    );
    if (
      mergeIndex >= 0 &&
      window.confirm(`分段点 ${index + 1} 已靠近分段点 ${mergeIndex + 1}，是否合并？`)
    ) {
      next = next.filter((_item, i) => i !== index);
      setSelectedPointIndex(next.length ? Math.min(index, next.length - 1) : null);
      setStatus("已合并重合的分段点");
    } else {
      setSelectedPointIndex(index);
    }
    setCurrentPoints(next);
    await refreshRoute(next);
  }

  function undoPoint() {
    if (!currentPoints.length) return;
    const next = currentPoints.slice(0, -1);
    setCurrentPoints(next);
    setSelectedPointIndex(next.length ? next.length - 1 : null);
    void refreshRoute(next);
  }

  function deleteSelectedPoint() {
    if (selectedPointIndex === null || selectedPointIndex < 0 || selectedPointIndex >= currentPoints.length) {
      setStatus("请先在画布中选择要删除的分段点");
      return;
    }
    const next = currentPoints.filter((_point, index) => index !== selectedPointIndex);
    setCurrentPoints(next);
    setSelectedPointIndex(next.length ? Math.min(selectedPointIndex, next.length - 1) : null);
    void refreshRoute(next);
  }

  async function handleBreakSkeletonChain() {
    if (!session) return;
    try {
      const result = await breakSkeletonChain(session.sid);
      setStatus(result.message);
    } catch (err) {
      setStatus(`断开连续加点失败：${String(err)}`);
    }
  }

  async function saveCurrent(reset = true) {
    if (!session || currentPoints.length < 2) return;
    const curve: DesignCurve = {
      id: activeCurveId || makeCurveId(),
      type: "manual_design_curve",
      semantic,
      manual_points: currentPoints.map((p, index) => ({ ...p, order: index + 1 })),
      cut_points: currentPoints.map((p, index) => ({ ...p, order: index + 1 })),
      closed,
      routed_points: routedPoints,
      route_segments: routeSegments,
      branch_choices: routeSegments.map((_segment, index) => branchChoices[index] || 0),
      route_ok: routeSegments.every((segment) => segment.ok !== false),
      source: "next_web_manual",
      created_at: new Date().toISOString()
    };
    const next = activeCurveId
      ? designCurves.map((item) => (item.id === activeCurveId ? curve : item))
      : [...designCurves, curve];
    setDesignCurves(next);
    await saveDesignCurves(session.sid, next);
    if (reset) {
      clearCurrent();
    }
    setStatus(`已保存 ${next.length} 条曲线`);
  }

  async function handleAutoSegment() {
    if (!session) return;
    setStatus("正在几何自动分段...");
    const size = session.graph.image_size || { width: 0, height: 0 };
    const result = await autoSegment(session.sid, autoSegmentMode, size);
    const curves = [...designCurves, ...(result.curves || [])];
    setDesignCurves(curves);
    await saveDesignCurves(session.sid, curves);
    setStatus(`几何自动分段完成：新增 ${result.curve_count} 条`);
  }

  async function handleExport() {
    if (!session) return;
    setStatus("正在创建导出任务...");
    const job = await startExport(session.sid, designCurves, degree, precisionFit);
    setExportJob({ ok: true, job_id: job.job_id, status: "queued", progress: 1 });
  }

  function deleteCurve(id: string) {
    if (!session) return;
    const next = designCurves.filter((curve) => curve.id !== id);
    setDesignCurves(next);
    setSelectedCurveIds((selected) => {
      const copy = new Set(selected);
      copy.delete(id);
      return copy;
    });
    if (activeCurveId === id) clearCurrent();
    void saveDesignCurves(session.sid, next);
  }

  function deleteSelectedCurves() {
    if (!session || selectedCurveIds.size === 0) return;
    const next = designCurves.filter((curve) => !selectedCurveIds.has(curve.id));
    setDesignCurves(next);
    setSelectedCurveIds(new Set());
    if (activeCurveId && selectedCurveIds.has(activeCurveId)) clearCurrent();
    void saveDesignCurves(session.sid, next);
    setStatus(`已删除 ${designCurves.length - next.length} 条曲线`);
  }

  function toggleCurveSelected(id: string) {
    setSelectedCurveIds((selected) => {
      const copy = new Set(selected);
      if (copy.has(id)) copy.delete(id);
      else copy.add(id);
      return copy;
    });
  }

  async function loadCurve(curve: DesignCurve) {
    const points = (curve.manual_points || curve.cut_points || []).map((point) => ({ ...point }));
    setActiveCurveId(curve.id);
    setSemantic(curve.semantic || "detail_line");
    setCurrentPoints(points);
    setSelectedPointIndex(null);
    setClosed(Boolean(curve.closed));
    setRoutedPoints(curve.routed_points || []);
    setRouteSegments(curve.route_segments || []);
    setBranchChoices(curve.branch_choices || []);
    setStatus(`已加载曲线：${curve.semantic || curve.id}`);
  }

  async function shiftBranchChoice(segmentIndex: number, delta: number) {
    const alternatives = routeSegments[segmentIndex]?.alternatives || [];
    if (alternatives.length <= 1) return;
    const next = [...branchChoices];
    const current = next[segmentIndex] || 0;
    next[segmentIndex] = (current + delta + alternatives.length) % alternatives.length;
    setBranchChoices(next);
    await refreshRoute(currentPoints, closed, next);
  }

  function handleWheel(event: KonvaEventObject<WheelEvent>) {
    event.evt.preventDefault();
    const stage = event.target.getStage();
    const pointer = stage?.getPointerPosition();
    if (!pointer) return;
    const scaleBy = 1.08;
    const oldScale = transform.scale;
    const nextScale = event.evt.deltaY > 0 ? oldScale / scaleBy : oldScale * scaleBy;
    const mousePointTo = {
      x: (pointer.x - transform.x) / oldScale,
      y: (pointer.y - transform.y) / oldScale
    };
    setTransform({
      scale: Math.max(0.03, Math.min(16, nextScale)),
      x: pointer.x - mousePointTo.x * nextScale,
      y: pointer.y - mousePointTo.y * nextScale
    });
  }

  const imageSize = session?.graph.image_size || { width: 0, height: 0 };
  const skeletonPoints = (session?.graph.full_skeleton_points || []) as number[][];
  const edgePolylines = session?.graph.edges || [];
  const designStrokePolylines = session?.graph.design_strokes || session?.graph.edges || [];
  const referenceWidth = Math.min(220, stageSize.width * 0.22);
  const referenceScale = imageSize.width > 0 ? referenceWidth / imageSize.width : 0;
  const referenceHeight = imageSize.height * referenceScale;

  return (
    <div className="appShell">
      <aside className="sidePanel">
        <div className="brand">
          <div>
            <span>AutoAlias Next</span>
            <h1>曲线分段工作台</h1>
          </div>
          <div className="brandGlyph" />
        </div>

        <section className="card compactCard">
          <h2>工程</h2>
          <input
            ref={projectInputRef}
            className="hiddenFile"
            type="file"
            accept="application/json,.json"
            onChange={(event) => void importProjectFile(event.target.files?.[0])}
          />
          <div className="buttonGrid">
            <button onClick={() => projectInputRef.current?.click()}>打开工程 JSON</button>
            <button onClick={downloadProject}>保存工程 JSON</button>
          </div>
          <button onClick={resetDefaults}>恢复默认设置</button>
        </section>

        <section className="card">
          <h2>图片与骨架</h2>
          <div className="miniMeta">
            <span>当前图片：{file?.name || "未选择"}</span>
            <span>骨架：{session?.graph.edge_count || 0} / 笔画：{Number(session?.graph.design_stroke_count || 0)}</span>
          </div>
          <input type="file" accept="image/*" onChange={(event) => setFile(event.target.files?.[0] || null)} />
          <div className="twoCols">
            <select value={inputPreprocess} onChange={(e) => setInputPreprocess(e.target.value)}>
              <option value="none">输入不预处理</option>
              <option value="raw_feature_lines">原图预处理</option>
              <option value="thick_stroke_contours">粗笔画轮廓</option>
            </select>
            <select value={extractionMode} onChange={(e) => setExtractionMode(e.target.value)}>
              <option value="auto">自动识别</option>
              <option value="black_on_white_line_art">白底黑线</option>
              <option value="white_on_black_sketch">黑底白线</option>
              <option value="pencil_weak_line_art">铅笔弱线</option>
              <option value="canny_edges">照片边缘</option>
            </select>
            <select value={weakLineThreshold} onChange={(e) => setWeakLineThreshold(e.target.value)}>
              <option value="18">弱线 18</option>
              <option value="24">弱线 24</option>
              <option value="32">弱线 32</option>
              <option value="45">弱线 45</option>
              <option value="60">弱线 60</option>
            </select>
            <select value={parallelCollapse} onChange={(e) => setParallelCollapse(e.target.value)}>
              <option value="off">不并线</option>
              <option value="soft">轻度并线</option>
              <option value="medium">中度并线</option>
              <option value="strong">强并线</option>
            </select>
          </div>
          <button className="primary" onClick={() => void handleUpload()}>上传并提取骨架</button>
          <button onClick={() => void handleReextract()} disabled={!session}>按当前选项重新提取</button>
        </section>

        <section className="card">
          <h2>手动分段</h2>
          <div className="twoCols">
            <label className="checkPill">
              <input type="checkbox" checked={skeletonEditMode} onChange={(e) => setSkeletonEditMode(e.target.checked)} />
              骨架修补
            </label>
            <select value={skeletonEditTool} onChange={(e) => setSkeletonEditTool(e.target.value as "add" | "delete")}>
              <option value="add">添加骨架点</option>
              <option value="delete">删除骨架点</option>
            </select>
            <select value={skeletonEditRadius} onChange={(e) => setSkeletonEditRadius(Number(e.target.value))}>
              <option value={12}>修补半径 12px</option>
              <option value={24}>修补半径 24px</option>
              <option value={48}>修补半径 48px</option>
              <option value={96}>修补半径 96px</option>
            </select>
            <button onClick={() => void handleBreakSkeletonChain()} disabled={!session}>断开连续加点</button>
          </div>
          <div className="twoCols">
            <select value={semantic} onChange={(e) => setSemantic(e.target.value)}>
              {semantics.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
            <select value={snapRadius} onChange={(e) => setSnapRadius(Number(e.target.value))}>
              <option value={12}>吸附 12px</option>
              <option value={24}>吸附 24px</option>
              <option value={48}>吸附 48px</option>
              <option value={96}>吸附 96px</option>
            </select>
          </div>
          <div className="buttonGrid">
            <button onClick={() => void saveCurrent(false)} disabled={currentPoints.length < 2}>保存当前</button>
            <button onClick={() => void saveCurrent(true)} disabled={currentPoints.length < 2}>保存并下一条</button>
            <button onClick={undoPoint} disabled={!currentPoints.length}>撤回点</button>
            <button onClick={deleteSelectedPoint} disabled={selectedPointIndex === null}>删除选中点</button>
            <button onClick={() => { setClosed(!closed); void refreshRoute(currentPoints, !closed); }} disabled={currentPoints.length < 3}>
              {closed ? "取消闭合" : "闭合"}
            </button>
            <button onClick={clearCurrent}>清空当前</button>
          </div>
        </section>

        <section className="card">
          <h2>自动分段与路径候选</h2>
          <select value={autoSegmentMode} onChange={(e) => setAutoSegmentMode(e.target.value)}>
            {autoSegmentModes.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
          <button onClick={() => void handleAutoSegment()} disabled={!session}>几何自动分段</button>
          <div className="branchList">
            {routeSegments.some((segment) => (segment.alternatives || []).length > 1)
              ? routeSegments.map((segment, index) => {
                  const alternatives = segment.alternatives || [];
                  if (alternatives.length <= 1) return null;
                  return (
                    <div className="branchItem" key={`branch-${index}`}>
                      <span>第 {index + 1} 段：方案 {(branchChoices[index] || 0) + 1} / {alternatives.length}</span>
                      <div>
                        <button onClick={() => void shiftBranchChoice(index, -1)}>上一候选</button>
                        <button onClick={() => void shiftBranchChoice(index, 1)}>下一候选</button>
                      </div>
                    </div>
                  );
                })
              : <span className="muted">当前路径没有明显分支候选</span>}
          </div>
        </section>

        <section className="card">
          <h2>显示与导出</h2>
          <div className="toggleRow">
            <label><input type="checkbox" checked={showImage} onChange={(e) => setShowImage(e.target.checked)} /> 原图</label>
            <label><input type="checkbox" checked={showFullSkeleton} onChange={(e) => setShowFullSkeleton(e.target.checked)} /> 完整骨架红点</label>
            <label><input type="checkbox" checked={showEdgeSkeleton} onChange={(e) => setShowEdgeSkeleton(e.target.checked)} /> 切段骨架绿线</label>
            <label><input type="checkbox" checked={showDesignStrokes} onChange={(e) => setShowDesignStrokes(e.target.checked)} /> 设计笔画绿线</label>
            <label><input type="checkbox" checked={showSavedCurves} onChange={(e) => setShowSavedCurves(e.target.checked)} /> 已保存蓝线</label>
            <label><input type="checkbox" checked={showReference} onChange={(e) => setShowReference(e.target.checked)} /> 右上角原图参考</label>
          </div>
          <select value={referenceOpacity} onChange={(e) => setReferenceOpacity(Number(e.target.value))}>
            <option value={40}>参考图透明度 40%</option>
            <option value={60}>参考图透明度 60%</option>
            <option value={78}>参考图透明度 78%</option>
            <option value={100}>参考图透明度 100%</option>
          </select>
          <div className="twoCols">
            <select value={degree} onChange={(e) => setDegree(e.target.value)}>
              <option value="auto">degree 自动</option>
              <option value="3">degree 3</option>
              <option value="5">degree 5</option>
              <option value="7">degree 7</option>
            </select>
            <label className="checkPill">
              <input type="checkbox" checked={precisionFit} onChange={(e) => setPrecisionFit(e.target.checked)} />
              精度优先
            </label>
          </div>
          <div className="buttonGrid">
            <button className="warning" onClick={() => void handleExport()} disabled={!session || designCurves.length === 0}>异步导出 Alias</button>
          </div>
          {exportJob && (
            <div className="jobBox">
              <div><strong>{exportJob.message || exportJob.status}</strong><span>{exportJob.progress || 0}%</span></div>
              <progress max={100} value={exportJob.progress || 0} />
              {exportJob.result?.exports && (
                <div className="links">
                  {Object.entries(exportJob.result.exports).map(([kind, item]) => <a key={kind} href={item.url}>{kind}</a>)}
                </div>
              )}
            </div>
          )}
        </section>

        <section className="card growCard">
          <h2>曲线列表</h2>
          <div className="buttonGrid">
            <button onClick={() => setSelectedCurveIds(new Set(designCurves.map((curve) => curve.id)))} disabled={!designCurves.length}>全选曲线</button>
            <button onClick={deleteSelectedCurves} disabled={selectedCurveIds.size === 0}>批量删除选中</button>
          </div>
          <div className="curveList">
            {designCurves.map((curve, index) => (
              <div key={curve.id} className={`curveItem ${activeCurveId === curve.id ? "active" : ""}`}>
                <input
                  type="checkbox"
                  checked={selectedCurveIds.has(curve.id)}
                  onChange={() => toggleCurveSelected(curve.id)}
                />
                <button onClick={() => void loadCurve(curve)}>
                  <strong>{index + 1}. {curve.semantic}</strong>
                  <span>{curve.closed ? "闭合" : "开放"} / {curve.manual_points.length} 点</span>
                </button>
                <button className="iconButton" onClick={() => deleteCurve(curve.id)}>删</button>
              </div>
            ))}
          </div>
        </section>
      </aside>

      <main className="workArea">
        <div className="toolbar">
          <span>{status}</span>
          <span>{routeStats ? `${routeStats.pointCount} 点 / ${Math.round(routeStats.length)}px` : "等待分段"}</span>
        </div>
        <div ref={stageWrapRef} className="canvasHost">
          <Stage width={stageSize.width} height={stageSize.height} onClick={(e) => void handleStageClick(e)} onWheel={handleWheel}>
            <Layer>
              <Group x={transform.x} y={transform.y} scaleX={transform.scale} scaleY={transform.scale}>
                {showImage && img && <KonvaImage image={img} width={imageSize.width} height={imageSize.height} opacity={0.58} />}
                {showFullSkeleton && (
                  <Shape
                    listening={false}
                    sceneFunc={(context, shape) => {
                      context.beginPath();
                      for (const [x, y] of skeletonPoints) context.rect(x - 0.55, y - 0.55, 1.1, 1.1);
                      context.fillStrokeShape(shape);
                    }}
                    fill="#ef4444"
                    opacity={0.82}
                  />
                )}
                {showEdgeSkeleton && (
                  <Shape
                    listening={false}
                    sceneFunc={(context, shape) => {
                      context.beginPath();
                      for (const edge of edgePolylines) {
                        const pts = edge.points || [];
                        if (pts.length < 2) continue;
                        context.moveTo(pts[0][0], pts[0][1]);
                        for (let i = 1; i < pts.length; i += 1) context.lineTo(pts[i][0], pts[i][1]);
                      }
                      context.fillStrokeShape(shape);
                    }}
                    stroke="rgba(0,140,115,.46)"
                    strokeWidth={1.25 / transform.scale}
                    lineCap="round"
                    lineJoin="round"
                  />
                )}
                {showDesignStrokes && (
                  <Shape
                    listening={false}
                    sceneFunc={(context, shape) => {
                      context.beginPath();
                      for (const stroke of designStrokePolylines) {
                        const pts = stroke.points || [];
                        if (pts.length < 2) continue;
                        context.moveTo(pts[0][0], pts[0][1]);
                        for (let i = 1; i < pts.length; i += 1) context.lineTo(pts[i][0], pts[i][1]);
                      }
                      context.fillStrokeShape(shape);
                    }}
                    stroke="rgba(0,95,55,.82)"
                    strokeWidth={1.8 / transform.scale}
                    lineCap="round"
                    lineJoin="round"
                  />
                )}
                {showSavedCurves && designCurves.map((curve) => (
                  <Line
                    key={curve.id}
                    points={flatten(curve.routed_points)}
                    stroke="#0b66ff"
                    strokeWidth={2.2 / transform.scale}
                    lineCap="round"
                    lineJoin="round"
                    listening={false}
                  />
                ))}
                {routeSegments.flatMap((segment, segmentIndex) =>
                  (segment.alternatives || []).map((alternative, alternativeIndex) => {
                    const selected = segment.selected_candidate ?? branchChoices[segmentIndex] ?? 0;
                    if (alternativeIndex === selected || !alternative.points?.length) return null;
                    const colors = ["#f97316", "#a855f7", "#0891b2"];
                    return (
                      <Line
                        key={`alternative-${segmentIndex}-${alternativeIndex}`}
                        points={flatten(alternative.points)}
                        stroke={colors[alternativeIndex % colors.length]}
                        strokeWidth={1.4 / transform.scale}
                        dash={[8 / transform.scale, 5 / transform.scale]}
                        opacity={0.72}
                        listening={false}
                      />
                    );
                  })
                )}
                {routedPoints.length > 1 && (
                  <Line points={flatten(routedPoints)} stroke="#006dff" strokeWidth={2.5 / transform.scale} lineCap="round" lineJoin="round" />
                )}
                {currentPoints.map((point, index) => point.snap_source === "saved_curve_anchor" && (
                  <Group key={`link-${index}`} listening={false}>
                    <Circle
                      x={point.x}
                      y={point.y}
                      radius={12 / transform.scale}
                      stroke="#dc2626"
                      strokeWidth={2.2 / transform.scale}
                    />
                    <Text
                      x={point.x + 13 / transform.scale}
                      y={point.y + 7 / transform.scale}
                      text="link"
                      fontSize={12 / transform.scale}
                      fill="#b91c1c"
                    />
                  </Group>
                ))}
                {currentPoints.map((point, index) => (
                  <Group key={`${point.x}-${point.y}-${index}`}>
                    <Circle
                      x={point.x}
                      y={point.y}
                      radius={5 / transform.scale}
                      fill={index === selectedPointIndex ? "#ff8a2a" : index === currentPoints.length - 1 ? "#ffd166" : "#695cff"}
                      stroke="#ffffff"
                      strokeWidth={1.4 / transform.scale}
                      draggable
                      onClick={(event) => {
                        event.cancelBubble = true;
                        setSelectedPointIndex(index);
                      }}
                      onDragEnd={(event) => {
                        const p = event.target.position();
                        void updatePoint(index, { x: p.x, y: p.y });
                      }}
                    />
                    <Text x={point.x + 6 / transform.scale} y={point.y - 14 / transform.scale} text={`${index + 1}`} fontSize={12 / transform.scale} fill="#172033" />
                  </Group>
                ))}
              </Group>
              {showReference && img && referenceScale > 0 && (
                <Group x={stageSize.width - referenceWidth - 20} y={76} listening={false}>
                  <Rect
                    x={-8}
                    y={-8}
                    width={referenceWidth + 16}
                    height={referenceHeight + 16}
                    fill="rgba(255,255,255,.82)"
                    stroke="rgba(198,211,225,.85)"
                    cornerRadius={14}
                  />
                  <KonvaImage
                    image={img}
                    width={referenceWidth}
                    height={referenceHeight}
                    opacity={referenceOpacity / 100}
                  />
                </Group>
              )}
            </Layer>
          </Stage>
        </div>
      </main>
    </div>
  );
}
