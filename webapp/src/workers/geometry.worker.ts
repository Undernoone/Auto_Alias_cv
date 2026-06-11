import type { RouteSegment } from "../types";

type MeasureRequest = {
  type: "measure-route";
  id: number;
  segments: RouteSegment[];
};

function measurePolyline(points: number[][]) {
  let length = 0;
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < points.length; i += 1) {
    const [x, y] = points[i];
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    if (i > 0) {
      const [px, py] = points[i - 1];
      length += Math.hypot(x - px, y - py);
    }
  }
  return {
    length,
    bbox: Number.isFinite(minX) ? { minX, minY, maxX, maxY } : null
  };
}

self.onmessage = (event: MessageEvent<MeasureRequest>) => {
  if (event.data.type !== "measure-route") return;
  const allPoints = event.data.segments.flatMap((segment) => segment.points || []);
  const result = measurePolyline(allPoints);
  self.postMessage({
    type: "route-measured",
    id: event.data.id,
    pointCount: allPoints.length,
    ...result
  });
};

export {};

