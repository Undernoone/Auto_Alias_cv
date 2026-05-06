from __future__ import annotations

import numpy as np


def chord_length_parameter(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    d = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(d)])
    if u[-1] <= 1e-12:
        return np.linspace(0.0, 1.0, len(pts))
    return u / u[-1]


def remove_duplicate_points(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return pts
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i, :2] - pts[keep[-1], :2]) > eps:
            keep.append(i)
    return pts[keep]


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    pts = remove_duplicate_points(points)
    if len(pts) < 2:
        raise ValueError("need at least two distinct points")
    d = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 1e-12:
        return np.repeat(pts[:1], n, axis=0)
    target = np.linspace(0.0, s[-1], n)
    out = np.empty((n, pts.shape[1]), dtype=float)
    for j in range(pts.shape[1]):
        out[:, j] = np.interp(target, s, pts[:, j])
    return out


def smooth_polyline(points: np.ndarray, window: int = 7) -> np.ndarray:
    if window < 3 or len(points) < window:
        return np.asarray(points, dtype=float)
    if window % 2 == 0:
        window += 1
    pts = np.asarray(points, dtype=float)
    pad = window // 2
    padded = np.pad(pts, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    out = np.vstack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(pts.shape[1])]).T
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


def estimate_polyline_curvature(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 5:
        return np.zeros(len(pts), dtype=float)
    p = smooth_polyline(pts[:, :2], window=5)
    d1 = np.gradient(p, axis=0)
    d2 = np.gradient(d1, axis=0)
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    speed = np.linalg.norm(d1, axis=1)
    return cross / np.maximum(speed**3, 1e-12)


def point_to_point_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.asarray(a, dtype=float)[:, :2]
    b2 = np.asarray(b, dtype=float)[:, :2]
    diff = a2[:, None, :] - b2[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))

