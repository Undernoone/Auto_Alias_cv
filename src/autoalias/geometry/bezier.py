from __future__ import annotations

from math import comb

import numpy as np


def bernstein_basis(degree: int, u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float).reshape(-1)
    basis = np.empty((len(u), degree + 1), dtype=float)
    for i in range(degree + 1):
        basis[:, i] = comb(degree, i) * (u**i) * ((1.0 - u) ** (degree - i))
    return basis


def evaluate_bezier(cvs: np.ndarray, u: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    cvs = np.asarray(cvs, dtype=float)
    degree = len(cvs) - 1
    basis = bernstein_basis(degree, u)
    if weights is None or np.allclose(weights, 1.0):
        return basis @ cvs
    weights = np.asarray(weights, dtype=float).reshape(1, -1)
    weighted = basis * weights
    denom = np.sum(weighted, axis=1, keepdims=True)
    return (weighted @ cvs) / np.maximum(denom, 1e-12)


def derivative_control_points(cvs: np.ndarray, order: int = 1) -> np.ndarray:
    out = np.asarray(cvs, dtype=float)
    degree = len(out) - 1
    for k in range(order):
        if degree <= 0:
            return np.zeros((1, out.shape[1]), dtype=float)
        out = degree * np.diff(out, axis=0)
        degree -= 1
    return out


def evaluate_derivative(cvs: np.ndarray, u: np.ndarray, order: int = 1) -> np.ndarray:
    d_cvs = derivative_control_points(cvs, order)
    if len(d_cvs) == 1:
        return np.repeat(d_cvs, len(np.asarray(u).reshape(-1)), axis=0)
    return evaluate_bezier(d_cvs, u)


def signed_curvature_2d(cvs: np.ndarray, u: np.ndarray) -> np.ndarray:
    d1 = evaluate_derivative(cvs[:, :2], u, 1)
    d2 = evaluate_derivative(cvs[:, :2], u, 2)
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    speed = np.linalg.norm(d1, axis=1)
    return cross / np.maximum(speed**3, 1e-12)


def sample_arclength(cvs: np.ndarray, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 1.0, n)
    pts = evaluate_bezier(cvs, u)
    seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] > 0:
        s /= s[-1]
    return u, s


def single_span_knots(degree: int) -> np.ndarray:
    return np.array([0.0] * (degree + 1) + [1.0] * (degree + 1), dtype=float)

