from __future__ import annotations

import math


def bernstein_basis_torch(degree, u):
    import torch

    values = []
    for i in range(degree + 1):
        c = math.comb(degree, i)
        values.append(c * (u**i) * ((1.0 - u) ** (degree - i)))
    return torch.stack(values, dim=-1)


def evaluate_bezier_torch(cvs, u, weights=None):
    import torch

    degree = cvs.shape[-2] - 1
    basis = bernstein_basis_torch(degree, u)
    while basis.ndim < cvs.ndim:
        basis = basis.unsqueeze(0)
    if weights is None:
        return basis @ cvs
    w = weights
    while w.ndim < basis.ndim:
        w = w.unsqueeze(-2)
    weighted = basis * w
    denom = torch.sum(weighted, dim=-1, keepdim=True).clamp_min(1e-9)
    return weighted @ cvs / denom


def derivative_cvs_torch(cvs, order=1):
    out = cvs
    degree = cvs.shape[-2] - 1
    for _ in range(order):
        out = degree * (out[..., 1:, :] - out[..., :-1, :])
        degree -= 1
    return out


def evaluate_derivative_torch(cvs, u, order=1):
    d_cvs = derivative_cvs_torch(cvs, order)
    return evaluate_bezier_torch(d_cvs, u)


def signed_curvature_2d_torch(cvs, u):
    d1 = evaluate_derivative_torch(cvs[..., :2], u, 1)
    d2 = evaluate_derivative_torch(cvs[..., :2], u, 2)
    cross = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
    speed = d1.norm(dim=-1).clamp_min(1e-9)
    return cross / (speed**3)

