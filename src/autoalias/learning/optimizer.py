from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autoalias.models import NURBSCurve


@dataclass(slots=True)
class TorchRefineOptions:
    steps: int = 80
    lr: float = 0.025
    samples: int = 180
    chamfer_weight: float = 1.0
    curvature_weight: float = 0.012
    bending_weight: float = 0.0005
    jerk_weight: float = 0.004
    cv_weight: float = 0.012
    tangent_weight: float = 0.02
    inflection_weight: float = 0.006


def refine_curve_torch(
    curve: NURBSCurve,
    target_points: np.ndarray,
    options: TorchRefineOptions | None = None,
) -> NURBSCurve:
    options = options or TorchRefineOptions()
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for differentiable refinement") from exc

    from autoalias.learning.losses import (
        bending_energy_loss,
        chamfer_loss_torch,
        curvature_jerk_loss,
        curvature_smoothness_loss,
        cv_distribution_loss,
        endpoint_tangent_loss,
        inflection_quality_loss,
    )
    from autoalias.learning.torch_bezier import (
        evaluate_bezier_torch,
        evaluate_derivative_torch,
        signed_curvature_2d_torch,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    original = torch.tensor(curve.cvs, dtype=dtype, device=device)
    target = torch.tensor(target_points, dtype=dtype, device=device)
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if target.shape[-1] == 2:
        z = torch.zeros((*target.shape[:-1], 1), dtype=dtype, device=device)
        target = torch.cat([target, z], dim=-1)

    interior = original[1:-1].clone().detach().requires_grad_(True)
    p0 = original[:1].detach()
    p1 = original[-1:].detach()
    opt = torch.optim.Adam([interior], lr=options.lr)
    u = torch.linspace(0.0, 1.0, options.samples, dtype=dtype, device=device)

    best_loss = float("inf")
    best_cvs = original.detach()
    for _ in range(options.steps):
        opt.zero_grad(set_to_none=True)
        cvs = torch.cat([p0, interior, p1], dim=0)
        samples = evaluate_bezier_torch(cvs, u).unsqueeze(0)
        curvature = signed_curvature_2d_torch(cvs, u)
        d2 = evaluate_derivative_torch(cvs, u, 2)
        loss = (
            options.chamfer_weight * chamfer_loss_torch(samples, target)
            + options.curvature_weight * curvature_smoothness_loss(curvature)
            + options.bending_weight * bending_energy_loss(d2)
            + options.jerk_weight * curvature_jerk_loss(curvature)
            + options.cv_weight * cv_distribution_loss(cvs.unsqueeze(0))
            + options.tangent_weight * endpoint_tangent_loss(cvs.unsqueeze(0), target)
            + options.inflection_weight * inflection_quality_loss(curvature)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([interior], 25.0)
        opt.step()
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_cvs = torch.cat([p0, interior.detach(), p1], dim=0).detach()

    refined = NURBSCurve.single_span(
        label=curve.label,
        degree=curve.degree,
        cvs=best_cvs.cpu().numpy(),
        weights=curve.weights.copy(),
        confidence=curve.confidence,
        source=f"{curve.source}+torch_refine",
        metadata={**curve.metadata, "torch_refine_loss": best_loss, "torch_refine_steps": options.steps},
    )
    return refined

