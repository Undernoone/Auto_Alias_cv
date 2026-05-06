from __future__ import annotations


def chamfer_loss_torch(a, b):
    import torch

    dist = torch.cdist(a[..., :2], b[..., :2])
    return 0.5 * (dist.min(dim=-1).values.mean() + dist.min(dim=-2).values.mean())


def curvature_smoothness_loss(curvature):
    dk = curvature[..., 1:] - curvature[..., :-1]
    return (dk * dk).mean()


def curvature_jerk_loss(curvature):
    if curvature.shape[-1] < 4:
        return curvature.new_tensor(0.0)
    dk = curvature[..., 1:] - curvature[..., :-1]
    d2k = dk[..., 1:] - dk[..., :-1]
    return (d2k * d2k).mean()


def bending_energy_loss(d2):
    return (d2 * d2).sum(dim=-1).mean()


def cv_distribution_loss(cvs):
    seg = (cvs[..., 1:, :2] - cvs[..., :-1, :2]).norm(dim=-1).clamp_min(1e-6)
    normalized = seg / seg.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    spacing = ((normalized - 1.0) ** 2).mean()
    if cvs.shape[-2] < 4:
        return spacing
    d2 = cvs[..., 2:, :2] - 2.0 * cvs[..., 1:-1, :2] + cvs[..., :-2, :2]
    polygon_fair = (d2 * d2).sum(dim=-1).mean() / (seg.mean() ** 2).clamp_min(1e-6)
    return spacing + 0.1 * polygon_fair


def endpoint_tangent_loss(cvs, target_points):
    import torch

    target_start = target_points[..., 1, :2] - target_points[..., 0, :2]
    target_end = target_points[..., -1, :2] - target_points[..., -2, :2]
    cv_start = cvs[..., 1, :2] - cvs[..., 0, :2]
    cv_end = cvs[..., -1, :2] - cvs[..., -2, :2]
    target_start = target_start / target_start.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    target_end = target_end / target_end.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    cv_start = cv_start / cv_start.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    cv_end = cv_end / cv_end.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return 0.5 * ((1.0 - (target_start * cv_start).sum(dim=-1)).mean() + (1.0 - (target_end * cv_end).sum(dim=-1)).mean())


def inflection_quality_loss(curvature):
    """Prefer clean, low-jerk zero crossing for S curves without adding fake oscillations."""
    import torch

    signs = torch.sign(curvature)
    sign_changes = (signs[..., 1:] * signs[..., :-1] < 0).float().sum(dim=-1)
    extra = torch.relu(sign_changes - 1.0).mean()
    near_zero = torch.exp(-50.0 * curvature.abs()).mean()
    return extra - 0.02 * near_zero

