from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def as_points3(points: np.ndarray | list[list[float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError("points must be an Nx2 or Nx3 array")
    if arr.shape[1] == 2:
        zeros = np.zeros((arr.shape[0], 1), dtype=float)
        arr = np.hstack([arr, zeros])
    return arr


@dataclass(slots=True)
class CurveCandidate:
    label: str
    points: np.ndarray
    confidence: float = 1.0
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = as_points3(self.points)
        if len(self.points) < 4:
            raise ValueError("a curve candidate needs at least 4 points")


@dataclass(slots=True)
class NURBSCurve:
    label: str
    degree: int
    cvs: np.ndarray
    weights: np.ndarray
    knots: np.ndarray
    u_min: float = 0.0
    u_max: float = 1.0
    confidence: float = 1.0
    source: str = "fitted"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cvs = as_points3(self.cvs)
        self.weights = np.asarray(self.weights, dtype=float).reshape(-1)
        self.knots = np.asarray(self.knots, dtype=float).reshape(-1)
        if self.degree < 1:
            raise ValueError("degree must be positive")
        if len(self.cvs) != len(self.weights):
            raise ValueError("weights count must match CV count")
        if np.any(self.weights <= 0):
            raise ValueError("all NURBS weights must be positive")

    @property
    def span_count(self) -> int:
        unique = np.unique(np.round(self.knots, 12))
        return max(0, len(unique) - 1)

    @property
    def is_single_span(self) -> bool:
        expected_cv = self.degree + 1
        if len(self.cvs) != expected_cv:
            return False
        expected_knots = np.array([0.0] * (self.degree + 1) + [1.0] * (self.degree + 1))
        return len(self.knots) == len(expected_knots) and np.allclose(self.knots, expected_knots)

    @classmethod
    def single_span(
        cls,
        label: str,
        degree: int,
        cvs: np.ndarray,
        weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "NURBSCurve":
        cvs = as_points3(cvs)
        if len(cvs) != degree + 1:
            raise ValueError(f"single-span degree {degree} requires {degree + 1} CVs")
        if weights is None:
            weights = np.ones(degree + 1, dtype=float)
        knots = np.array([0.0] * (degree + 1) + [1.0] * (degree + 1), dtype=float)
        return cls(label=label, degree=degree, cvs=cvs, weights=weights, knots=knots, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "degree": self.degree,
            "span": self.span_count,
            "single_span": self.is_single_span,
            "cv": self.cvs.tolist(),
            "weights": self.weights.tolist(),
            "knots": self.knots.tolist(),
            "u_min": self.u_min,
            "u_max": self.u_max,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class QualityReport:
    label: str
    passed: bool
    metrics: dict[str, float | int | bool | list[float] | str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "passed": self.passed,
            "metrics": self.metrics,
            "warnings": self.warnings,
        }

