from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from autoalias.geometry.polyline import resample_polyline


DEGREE_TO_CLASS = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4}
CLASS_TO_DEGREE = {v: k for k, v in DEGREE_TO_CLASS.items()}


class CurveSupervisionDataset:
    """JSON dataset for supervised neural NURBS training.

    Each item must contain:
      points: ordered target points
      degree: 3..7
      cv or cvs: ground-truth control vertices
    """

    def __init__(self, paths, n_points: int = 128, max_cvs: int = 8):
        self.paths = [Path(p) for p in paths]
        self.n_points = n_points
        self.max_cvs = max_cvs
        self.items = []
        for path in self.paths:
            self.items.extend(_load_items(path))
        self.items = [item for item in self.items if _has_supervision(item)]
        if not self.items:
            raise ValueError("no supervised curves found; expected points + degree + cv")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import torch

        item = self.items[idx]
        points = np.asarray(item["points"], dtype=np.float32)
        if points.shape[1] == 2:
            points = np.column_stack([points, np.zeros(len(points), dtype=np.float32)])
        points = resample_polyline(points, self.n_points).astype(np.float32)
        degree = int(item["degree"])
        cv = np.asarray(item.get("cv", item.get("cvs")), dtype=np.float32)
        if cv.shape[1] == 2:
            cv = np.column_stack([cv, np.zeros(len(cv), dtype=np.float32)])
        cv_padded = np.zeros((self.max_cvs, 3), dtype=np.float32)
        mask = np.zeros((self.max_cvs,), dtype=np.float32)
        count = min(len(cv), self.max_cvs)
        cv_padded[:count] = cv[:count]
        mask[:count] = 1.0
        return {
            "points": torch.from_numpy(points),
            "degree_class": torch.tensor(DEGREE_TO_CLASS[degree], dtype=torch.long),
            "degree": torch.tensor(degree, dtype=torch.long),
            "cv": torch.from_numpy(cv_padded),
            "cv_mask": torch.from_numpy(mask),
        }


def _load_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if "curves" in data:
        return data["curves"]
    if "points" in data:
        return [data]
    raise ValueError(f"unsupported dataset JSON: {path}")


def _has_supervision(item: dict) -> bool:
    degree = int(item.get("degree", -1))
    return "points" in item and degree in DEGREE_TO_CLASS and ("cv" in item or "cvs" in item)

