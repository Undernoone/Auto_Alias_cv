from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from autoalias.geometry.bezier import evaluate_bezier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate supervised synthetic AutoAlias curves.")
    parser.add_argument("--out", type=Path, default=Path("data/synthetic_curves.json"))
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    curves = []
    for i in range(args.count):
        degree = random.choice([3, 5, 7])
        cvs = _random_designer_cvs(rng, degree)
        u = np.linspace(0, 1, 96)
        points = evaluate_bezier(cvs, u)
        points[:, :2] += rng.normal(0, 0.35, size=(len(points), 2))
        curves.append(
            {
                "label": f"synthetic_degree_{degree}",
                "degree": degree,
                "span": 1,
                "points": points[:, :2].round(4).tolist(),
                "cv": cvs.round(4).tolist(),
                "weights": [1.0] * (degree + 1),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"curves": curves}, indent=2), encoding="utf-8")
    print(f"wrote {len(curves)} curves to {args.out}")
    return 0


def _random_designer_cvs(rng, degree: int) -> np.ndarray:
    x = np.linspace(0, rng.uniform(160, 420), degree + 1)
    base_y = rng.uniform(-40, 40)
    amplitude = rng.uniform(5, 55)
    phase = rng.uniform(-0.8, 0.8)
    s_shape = rng.random() < 0.45
    t = np.linspace(0, 1, degree + 1)
    if s_shape:
        y = base_y + amplitude * np.sin((t - 0.5 + phase * 0.1) * np.pi)
    else:
        y = base_y + amplitude * (t - 0.5) ** 2 * rng.choice([-1, 1])
    y += rng.normal(0, amplitude * 0.08, size=degree + 1)
    y[0] += rng.normal(0, 2)
    y[-1] += rng.normal(0, 2)
    return np.column_stack([x, y, np.zeros(degree + 1)])


if __name__ == "__main__":
    raise SystemExit(main())

