from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from autoalias.exporters import write_iges, write_json_bundle, write_svg_preview
from autoalias.geometry.polyline import resample_polyline
from autoalias.learning.dataset import CLASS_TO_DEGREE
from autoalias.models import CurveCandidate, NURBSCurve
from autoalias.quality import ClassAValidator
from autoalias.vision.extractor import ExtractorOptions, OpenCVCurveExtractor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decode AutoAlias curves from a trained checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("json", type=Path)
    parser.add_argument("--out", type=Path, default=Path("out_checkpoint"))
    parser.add_argument("--degree-source", choices=("input", "predicted"), default="input")
    parser.add_argument("--n-points", type=int, default=128)
    parser.add_argument("--max-curves", type=int, default=200)
    return decode(parser.parse_args(argv))


def decode_image(
    checkpoint_path: str | Path,
    image_path: str | Path,
    out: str | Path,
    *,
    degree_source: str = "predicted",
    n_points: int = 128,
    max_curves: int = 80,
) -> int:
    extractor = OpenCVCurveExtractor(ExtractorOptions(max_curves=max_curves))
    candidates = extractor.extract(image_path)
    items = [_candidate_to_item(candidate) for candidate in candidates]
    return decode_items(
        Path(checkpoint_path),
        items,
        Path(out),
        degree_source=degree_source,
        n_points=n_points,
        max_curves=max_curves,
        background_image=Path(image_path),
        preview_candidates=candidates,
    )


def decode(args: argparse.Namespace) -> int:
    items = _load_curve_items(args.json)
    return decode_items(
        args.checkpoint,
        items,
        args.out,
        degree_source=args.degree_source,
        n_points=args.n_points,
        max_curves=args.max_curves,
    )


def decode_items(
    checkpoint_path: Path,
    items: list[dict[str, Any]],
    out: Path,
    *,
    degree_source: str = "input",
    n_points: int = 128,
    max_curves: int = 200,
    background_image: Path | None = None,
    preview_candidates: list[CurveCandidate] | None = None,
) -> int:
    import torch

    from autoalias.learning.decoder import CurveTokenNURBSDecoder

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_args = checkpoint.get("args", {})
    model = CurveTokenNURBSDecoder(
        hidden_dim=int(train_args.get("hidden_dim", 256)),
        layers=int(train_args.get("layers", 4)),
        heads=int(train_args.get("heads", 8)),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    curves: list[NURBSCurve] = []
    items = items[: max(int(max_curves), 1)]
    with torch.no_grad():
        for index, item in enumerate(items):
            points = _points_tensor(item, n_points)
            pred = model(points)
            pred_class = int(torch.argmax(pred["degree_logits"], dim=-1).item())
            predicted_degree = CLASS_TO_DEGREE[pred_class]
            degree = int(item.get("degree", predicted_degree)) if degree_source == "input" else predicted_degree
            cv_count = degree + 1
            cvs = pred["cv"][0, :cv_count].detach().cpu().numpy()
            weights = pred["weights"][0, :cv_count].detach().cpu().numpy()
            curves.append(
                NURBSCurve.single_span(
                    label=str(item.get("label", f"decoded_curve_{index:04d}")),
                    degree=degree,
                    cvs=cvs,
                    weights=np.maximum(weights, 1e-4),
                    confidence=float(pred["confidence"][0, 0].detach().cpu()),
                    source=str(checkpoint_path),
                    metadata={
                        "checkpoint": str(checkpoint_path),
                        "degree_source": degree_source,
                        "predicted_degree": predicted_degree,
                        "input_degree": item.get("degree"),
                        "input_source": item.get("source"),
                    },
                )
            )

    out.mkdir(parents=True, exist_ok=True)
    validator = ClassAValidator()
    reports = [validator.validate(curve) for curve in curves]
    write_json_bundle(out / "decoded_curves.json", curves, reports)
    if curves:
        write_iges(out / "decoded_curves.igs", curves)
        candidates = preview_candidates or []
        write_svg_preview(out / "preview.svg", curves, candidates, background_image=background_image)
        write_svg_preview(out / "decoded_preview.svg", curves, candidates, background_image=background_image)
        write_svg_preview(
            out / "clean_preview.svg",
            curves,
            [],
            background_image=background_image,
            show_labels=False,
            show_comb=False,
            show_cvs=False,
            show_candidates=False,
        )
    print(f"decoded {len(curves)} curve(s) to {out}")
    if curves:
        print(f"Alias IGES: {out / 'decoded_curves.igs'}")
    return 0 if curves else 1


def _load_curve_items(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if "curves" in data:
        return list(data["curves"])
    if "points" in data:
        return [data]
    raise ValueError(f"unsupported curve JSON: {path}")


def _points_tensor(item: dict[str, Any], n_points: int):
    import torch

    points = np.asarray(item["points"], dtype=np.float32)
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points), dtype=np.float32)])
    points = resample_polyline(points, n_points).astype(np.float32)
    return torch.from_numpy(points).unsqueeze(0)


def _candidate_to_item(candidate: CurveCandidate) -> dict[str, Any]:
    return {
        "label": candidate.label,
        "points": candidate.points[:, :2].round(3).tolist(),
        "confidence": candidate.confidence,
        "source": candidate.source,
        "metadata": candidate.metadata,
    }


if __name__ == "__main__":
    raise SystemExit(main())
