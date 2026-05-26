from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autoalias.models import NURBSCurve
from autoalias.pipeline import CurveReconstructionPipeline, PipelineOptions
from autoalias.quality import ClassAValidator


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "fit-image":
        return _fit_image(args)
    if args.command == "fit-points":
        return _fit_points(args)
    if args.command == "review-image":
        return _review_image(args)
    if args.command == "skeleton-review":
        return _skeleton_review(args)
    if args.command == "desktop-review":
        return _desktop_review(args)
    if args.command == "fit-reviewed":
        return _fit_reviewed(args)
    if args.command == "build-training-set":
        return _build_training_set(args)
    if args.command == "train-decoder":
        return _train_decoder(args)
    if args.command == "decode-checkpoint":
        return _decode_checkpoint(args)
    if args.command == "infer-image-checkpoint":
        return _infer_image_checkpoint(args)
    if args.command == "validate-json":
        return _validate_json(args)
    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoalias",
        description="Reconstruct Alias-ready single-span NURBS curves from automotive images.",
    )
    sub = parser.add_subparsers(dest="command")

    p_img = sub.add_parser("fit-image", help="fit curves from an image")
    p_img.add_argument("image", type=Path)
    _add_common_fit_args(p_img)

    p_pts = sub.add_parser("fit-points", help="fit curves from ordered point JSON")
    p_pts.add_argument("points_json", type=Path)
    _add_common_fit_args(p_pts)

    p_review = sub.add_parser("review-image", help="open the interactive topology correction tool")
    p_review.add_argument("image", type=Path)
    p_review.add_argument("--out", type=Path, default=Path("corrections"))
    p_review.add_argument("--host", default="127.0.0.1")
    p_review.add_argument("--port", type=int, default=8765)
    p_review.add_argument("--no-browser", action="store_true")

    p_skeleton_review = sub.add_parser(
        "skeleton-review",
        help="serve a LAN web tool for image upload, skeleton routing, manual splitting and IGES export",
    )
    p_skeleton_review.add_argument("--out", type=Path, default=Path("lan_reviews"))
    p_skeleton_review.add_argument("--host", default="0.0.0.0")
    p_skeleton_review.add_argument("--port", type=int, default=8765)
    p_skeleton_review.add_argument("--no-browser", action="store_true")

    p_desktop = sub.add_parser(
        "desktop-review",
        help="open the PySide6 desktop GUI for skeleton splitting and IGES export",
    )
    p_desktop.add_argument("image", nargs="?", type=Path)
    p_desktop.add_argument("--out", type=Path, default=Path("lan_reviews"))

    p_fit_reviewed = sub.add_parser(
        "fit-reviewed",
        help="fit Alias-ready curves from manually saved review design curves",
    )
    p_fit_reviewed.add_argument("annotations", nargs="+", type=Path)
    p_fit_reviewed.add_argument("--out", type=Path, default=Path("out_reviewed"))
    p_fit_reviewed.add_argument("--degree", default="auto", help="auto, 3, 4, 5, 6 or 7")
    p_fit_reviewed.add_argument("--min-points", type=int, default=8)
    p_fit_reviewed.add_argument("--fast", action="store_true", help="use the fast export path for large auto-segmented jobs")
    p_fit_reviewed.add_argument("--max-fit-points", type=int, default=None)
    p_fit_reviewed.add_argument(
        "--fit-mode",
        choices=("manual_class_a_g2", "standard", "precision"),
        default="manual_class_a_g2",
        help="manual_class_a_g2 keeps Class-A/G2; precision ignores CV aesthetics and fits routed points as closely as possible",
    )
    p_fit_reviewed.add_argument(
        "--wire",
        action="store_true",
        help="also convert reviewed_curves.igs to reviewed_curves.wire through Autodesk IgesToAl",
    )
    p_fit_reviewed.add_argument(
        "--iges-to-al",
        type=Path,
        default=None,
        help="full path to Autodesk Alias IgesToAl.exe; otherwise AUTOALIAS_IGES_TO_AL/PATH is used",
    )
    p_fit_reviewed.add_argument("--diagnostic-preview", action="store_true", help="also write the heavy CV/comb diagnostic SVG")

    p_build = sub.add_parser(
        "build-training-set",
        help="convert reviewed manual curves into neural decoder supervision JSON",
    )
    p_build.add_argument("annotations", nargs="+", type=Path)
    p_build.add_argument("--out", type=Path, default=Path("data/manual_curve_supervision.json"))
    p_build.add_argument("--degree", default="auto", help="auto, 3, 4, 5, 6 or 7")
    p_build.add_argument("--min-points", type=int, default=8)

    p_train = sub.add_parser("train-decoder", help="train the neural NURBS decoder")
    p_train.add_argument("json", nargs="+", type=Path)
    p_train.add_argument("--out", type=Path, default=Path("checkpoints/manual_curve_decoder.pt"))
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--hidden-dim", type=int, default=256)
    p_train.add_argument("--layers", type=int, default=4)
    p_train.add_argument("--heads", type=int, default=8)

    p_decode = sub.add_parser(
        "decode-checkpoint",
        help="load a trained decoder checkpoint and export Alias-ready curves",
    )
    p_decode.add_argument("checkpoint", type=Path)
    p_decode.add_argument("json", type=Path)
    p_decode.add_argument("--out", type=Path, default=Path("out_checkpoint"))
    p_decode.add_argument("--degree-source", choices=("input", "predicted"), default="input")
    p_decode.add_argument("--n-points", type=int, default=128)
    p_decode.add_argument("--max-curves", type=int, default=200)

    p_img_decode = sub.add_parser(
        "infer-image-checkpoint",
        help="extract image candidates, decode with checkpoint, and export IGES",
    )
    p_img_decode.add_argument("image", type=Path)
    p_img_decode.add_argument("--checkpoint", required=True, type=Path)
    p_img_decode.add_argument("--out", type=Path, default=Path("out_image_checkpoint"))
    p_img_decode.add_argument("--degree-source", choices=("input", "predicted"), default="predicted")
    p_img_decode.add_argument("--n-points", type=int, default=128)
    p_img_decode.add_argument("--max-curves", type=int, default=80)

    p_val = sub.add_parser("validate-json", help="validate an AutoAlias curves.json file")
    p_val.add_argument("curves_json", type=Path)

    return parser


def _add_common_fit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--degree", default="auto", help="auto, 3, 4, 5, 6 or 7")
    parser.add_argument("--max-curves", type=int, default=400)
    parser.add_argument("--unit", default="px")
    parser.add_argument(
        "--exports",
        default="json,svg,dxf,iges",
        help="comma-separated: json,svg,dxf,iges",
    )
    parser.add_argument(
        "--torch-refine",
        action="store_true",
        help="run differentiable PyTorch CV refinement after deterministic fitting",
    )
    parser.add_argument("--torch-steps", type=int, default=80)
    parser.add_argument(
        "--no-fill-missing",
        action="store_true",
        help="disable second-pass tracing of uncovered line-art pixels",
    )
    parser.add_argument("--fill-passes", type=int, default=4)


def _fit_image(args: argparse.Namespace) -> int:
    degree = _parse_degree(args.degree)
    pipeline = CurveReconstructionPipeline(
        PipelineOptions(
            degree=degree,
            max_curves=args.max_curves,
            unit=args.unit,
            exports=_parse_exports(args.exports),
            torch_refine=args.torch_refine,
            torch_steps=args.torch_steps,
            fill_missing_line_art=not args.no_fill_missing,
            fill_missing_passes=args.fill_passes,
        )
    )
    result = pipeline.run_image(args.image, args.out)
    return _print_result(result.curves, result.reports, result.output_dir)


def _fit_points(args: argparse.Namespace) -> int:
    degree = _parse_degree(args.degree)
    pipeline = CurveReconstructionPipeline(
        PipelineOptions(
            degree=degree,
            max_curves=args.max_curves,
            unit=args.unit,
            exports=_parse_exports(args.exports),
            torch_refine=args.torch_refine,
            torch_steps=args.torch_steps,
            fill_missing_line_art=not args.no_fill_missing,
            fill_missing_passes=args.fill_passes,
        )
    )
    result = pipeline.run_points_json(args.points_json, args.out)
    return _print_result(result.curves, result.reports, result.output_dir)


def _review_image(args: argparse.Namespace) -> int:
    from autoalias.review.server import run_review_app

    run_review_app(
        args.image,
        args.out,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )
    return 0


def _skeleton_review(args: argparse.Namespace) -> int:
    from autoalias.review.workflow_server import run_skeleton_review_server

    run_skeleton_review_server(
        args.out,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )
    return 0


def _desktop_review(args: argparse.Namespace) -> int:
    from autoalias.gui.desktop_editor import main as gui_main

    argv: list[str] = []
    if args.image:
        argv.append(str(args.image))
    argv.extend(["--out", str(args.out)])
    return gui_main(argv)


def _fit_reviewed(args: argparse.Namespace) -> int:
    from autoalias.review.fit_reviewed import fit_reviewed_annotations

    degree = _parse_degree(args.degree)
    result = fit_reviewed_annotations(
        args.annotations,
        args.out,
        degree=degree,
        min_points=args.min_points,
        max_fit_points=args.max_fit_points,
        diagnostic_preview=args.diagnostic_preview,
        fast_mode=args.fast,
        fit_mode=args.fit_mode,
        wire_export=args.wire,
        iges_to_al=args.iges_to_al,
    )
    passed = sum(1 for report in result.reports if report.passed)
    print(f"AutoAlias fitted {len(result.curves)} manually reviewed curve(s) to {result.out}")
    print(f"Quality: {passed}/{len(result.reports)} passed")
    print(f"Alias file: {result.out / 'reviewed_curves.igs'}")
    if result.wire_result is not None:
        if result.wire_result.ok:
            print(f"Alias WIRE file: {result.wire_result.wire_path}")
        else:
            print(f"WIRE not generated: {result.wire_result.message}")
            print(f"WIRE status: {result.out / 'reviewed_curves.wire_status.json'}")
    print(f"Compact curve JSON: {result.out / 'reviewed_curves.json'}")
    print(f"Preview SVG: {result.out / 'reviewed_clean_preview.svg'}")
    if result.skipped_count:
        print(f"Skipped {result.skipped_count} incomplete reviewed curve(s)")
    for report in result.reports[:40]:
        status = "PASS" if report.passed else "WARN"
        degree_value = report.metrics.get("degree", "?")
        span = report.metrics.get("span", "?")
        print(f"{status} {report.label}: degree={degree_value} span={span}")
        for warning in report.warnings[:3]:
            print(f"  - {warning}")
    if len(result.reports) > 40:
        print(f"... {len(result.reports) - 40} more curve report(s) omitted")
    return 0 if result.curves else 1


def _build_training_set(args: argparse.Namespace) -> int:
    from autoalias.learning.annotation_dataset import build_supervision_from_annotations

    degree = _parse_degree(args.degree)
    result = build_supervision_from_annotations(
        args.annotations,
        args.out,
        degree=degree,
        min_points=args.min_points,
    )
    print(f"AutoAlias wrote {result.item_count} training curve(s) to {result.out}")
    if result.skipped_count:
        print(f"Skipped {result.skipped_count} incomplete or unfittable curve(s)")
    return 0 if result.item_count else 1


def _train_decoder(args: argparse.Namespace) -> int:
    from autoalias.learning.train_supervised import train

    return train(args)


def _decode_checkpoint(args: argparse.Namespace) -> int:
    from autoalias.learning.predict_checkpoint import decode

    return decode(args)


def _infer_image_checkpoint(args: argparse.Namespace) -> int:
    from autoalias.learning.predict_checkpoint import decode_image

    return decode_image(
        args.checkpoint,
        args.image,
        args.out,
        degree_source=args.degree_source,
        n_points=args.n_points,
        max_curves=args.max_curves,
    )


def _validate_json(args: argparse.Namespace) -> int:
    data = json.loads(args.curves_json.read_text(encoding="utf-8"))
    validator = ClassAValidator()
    reports = []
    for item in data.get("curves", []):
        curve = NURBSCurve(
            label=item["label"],
            degree=int(item["degree"]),
            cvs=item["cv"],
            weights=item["weights"],
            knots=item["knots"],
            confidence=float(item.get("confidence", 1.0)),
            source=item.get("source", "json"),
            metadata=item.get("metadata", {}),
        )
        reports.append(validator.validate(curve))
    failed = [r for r in reports if not r.passed]
    for report in reports:
        status = "PASS" if report.passed else "WARN"
        print(f"{status} {report.label}: {len(report.warnings)} warning(s)")
        for warning in report.warnings:
            print(f"  - {warning}")
    return 1 if failed else 0


def _print_result(curves, reports, out: Path) -> int:
    passed = sum(1 for report in reports if report.passed)
    print(f"AutoAlias wrote {len(curves)} curve(s) to {out}")
    print(f"Quality: {passed}/{len(reports)} passed")
    if passed:
        print(f"Accepted Alias file: {out / 'accepted_curves.igs'}")
        print(f"Beautified Alias file: {out / 'beautified_curves.igs'}")
    limit = 40
    for report in reports[:limit]:
        status = "PASS" if report.passed else "WARN"
        span = report.metrics.get("span", "?")
        degree = report.metrics.get("degree", "?")
        print(f"{status} {report.label}: degree={degree} span={span}")
        for warning in report.warnings[:4]:
            print(f"  - {warning}")
    if len(reports) > limit:
        print(f"... {len(reports) - limit} more curve report(s) omitted")
    return 0 if curves else 1


def _parse_degree(value: str) -> int | str:
    if value == "auto":
        return value
    degree = int(value)
    if degree not in (3, 4, 5, 6, 7):
        raise argparse.ArgumentTypeError("degree must be auto, 3, 4, 5, 6 or 7")
    return degree


def _parse_exports(value: str) -> tuple[str, ...]:
    allowed = {"json", "svg", "dxf", "iges"}
    exports = tuple(v.strip().lower() for v in value.split(",") if v.strip())
    bad = set(exports) - allowed
    if bad:
        raise argparse.ArgumentTypeError(f"unknown export(s): {', '.join(sorted(bad))}")
    return exports


if __name__ == "__main__":
    sys.exit(main())
