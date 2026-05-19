#!/usr/bin/env python3
"""
stp_to_iges.py

Convert STEP/STP files to IGES/IGS using pythonocc-core / OpenCascade.

Install:
    conda install -c conda-forge pythonocc-core

Usage:
    python stp_to_iges.py input.stp output.iges
    python stp_to_iges.py input.step
    python stp_to_iges.py ./step_files --output ./iges_files --batch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IGESControl import IGESControl_Writer
    from OCC.Core.IFSelect import IFSelect_RetDone
except ImportError as exc:
    raise SystemExit(
        "缺少依赖 pythonocc-core。请先安装：\n"
        "    conda install -c conda-forge pythonocc-core\n"
        "注意：pythonocc-core 通常推荐用 conda 安装，而不是 pip。"
    ) from exc


STEP_EXTS = {".stp", ".step"}
IGES_EXTS = {".igs", ".iges"}


def convert_step_to_iges(input_file: str | Path, output_file: str | Path | None = None) -> Path:
    """Convert one STEP/STP file to IGES/IGS.

    Args:
        input_file: Path to .stp or .step file.
        output_file: Optional output .iges/.igs path. If omitted, uses input stem + .iges.

    Returns:
        The generated IGES file path.
    """
    input_path = Path(input_file).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")
    if input_path.suffix.lower() not in STEP_EXTS:
        raise ValueError(f"输入文件后缀应为 .stp 或 .step：{input_path}")

    if output_file is None:
        output_path = input_path.with_suffix(".iges")
    else:
        output_path = Path(output_file).expanduser().resolve()
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".iges")
        if output_path.suffix.lower() not in IGES_EXTS:
            raise ValueError(f"输出文件后缀应为 .igs 或 .iges：{output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) 读取 STEP/STP
    reader = STEPControl_Reader()
    read_status = reader.ReadFile(str(input_path))
    if read_status != IFSelect_RetDone:
        raise RuntimeError(f"STEP 文件读取失败：{input_path}")

    # 2) 转换为 OpenCascade 的 Shape
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError(f"STEP 文件转换失败，未得到有效几何体：{input_path}")

    # 3) 写出 IGES
    # unit='MM' 表示毫米；modecr=0 为 Face 模式，兼容性通常较好。
    writer = IGESControl_Writer("MM", 0)
    writer.AddShape(shape)
    ok = writer.Write(str(output_path))
    if not ok:
        raise RuntimeError(f"IGES 文件写入失败：{output_path}")

    return output_path


def batch_convert(input_dir: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    """Convert all .stp/.step files in a directory, non-recursively."""
    src_dir = Path(input_dir).expanduser().resolve()
    if not src_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是文件夹：{src_dir}")

    dst_dir = Path(output_dir).expanduser().resolve() if output_dir else src_dir
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in STEP_EXTS)
    if not files:
        raise FileNotFoundError(f"文件夹中没有 .stp/.step 文件：{src_dir}")

    outputs: list[Path] = []
    for step_file in files:
        out_file = dst_dir / f"{step_file.stem}.iges"
        print(f"Converting: {step_file.name} -> {out_file.name}")
        outputs.append(convert_step_to_iges(step_file, out_file))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert STEP/STP files to IGES/IGS.")
    parser.add_argument("input", help="输入 .stp/.step 文件；批量模式下为输入文件夹")
    parser.add_argument(
        "output",
        nargs="?",
        help="输出 .iges/.igs 文件；批量模式下为输出文件夹，可选",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="批量转换文件夹中的所有 .stp/.step 文件，非递归",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.batch:
            outputs = batch_convert(args.input, args.output)
            print(f"完成：共转换 {len(outputs)} 个文件")
            for p in outputs:
                print(f"  {p}")
        else:
            output = convert_step_to_iges(args.input, args.output)
            print(f"转换成功：{output}")
        return 0
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
