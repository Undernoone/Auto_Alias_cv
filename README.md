# AutoAlias Curves

AutoAlias Curves reconstructs automotive design-intent curves from images or ordered point data and exports Alias-friendly single-span NURBS/Bezier curves.

This repository is built as an engineering base, not a visual toy:

- single-span degree 3/5/7 curve generation;
- CV count is locked to `degree + 1`;
- open clamped knot vectors only;
- fairing losses for curvature, bending, jerk and CV polygon quality;
- S-curve inflection validation;
- L-corner/blend validation;
- JSON, SVG preview, DXF SPLINE and IGES export;
- optional hooks for SAM2, DINOv2 and CAD-grade OpenCascade export.

## Install

```powershell
conda env create -f environment.yml
conda activate autoalias
pip install -e .
```

If you already have Python 3.10+:

```powershell
pip install -e .
```

For OpenCascade IGES export:

```powershell
conda install -c conda-forge pythonocc-core
```

## Run From An Image

```powershell
autoalias fit-image path\to\car.jpg --out out --max-curves 400 --degree auto
```

Outputs:

- `curves.json`: exact NURBS/CV data;
- `curves.igs`: IGES BSpline curves, OpenCascade when available, manual writer fallback;
- `curves.dxf`: DXF SPLINE entities;
- `preview.svg`: target points, fitted curve, CV polygon and curvature comb;
- `quality.json`: Class-A and Alias-readiness metrics.

## Run From Ordered Points

Use this when another segmentation system already provides ordered curve points.

```json
{
  "curves": [
    {
      "label": "beltline",
      "points": [[0, 0], [20, 3], [60, 8], [100, 7]]
    }
  ]
}
```

```powershell
autoalias fit-points points.json --out out --degree 7
```

Without installing the package, run it directly from this workspace:

```powershell
.\scripts\autoalias.cmd fit-points examples\points_s_curve.json --out out --degree 7 --torch-refine
```

The `.cmd` wrapper uses `F:\ComfyUI\.venv\Scripts\python.exe` automatically when it exists.

## Interactive Topology Correction

Use this before training the junction resolver. The tool opens a local browser page where you can
select two stroke branches and mark the intended relationship: same design line, fair blend, break,
or do-not-connect.

The same page also supports manual design-curve annotation. In manual mode, AutoAlias does not ask
you to use its automatic edge split: click the curve split/blend points directly on the image,
choose a semantic label, and save the result as one `manual_design_curve` training sample.
After two or more points are clicked, the page routes between them on the extracted stroke
skeleton and saves the routed centerline as `routed_points`. Add an intermediate guide point if a
junction chooses the wrong branch.
Use `保存设计曲线` to update the current line, or `保存并下一条` to store the current manual
line and immediately clear the point buffer for the next line. A mistaken point can be removed
with `撤回上一点`, or by clicking that point and using `删除选中点` / `Delete`.
For closed outlines such as a lamp loop, wheel detail, or closed styling island, click at least
three points and enable `闭合曲线`; the saved sample records `closed: true` and routes the last
point back to the first point on the stroke skeleton.

```powershell
.\scripts\autoalias.cmd review-image F:\430AutoAlias\test.png --out F:\430AutoAlias\corrections
```

Saved annotations are written as:

```text
F:\430AutoAlias\corrections\<image-name>.topology_corrections.json
```

These files are the first training dataset for the curve-topology model.

Export Alias curves directly from manually reviewed segmentation:

```powershell
.\scripts\autoalias.cmd fit-reviewed F:\430AutoAlias\corrections\*.topology_corrections.json --out F:\430AutoAlias\out_reviewed --degree auto
```

This command reads only the saved manual `design_curves`, fits each one as a compact
degree 3/5/7 single-span NURBS curve, and writes:

- `reviewed_curves.igs`: Alias import file;
- `reviewed_curves.json`: compact CV/degree/weight/knot data;
- `reviewed_clean_preview.svg`: clean curve preview;
- `reviewed_preview.svg`: debug preview with skeleton target, CV polygon and curvature comb.

When a saved manual design curve contains several boundary points, AutoAlias treats
them as split boundaries, not as one long fitting guide. For example, points
`1, 2, 3, 4` export as three independent single-span curves: `1->2`, `2->3`,
and `3->4`. If the curve is closed, the final `N->1` segment is exported too.

Convert reviewed curves into supervised decoder data:

```powershell
.\scripts\autoalias.cmd build-training-set F:\430AutoAlias\corrections\test.topology_corrections.json --out F:\430AutoAlias\data\manual_curve_supervision.json --degree auto
```

Train the neural decoder from one or more supervision JSON files:

```powershell
.\scripts\autoalias.cmd train-decoder F:\430AutoAlias\data\manual_curve_supervision.json --out F:\430AutoAlias\checkpoints\manual_curve_decoder.pt --epochs 50 --batch-size 16
```

## Industrial Integration Points

The production architecture is intentionally modular:

- `autoalias.vision.extractor.OpenCVCurveExtractor`: local fallback extractor;
- `autoalias.vision.foundation.FoundationBackends`: SAM2/DINOv2 adapter interface;
- `autoalias.geometry.fitting.SingleSpanFitter`: deterministic fair single-span fitting;
- `autoalias.quality.ClassAValidator`: Alias and Class-A metric gate;
- `autoalias.exporters`: JSON/SVG/DXF/IGES exporters.

In a deployed system, replace or augment the OpenCV extractor with SAM2/Grounded-SAM/DINOv2 candidates while keeping the same curve fitting, validation and export path.
