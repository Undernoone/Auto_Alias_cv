# Industrial Backend Contract

The local OpenCV extractor is deliberately replaceable. Production deployments should provide curve candidates in this shape:

```json
{
  "label": "roofline",
  "points": [[x, y], [x, y]],
  "confidence": 0.93,
  "source": "sam2_dinov2",
  "semantic": {
    "view": "side",
    "part": "upper_body",
    "curve_type": "silhouette"
  }
}
```

Required guarantees before fitting:

- points are ordered along the intended curve;
- reflection/shadow/background edges are already down-weighted or removed;
- coordinates are in a consistent image or rectified plane;
- a semantic label is provided when possible.

The fitter will preserve the Alias contract:

- degree in 3..7;
- `CV count = degree + 1`;
- knot vector `[0...0, 1...1]`;
- span count exactly 1;
- no polyline or mesh output.

