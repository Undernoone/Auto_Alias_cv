from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency path
    torch = None
    nn = None
    F = None


if nn is not None:

    class CurveTokenNURBSDecoder(nn.Module):
        """Transformer decoder that predicts single-span NURBS control data.

        Input:
          points: [batch, n_points, point_dim]
          semantic_id: optional [batch]

        Output:
          degree_logits: [batch, 5] for degree {3,4,5,6,7}
          cv: [batch, 8, 3] max degree-7 CVs
          weights: [batch, 8]
          confidence: [batch, 1]

        The downstream constraint chooses degree p and uses the first p+1 CVs with
        clamped knots [0*(p+1), 1*(p+1)].
        """

        def __init__(
            self,
            point_dim: int = 3,
            hidden_dim: int = 256,
            layers: int = 4,
            heads: int = 8,
            max_cvs: int = 8,
            semantic_classes: int = 32,
        ):
            super().__init__()
            self.max_cvs = max_cvs
            self.semantic_embed = nn.Embedding(semantic_classes, hidden_dim)
            self.point_proj = nn.Sequential(
                nn.Linear(point_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.pool = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.degree_head = nn.Linear(hidden_dim, 5)
            self.cv_head = nn.Linear(hidden_dim, max_cvs * 3)
            self.weight_head = nn.Linear(hidden_dim, max_cvs)
            self.conf_head = nn.Linear(hidden_dim, 1)

        def forward(self, points, semantic_id=None):
            x = self.point_proj(points)
            if semantic_id is not None:
                x = x + self.semantic_embed(semantic_id).unsqueeze(1)
            x = self.encoder(x)
            pooled = self.pool(x.mean(dim=1))
            return {
                "degree_logits": self.degree_head(pooled),
                "cv": self.cv_head(pooled).reshape(points.shape[0], self.max_cvs, 3),
                "weights": F.softplus(self.weight_head(pooled)) + 1e-4,
                "confidence": torch.sigmoid(self.conf_head(pooled)),
            }

else:

    class CurveTokenNURBSDecoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for CurveTokenNURBSDecoder")
