from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Segmenter(Protocol):
    def segment(self, image: np.ndarray, prompts: list[str]) -> dict[str, np.ndarray]:
        ...


class DenseFeatureEncoder(Protocol):
    def encode(self, image: np.ndarray) -> np.ndarray:
        ...


@dataclass(slots=True)
class FoundationBackends:
    """Optional SAM2/DINOv2 backend registry.

    The production system should inject real implementations here. The local OpenCV extractor
    remains available so the project can run without model weights.
    """

    segmenter: Segmenter | None = None
    encoder: DenseFeatureEncoder | None = None


class Sam2Unavailable(RuntimeError):
    pass


class DinoV2Unavailable(RuntimeError):
    pass


class Sam2Segmenter:
    def __init__(self, checkpoint: str, config: str | None = None):
        self.checkpoint = checkpoint
        self.config = config
        try:
            import torch  # noqa: F401
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
        except Exception as exc:  # pragma: no cover - optional heavy dependency
            raise Sam2Unavailable(
                "SAM2 is not installed. Install Meta SAM2 and pass a local checkpoint."
            ) from exc

        if config is None:
            raise Sam2Unavailable("SAM2 config path is required for this adapter")
        model = build_sam2(config, checkpoint)
        self.predictor = SAM2ImagePredictor(model)

    def segment(self, image: np.ndarray, prompts: list[str]) -> dict[str, np.ndarray]:
        raise NotImplementedError(
            "Text-prompt SAM2 needs a detector such as GroundingDINO. "
            "Inject masks through the pipeline or implement project-specific prompts here."
        )


class DinoV2Encoder:
    def __init__(self, model_name: str = "dinov2_vitl14"):
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional heavy dependency
            raise DinoV2Unavailable("PyTorch is required for DINOv2.") from exc
        self.torch = torch
        self.model_name = model_name
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)  # pragma: no cover
        self.model.eval()

    def encode(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "DINOv2 preprocessing is deployment-specific. Use this adapter as the integration point."
        )

