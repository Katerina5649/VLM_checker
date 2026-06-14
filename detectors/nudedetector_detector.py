"""
detectors/nudedetector_detector.py
-----------------------------------
Adapter for a ViT-based NSFW classifier hosted on HuggingFace.

Default model: ``AdamCodd/vit-base-nsfw-detector``
  Labels: ``normal``, ``explicit``, ``suggestive``

The model can be overridden via the ``NUDITY_NUDEDETECTOR_MODEL`` env var
or via ``Config.nudedetector_model_name``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from .base import BaseDetector, Category, DetectionResult

logger = logging.getLogger(__name__)

# Labels that indicate nudity/explicit content.
_NSFW_LABELS: frozenset[str] = frozenset({"explicit", "suggestive", "nsfw", "porn"})

# Label taxonomy → our normalised label names.
_LABEL_MAP: dict[str, str] = {
    "explicit": "EXPLICIT_CONTENT",
    "suggestive": "SUGGESTIVE_CONTENT",
    "normal": "SAFE",
    "nsfw": "NSFW",
    "sfw": "SAFE",
    "porn": "EXPLICIT_CONTENT",
    "sexy": "SUGGESTIVE_CONTENT",
}


class NudeDetectorDetector(BaseDetector):
    """
    ViT-based NSFW image classifier via HuggingFace Transformers.

    Uses pipeline("image-classification") for simplicity and GPU/CPU
    fallback out of the box.
    """

    name = "nudedetector"

    def __init__(
        self,
        threshold: float = 0.7,
        device: str | None = None,
        model_name: str = "AdamCodd/vit-base-nsfw-detector",
    ) -> None:
        super().__init__(threshold=threshold, device=device)
        self._model_name = model_name
        self._pipeline: Any = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            from transformers import pipeline  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed.  Run: pip install transformers"
            ) from exc

        # Map our device string to transformers device index.
        device_arg: int | str
        if self.device == "cuda":
            device_arg = 0  # first GPU
        elif self.device == "mps":
            device_arg = "mps"
        else:
            device_arg = -1  # CPU

        logger.info(
            "Loading HuggingFace pipeline '%s' on device=%s",
            self._model_name,
            self.device,
        )
        self._pipeline = pipeline(
            "image-classification",
            model=self._model_name,
            device=device_arg,
        )
        self._model = self._pipeline

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict(self, image: Image.Image) -> DetectionResult:
        raw: list[dict[str, Any]] = self._pipeline(image, top_k=None)
        return self._parse_pipeline_output(raw)

    def _parse_pipeline_output(
        self, raw: list[dict[str, Any]]
    ) -> DetectionResult:
        categories: list[Category] = []
        nsfw_confidence = 0.0

        for item in raw:
            label_raw: str = item["label"].lower().strip()
            score: float = float(item["score"])
            norm_label = _LABEL_MAP.get(label_raw, label_raw.upper())
            categories.append(Category(label=norm_label, score=score))
            if label_raw in _NSFW_LABELS:
                nsfw_confidence = max(nsfw_confidence, score)

        logger.debug("NudeDetector raw output: %s", raw)
        return self._make_result(nsfw_confidence, categories)

    # ------------------------------------------------------------------
    # Batched inference
    # ------------------------------------------------------------------

    def predict_batch(
        self, image_paths: list[str | Path]
    ) -> list[DetectionResult]:
        """Use transformers pipeline batching for efficiency."""
        self.load()
        images = [self.load_image(p) for p in image_paths]
        logger.info(
            "Running NudeDetector pipeline batch on %d images", len(images)
        )
        batch_raw: list[list[dict[str, Any]]] = self._pipeline(
            images, top_k=None, batch_size=8
        )
        return [self._parse_pipeline_output(r) for r in batch_raw]
