"""
detectors/laion_detector.py
----------------------------
Adapter for the LAION NSFW image classifier.

Default model: ``Falconsai/nsfw_image_detection``  (ViT-based, trained on
LAION-5B NSFW annotations).

Labels: ``normal``, ``nsfw``

GPU / MPS / CPU support is handled by HuggingFace pipeline automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from .base import BaseDetector, Category, DetectionResult

logger = logging.getLogger(__name__)

# Raw label → normalised label
_LABEL_MAP: dict[str, str] = {
    "normal": "SAFE",
    "nsfw": "NSFW",
    "sfw": "SAFE",
    "explicit": "EXPLICIT_CONTENT",
    "suggestive": "SUGGESTIVE_CONTENT",
}

# Labels that contribute to nudity confidence.
_NSFW_LABELS: frozenset[str] = frozenset(
    {"nsfw", "explicit", "suggestive", "porn", "sexy"}
)


class LaionNSFWDetector(BaseDetector):
    """
    LAION NSFW classifier using HuggingFace Transformers.

    Uses ``pipeline("image-classification")`` which handles device placement,
    batching, and preprocessing automatically.
    """

    name = "laion_nsfw"

    def __init__(
        self,
        threshold: float = 0.7,
        device: str | None = None,
        model_name: str = "Falconsai/nsfw_image_detection",
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

        device_arg: int | str
        if self.device == "cuda":
            device_arg = 0
        elif self.device == "mps":
            device_arg = "mps"
        else:
            device_arg = -1

        logger.info(
            "Loading LAION NSFW pipeline '%s' on device=%s",
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
        return self._parse_output(raw)

    def _parse_output(self, raw: list[dict[str, Any]]) -> DetectionResult:
        categories: list[Category] = []
        nsfw_confidence = 0.0

        for item in raw:
            label_raw: str = item["label"].lower().strip()
            score: float = float(item["score"])
            norm_label = _LABEL_MAP.get(label_raw, label_raw.upper())
            categories.append(Category(label=norm_label, score=score))
            if label_raw in _NSFW_LABELS:
                nsfw_confidence = max(nsfw_confidence, score)

        logger.debug("LAION NSFW raw output: %s", raw)
        return self._make_result(nsfw_confidence, categories)

    # ------------------------------------------------------------------
    # Batched inference
    # ------------------------------------------------------------------

    def predict_batch(
        self, image_paths: list[str | Path]
    ) -> list[DetectionResult]:
        """Use HuggingFace pipeline's native batching."""
        self.load()
        images = [self.load_image(p) for p in image_paths]
        logger.info(
            "Running LAION NSFW pipeline batch on %d images", len(images)
        )
        batch_raw: list[list[dict[str, Any]]] = self._pipeline(
            images, top_k=None, batch_size=8
        )
        return [self._parse_output(r) for r in batch_raw]
