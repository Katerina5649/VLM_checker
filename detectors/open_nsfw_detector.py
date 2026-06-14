"""
detectors/open_nsfw_detector.py
--------------------------------
Adapter for Yahoo's Open-NSFW model via the ``opennsfw2`` package.

opennsfw2 ships a Keras ResNet-50 model.  Keras 3.x is backend-agnostic, so
we configure it to use the PyTorch backend (already installed) instead of
TensorFlow — no TF dependency needed.

Requirements:
    pip install opennsfw2 keras

On first run, opennsfw2 downloads ~97 MB of pre-trained weights from GitHub
into ~/.opennsfw2/weights/open_nsfw_weights.h5.

Output: two categories — NSFW and SFW.
confidence = NSFW probability.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from PIL import Image

from .base import BaseDetector, Category, DetectionResult

logger = logging.getLogger(__name__)


class OpenNSFWDetector(BaseDetector):
    """
    Yahoo Open-NSFW ResNet-50 classifier via ``opennsfw2`` + Keras/PyTorch.

    Keras 3.x is set to use the PyTorch backend automatically so TensorFlow
    is not required.
    """

    name = "open_nsfw"

    def __init__(
        self,
        threshold: float = 0.7,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, device=device)
        self._n2: Any = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        # ── Set Keras backend to PyTorch BEFORE any keras import ──────────
        # Keras 3.x reads KERAS_BACKEND at first import time.
        # We use setdefault so a user-set env var takes precedence.
        os.environ.setdefault("KERAS_BACKEND", "torch")
        logger.debug(
            "KERAS_BACKEND=%s (using PyTorch backend for open_nsfw)",
            os.environ["KERAS_BACKEND"],
        )

        try:
            import opennsfw2 as n2  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "opennsfw2 or keras is not installed.\n"
                "Run: pip install opennsfw2 keras"
            ) from exc

        self._n2 = n2
        self._model = n2

        # Trigger a warm-up load so weight download happens at load time,
        # not silently during the first predict call.
        logger.info(
            "open_nsfw: first run will download weights (~97 MB) to "
            "~/.opennsfw2/weights/ if not already cached"
        )

    # ------------------------------------------------------------------
    # Inference  (override predict() to skip PIL round-trip)
    # ------------------------------------------------------------------

    def predict(self, image_path: str | Path) -> DetectionResult:
        self.load()
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        logger.debug("open_nsfw predicting: %s", path)
        nsfw_prob = self._call_n2(str(path))
        return self._build_result(nsfw_prob)

    def _predict(self, image: Image.Image) -> DetectionResult:  # pragma: no cover
        raise NotImplementedError  # predict() is fully overridden above

    def _call_n2(self, image_path: str) -> float:
        """
        Call opennsfw2.predict_image and extract the NSFW probability.

        opennsfw2 returns a plain float.
        """
        result = self._n2.predict_image(image_path)
        # Guard against future API changes that might return a tuple
        if isinstance(result, (tuple, list)):
            nsfw_prob = float(result[-1])
        else:
            nsfw_prob = float(result)
        logger.debug("open_nsfw raw NSFW probability: %.4f", nsfw_prob)
        return nsfw_prob

    def _build_result(self, nsfw_prob: float) -> DetectionResult:
        sfw_prob = 1.0 - nsfw_prob
        categories = [
            Category(label="NSFW", score=round(nsfw_prob, 4)),
            Category(label="SFW", score=round(sfw_prob, 4)),
        ]
        return self._make_result(nsfw_prob, categories)

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def predict_batch(
        self, image_paths: list[str | Path]
    ) -> list[DetectionResult]:
        self.load()
        str_paths = [str(p) for p in image_paths]
        logger.info("Running open_nsfw batch on %d images", len(str_paths))

        raw: list[Any] = self._n2.predict_images(str_paths)
        results: list[DetectionResult] = []
        for item in raw:
            nsfw_prob = float(item[-1] if isinstance(item, (tuple, list)) else item)
            results.append(self._build_result(nsfw_prob))
        return results
