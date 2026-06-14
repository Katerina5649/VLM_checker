"""
detectors/base.py
-----------------
Abstract base class and shared data structures for all nudity detectors.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Category:
    """A single detected category with its confidence score."""

    label: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "score": round(self.score, 4)}


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Normalised output returned by every detector."""

    contains_nudity: bool
    confidence: float
    model: str
    categories: list[Category] = field(default_factory=list)
    # Optional extra metadata (e.g. raw logits) – never serialised to JSON
    _meta: dict[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contains_nudity": self.contains_nudity,
            "confidence": round(self.confidence, 4),
            "model": self.model,
            "categories": [c.to_dict() for c in self.categories],
        }


# ---------------------------------------------------------------------------
# Abstract detector
# ---------------------------------------------------------------------------


class BaseDetector(abc.ABC):
    """
    Common interface that every model adapter must implement.

    Subclasses override ``_load_model`` and ``_predict``.
    The public ``predict`` method handles device selection, image loading,
    timing, and logging.
    """

    #: Registry name used by the factory (set in each subclass).
    name: str = ""

    def __init__(
        self,
        threshold: float = 0.7,
        device: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        threshold:
            Minimum confidence required for ``contains_nudity = True``.
        device:
            ``"cuda"``, ``"mps"``, or ``"cpu"``.  Pass ``None`` to auto-detect.
        """
        self.threshold = threshold
        self.device = self._resolve_device(device)
        self._model: Any = None
        self._loaded = False
        logger.debug("Initialised %s (device=%s, threshold=%s)", self.name, self.device, threshold)

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        if device is not None:
            return device
        try:
            import torch  # local import — torch may not be installed at import time
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_image(path: str | Path) -> Image.Image:
        """Load and convert image to RGB, raising a clear error on failure."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        if not p.is_file():
            raise ValueError(f"Path is not a file: {p}")
        try:
            img = Image.open(p).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Cannot open image '{p}': {exc}") from exc
        logger.debug("Loaded image %s (%dx%d)", p.name, *img.size)
        return img

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Idempotent model loader — called automatically before first predict."""
        if self._loaded:
            return
        logger.info("Loading model: %s", self.name)
        t0 = time.perf_counter()
        self._load_model()
        self._loaded = True
        logger.info("Model %s loaded in %.2fs", self.name, time.perf_counter() - t0)

    @abc.abstractmethod
    def _load_model(self) -> None:
        """Download / initialise the model.  Must set ``self._model``."""

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _predict(self, image: Image.Image) -> DetectionResult:
        """
        Run inference on a pre-loaded PIL Image.

        Must return a :class:`DetectionResult` with ``contains_nudity`` set
        based on ``self.threshold``.
        """

    def predict(self, image_path: str | Path) -> DetectionResult:
        """
        Full pipeline: load model (once), load image, run inference.

        Parameters
        ----------
        image_path:
            Path to the image file.

        Returns
        -------
        DetectionResult
        """
        self.load()
        image = self.load_image(image_path)
        t0 = time.perf_counter()
        result = self._predict(image)
        logger.debug(
            "%s inference took %.3fs  confidence=%.4f  nudity=%s",
            self.name,
            time.perf_counter() - t0,
            result.confidence,
            result.contains_nudity,
        )
        return result

    def predict_batch(
        self, image_paths: list[str | Path]
    ) -> list[DetectionResult]:
        """
        Run inference on a list of images.

        The default implementation calls ``predict`` sequentially.
        Subclasses may override for true batched GPU inference.
        """
        self.load()
        results: list[DetectionResult] = []
        for i, path in enumerate(image_paths, 1):
            logger.info("Processing %d/%d: %s", i, len(image_paths), path)
            try:
                results.append(self.predict(path))
            except Exception as exc:
                logger.error("Failed to process %s: %s", path, exc)
                raise
        return results

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    def _make_result(
        self,
        confidence: float,
        categories: list[Category],
    ) -> DetectionResult:
        """Convenience factory that applies the threshold decision."""
        return DetectionResult(
            contains_nudity=confidence >= self.threshold,
            confidence=round(confidence, 4),
            model=self.name,
            categories=sorted(categories, key=lambda c: c.score, reverse=True),
        )
