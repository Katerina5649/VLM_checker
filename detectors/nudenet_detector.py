"""
detectors/nudenet_detector.py
-----------------------------
Adapter for NudeNet ≥ 3.x (https://github.com/notAI-tech/NudeNet).

NudeNet 3.x ships a bundled ONNX model (320n.onnx) — no download needed.
It reads images directly from file paths via cv2 and returns bounding-box
detections with labels from this fixed vocabulary (v3.4.2):

  FEMALE_GENITALIA_COVERED  FEMALE_GENITALIA_EXPOSED
  FEMALE_BREAST_COVERED     FEMALE_BREAST_EXPOSED
  MALE_BREAST_EXPOSED       MALE_GENITALIA_EXPOSED
  BUTTOCKS_COVERED          BUTTOCKS_EXPOSED
  ANUS_COVERED              ANUS_EXPOSED
  FEET_COVERED              FEET_EXPOSED
  ARMPITS_COVERED           ARMPITS_EXPOSED
  BELLY_COVERED             BELLY_EXPOSED
  FACE_FEMALE               FACE_MALE

``confidence`` = max score across all EXPOSED / explicit categories.
Covered / face / safe categories are NOT counted toward nudity confidence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .base import BaseDetector, Category, DetectionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label taxonomy — sourced directly from nudenet 3.4.2 source
# ---------------------------------------------------------------------------

# All labels the model can emit.
ALL_LABELS: frozenset[str] = frozenset(
    {
        "FEMALE_GENITALIA_COVERED",
        "FACE_FEMALE",
        "BUTTOCKS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_BREAST_EXPOSED",
        "ANUS_EXPOSED",
        "FEET_EXPOSED",
        "BELLY_COVERED",
        "FEET_COVERED",
        "ARMPITS_COVERED",
        "ARMPITS_EXPOSED",
        "FACE_MALE",
        "BELLY_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_COVERED",
        "FEMALE_BREAST_COVERED",
        "BUTTOCKS_COVERED",
    }
)

# Labels that directly indicate nudity / explicit content.
# Covered / face / non-sexual body parts are excluded from confidence.
NUDITY_LABELS: frozenset[str] = frozenset(
    {
        "FEMALE_GENITALIA_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "MALE_BREAST_EXPOSED",
        "BUTTOCKS_EXPOSED",
        "ANUS_EXPOSED",
    }
)

# Labels that are suggestive but not fully explicit (contribute at half-weight).
SUGGESTIVE_LABELS: frozenset[str] = frozenset(
    {
        "FEMALE_GENITALIA_COVERED",
        "FEMALE_BREAST_COVERED",
    }
)


class NudeNetDetector(BaseDetector):
    """NudeNet 3.x ONNX object-detection nudity detector."""

    name = "nudenet"

    def __init__(self, threshold: float = 0.7, device: str | None = None) -> None:
        super().__init__(threshold=threshold, device=device)
        self._nudenet: Any = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            from nudenet import NudeDetector as _NudeNet  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "nudenet is not installed.  Run: pip install nudenet"
            ) from exc

        # NudeNet 3.x bundles 320n.onnx inside the package — no download.
        self._nudenet = _NudeNet()
        self._model = self._nudenet
        logger.debug(
            "NudeNet loaded bundled ONNX model; explicit labels: %s",
            sorted(NUDITY_LABELS),
        )

    # ------------------------------------------------------------------
    # Inference — override predict() to pass path directly (no PIL round-trip)
    # ------------------------------------------------------------------

    def predict(self, image_path: str | Path) -> DetectionResult:
        """Pass the file path straight to NudeNet (it reads via cv2 internally)."""
        self.load()
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        logger.debug("NudeNet detecting: %s", path)
        detections: list[dict[str, Any]] = self._nudenet.detect(str(path))
        logger.debug("NudeNet raw detections (%d): %s", len(detections), detections)
        return self._parse_detections(detections)

    def _predict(self, image: Any) -> DetectionResult:  # pragma: no cover
        # Unused — predict() is fully overridden above.
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_detections(
        self, detections: list[dict[str, Any]]
    ) -> DetectionResult:
        """
        Convert NudeNet raw detections to a normalised :class:`DetectionResult`.

        Strategy:
          - Keep max score per label across all bounding boxes.
          - confidence = max score among NUDITY_LABELS (explicit categories).
          - All detected categories (including covered / suggestive) are reported.
        """
        # Max score per label across all boxes.
        label_scores: dict[str, float] = {}
        for det in detections:
            label: str = det.get("class", "")
            score: float = float(det.get("score", 0.0))
            if label:
                label_scores[label] = max(label_scores.get(label, 0.0), score)

        logger.debug("NudeNet per-label scores: %s", label_scores)

        # Build category list (all detected labels, sorted by score desc).
        categories = [
            Category(label=lbl, score=scr)
            for lbl, scr in label_scores.items()
        ]

        # Confidence driven purely by explicit/exposed labels.
        confidence = max(
            (label_scores[lbl] for lbl in NUDITY_LABELS if lbl in label_scores),
            default=0.0,
        )

        return self._make_result(confidence, categories)

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def predict_batch(
        self, image_paths: list[str | Path]
    ) -> list[DetectionResult]:
        """Use NudeNet's built-in detect_batch for efficiency."""
        self.load()
        str_paths = [str(p) for p in image_paths]
        logger.info("Running NudeNet detect_batch on %d images", len(str_paths))

        batch_detections: list[list[dict[str, Any]]] = self._nudenet.detect_batch(
            str_paths
        )

        return [self._parse_detections(dets) for dets in batch_detections]
