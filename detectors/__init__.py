"""
detectors/__init__.py
---------------------
Model factory — the single place that maps model name strings to detector
classes.  All external code should import from here rather than from the
individual adapter modules.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import BaseDetector, Category, DetectionResult
from .laion_detector import LaionNSFWDetector
from .nudenet_detector import NudeNetDetector
from .nudedetector_detector import NudeDetectorDetector
from .open_nsfw_detector import OpenNSFWDetector

if TYPE_CHECKING:
    from config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseDetector]] = {
    "nudenet": NudeNetDetector,
    "nudedetector": NudeDetectorDetector,
    "open_nsfw": OpenNSFWDetector,
    "laion_nsfw": LaionNSFWDetector,
}


def list_models() -> list[str]:
    """Return the names of all registered models."""
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_detector(
    model: str,
    threshold: float = 0.7,
    device: str | None = None,
    **kwargs: object,
) -> BaseDetector:
    """
    Instantiate a detector by name.

    Parameters
    ----------
    model:
        One of the registered model names (see :func:`list_models`).
    threshold:
        Confidence threshold for ``contains_nudity = True``.
    device:
        ``"cuda"``, ``"mps"``, or ``"cpu"``.  ``None`` = auto-detect.
    **kwargs:
        Additional keyword arguments forwarded to the detector constructor.

    Returns
    -------
    BaseDetector
        An uninitialised detector instance (weights are loaded lazily on the
        first call to ``predict``).

    Raises
    ------
    ValueError
        If *model* is not registered.
    """
    key = model.lower().strip()
    if key not in _REGISTRY:
        available = "\n  ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Model '{model}' is not implemented.\n"
            f"Available models:\n  {available}"
        )

    cls = _REGISTRY[key]
    logger.debug("Creating detector %s (threshold=%s, device=%s)", key, threshold, device)
    return cls(threshold=threshold, device=device, **kwargs)


def create_detector_from_config(model: str, config: "Config") -> BaseDetector:
    """
    Convenience wrapper that reads threshold and device from a :class:`Config`.
    """
    extra: dict[str, object] = {}

    if model == "open_nsfw":
        extra["model_cache_dir"] = config.open_nsfw_model_path
    elif model == "nudedetector":
        extra["model_name"] = config.nudedetector_model_name
    elif model == "laion_nsfw":
        extra["model_name"] = config.laion_model_name

    return create_detector(
        model=model,
        threshold=config.threshold,
        device=config.device,
        **extra,
    )


__all__ = [
    "BaseDetector",
    "Category",
    "DetectionResult",
    "NudeNetDetector",
    "NudeDetectorDetector",
    "OpenNSFWDetector",
    "LaionNSFWDetector",
    "create_detector",
    "create_detector_from_config",
    "list_models",
]
