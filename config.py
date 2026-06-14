"""
config.py
---------
Centralised configuration for the nudity detection pipeline.

All settings can be overridden via environment variables (see field
docstrings) or by constructing a :class:`Config` instance directly in code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Main config dataclass
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """
    Runtime configuration.

    Environment variable overrides are shown next to each field.

    Parameters
    ----------
    threshold:
        Confidence threshold for ``contains_nudity = True``.
        Env: ``NUDITY_THRESHOLD``  (default: ``0.7``)
    device:
        Compute device: ``"cuda"``, ``"mps"``, or ``"cpu"``.
        ``None`` means auto-detect.
        Env: ``NUDITY_DEVICE``
    log_level:
        Python logging level name, e.g. ``"DEBUG"``, ``"INFO"``.
        Env: ``NUDITY_LOG_LEVEL``  (default: ``"WARNING"``)
    batch_size:
        Maximum images per forward pass for models that support true batching.
        Env: ``NUDITY_BATCH_SIZE``  (default: ``8``)
    model_cache_dir:
        Directory where model weights are cached.
        Env: ``NUDITY_MODEL_CACHE_DIR``  (default: ``~/.cache/nudity_models``)
    """

    threshold: float = field(
        default_factory=lambda: _env_float("NUDITY_THRESHOLD", 0.7)
    )
    device: str | None = field(
        default_factory=lambda: _env_str("NUDITY_DEVICE", "") or None
    )
    log_level: str = field(
        default_factory=lambda: _env_str("NUDITY_LOG_LEVEL", "WARNING")
    )
    batch_size: int = field(
        default_factory=lambda: _env_int("NUDITY_BATCH_SIZE", 8)
    )
    model_cache_dir: str = field(
        default_factory=lambda: _env_str(
            "NUDITY_MODEL_CACHE_DIR",
            os.path.join(os.path.expanduser("~"), ".cache", "nudity_models"),
        )
    )

    # -----------------------------------------------------------------------
    # Derived / computed properties
    # -----------------------------------------------------------------------

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                f"threshold must be in [0, 1], got {self.threshold!r}"
            )
        os.makedirs(self.model_cache_dir, exist_ok=True)

    @property
    def nudenet_model_path(self) -> str:
        """Path where NudeNet weights are cached."""
        return os.path.join(self.model_cache_dir, "nudenet")

    @property
    def open_nsfw_model_path(self) -> str:
        """Path where Open-NSFW ONNX weights are cached."""
        return os.path.join(self.model_cache_dir, "open_nsfw")

    @property
    def laion_model_name(self) -> str:
        """HuggingFace model ID for the LAION NSFW classifier."""
        return _env_str(
            "NUDITY_LAION_MODEL",
            "Falconsai/nsfw_image_detection",
        )

    @property
    def nudedetector_model_name(self) -> str:
        """HuggingFace model ID for the NudeDetector ViT model."""
        return _env_str(
            "NUDITY_NUDEDETECTOR_MODEL",
            "AdamCodd/vit-base-nsfw-detector",
        )


# ---------------------------------------------------------------------------
# Supported model identifiers
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: tuple[str, ...] = (
    "nudenet",
    "nudedetector",
    "open_nsfw",
    "laion_nsfw",
)
