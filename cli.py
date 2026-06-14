#!/usr/bin/env python3
"""
cli.py
------
Command-line interface for the nudity detection pipeline.

Usage (single image):
    python cli.py --image_path image.jpg --model nudenet

Usage (batch — pass multiple paths):
    python cli.py --image_path a.jpg b.jpg c.png --model laion_nsfw

Usage (batch from file):
    python cli.py --image_list paths.txt --model open_nsfw

Output is always newline-delimited JSON (one object per image).
Use --pretty for human-readable indented JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from config import Config, SUPPORTED_MODELS
from detectors import create_detector_from_config, list_models

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nudity_detect",
        description="Detect nudity in images using configurable AI models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python cli.py --image_path photo.jpg --model nudenet

  # Multiple images (batch)
  python cli.py --image_path a.jpg b.jpg c.png --model laion_nsfw

  # Images listed in a text file (one path per line)
  python cli.py --image_list images.txt --model open_nsfw

  # Lower threshold, verbose logging, pretty JSON
  python cli.py --image_path photo.jpg --model nudenet \\
                --threshold 0.5 --log_level DEBUG --pretty

  # Force CPU even when GPU is available
  python cli.py --image_path photo.jpg --model nudedetector --device cpu
""",
    )

    # --- Input ---
    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument(
        "--image_path",
        nargs="+",
        metavar="PATH",
        help="Path(s) to the image file(s) to analyse.",
    )
    input_grp.add_argument(
        "--image_list",
        metavar="FILE",
        help="Text file with one image path per line.",
    )

    # --- Model ---
    parser.add_argument(
        "--model",
        required=True,
        choices=SUPPORTED_MODELS,
        metavar="MODEL",
        help=f"Model to use for detection. Choices: {', '.join(SUPPORTED_MODELS)}",
    )

    # --- Config overrides ---
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Confidence threshold for contains_nudity=true (default: 0.7).",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        default=None,
        metavar="DEVICE",
        help="Compute device override (default: auto-detect).",
    )

    # --- Output ---
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output (indent=2).",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Write JSON output to FILE instead of stdout.",
    )

    # --- Logging ---
    parser.add_argument(
        "--log_level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        metavar="LEVEL",
        help="Logging verbosity (default: WARNING).",
    )

    return parser


# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------


def _resolve_image_paths(args: argparse.Namespace) -> list[Path]:
    if args.image_path:
        paths = [Path(p) for p in args.image_path]
    else:
        list_file = Path(args.image_list)
        if not list_file.exists():
            print(
                f"ERROR: image list file not found: {list_file}",
                file=sys.stderr,
            )
            sys.exit(1)
        paths = [
            Path(line.strip())
            for line in list_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    # Validate all paths up-front so we fail fast.
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print(
            "ERROR: The following images were not found:\n  "
            + "\n  ".join(missing),
            file=sys.stderr,
        )
        sys.exit(1)

    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """
    Entry point.  Returns 0 on success, 1 on error.

    The function is importable so it can be unit-tested without spawning a
    subprocess.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Build config, applying any CLI overrides.
    cfg = Config()
    if args.threshold is not None:
        cfg.threshold = args.threshold
    if args.device is not None:
        cfg.device = args.device
    if args.log_level is not None:
        cfg.log_level = args.log_level

    _setup_logging(cfg.log_level)
    log = logging.getLogger(__name__)
    log.debug("Config: %s", cfg)

    # Resolve image paths.
    image_paths = _resolve_image_paths(args)
    log.info("Processing %d image(s) with model '%s'", len(image_paths), args.model)

    # Create detector (lazy — weights load on first predict call).
    try:
        detector = create_detector_from_config(args.model, cfg)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Run inference.
    results_json: list[dict] = []
    exit_code = 0

    if len(image_paths) == 1:
        # Single image — use predict() directly.
        try:
            result = detector.predict(image_paths[0])
            d = result.to_dict()
            d["image_path"] = str(image_paths[0])
            results_json.append(d)
        except Exception as exc:
            log.error("Inference failed: %s", exc, exc_info=True)
            results_json.append(
                {
                    "error": str(exc),
                    "image_path": str(image_paths[0]),
                    "model": args.model,
                }
            )
            exit_code = 1
    else:
        # Batch — delegate to predict_batch for efficiency.
        try:
            batch_results = detector.predict_batch(image_paths)
            for path, result in zip(image_paths, batch_results):
                d = result.to_dict()
                d["image_path"] = str(path)
                results_json.append(d)
        except Exception as exc:
            log.error("Batch inference failed: %s", exc, exc_info=True)
            results_json.append(
                {
                    "error": str(exc),
                    "model": args.model,
                }
            )
            exit_code = 1

    # Serialise output.
    indent = 2 if args.pretty else None
    if len(results_json) == 1:
        # Single result — output a plain object, not a list.
        output_str = json.dumps(results_json[0], ensure_ascii=False, indent=indent)
    else:
        output_str = json.dumps(results_json, ensure_ascii=False, indent=indent)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(output_str + "\n", encoding="utf-8")
        log.info("Results written to %s", out_path)
    else:
        print(output_str)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
