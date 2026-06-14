# Nudity Detection CLI

A Python CLI for image nudity/NSFW detection using four pluggable AI models.

---

## Installation

Requires **Python 3.11+** and **Apple Silicon M-series Mac** (or adapt for CUDA).

### 1. Create and activate a virtual environment

```bash
cd VLM_checker
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install opennsfw2 keras       # required for the open_nsfw model
```

> **Note:** `torch` and `torchvision` are in `requirements.txt`. On Apple Silicon,
> MPS (GPU) acceleration is enabled by default — no extra steps needed.

---

## Usage

### Single image

```bash
python cli.py --image_path path/to/image.jpg --model nudenet
```

### Multiple images (batch)

```bash
python cli.py --image_path a.jpg b.jpg c.png --model laion_nsfw
```

### Images listed in a text file

```bash
python cli.py --image_list paths.txt --model open_nsfw
```

### Save output to a file

```bash
python cli.py --image_path image.jpg --model nudenet --output result.json
```

### Pretty-printed JSON

```bash
python cli.py --image_path image.jpg --model nudenet --pretty
```

### Adjust confidence threshold (default: 0.7)

```bash
python cli.py --image_path image.jpg --model nudenet --threshold 0.5
```

### Force CPU (disable GPU)

```bash
python cli.py --image_path image.jpg --model nudenet --device cpu
```

### Verbose logging

```bash
python cli.py --image_path image.jpg --model nudenet --log_level DEBUG
```

---

## Supported Models

| Model flag      | Description                                              | Extra install needed |
|-----------------|----------------------------------------------------------|-----------------------|
| `nudenet`       | Object-detection — returns bounding-box labels (FEMALE_BREAST_EXPOSED, etc.) | none |
| `nudedetector`  | ViT classifier via HuggingFace (`AdamCodd/vit-base-nsfw-detector`) | none |
| `laion_nsfw`    | ViT classifier via HuggingFace (`Falconsai/nsfw_image_detection`) | none |
| `open_nsfw`     | Yahoo ResNet-50 via opennsfw2 + Keras/PyTorch backend   | `pip install opennsfw2 keras` |

> On first run, HuggingFace models download weights automatically (~300–500 MB each).
> `open_nsfw` downloads ~97 MB to `~/.opennsfw2/weights/`.

---

## Output Format

All models return the same JSON schema:

```json
{
  "contains_nudity": true,
  "confidence": 0.97,
  "model": "nudenet",
  "categories": [
    { "label": "FEMALE_BREAST_EXPOSED", "score": 0.97 },
    { "label": "BUTTOCKS_EXPOSED",      "score": 0.84 }
  ],
  "image_path": "path/to/image.jpg"
}
```

- **`contains_nudity`** — `true` if `confidence >= threshold` (default `0.7`)
- **`confidence`** — highest score among explicit categories
- **`categories`** — all detected labels, sorted by score descending

---

## Configuration via Environment Variables

| Variable                    | Default                           | Description                    |
|-----------------------------|-----------------------------------|--------------------------------|
| `NUDITY_THRESHOLD`          | `0.7`                             | Confidence threshold           |
| `NUDITY_DEVICE`             | auto                              | `cuda`, `mps`, or `cpu`        |
| `NUDITY_LOG_LEVEL`          | `WARNING`                         | Logging verbosity              |
| `NUDITY_BATCH_SIZE`         | `8`                               | Images per batch               |
| `NUDITY_MODEL_CACHE_DIR`    | `~/.cache/nudity_models`          | Model weight cache directory   |
| `NUDITY_LAION_MODEL`        | `Falconsai/nsfw_image_detection`  | HuggingFace model ID           |
| `NUDITY_NUDEDETECTOR_MODEL` | `AdamCodd/vit-base-nsfw-detector` | HuggingFace model ID           |
| `KERAS_BACKEND`             | `torch`                           | Keras backend for `open_nsfw`  |

---

## Error: unknown model

Passing an unsupported model name prints the available options and exits:

```
ERROR: Model 'xyz' is not implemented.
Available models:
  laion_nsfw
  nudedetector
  nudenet
  open_nsfw
```
