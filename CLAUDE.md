# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a YOLO11-based computer vision project for plastic bottle classification. The primary task is training object detection models to identify bottle sizes across multiple class groupings (Kelas 5-9, 10-14, 15-16, 17-22). The project root is `/home/aandong/projectYOLO/`; `bottle_test/` contains test images used for inference validation.

## Common Commands

**Training via script:**
```bash
python /home/aandong/projectYOLO/train_py.py
```

**Training via YOLO CLI:**
```bash
yolo task=detect mode=train model=<model_or_weights.pt> data=<data.yaml> epochs=50 imgsz=640 batch=32
```

**Validation:**
```bash
yolo task=detect mode=val model=<weights/best.pt> data=<data.yaml> imgsz=640 batch=16
```

**Inference on test images:**
```bash
yolo task=detect mode=predict source=bottle_test/ model=<weights/best.pt> conf=0.5 save=True
```

**Export to ONNX:**
```bash
yolo export model=<weights/best.pt> format=onnx
```

## Architecture

### Key Paths

| Path | Purpose |
|------|---------|
| `train_py.py` | Main training script — edit `model`, `dataset` variables at top to switch targets |
| `datasets/Botol-Plastik-Kelas-*/data.yaml` | Dataset configs for training (primary location) |
| `bottle_test/dataset/Botol-Plastik-Kelas-*/data.yaml` | Dataset copies for inference testing |
| `bottle_test/sample_test/` | 3 quick-test images for fast inference validation |
| `runs/detect/<run-name>/weights/best.pt` | Trained model weights |
| `bottle.ipynb` | Notebook for full download→train→validate→predict workflow |

### Dataset Organization

Each bottle class range is a separate Roboflow-sourced dataset (`ittp` workspace) with its own `data.yaml`. Class names in `data.yaml` are the bottle size numbers themselves (e.g., `names: ['5','6','7','8','9']` for Kelas 5-9 — 5 classes). Available datasets:

| Dataset dir suffix | Classes (nc) |
|--------------------|-------------|
| `Kelas-5---9-1` | 5 classes: 5–9 |
| `Kelas-10---14--1` | 5 classes: 10–14 |
| `Kelas-15---16--1` | 2 classes: 15–16 |
| `Kelas-17---22--1` | 6 classes: 17–22 |

Training runs in `runs/detect/` use the naming convention `bottle-<date>-kelas<range>`. YOLO auto-appends a number suffix when a run name already exists (e.g., `bottle-ags4-kelas5-92` is a second run with that name) — use unique names to avoid confusion.

### Training Script Pattern

`train_py.py` has hardcoded path variables at the top (`model`, `dataset`, `model_yg_dipake`, etc.). To retrain on a different class group, update those variables before running. The script calls `train_bottle_model()` → `validate_model()` → `predict_with_model()` in sequence.

**Known inconsistency in `train_py.py`:** `BOTTLE_DATASET_5_9` and `BOTTLE_DATASET_10_14` are defined as directory paths (no `/data.yaml` suffix), while `BOTTLE_DATASET_5_8`, `BOTTLE_DATASET_17_22`, and `BOTTLE_DATASET_15_16` include the `/data.yaml` suffix. The `dataset_yg_dipake` variable appends `/data.yaml` manually, so use the bare directory form for those two.

### Model Performance

Best models achieve mAP50 ≈ 0.995. Inference runs ~3.4ms/image on RTX 4070 Ti SUPER (CUDA 12.4). Models are exported to ONNX/TorchScript for deployment.

## Environment

- Python 3.11.11, PyTorch 2.6.0 (CUDA 12.4), Ultralytics YOLO11
- GPU: NVIDIA RTX 4070 Ti SUPER
- No `requirements.txt` — dependencies managed via conda/pip directly
- Not a git repository
