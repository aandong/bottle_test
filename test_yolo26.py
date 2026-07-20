"""
YOLO26-small (yolo26s) testing / inference for plastic-bottle detection.

Set DATASET_DIR below (or pass a dataset folder path) -- it must match the
folder you trained on, so the script can find the trained weights.

Predict (saves annotated images, prints default per-image output):
    python test_yolo26.py
    python test_yolo26.py --source /path/to/imgs_or_video
    python test_yolo26.py --conf 0.5

Metrics on the test split (prints the default Ultralytics metrics table):
    python test_yolo26.py --val

Output lands inside the SAME run folder as training:
    runs/detect/yolo26s-<dataset-folder-name>/predict/        annotated images
    runs/detect/yolo26s-<dataset-folder-name>/test-metrics/   --val metrics
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# --------------------------------------------------------------------- paths
PROJECT_DIR = Path("/home/aandong/projectYOLO/bottle_test")

# Dataset folder -- must match the folder used for training.
DATASET_DIR = PROJECT_DIR / "dataset" / "Botol-Plastik-Kelas-5---9-1"

RUNS_DIR = PROJECT_DIR / "runs" / "detect"
IMGSZ    = 640


def tune_gpu():
    """Enable GPU fast-paths. Returns True if CUDA is available."""
    if not torch.cuda.is_available():
        return False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return True


def test_images_dir(data_yaml):
    """Resolve the test-image directory from a data.yaml the way Ultralytics does."""
    cfg = yaml.safe_load(data_yaml.read_text())
    base = data_yaml.parent
    if cfg.get("path"):
        base = (data_yaml.parent / cfg["path"]).resolve()
    rel = cfg.get("test") or cfg.get("val") or "test/images"
    return (base / rel).resolve()


def main():
    ap = argparse.ArgumentParser(description="YOLO26s testing / inference")
    ap.add_argument("dataset", nargs="?",
                    help="path to dataset folder (contains data.yaml); default DATASET_DIR")
    ap.add_argument("--source", help="image/dir/video to predict (default: dataset test split)")
    ap.add_argument("--val", action="store_true",
                    help="run metrics on the test split instead of predicting")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence threshold (default 0.25)")
    ap.add_argument("--weights", help="override weights path (.pt)")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset).resolve() if args.dataset else DATASET_DIR
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        sys.exit(f"data.yaml not found in: {dataset_dir}")

    run_name = f"yolo26s-{dataset_dir.name}"
    run_dir  = RUNS_DIR / run_name

    weights = Path(args.weights) if args.weights else run_dir / "weights" / "best.pt"
    if not weights.is_file():
        sys.exit(f"weights not found: {weights}\n"
                 f"Train first:  python train_yolo26.py {dataset_dir}")

    has_gpu = tune_gpu()
    device  = 0 if has_gpu else "cpu"
    print(f"weights : {weights}")
    print(f"device  : {torch.cuda.get_device_name(0) if has_gpu else 'CPU'}")

    model = YOLO(str(weights))

    if args.val:
        # default Ultralytics metrics table printed to the console
        model.val(
            data=str(data_yaml),
            split="test",
            imgsz=IMGSZ,
            batch=16,
            device=device,
            half=has_gpu,
            project=str(RUNS_DIR),
            name=f"{run_name}/test-metrics",
            exist_ok=True,
            plots=True,
            verbose=True,
        )
        return

    # predict mode
    if args.source:
        source = args.source
    else:
        td = test_images_dir(data_yaml)
        if not td.exists():
            sys.exit(f"test images not found at {td}\nPass an explicit path with --source")
        source = str(td)
    print(f"source  : {source}")

    model.predict(
        source=source,
        conf=args.conf,
        imgsz=IMGSZ,
        device=device,
        half=has_gpu,
        save=True,
        project=str(RUNS_DIR),
        name=f"{run_name}/predict",
        exist_ok=True,
        verbose=True,   # default per-image output
    )


if __name__ == "__main__":
    main()
