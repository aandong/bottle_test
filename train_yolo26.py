"""
YOLO26-small (yolo26s) training + validation for plastic-bottle detection.

Train one dataset folder at a time (staged training). Set DATASET_DIR below,
or pass a dataset folder path on the command line:
    python train_yolo26.py
    python train_yolo26.py /path/to/Botol-Plastik-Kelas-10---14--1

Resume:
    An interrupted run auto-resumes from weights/last.pt the next time you
    run it for the SAME dataset. Force a clean restart with --fresh.

Output (everything in ONE folder):
    runs/detect/yolo26s-<dataset-folder-name>/
        weights/            best.pt, last.pt
        train-*.log         text log of the training session
        results.csv/png     metric curves
        *_batch*.jpg        training / validation result images
        val/                final metrics on the test split

Tuning:
    100 epochs, early stopping via `patience`, batch/cache sized to fill
    GPU VRAM + system RAM, TF32 + AMP + cuDNN autotune enabled for speed.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

# --------------------------------------------------------------------- paths
PROJECT_DIR = Path("/home/aandong/projectYOLO/bottle_test")

# Dataset folder to train on -- must contain data.yaml directly inside it.
# For staged training, edit this per folder or pass a folder path as a CLI arg.
DATASET_DIR = PROJECT_DIR / "dataset" / "merged"

# Parent of every run folder. Each run -> runs/detect/yolo26s-<folder>/ holding
# weights, plots, result images, and the training log all together.
RUNS_DIR = PROJECT_DIR / "runs" / "detect"

# ----------------------------------------------------------- training config
BASE_MODEL = "/home/aandong/projectYOLO/yolo26n.pt"  # pretrained COCO nano — fine-tune ke dataset gabungan
EPOCHS = 100
PATIENCE = 20  # early stopping: stop after N epochs with no val gain
IMGSZ = 640
# 0-1 float => AutoBatch fills that fraction of VRAM. 0.70 leaves ~30% VRAM
# free for the desktop/browser. Raise toward 0.90 if not using the PC.
BATCH = 16   # fixed batch — AutoBatch (float) probe spikes RAM; aman untuk 16GB VRAM
# 'disk' caches decoded images to disk (low RAM use); 'ram' is fastest but
# eats GB of RAM; False = no cache (decode every epoch).
CACHE = False  # disk cache thrash epoch-1 + no swap = freeze; False aman
# 2 workers cukup untuk RTX 4070 Ti SUPER; lebih banyak = RAM pressure fatal tanpa swap
WORKERS = 2


def tune_gpu():
    """Enable GPU fast-paths. Returns True if CUDA is available."""
    if not torch.cuda.is_available():
        return False
    torch.backends.cudnn.benchmark = True  # autotune conv kernels per shape
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 matmul on Ada tensor cores
    torch.backends.cudnn.allow_tf32 = True
    return True


def setup_logger(run_dir):
    """File + console logger. The log file lives inside the run folder."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = run_dir / f"train-{stamp}.log"
    log = logging.getLogger("train_yolo26")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    for h in (
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log, log_file


def make_callbacks(log):
    """Callbacks that record per-epoch metrics and early-stop events to the log."""

    def on_fit_epoch_end(trainer):
        m = trainer.metrics or {}
        ep = trainer.epoch + 1
        log.info(
            f"epoch {ep:>3}/{trainer.epochs}  "
            f"mAP50={m.get('metrics/mAP50(B)', 0):.4f}  "
            f"mAP50-95={m.get('metrics/mAP50-95(B)', 0):.4f}  "
            f"P={m.get('metrics/precision(B)', 0):.4f}  "
            f"R={m.get('metrics/recall(B)', 0):.4f}"
        )
        if getattr(trainer, "stop", False):
            stopper = getattr(trainer, "stopper", None)
            best = getattr(stopper, "best_epoch", "?")
            log.info(f"early stopping triggered at epoch {ep} (best epoch {best})")

    def on_train_end(trainer):
        log.info(f"training finished -- best weights: {trainer.best}")

    return {"on_fit_epoch_end": on_fit_epoch_end, "on_train_end": on_train_end}


def main():
    ap = argparse.ArgumentParser(description="YOLO26s staged training + validation")
    ap.add_argument(
        "dataset",
        nargs="?",
        help="path to dataset folder (contains data.yaml); default DATASET_DIR",
    )
    ap.add_argument(
        "--fresh", action="store_true", help="ignore last.pt, train from scratch"
    )
    ap.add_argument(
        "--epochs", type=int, default=EPOCHS, help=f"epochs (default {EPOCHS})"
    )
    args = ap.parse_args()

    dataset_dir = Path(args.dataset).resolve() if args.dataset else DATASET_DIR
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        sys.exit(f"data.yaml not found in: {dataset_dir}")

    run_name = f"yolo26s-{dataset_dir.name}"
    run_dir = RUNS_DIR / run_name
    last_pt = run_dir / "weights" / "last.pt"
    run_dir.mkdir(parents=True, exist_ok=True)  # so the log sits beside the outputs

    log, log_file = setup_logger(run_dir)
    has_gpu = tune_gpu()

    log.info("=" * 72)
    log.info(f"dataset   : {dataset_dir}")
    log.info(f"data.yaml : {data_yaml}")
    log.info(f"run dir   : {run_dir}")
    log.info(f"log file  : {log_file}")
    log.info(f"GPU       : {torch.cuda.get_device_name(0) if has_gpu else 'CPU only'}")
    log.info(
        f"epochs={args.epochs}  patience={PATIENCE}  imgsz={IMGSZ}  "
        f"batch={BATCH}  cache={CACHE}  workers={WORKERS}"
    )
    log.info("=" * 72)

    resume = last_pt.is_file() and not args.fresh
    callbacks = make_callbacks(log)

    try:
        if resume:
            log.info(f"resuming interrupted run from {last_pt}")
            model = YOLO(str(last_pt))
            for evt, fn in callbacks.items():
                model.add_callback(evt, fn)
            model.train(resume=True)
        else:
            if last_pt.is_file():
                log.info("--fresh set: ignoring last.pt, training from scratch")
            log.info(f"starting fresh from {BASE_MODEL}")
            model = YOLO(BASE_MODEL)
            for evt, fn in callbacks.items():
                model.add_callback(evt, fn)
            model.train(
                data=str(data_yaml),
                project=str(RUNS_DIR),
                name=run_name,
                exist_ok=True,
                epochs=args.epochs,
                patience=PATIENCE,  # built-in early stopping
                imgsz=IMGSZ,
                batch=BATCH,  # AutoBatch -> ~85% VRAM
                cache=CACHE,  # dataset cached in RAM
                workers=WORKERS,
                device=0 if has_gpu else "cpu",
                amp=True,  # mixed precision (FP16) on tensor cores
                deterministic=False,  # let cuDNN autotuner pick faster kernels
                plots=True,
                val=True,
            )
    except Exception as e:
        log.exception(f"training stopped with an error: {e}")
        log.info("re-run the SAME command to auto-resume from last.pt")
        raise

    log.info("training complete -- running final validation on the test split")
    log.info("=" * 72)

    # final validation: default Ultralytics metrics tables printed to the console;
    # output saved inside the run folder at  runs/detect/<run_name>/val/
    best_pt = run_dir / "weights" / "best.pt"
    val_model = YOLO(str(best_pt if best_pt.is_file() else last_pt))
    val_model.val(
        data=str(data_yaml),
        split="test",  # Roboflow exports include a test split
        imgsz=IMGSZ,
        batch=16,
        device=0 if has_gpu else "cpu",
        half=has_gpu,  # FP16 inference for faster validation
        project=str(RUNS_DIR),
        name=f"{run_name}/val",
        exist_ok=True,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
