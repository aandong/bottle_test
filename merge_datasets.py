"""
merge_datasets.py — Gabung semua dataset Roboflow YOLO jadi satu folder.

Cara kerja:
  - Baca data.yaml tiap folder (names seperti ['5','6','7','8','9'])
  - Map nama kelas string → global class ID (int)
  - Copy gambar (.jpg/.png saja, skip .npy)
  - Remap class ID di label .txt
  - Tulis data.yaml tunggal (nc=23, kelas 0-22)

Output: bottle_test/dataset/merged/
  merged/
    train/images/   merged/train/labels/
    valid/images/   merged/valid/labels/
    test/images/    merged/test/labels/
    data.yaml

Usage:
    python merge_datasets.py              # jalankan merge
    python merge_datasets.py --dry-run    # preview saja, tidak menulis file
"""

import argparse
import shutil
from pathlib import Path

import yaml

# ------------------------------------------------------------------ paths
PROJECT_DIR = Path("/home/aandong/projectYOLO/bottle_test")
DATASET_DIR = PROJECT_DIR / "dataset"
OUT_DIR = DATASET_DIR / "merged"

# Semua folder sumber. Urutan tidak mempengaruhi hasil.
SOURCE_DIRS = [
    DATASET_DIR / "Botol Plastik Kelas 0-4.v1i.yolo26",
    DATASET_DIR / "Botol-Plastik-Kelas-5---9--1",
    DATASET_DIR / "Botol-Plastik-Kelas-10---14--1",
    DATASET_DIR / "Botol-Plastik-Kelas-15---16--1",
    DATASET_DIR / "Botol-Plastik-Kelas-17---22--1",
    # DATASET_DIR / "lotob-asri2-1",
    # DATASET_DIR / "Prima-Ades-Sosro-1",
]

SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Ruang kelas global: 0-22 (semua kelas botol plastik)
NC = 23
GLOBAL_NAMES = [str(i) for i in range(NC)]


# ------------------------------------------------------------------ core
def load_remap(data_yaml: Path) -> dict:
    """
    Baca data.yaml, kembalikan {local_id: global_id}.
    Nama kelas di Roboflow export adalah angka string (misal '10', '11', ...).
    Global ID = int(nama_kelas).
    """
    with open(data_yaml, encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    names = meta["names"]
    return {local_id: int(name) for local_id, name in enumerate(names)}


def remap_label(src: Path, dst: Path, remap: dict) -> None:
    """Copy file label .txt dengan class ID yang sudah diremapping."""
    lines_out = []
    for line in src.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        global_id = remap[int(parts[0])]
        lines_out.append(f"{global_id} " + " ".join(parts[1:]))
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")


def merge(dry_run: bool = False) -> None:
    print(f"{'[DRY RUN] ' if dry_run else ''}Output: {OUT_DIR}\n")

    stats = {s: {"images": 0, "labels": 0, "skipped": 0} for s in SPLITS}

    for src_dir in SOURCE_DIRS:
        data_yaml = src_dir / "data.yaml"
        if not data_yaml.exists():
            print(f"  SKIP (tidak ada data.yaml): {src_dir.name}")
            continue

        remap = load_remap(data_yaml)
        # Prefix pendek dari nama folder agar nama file tetap unik lintas folder
        prefix = src_dir.name.replace(" ", "_")[:24]
        print(f"  {src_dir.name}")
        print(f"    remap: {remap}")

        for split in SPLITS:
            img_src = src_dir / split / "images"
            lbl_src = src_dir / split / "labels"
            img_dst = OUT_DIR / split / "images"
            lbl_dst = OUT_DIR / split / "labels"

            if not img_src.is_dir():
                continue

            if not dry_run:
                img_dst.mkdir(parents=True, exist_ok=True)
                lbl_dst.mkdir(parents=True, exist_ok=True)

            for img_file in sorted(img_src.iterdir()):
                if img_file.suffix.lower() not in IMG_EXTS:
                    stats[split]["skipped"] += 1
                    continue

                new_img_name = f"{prefix}__{img_file.name}"
                if not dry_run:
                    shutil.copy2(img_file, img_dst / new_img_name)
                stats[split]["images"] += 1

                lbl_file = lbl_src / img_file.with_suffix(".txt").name
                if lbl_file.exists():
                    new_lbl_name = f"{prefix}__{lbl_file.name}"
                    if not dry_run:
                        remap_label(lbl_file, lbl_dst / new_lbl_name, remap)
                    stats[split]["labels"] += 1

        print()

    # Tulis data.yaml gabungan
    out_yaml = OUT_DIR / "data.yaml"
    yaml_content = {
        "path": str(OUT_DIR),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": NC,
        "names": GLOBAL_NAMES,
    }
    if not dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_yaml, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    print("=" * 60)
    print(f"{'[DRY RUN] ' if dry_run else ''}Selesai.")
    for split, s in stats.items():
        print(
            f"  {split:5s}: {s['images']:4d} images | {s['labels']:4d} labels"
            f" | {s['skipped']:4d} file non-image dilewati (.npy dll)"
        )
    print(f"  nc={NC}, names={GLOBAL_NAMES}")
    print(f"  data.yaml → {out_yaml}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge Roboflow YOLO datasets")
    ap.add_argument("--dry-run", action="store_true", help="Preview tanpa menulis file")
    args = ap.parse_args()
    merge(dry_run=args.dry_run)
