"""YOLO inference on conveyor_output.mp4 using merged best.pt weights."""

import csv
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs/detect/yolo26s-merged/weights/best.pt"
SOURCE = BASE_DIR / "conveyor_output.mp4"
CSV_PATH = BASE_DIR / "Kelas-NamaProduct-Perusahaan.csv"


def load_class_names(csv_path: Path) -> dict[int, str]:
    names = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kelas = row.get("Kelas")
            nama = row.get("Nama")
            if kelas is None or kelas.strip() == "":
                continue
            names[int(kelas)] = nama.strip()
    return names


def main():
    class_names = load_class_names(CSV_PATH)

    model = YOLO(str(MODEL_PATH))
    model.model.names = class_names

    model.predict(
        source=str(SOURCE),
        conf=0.5,
        save=True,
        project=str(BASE_DIR / "runs/detect"),
        name="conveyor_inference",
    )


if __name__ == "__main__":
    main()
