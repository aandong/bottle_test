"""YOLO tracking + line-crossing counter on conveyor_output.mp4.

Purple vertical line at horizontal center. Tracked objects crossing the
line right-to-left increment a per-class counter. Classes whose count is
above 1 are printed in a panel below the video.
"""

import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs/detect/yolo26s-merged/weights/best.pt"
SOURCE = BASE_DIR / "conveyor_output.mp4"
CSV_PATH = BASE_DIR / "Kelas-NamaProduct-Perusahaan.csv"
OUT_PATH = BASE_DIR / "runs/detect/conveyor_inference/conveyor_count.mp4"

LINE_COLOR = (128, 0, 128)  # purple, BGR
LINE_THICKNESS = 3
PANEL_HEIGHT = 160
PANEL_BG = (30, 30, 30)
TEXT_COLOR = (255, 255, 255)
COLS_PER_ROW = 4


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


def draw_panel(width: int, class_names: dict[int, str], counts: dict[int, int]) -> np.ndarray:
    panel = np.full((PANEL_HEIGHT, width, 3), PANEL_BG, dtype=np.uint8)
    entries = [
        (class_names.get(cls, str(cls)), n)
        for cls, n in sorted(counts.items())
        if n > 1
    ]
    if not entries:
        cv2.putText(panel, "No class count > 1 yet", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
        return panel

    col_w = width // COLS_PER_ROW
    row_h = 30
    for i, (name, n) in enumerate(entries):
        row, col = divmod(i, COLS_PER_ROW)
        x = 15 + col * col_w
        y = 30 + row * row_h
        if y > PANEL_HEIGHT - 10:
            break
        cv2.putText(panel, f"{name}: {n}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)
    return panel


def main():
    class_names = load_class_names(CSV_PATH)

    model = YOLO(str(MODEL_PATH))
    model.model.names = class_names

    cap = cv2.VideoCapture(str(SOURCE))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    line_x = width // 2

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_PATH), fourcc, fps, (width, height + PANEL_HEIGHT))

    counts: dict[int, int] = {}
    counted_ids: set[int] = set()
    prev_x: dict[int, float] = {}

    results = model.track(
        source=str(SOURCE),
        conf=0.5,
        persist=True,
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False,
    )

    for r in results:
        annotated = r.plot()

        boxes = r.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().tolist()
            clss = boxes.cls.int().tolist()
            xyxy = boxes.xyxy.tolist()

            for tid, cls, (x1, _, x2, _) in zip(ids, clss, xyxy):
                cx = (x1 + x2) / 2
                if tid in prev_x:
                    prev_cx = prev_x[tid]
                    if prev_cx > line_x >= cx and tid not in counted_ids:
                        counts[cls] = counts.get(cls, 0) + 1
                        counted_ids.add(tid)
                prev_x[tid] = cx

        cv2.line(annotated, (line_x, 0), (line_x, height), LINE_COLOR, LINE_THICKNESS)

        panel = draw_panel(width, class_names, counts)
        frame_out = np.vstack([annotated, panel])
        writer.write(frame_out)

    writer.release()
    print(f"Saved to {OUT_PATH}")
    print("Final counts:", {class_names.get(c, c): n for c, n in counts.items()})


if __name__ == "__main__":
    main()
