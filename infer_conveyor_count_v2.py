"""YOLO tracking + line-crossing counter on conveyor_output.mp4 (v2).

Improvements over v1:
- Anti-blink: per-track bbox cache with grace period — when a track drops
  out for a few frames, its last box is still drawn; coordinates are
  EMA-smoothed to remove jitter.
- Custom drawing (no r.plot()): distinct color per class, small readable
  label with filled background.
- Panel below video: fixed-width columns, per-class colored text, and a
  left-arrow marker on the class that was counted most recently.
"""

import colorsys
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs/detect/yolo26s-merged/weights/best.pt"
SOURCE = BASE_DIR / "conveyor_output.mp4"
CSV_PATH = BASE_DIR / "Kelas-NamaProduct-Perusahaan.csv"
OUT_PATH = BASE_DIR / "runs/detect/conveyor_inference/conveyor_count_v2.mp4"

CONF = 0.3  # lower than v1 to reduce detection dropout (blinking)
GRACE_FRAMES = 8  # keep drawing a lost track's box for this many frames
EMA_ALPHA = 0.6  # bbox smoothing: new = a*current + (1-a)*previous

LINE_COLOR = (200, 0, 200)  # purple, BGR
LINE_THICKNESS = 2

PANEL_HEIGHT = 170
PANEL_BG = (24, 24, 24)
PANEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_FONT_SCALE = 0.42
PANEL_FONT_THICK = 1
COLS_PER_ROW = 3
ROW_H = 22

BOX_FONT_SCALE = 0.45
BOX_FONT_THICK = 1


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


def make_class_colors(n: int) -> dict[int, tuple[int, int, int]]:
    """Distinct BGR color per class via evenly spaced hues."""
    colors = {}
    for i in range(n):
        h = (i * 0.618033988749895) % 1.0  # golden-ratio hue spacing
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors[i] = (int(b * 255), int(g * 255), int(r * 255))
    return colors


def text_color_for(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = bg
    lum = 0.114 * b + 0.587 * g + 0.299 * r
    return (0, 0, 0) if lum > 140 else (255, 255, 255)


def draw_box(frame, x1, y1, x2, y2, color, label):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), base = cv2.getTextSize(label, PANEL_FONT, BOX_FONT_SCALE, BOX_FONT_THICK)
    ty = y1 - 4 if y1 - th - base - 4 > 0 else y2 + th + base + 4
    cv2.rectangle(frame, (x1, ty - th - base), (x1 + tw + 4, ty + base), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 2, ty),
        PANEL_FONT,
        BOX_FONT_SCALE,
        text_color_for(color),
        BOX_FONT_THICK,
        cv2.LINE_AA,
    )


def draw_panel(width, class_names, class_colors, counts, last_counted_cls):
    panel = np.full((PANEL_HEIGHT, width, 3), PANEL_BG, dtype=np.uint8)
    entries = [(cls, n) for cls, n in sorted(counts.items()) if n > 0]
    if not entries:
        cv2.putText(
            panel,
            "Belum ada kelas dengan count > 0",
            (15, 35),
            PANEL_FONT,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return panel

    col_w = width // COLS_PER_ROW
    for i, (cls, n) in enumerate(entries):
        row, col = divmod(i, COLS_PER_ROW)
        x = 12 + col * col_w
        y = 26 + row * ROW_H
        if y > PANEL_HEIGHT - 8:
            break
        color = class_colors[cls]
        name = class_names.get(cls, str(cls))
        # truncate name so it never bleeds into the next column
        max_name_chars = 20
        if len(name) > max_name_chars:
            name = name[: max_name_chars - 2] + ".."
        text = f"{name}: {n}"
        cv2.circle(panel, (x + 4, y - 5), 4, color, -1)
        cv2.putText(
            panel,
            text,
            (x + 14, y),
            PANEL_FONT,
            PANEL_FONT_SCALE,
            (235, 235, 235),
            PANEL_FONT_THICK,
            cv2.LINE_AA,
        )
        if cls == last_counted_cls:
            (tw, _), _ = cv2.getTextSize(
                text, PANEL_FONT, PANEL_FONT_SCALE, PANEL_FONT_THICK
            )
            ax = x + 14 + tw + 8
            ay = y - 5
            pts = np.array(
                [[ax + 10, ay - 6], [ax + 10, ay + 6], [ax, ay]], dtype=np.int32
            )
            cv2.fillPoly(panel, [pts], (0, 255, 255))
    return panel


def main():
    class_names = load_class_names(CSV_PATH)
    class_colors = make_class_colors(len(class_names))

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
    last_counted_cls: int | None = None

    # track state: id -> {"box": np.array(4), "cls": int, "conf": float,
    #                     "miss": int, "prev_cx": float}
    tracks: dict[int, dict] = {}

    results = model.track(
        source=str(SOURCE),
        conf=CONF,
        persist=True,
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False,
    )

    for r in results:
        frame = r.orig_img.copy()

        seen_ids = set()
        boxes = r.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().tolist()
            clss = boxes.cls.int().tolist()
            confs = boxes.conf.tolist()
            xyxy = boxes.xyxy.cpu().numpy()

            for tid, cls, cf, box in zip(ids, clss, confs, xyxy):
                seen_ids.add(tid)
                st = tracks.get(tid)
                if st is None:
                    st = {
                        "box": box.astype(float),
                        "cls": cls,
                        "conf": cf,
                        "miss": 0,
                        "prev_cx": (box[0] + box[2]) / 2,
                    }
                    tracks[tid] = st
                else:
                    st["box"] = EMA_ALPHA * box + (1 - EMA_ALPHA) * st["box"]
                    st["cls"] = cls
                    st["conf"] = cf
                    st["miss"] = 0

                cx = (st["box"][0] + st["box"][2]) / 2
                if st["prev_cx"] > line_x >= cx and tid not in counted_ids:
                    counts[cls] = counts.get(cls, 0) + 1
                    counted_ids.add(tid)
                    last_counted_cls = cls
                st["prev_cx"] = cx

        # age out unseen tracks; draw survivors (incl. grace-period ghosts)
        for tid in list(tracks):
            st = tracks[tid]
            if tid not in seen_ids:
                st["miss"] += 1
                if st["miss"] > GRACE_FRAMES:
                    del tracks[tid]
                    continue
            x1, y1, x2, y2 = st["box"]
            color = class_colors[st["cls"]]
            name = class_names.get(st["cls"], str(st["cls"]))
            label = f"{name} {st['conf']:.2f}"
            draw_box(frame, x1, y1, x2, y2, color, label)

        cv2.line(frame, (line_x, 0), (line_x, height), LINE_COLOR, LINE_THICKNESS)

        panel = draw_panel(width, class_names, class_colors, counts, last_counted_cls)
        writer.write(np.vstack([frame, panel]))

    writer.release()
    print(f"Saved to {OUT_PATH}")
    print("Final counts:", {class_names.get(c, c): n for c, n in counts.items()})


if __name__ == "__main__":
    main()
