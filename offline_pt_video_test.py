#!/usr/bin/env python3

"""
Offline CPU PyTorch (.pt) inference on video or single image.

Parallels offline_hef_video_test.py:
- Reads --input (MP4 or image)
 - Runs a .pt model on CPU using Ultralytics YOLO
- Draws boxes + labels
- Writes:
  - outputs/best_annotated.avi (or best_annotated.jpg for images)
  - outputs/best_summary.json
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

# Ultralytics YOLO is required
try:
    from ultralytics import YOLO
except Exception:
    print("[ERROR] ultralytics is required. Install with: pip install ultralytics")
    raise


# Default labels (matches resources/labels_custom.json) used when --labels-mode=custom
CUSTOM_LABELS: List[str] = [
    "T90",
    "T72",
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
    "Others",
    "BTR",
    "KV1",
]


@dataclass
class Summary:
    label_counts: Dict[str, int] = field(default_factory=dict)
    frames_processed: int = 0

    def bump(self, label: str) -> None:
        self.label_counts[label] = self.label_counts.get(label, 0) + 1


def looks_like_image(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# (No letterbox needed: Ultralytics handles resizing internally.)


def ultralytics_predict(model: YOLO, frame_bgr: np.ndarray, imgsz: Tuple[int, int], conf_thres: float) -> List[Dict[str, Any]]:
    res = model.predict(
        source=frame_bgr,
        imgsz=(int(imgsz[1]), int(imgsz[0])),  # (h, w)
        verbose=False,
        device="cpu",
        conf=float(conf_thres),
    )
    if not res:
        return []
    r = res[0]
    dets: List[Dict[str, Any]] = []
    if not hasattr(r, "boxes") or r.boxes is None or len(r.boxes) == 0:
        return dets
    boxes = r.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), sc, c in zip(xyxy, conf, cls):
        dets.append({
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            "conf": float(sc), "cls": int(c)
        })
    return dets


def draw_and_count(frame_bgr: np.ndarray, dets: List[Dict[str, Any]], label_fn, summary: Summary) -> np.ndarray:
    for d in dets:
        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        if x2 <= x1 or y2 <= y1:
            continue
        conf = float(d.get("conf", 0.0))
        cls_id = int(d.get("cls", -1))
        label = label_fn(cls_id) if cls_id >= 0 else "unknown"
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_bgr, f"{label}:{conf:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
        )
        summary.bump(label)
    summary.frames_processed += 1
    return frame_bgr


def load_labels_from_json(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "labels" in data and isinstance(data["labels"], list):
            return [str(x) for x in data["labels"]]
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data  # direct list
    except Exception as e:
        print(f"[WARN] Could not load labels from {path}: {e}")
    return None


def run(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.input):
        print(f"[ERROR] Input not found: {args.input}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        sys.exit(1)

    # Labels selection (custom vs model-provided)
    labels: Optional[List[str]] = None
    if args.labels_mode == "custom":
        labels = load_labels_from_json(args.labels_json) or CUSTOM_LABELS

    imgsz = (int(args.width), int(args.height))
    model = YOLO(args.model)
    # Try to read model-provided names
    model_names: Optional[Dict[int, str]] = None
    try:
        model_names = {int(k): v for k, v in model.model.names.items()}  # type: ignore[attr-defined]
    except Exception:
        try:
            model_names = {int(i): v for i, v in enumerate(model.names)}  # type: ignore[attr-defined]
        except Exception:
            model_names = None

    # Label resolution
    def get_label(cls_id: int) -> str:
        if args.labels_mode == "custom":
            lbls = labels or CUSTOM_LABELS
            return lbls[cls_id] if 0 <= cls_id < len(lbls) else "unknown"
        if model_names and cls_id in model_names:
            return str(model_names[cls_id])
        return "unknown"

    os.makedirs(args.outdir, exist_ok=True)
    summary = Summary()
    is_image = looks_like_image(args.input)

    # Output paths (keep parity with Hailo script)
    out_video = os.path.join(args.outdir, "best_annotated.avi")
    out_image = os.path.join(args.outdir, "best_annotated.jpg")
    out_json = os.path.join(args.outdir, "best_summary.json")

    if is_image:
        img = cv2.imread(args.input, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[ERROR] Failed to read image: {args.input}")
            sys.exit(1)
        dets = ultralytics_predict(model, img, imgsz, args.conf_thres)
        img = draw_and_count(img, dets, get_label, summary)
        cv2.imwrite(out_image, img)
        print(f"[INFO] Wrote annotated image to: {out_image}")
    else:
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video: {args.input}")
            sys.exit(1)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 0:
            fps = 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))
        if not writer.isOpened():
            print("[ERROR] Failed to open VideoWriter for output.")
            sys.exit(1)
        print(f"[INFO] Writing annotated video to: {out_video} @ {fps:.2f} FPS")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                dets = ultralytics_predict(model, frame, imgsz, args.conf_thres)
                frame = draw_and_count(frame, dets, get_label, summary)
                writer.write(frame)
        finally:
            cap.release()
            writer.release()

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {"frames_processed": summary.frames_processed, "label_counts": summary.label_counts},
            f,
            indent=2,
        )
    print(f"[INFO] Wrote summary: {out_json}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline CPU PyTorch (.pt) inference on video or single image")
    p.add_argument("--input", default="resources/face_recognition.mp4", help="Input media path (MP4 or image)")
    p.add_argument("--model", default="best.pt", help=".pt model path (Ultralytics YOLO)")
    p.add_argument("--outdir", default="outputs", help="Directory to write outputs")
    p.add_argument("--width", type=int, default=640, help="Resize width for inference (used by some backends)")
    p.add_argument("--height", type=int, default=640, help="Resize height for inference (used by some backends)")
    p.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--labels-json", default=None, help="Optional JSON file with label list (overrides built-in list)")
    p.add_argument(
        "--labels-mode", choices=["custom", "model"], default="model",
        help="Use model-provided class names ('model') or custom list ('custom').",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
