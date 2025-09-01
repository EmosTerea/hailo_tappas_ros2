#!/usr/bin/env python3

"""
Annotate images from previously saved raw tensors (NPZ files).

Inputs a run directory produced by run_hef_on_images.py or
run_hef_on_images_gststream.py, loads raw_tensors/out_*.npz and
summary.json, decodes YOLOv8 heads (DFL) and writes annotated images.

Usage example:
  python3 annotate_saved_tensors.py \
    --run-dir outputs/hef_images_gst_filter_8 \
    --outdir outputs/hef_images_gst_filter_8/ann_from_npz \
    --yolo-classes 15 --letterbox --letterbox-pad 114 --max-print 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Annotate images from saved raw tensors (YOLOv8 decode)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", type=Path, required=True, help="Run dir with raw_tensors/ and summary.json")
    p.add_argument("--outdir", type=Path, required=True, help="Output dir for annotated images")
    p.add_argument("--yolo-classes", type=int, required=True, help="Number of classes in the model")
    p.add_argument("--dfl-bins", type=int, default=16, help="DFL bins (usually 16)")
    p.add_argument("--nms-iou", type=float, default=0.01, help="NMS IoU threshold")
    p.add_argument("--score-thresh", type=float, default=0.01, help="Score threshold")
    p.add_argument("--max-dets", type=int, default=300, help="Max dets after NMS")
    p.add_argument("--max-print", type=int, default=8, help="Max dets to draw per image")
    p.add_argument("--labels", type=Path, default=None, help="Optional JSON list of class names")
    p.add_argument("--letterbox", action="store_true", help="Assume letterbox preprocessing for viz geometry")
    p.add_argument("--letterbox-pad", type=int, default=114, help="Pad used during letterbox")
    return p.parse_args()


def overlay_text(img: np.ndarray, lines: Iterable[str]) -> np.ndarray:
    out = img.copy()
    y = 22
    for line in lines:
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        y += 20
    return out


def make_viz(image_path: Path, whc: Tuple[int, int, int], letterbox: bool, pad: int) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    h, w, c = whc
    if letterbox:
        ih, iw = img.shape[:2]
        r = min(w / iw, h / ih)
        nw, nh = int(round(iw * r)), int(round(ih * r))
        inner = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((h, w, c), int(pad), dtype=inner.dtype)
        top = (h - nh) // 2
        left = (w - nw) // 2
        canvas[top : top + nh, left : left + nw] = inner
        return canvas
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def main() -> int:
    args = parse_args()

    # Import decoder
    try:
        from run_har_on_images import decode_yolov8_dfl  # type: ignore
    except Exception:
        try:
            from hailo_model_converter.shared_with_docker.run_har_on_images import (  # type: ignore
                decode_yolov8_dfl,
            )
        except Exception as e:  # pylint: disable=broad-except
            print("[ERROR] Could not import decode_yolov8_dfl."
                  " Ensure hailo_model_converter is available.")
            print(f"        Import error: {e}")
            return 2

    run_dir = args.run_dir
    raw_dir = run_dir / "raw_tensors"
    summary_path = run_dir / "summary.json"
    if not raw_dir.is_dir() or not summary_path.is_file():
        print(f"[ERROR] Missing raw_tensors/ or summary.json under: {run_dir}")
        return 2

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Load summary
    summary = json.loads(summary_path.read_text())
    in_h, in_w, in_c = summary["input_shape"]
    images: List[str] = summary.get("images", [])
    if not images:
        print("[ERROR] No images listed in summary.json")
        return 2

    # Optional labels
    labels: List[str] | None = None
    if args.labels and args.labels.is_file():
        try:
            labels = json.loads(args.labels.read_text())
        except Exception:
            labels = None

    wrote = 0
    for i, img_path in enumerate(images):
        npz_path = raw_dir / f"out_{i:03d}.npz"
        if not npz_path.is_file():
            print(f"[WARN] NPZ missing for index {i}: {npz_path}")
            continue
        data = np.load(npz_path)
        per_image = {k: data[k] for k in data.files}
        # Try to reuse pre-decoded boxes
        if {"boxes", "classes", "scores"}.issubset(per_image.keys()):
            boxes = np.array(per_image["boxes"])  # xyxy
            scores = np.array(per_image["scores"])
            classes = np.array(per_image["classes"]).astype(int)
        else:
            # Decode from raw heads
            boxes, scores, classes = decode_yolov8_dfl(
                per_image, in_h, in_w, args.yolo_classes,
                dfl_bins=int(args.dfl_bins),
                score_thresh=float(args.score_thresh),
                iou_thresh=float(args.nms_iou),
                max_dets=int(args.max_dets),
            )

        # Prepare viz (preprocessed geometry)
        viz = make_viz(Path(img_path), (in_h, in_w, in_c), args.letterbox, int(args.letterbox_pad))
        vis = viz.copy()

        # Draw top-k
        if boxes.size and scores.size:
            order = np.argsort(-scores)[: min(scores.size, args.max_print)]
            for j in order:
                x1, y1, x2, y2 = boxes[j].astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cls_id = int(classes[j]) if classes.size else -1
                name = labels[cls_id] if labels and 0 <= cls_id < len(labels) else str(cls_id)
                cv2.putText(
                    vis,
                    f"{name}:{float(scores[j]):.2f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        # Add overlay text
        lines = [
            f"src: {Path(img_path).name}",
            f"decoded: {boxes.shape[0]} boxes (top {min(boxes.shape[0], args.max_print)})",
        ]
        vis = overlay_text(vis, lines)
        out_path = outdir / f"ann_{i:03d}.jpg"
        cv2.imwrite(str(out_path), vis)
        wrote += 1

    print(f"[OK] Annotated {wrote} image(s) to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

