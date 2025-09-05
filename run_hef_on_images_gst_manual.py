#!/usr/bin/env python3

"""
Manual post-processing (no .so) runner for Hailo HEFs via GStreamer.

What it does
- Preprocesses images on host (resize or letterbox) to the HEF input size.
- Pushes frames into a GStreamer pipeline: appsrc -> hailonet -> identity -> fakesink.
- Pulls raw output tensors from Hailo metadata (no hailofilter used).
- Optionally decodes YOLOv8 (DFL) in pure Python and rescales boxes back
  to the original image using letterbox parameters.
- Saves per-image NPZs (raw tensors and optional decoded boxes) and
  annotated JPEGs with drawn detections.

Why another script?
- Avoids Hailo post-processing .so libraries entirely.
- Keeps the pipeline minimal and robust to plugin differences.
- Makes letterboxing and shape handling explicit and reproducible in Python.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

import gi  # type: ignore

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # type: ignore
import hailo  # type: ignore


# ----------------------------- CLI & Utils -----------------------------


def parse_args() -> argparse.Namespace:
    default_outdir = (
        Path("shared_with_docker/outputs")
        if Path("shared_with_docker").exists()
        else Path("outputs")
    )
    p = argparse.ArgumentParser(
        description="Run HEF on images via GStreamer with manual postprocessing (no .so).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--hef", required=True, type=Path, help="Path to .hef file")
    p.add_argument("--images-dir", type=Path, help="Folder with images")
    p.add_argument("--single-image", type=Path, default=None, help="Run on this image only")
    p.add_argument("--outdir", type=Path, default=default_outdir, help="Output directory")
    p.add_argument("--limit", type=int, default=8, help="# random images to run")
    p.add_argument("--seed", type=int, default=0, help="Random shuffle seed")

    # Input preprocessing
    p.add_argument("--input-h", type=int, default=640, help="Model input height")
    p.add_argument("--input-w", type=int, default=640, help="Model input width")
    p.add_argument("--input-c", type=int, default=3, help="Model input channels")
    p.add_argument("--bgr", action="store_true", help="Keep BGR order (default RGB)")
    p.add_argument("--letterbox", action="store_true", help="Use letterbox instead of stretch resize")
    p.add_argument("--letterbox-pad", type=int, default=114, help="Letterbox pad value 0-255")
    p.add_argument("--max-side", type=int, default=None, help="Downscale large images before preprocess")

    # Decoding options
    p.add_argument(
        "--decoder",
        type=str,
        default="none",
        choices=["none", "yolov8"],
        help="Pure-Python decoder to run on raw tensors",
    )
    p.add_argument("--classes", type=int, default=None, help="# classes for decoder (YOLOv8)")
    p.add_argument("--dfl-bins", type=int, default=16, help="DFL bins per side (YOLOv8)")
    p.add_argument("--score-thresh", type=float, default=0.25, help="Score threshold for drawing")
    p.add_argument("--nms-iou", type=float, default=0.45, help="IoU for NMS")
    p.add_argument("--max-dets", type=int, default=300, help="Max dets per image after NMS")
    p.add_argument("--max-print", type=int, default=20, help="Max dets to print/draw")

    return p.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files


def pick_random(xs: Sequence[Path], k: int, seed: int) -> List[Path]:
    rng = random.Random(seed)
    xs = list(xs)
    if not xs:
        return []
    if k >= len(xs):
        rng.shuffle(xs)
        return xs
    return rng.sample(xs, k)


@dataclass
class LetterboxMeta:
    scale: float
    pad_left: int
    pad_top: int
    new_w: int
    new_h: int
    orig_w: int
    orig_h: int


@dataclass
class RunSummary:
    hef: str
    images: List[str]
    outdir: str
    input_shape: Tuple[int, int, int]
    output_shapes: Dict[str, Tuple[int, ...]]
    decoder: str
    classes: Optional[int]
    dfl_bins: int
    letterbox: bool


def overlay_text(img: np.ndarray, lines: Iterable[str]) -> np.ndarray:
    out = img.copy()
    y = 22
    for line in lines:
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        y += 20
    return out


def letterbox_resize(img: np.ndarray, out_h: int, out_w: int, pad_val: int = 114) -> Tuple[np.ndarray, LetterboxMeta]:
    ih, iw = img.shape[:2]
    r = min(out_w / iw, out_h / ih)
    nw, nh = int(round(iw * r)), int(round(ih * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((out_h, out_w, img.shape[2]), int(pad_val), dtype=resized.dtype)
    top = (out_h - nh) // 2
    left = (out_w - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    meta = LetterboxMeta(scale=r, pad_left=left, pad_top=top, new_w=nw, new_h=nh, orig_w=iw, orig_h=ih)
    return canvas, meta


def stretch_resize(img: np.ndarray, out_h: int, out_w: int) -> Tuple[np.ndarray, LetterboxMeta]:
    ih, iw = img.shape[:2]
    resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    meta = LetterboxMeta(scale=out_w / iw, pad_left=0, pad_top=0, new_w=out_w, new_h=out_h, orig_w=iw, orig_h=ih)
    return resized, meta


def prepare_frame(
    img_path: Path,
    in_h: int,
    in_w: int,
    use_bgr: bool,
    max_side: Optional[int],
    letterbox: bool,
    lb_pad: int,
) -> Tuple[np.ndarray, np.ndarray, LetterboxMeta]:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    if not use_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if max_side is not None and max(img.shape[:2]) > max_side:
        scale = max_side / float(max(img.shape[:2]))
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    if letterbox:
        resized, meta = letterbox_resize(img, in_h, in_w, pad_val=int(lb_pad))
    else:
        resized, meta = stretch_resize(img, in_h, in_w)

    arr = resized.astype(np.uint8, copy=False)
    viz = resized if resized.dtype == np.uint8 else np.clip(resized, 0, 255).astype(np.uint8)
    return arr, viz, meta


def to_str_shape(arr: np.ndarray | Tuple[int, ...]) -> str:
    shape = arr.shape if isinstance(arr, np.ndarray) else arr
    return "x".join(str(int(x)) for x in shape)


# ----------------------------- GStreamer Driver -----------------------------


class GstHefNoPost:
    """Minimal pipeline: appsrc -> hailonet -> identity -> fakesink.

    Extracts raw Hailo tensors from each buffer (no hailofilter involved).
    Enqueues a dict[str, np.ndarray] per frame.
    """

    def __init__(self, hef_path: Path, in_h: int, in_w: int) -> None:
        self.hef_path = str(hef_path)
        self.in_h = int(in_h)
        self.in_w = int(in_w)
        self.pipeline: Optional[Gst.Element] = None
        self.appsrc: Optional[Gst.Element] = None
        self.bus: Optional[Gst.Bus] = None
        self.mainloop: Optional[GLib.MainLoop] = None
        self.mainloop_thread: Optional[threading.Thread] = None
        self.q: "queue.Queue[Dict[str, np.ndarray]]" = queue.Queue()
        self.err: Optional[str] = None

    def build(self) -> None:
        Gst.init(None)
        caps = f"video/x-raw,format=RGB,width={self.in_w},height={self.in_h},framerate=30/1"
        pipeline_desc = f"""
            appsrc name=src is-live=false format=time do-timestamp=true block=true caps={caps} !
            queue max-size-buffers=8 leaky=downstream !
            hailonet hef-path={self.hef_path} output-format-type=HAILO_FORMAT_TYPE_FLOAT32 force-writable=true !
            identity name=tap !
            fakesink sync=false
        """
        try:
            self.pipeline = Gst.parse_launch(pipeline_desc)
        except Exception as e:  # pylint: disable=broad-except
            raise RuntimeError(f"Failed to create pipeline: {e}") from e

        self.appsrc = self.pipeline.get_by_name("src")
        if self.appsrc is None:
            raise RuntimeError("appsrc not found")
        identity = self.pipeline.get_by_name("tap")
        if identity is None:
            raise RuntimeError("identity tap not found")
        srcpad = identity.get_static_pad("src")
        if srcpad is None:
            raise RuntimeError("tap src pad not found")
        srcpad.add_probe(Gst.PadProbeType.BUFFER, self._on_buffer)

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_bus)

    def start(self) -> None:
        assert self.pipeline is not None
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.pipeline.get_state(timeout=Gst.SECOND * 3)
        except Exception:
            pass
        self.mainloop = GLib.MainLoop()
        self.mainloop_thread = threading.Thread(target=self.mainloop.run, daemon=True)
        self.mainloop_thread.start()

    def stop(self) -> None:
        try:
            if self.pipeline is not None:
                self.pipeline.set_state(Gst.State.NULL)
            if self.mainloop is not None:
                self.mainloop.quit()
        except Exception:
            pass

    def push_frame(self, rgb8: np.ndarray) -> None:
        if self.appsrc is None:
            raise RuntimeError("appsrc not ready")
        assert rgb8.dtype == np.uint8 and rgb8.ndim == 3
        assert rgb8.shape[0] == self.in_h and rgb8.shape[1] == self.in_w
        payload = rgb8.tobytes(order="C")
        buf = Gst.Buffer.new_allocate(None, len(payload), None)
        buf.fill(0, payload)
        ret = self.appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            raise RuntimeError(f"push-buffer failed: {ret}")

    def end_stream(self) -> None:
        if self.appsrc is not None:
            self.appsrc.emit("end-of-stream")

    def get_result(self, timeout: float = 10.0) -> Dict[str, np.ndarray]:
        return self.q.get(timeout=timeout)

    # ---- Callbacks ----
    def _on_buffer(self, pad, info):  # type: ignore[override]
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
        try:
            roi = hailo.get_roi_from_buffer(buf)
            result: Dict[str, np.ndarray] = {}
            if roi:
                tensors = []
                try:
                    tensors = list(roi.get_tensors())
                except Exception:
                    tensors = []
                for t in tensors:
                    # Collect dims if available for reliable reshape
                    h = w = c = None
                    try:
                        h = int(t.height()); w = int(t.width()); c = int(t.features())
                    except Exception:
                        pass
                    try:
                        name = t.name()
                    except Exception:
                        name = f"tensor_{len(result)}"
                    try:
                        arr = np.array(t.get_full_percision(), copy=False)  # float (hailo typo)
                    except Exception:
                        try:
                            arr = np.array(t.get_full_precision(), copy=False)  # alt spelling
                        except Exception:
                            try:
                                arr = np.array(t, copy=False)
                            except Exception:
                                arr = np.array([])
                    arr = np.array(arr, copy=True)  # own memory
                    if h and w and c and arr.size == h * w * c:
                        arr = arr.reshape((h, w, c))
                    result[name] = arr
                # Fallback: matrices (raw outputs) if tensors list is empty
                if not result:
                    try:
                        mats = roi.get_objects_typed(hailo.HAILO_MATRIX)
                    except Exception:
                        mats = []
                    for idx, m in enumerate(mats):
                        try:
                            data = m.get_data()
                            arr = np.array(data)
                            try:
                                shp = m.shape()
                                if isinstance(shp, (list, tuple)):
                                    shp = tuple(int(x) for x in shp)
                                else:
                                    shp = tuple(int(x) for x in list(shp))
                                if arr.size == int(np.prod(shp)):
                                    arr = arr.reshape(shp)
                            except Exception:
                                pass
                            result[f"matrix_{idx}"] = arr
                        except Exception:
                            continue
            self.q.put(result)
        except Exception as e:  # pylint: disable=broad-except
            self.err = f"Probe error: {e}"
        return Gst.PadProbeReturn.OK

    def _on_bus(self, bus, message):  # type: ignore[override]
        if message.type == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            self.err = f"GStreamer error: {err} | {dbg}"
        return True


# ----------------------------- YOLOv8 Decode (Python) -----------------------------


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float, max_dets: int) -> List[int]:
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0 and len(keep) < max_dets:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def build_grids(input_w: int, input_h: int, feat_w: int, feat_h: int) -> Tuple[np.ndarray, np.ndarray, int]:
    stride_w = input_w // feat_w
    stride_h = input_h // feat_h
    assert stride_w == stride_h, "Non-square stride not supported for YOLOv8"
    s = stride_w
    gy, gx = np.meshgrid(np.arange(feat_h), np.arange(feat_w), indexing="ij")
    cx = (gx + 0.5) * s
    cy = (gy + 0.5) * s
    return cx.astype(np.float32), cy.astype(np.float32), s


def dfl_expectation(probs: np.ndarray) -> np.ndarray:
    # probs: (..., bins)
    bins = np.arange(probs.shape[-1], dtype=np.float32)
    return np.sum(probs * bins, axis=-1)


def decode_yolov8_dfl(
    tensors: Mapping[str, np.ndarray],
    input_h: int,
    input_w: int,
    num_classes: Optional[int],
    dfl_bins: int = 16,
    score_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_dets: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Heuristic YOLOv8 DFL decode from raw feature maps.

    Assumptions
    - Outputs provide per-scale tensors shaped like (H, W, C).
    - Each scale either has two tensors (cls and reg) or a single concatenated tensor.
    - Distances l,t,r,b are predicted via DFL (softmax over `dfl_bins`).
    - Grid centers align at stride s = input/feat.
    """
    # Group tensors by spatial dims (H, W)
    by_hw: Dict[Tuple[int, int], List[Tuple[str, np.ndarray]]] = {}
    for name, arr in tensors.items():
        if arr.ndim == 3:
            h, w, c = arr.shape
            by_hw.setdefault((h, w), []).append((name, arr))

    boxes_all: List[np.ndarray] = []
    scores_all: List[np.ndarray] = []
    classes_all: List[np.ndarray] = []

    for (fh, fw), items in sorted(by_hw.items(), key=lambda kv: -kv[0][0]):
        # Identify cls/reg from channels
        cls_arr: Optional[np.ndarray] = None
        reg_arr: Optional[np.ndarray] = None
        concat_arr: Optional[np.ndarray] = None

        if num_classes is not None:
            for name, arr in items:
                c = arr.shape[2]
                if c == num_classes:
                    cls_arr = arr
                elif c == 4 * dfl_bins:
                    reg_arr = arr
                elif c == num_classes + 4 * dfl_bins:
                    concat_arr = arr
        else:
            # Try to guess classes from the non-reg tensor with largest C
            for name, arr in items:
                c = arr.shape[2]
                if c % dfl_bins == 0 and c // dfl_bins == 4:
                    reg_arr = arr
            if reg_arr is None and items:
                # look for a concatenated tensor
                for _, arr in items:
                    c = arr.shape[2]
                    if c > 4 * dfl_bins:
                        concat_arr = arr
                        break
            if num_classes is None and concat_arr is not None:
                num_classes = concat_arr.shape[2] - 4 * dfl_bins
            if num_classes is None and reg_arr is not None and items:
                # guess classes from a remaining tensor in this scale
                for _, arr in items:
                    if arr is reg_arr:
                        continue
                    num_classes = arr.shape[2]
                    cls_arr = arr
                    break

        if concat_arr is not None:
            cls_arr = concat_arr[:, :, : num_classes]
            reg_arr = concat_arr[:, :, num_classes : num_classes + 4 * dfl_bins]

        if reg_arr is None or cls_arr is None:
            # Cannot decode this scale
            continue

        # Softmax over last dim per side
        reg = reg_arr.reshape(fh, fw, 4, dfl_bins)
        reg = reg - reg.max(axis=-1, keepdims=True)
        reg = np.exp(reg)
        reg = reg / np.sum(reg, axis=-1, keepdims=True)
        ltrb = dfl_expectation(reg)  # (fh, fw, 4)

        # Grid centers
        cx, cy, stride = build_grids(input_w, input_h, fw, fh)
        cx = np.broadcast_to(cx[..., None], (fh, fw, 1)).astype(np.float32)
        cy = np.broadcast_to(cy[..., None], (fh, fw, 1)).astype(np.float32)

        # Distances are in bins; multiply by stride to get pixels
        l = ltrb[:, :, 0:1] * stride
        t = ltrb[:, :, 1:2] * stride
        r = ltrb[:, :, 2:3] * stride
        b = ltrb[:, :, 3:4] * stride
        x1 = cx - l
        y1 = cy - t
        x2 = cx + r
        y2 = cy + b
        boxes = np.concatenate([x1, y1, x2, y2], axis=2).reshape(-1, 4)

        # Class scores (sigmoid)
        cls = sigmoid(cls_arr).reshape(-1, cls_arr.shape[2])
        scores = cls.max(axis=1)
        classes = cls.argmax(axis=1)

        # Threshold and NMS per-scale
        keep = np.where(scores >= float(score_thresh))[0]
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]
        keep_nms = nms_xyxy(boxes, scores, float(iou_thresh), int(max_dets))
        boxes_all.append(boxes[keep_nms])
        scores_all.append(scores[keep_nms])
        classes_all.append(classes[keep_nms])

    if not boxes_all:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32)

    boxes = np.concatenate(boxes_all, axis=0)
    scores = np.concatenate(scores_all, axis=0)
    classes = np.concatenate(classes_all, axis=0)
    return boxes.astype(np.float32), scores.astype(np.float32), classes.astype(np.int32)


def deletterbox_boxes(boxes: np.ndarray, lb: LetterboxMeta) -> np.ndarray:
    # Boxes are in input-space pixels (after resize/letterbox). Map back to original image.
    if boxes.size == 0:
        return boxes
    x = boxes.copy()
    x[:, [0, 2]] -= lb.pad_left
    x[:, [1, 3]] -= lb.pad_top
    x[:, :4] /= max(lb.scale, 1e-9)
    # Clip to original image size
    x[:, 0] = np.clip(x[:, 0], 0, lb.orig_w - 1)
    x[:, 1] = np.clip(x[:, 1], 0, lb.orig_h - 1)
    x[:, 2] = np.clip(x[:, 2], 0, lb.orig_w - 1)
    x[:, 3] = np.clip(x[:, 3], 0, lb.orig_h - 1)
    return x


# ----------------------------- Main Flow -----------------------------


def main() -> int:
    args = parse_args()

    if not args.hef.exists():
        print(f"[ERROR] HEF not found: {args.hef}")
        return 2
    if args.single_image is None and (args.images_dir is None or not args.images_dir.exists()):
        print("[ERROR] Provide --images-dir or --single-image")
        return 2

    outdir = ensure_dir(args.outdir)
    tensors_dir = ensure_dir(outdir / "raw_tensors")
    ann_dir = ensure_dir(outdir / "annotated")

    # Select images
    if args.single_image is not None:
        chosen = [args.single_image]
    else:
        chosen = pick_random(list_images(args.images_dir), args.limit, args.seed)
        if not chosen:
            print(f"[ERROR] No images found in {args.images_dir}")
            return 2
    print(f"[INFO] Selected {len(chosen)} image(s)")
    for p in chosen:
        print(f"       - {p}")

    in_h, in_w, in_c = int(args.input_h), int(args.input_w), int(args.input_c)
    runner = GstHefNoPost(args.hef, in_h, in_w)
    try:
        runner.build(); runner.start()
    except Exception as e:  # pylint: disable=broad-except
        print(f"[ERROR] Failed to start pipeline: {e}")
        return 2

    # Push frames and collect outputs
    t0 = time.time()
    results: List[Dict[str, np.ndarray]] = []
    viz_imgs: List[np.ndarray] = []
    metas: List[LetterboxMeta] = []
    for img_path in chosen:
        arr, viz, lb = prepare_frame(
            img_path,
            in_h,
            in_w,
            use_bgr=args.bgr,
            max_side=args.max_side,
            letterbox=args.letterbox,
            lb_pad=int(args.letterbox_pad),
        )
        viz_imgs.append(viz)
        metas.append(lb)
        runner.push_frame(arr)
        try:
            res = runner.get_result(timeout=30.0)
        except Exception as e:  # pylint: disable=broad-except
            print(f"[ERROR] Timed out waiting for inference result: {e}")
            runner.stop()
            return 2
        if runner.err:
            print(f"[ERROR] Pipeline error: {runner.err}")
            runner.stop()
            return 2
        results.append(res)

    runner.end_stream(); time.sleep(0.2); runner.stop()

    if not results or any(r is None for r in results):
        print("[ERROR] No results collected.")
        return 2

    # Determine output shapes (from first image)
    output_shapes: Dict[str, Tuple[int, ...]] = {}
    for r in results:
        if r:
            output_shapes = {k: tuple(v.shape) for k, v in r.items()}
            break

    # Postprocess per-image
    for i, (img_path, res, viz, lb) in enumerate(zip(chosen, results, viz_imgs, metas)):
        per_image = dict(res)

        boxes = np.zeros((0, 4), np.float32)
        scores = np.zeros((0,), np.float32)
        classes = np.zeros((0,), np.int32)
        if args.decoder == "yolov8":
            boxes, scores, classes = decode_yolov8_dfl(
                per_image,
                input_h=in_h,
                input_w=in_w,
                num_classes=args.classes,
                dfl_bins=int(args.dfl_bins),
                score_thresh=float(args.score_thresh),
                iou_thresh=float(args.nms_iou),
                max_dets=int(args.max_dets),
            )
            # Map boxes back to original image
            boxes = deletterbox_boxes(boxes, lb)
            per_image["boxes"] = boxes
            per_image["scores"] = scores
            per_image["classes"] = classes

        # Save raw and decoded tensors
        np.savez(str(tensors_dir / f"out_{i:03d}.npz"), **per_image)

        # Annotate
        ann = viz.copy()
        lines = [
            f"src: {Path(img_path).name}",
            f"input: {in_h}x{in_w}x{in_c}",
        ]
        if res:
            k0 = sorted(res.keys())[0]
            flat = np.array(res[k0]).reshape(-1)
            lines.append(f"out[{k0}]: {to_str_shape(res[k0])} | {', '.join(f'{float(v):.3g}' for v in flat[:5])} ...")
        ann = overlay_text(ann, lines)

        # Draw detections (on letterboxed viz)
        if boxes.size:
            order = np.argsort(-scores)[: min(len(scores), args.max_print)]
            for j in order:
                x1, y1, x2, y2 = boxes[j].astype(int)
                score = float(scores[j]); cid = int(classes[j])
                cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(ann, f"{cid}:{score:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(str(ann_dir / f"ann_{i:03d}.jpg"), ann)

    # Summary JSON
    summary = RunSummary(
        hef=str(args.hef.resolve()),
        images=[str(p.resolve()) for p in chosen],
        outdir=str(outdir.resolve()),
        input_shape=(in_h, in_w, in_c),
        output_shapes=output_shapes,
        decoder=args.decoder,
        classes=args.classes,
        dfl_bins=int(args.dfl_bins),
        letterbox=bool(args.letterbox),
    )
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    dt = time.time() - t0
    print("[OK] Inference + manual postprocessing complete.")
    print(f"     Raw tensors: {tensors_dir}")
    print(f"     Annotated:   {ann_dir}")
    print(f"     Summary:     {outdir / 'summary.json'}")
    print(f"     Elapsed:     {dt:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
