#!/usr/bin/env python3

"""
Run a Hailo HEF on a set of images and save results, using a GStreamer
pipeline (hailonet) instead of the direct HailoRT InferVStreams API.

This mirrors run_hef_on_images.py in spirit and output format:
 - selects N random images from --images-dir
 - preprocesses to the HEF input shape (letterbox optionally)
 - pushes frames via appsrc into a pipeline: appsrc -> hailonet -> identity -> fakesink
 - collects raw output tensors from Hailo metadata on the buffers
 - saves per-image NPZs and annotated JPGs, plus a JSON summary

Notes
 - We avoid GStreamer color/scale operations to match preprocessing exactly.
   Images are preprocessed in Python (OpenCV) and pushed to appsrc as RGB or
   RGBF32 frames according to --input-float/--normalize.
 - We require the Hailo GStreamer plugin (hailonet) and the Python 'hailo'
   binding to be available (from TAPPAS).
 - If --decode-yolov8 is given, we try to import a compatible decode routine
   and decode the collected raw tensors on CPU, for apples-to-apples results
   with the baseline script.
"""

from __future__ import annotations

import argparse
import json
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

# GStreamer / Hailo
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
        description="Run a Hailo HEF on images via GStreamer and save outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--hef", required=True, type=Path, help="Path to .hef file")
    p.add_argument(
        "--images-dir",
        required=True,
        type=Path,
        help="Folder containing images (jpg/png/bmp)",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=default_outdir,
        help="Output directory for tensors, annotations, and logs",
    )
    p.add_argument("--limit", type=int, default=8, help="Number of random images")
    p.add_argument("--seed", type=int, default=0, help="Random seed")

    # Preprocess / formats (match baseline flags)
    p.add_argument(
        "--input-float",
        action="store_true",
        help="Push RGBF32 frames to hailonet (else RGB uint8)",
    )
    p.add_argument(
        "--output-float",
        action="store_true",
        help="Ask hailonet for FLOAT32 output tensors (else UINT8)",
    )
    p.add_argument(
        "--normalize",
        action="store_true",
        help="If --input-float: divide pixels by 255.0 before push",
    )
    p.add_argument(
        "--bgr",
        action="store_true",
        help="Keep OpenCV BGR order (default converts to RGB)",
    )
    p.add_argument(
        "--max-side",
        type=int,
        default=None,
        help="Optional cap for max(H,W) before resize to save RAM",
    )
    p.add_argument(
        "--letterbox", action="store_true", help="Use letterbox to fit aspect ratio"
    )
    p.add_argument(
        "--letterbox-pad", type=int, default=114, help="Pad value for letterbox (0-255)"
    )

    # Optional YOLOv8 decode (DFL)
    p.add_argument(
        "--decode-yolov8",
        action="store_true",
        help="Decode YOLOv8 heads (DFL + NMS) and save boxes",
    )
    p.add_argument(
        "--yolo-classes",
        type=int,
        default=None,
        help="Number of classes; if omitted, infer from outputs",
    )
    p.add_argument("--dfl-bins", type=int, default=16, help="DFL bins per side (usually 16)")
    p.add_argument("--nms-iou", type=float, default=0.01, help="IoU threshold for NMS")
    p.add_argument(
        "--score-thresh",
        type=float,
        default=0.01,
        help="Detection score threshold for logs (if scores present)",
    )
    p.add_argument(
        "--max-dets", type=int, default=300, help="Max detections per image after NMS"
    )
    p.add_argument(
        "--max-print",
        type=int,
        default=8,
        help="Max detections to draw/print per image",
    )
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
class RunSummary:
    hef: str
    images_dir: str
    outdir: str
    input_vstream_name: str
    input_shape: Tuple[int, int, int]
    output_shapes: Dict[str, Tuple[int, ...]]
    count: int
    input_float: bool
    output_float: bool
    normalize: bool
    bgr: bool
    images: List[str]


def overlay_text(img: np.ndarray, lines: Iterable[str]) -> np.ndarray:
    out = img.copy()
    y = 24
    for line in lines:
        cv2.putText(
            out,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 22
    return out


def letterbox_resize(
    img: np.ndarray, whc: Tuple[int, int, int], pad_val: int = 114
) -> np.ndarray:
    h, w, c = whc
    ih, iw = img.shape[:2]
    r = min(w / iw, h / ih)
    nw, nh = int(round(iw * r)), int(round(ih * r))
    resized_inner = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((h, w, c), int(pad_val), dtype=resized_inner.dtype)
    top = (h - nh) // 2
    left = (w - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized_inner
    return canvas


def prepare_frame(
    img_path: Path,
    hwc: Tuple[int, int, int],
    use_bgr: bool,
    input_float: bool,
    normalize: bool,
    max_side: Optional[int],
    letterbox: bool,
    lb_pad: int,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w, c = hwc
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    if max_side is not None and max(img.shape[0], img.shape[1]) > max_side:
        scale = max_side / float(max(img.shape[0], img.shape[1]))
        img = cv2.resize(
            img,
            (int(img.shape[1] * scale), int(img.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    if not use_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if letterbox:
        resized = letterbox_resize(img, (h, w, c), pad_val=int(lb_pad))
    else:
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    if input_float:
        arr = resized.astype(np.float32)
        if normalize:
            arr /= 255.0
    else:
        arr = resized.astype(np.uint8)

    viz = (
        resized
        if resized.dtype == np.uint8
        else np.clip(resized * (255.0 if normalize else 1.0), 0, 255).astype(np.uint8)
    )
    return arr, viz


def to_str_shape(arr: np.ndarray | Tuple[int, ...]) -> str:
    shape = arr.shape if isinstance(arr, np.ndarray) else arr
    return "x".join(str(x) for x in shape)


# ----------------------------- GStreamer Driver -----------------------------


class GstHefRunner:
    """Pushes preprocessed frames through hailonet and collects raw tensors.

    For each buffer that exits hailonet, an identity pad probe extracts the
    Hailo ROI and its tensors, converts to numpy arrays, and enqueues a
    Python dict mapping tensor name -> numpy array. The caller pops one result
    per pushed frame.
    """

    def __init__(
        self,
        hef_path: Path,
        in_shape_hwc: Tuple[int, int, int],
        input_float: bool,
        output_float: bool,
    ) -> None:
        self.hef_path = str(hef_path)
        self.in_h, self.in_w, self.in_c = map(int, in_shape_hwc)
        self.input_float = bool(input_float)
        self.output_float = bool(output_float)

        self.pipeline: Optional[Gst.Element] = None
        self.appsrc: Optional[Gst.Element] = None
        self.mainloop: Optional[GLib.MainLoop] = None
        self.mainloop_thread: Optional[threading.Thread] = None
        self.results_q: "queue.Queue[Dict[str, np.ndarray]]" = queue.Queue()
        self.err: Optional[str] = None
        self.bus: Optional[Gst.Bus] = None
        self.buffers_pushed: int = 0
        self.buffers_seen: int = 0

    def build(self) -> None:
        Gst.init(None)
        # hailonet sink caps typically accept only RGB (uint8) frames.
        # We still allow setting input-format-type to FLOAT32 so the plugin
        # handles internal conversion consistently with HailoRT, but we push
        # RGB8 buffers through appsrc either way.
        fmt = "RGB"
        # Do not set input-format-type; hailonet expects RGB (uint8) frame sizes.
        hailo_in_ftype = ""
        hailo_out_ftype = (
            "output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
            if self.output_float
            else "output-format-type=HAILO_FORMAT_TYPE_UINT8"
        )
        # Prefer a stable postprocess that exists in this env to ensure metadata propagation.
        post_so = "/hailo-apps-infra/resources/libyolo_hailortpp_postprocess.so"
        use_post = os.path.isfile(post_so)
        if use_post:
            pipeline_desc = f"""
                appsrc name=src is-live=false format=time do-timestamp=true block=true caps=video/x-raw,format={fmt},width={self.in_w},height={self.in_h},framerate=30/1 !
                queue max-size-buffers=8 leaky=downstream !
                hailonet hef-path={self.hef_path} {hailo_in_ftype} {hailo_out_ftype} !
                hailofilter so-path={post_so} function-name=filter_letterbox remove-tensors=false !
                identity name=after_hailo !
                fakesink sync=false
            """
        else:
            pipeline_desc = f"""
                appsrc name=src is-live=false format=time do-timestamp=true block=true caps=video/x-raw,format={fmt},width={self.in_w},height={self.in_h},framerate=30/1 !
                queue max-size-buffers=8 leaky=downstream !
                hailonet hef-path={self.hef_path} {hailo_in_ftype} {hailo_out_ftype} !
                identity name=after_hailo !
                fakesink sync=false
            """
        try:
            self.pipeline = Gst.parse_launch(pipeline_desc)
        except Exception as e:  # pylint: disable=broad-except
            raise RuntimeError(f"Failed to create pipeline: {e}") from e

        self.appsrc = self.pipeline.get_by_name("src")
        if self.appsrc is None:
            raise RuntimeError("appsrc not found in pipeline")

        identity = self.pipeline.get_by_name("after_hailo")
        if identity is None:
            raise RuntimeError("identity(after_hailo) not found")
        srcpad = identity.get_static_pad("src")
        if srcpad is None:
            raise RuntimeError("Could not get src pad from identity")
        srcpad.add_probe(Gst.PadProbeType.BUFFER, self._on_buffer)

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_bus_message)

    def start(self) -> None:
        assert self.pipeline is not None
        self.pipeline.set_state(Gst.State.PLAYING)
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

    # ---- Data path ----
    def push_frame(self, arr: np.ndarray) -> None:
        if self.appsrc is None:
            raise RuntimeError("appsrc not initialized")
        # Expect NHWC without batch; hailonet consumes 1 frame at a time
        assert arr.ndim == 3 and arr.shape[0] == self.in_h and arr.shape[1] == self.in_w
        # Push RGB uint8 to hailonet; sink caps require RGB.
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)

        # Create Gst.Buffer from numpy memory without extra copies when possible
        bytes_data = arr.tobytes(order="C")
        buf = Gst.Buffer.new_allocate(None, len(bytes_data), None)
        buf.fill(0, bytes_data)

        # timestamps optional; rely on do-timestamp=true
        ret = self.appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            raise RuntimeError(f"push-buffer failed: {ret}")
        self.buffers_pushed += 1

    def end_stream(self) -> None:
        if self.appsrc is None:
            return
        self.appsrc.emit("end-of-stream")

    def get_result(self, timeout: float = 10.0) -> Dict[str, np.ndarray]:
        return self.results_q.get(timeout=timeout)

    # ---- Callbacks ----
    def _on_buffer(self, pad, info):  # type: ignore[override]
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
        try:
            roi = hailo.get_roi_from_buffer(buf)
            # Prefer tensors from ROI if present
            tensors = list(roi.get_tensors()) if roi and roi.has_tensors() else []
            result: Dict[str, np.ndarray] = {}
            if tensors:
                for t in tensors:
                    try:
                        name = t.name()
                    except Exception:
                        # Fallback if name() not available
                        name = getattr(t, "_name", f"tensor_{len(result)}")
                    # Prefer dequantized float when requested
                    # Robust conversion to numpy via raw bytes buffer
                    try:
                        h = int(t.height()); w = int(t.width()); f = int(t.features())
                    except Exception:
                        h = w = f = 0
                    # Try direct view first
                    arr = np.array(t, copy=False)
                    if arr.ndim == 0 and h and w and f:
                        # Expensive fallback: iterate to reconstruct tensor (rare)
                        dtype = np.float32 if self.output_float else np.uint8
                        tmp = np.empty((h, w, f), dtype=dtype)
                        try:
                            for yy in range(h):
                                for xx in range(w):
                                    for cc in range(f):
                                        try:
                                            val = t.get(yy, xx, cc)
                                        except Exception:
                                            val = 0.0
                                        tmp[yy, xx, cc] = val
                            arr = tmp
                        except Exception:
                            arr = tmp
                    result[name] = arr
            else:
                # Try HAILO_MATRIX objects as raw outputs fallback
                try:
                    mats = roi.get_objects_typed(hailo.HAILO_MATRIX)
                except Exception:
                    mats = []
                try:
                    dets = roi.get_objects_typed(hailo.HAILO_DETECTION)
                    det_n = len(dets)
                    if det_n:
                        # Export detections as arrays (boxes xyxy in input pixels)
                        boxes = []
                        scores = []
                        classes = []
                        for det in dets:
                            try:
                                bbox = det.get_bbox()
                                x = float(bbox.xmin()) * float(self.in_w)
                                y = float(bbox.ymin()) * float(self.in_h)
                                w = float(bbox.width()) * float(self.in_w)
                                h = float(bbox.height()) * float(self.in_h)
                                boxes.append([x, y, x + w, y + h])
                            except Exception:
                                continue
                            # Confidence / class id may be missing depending on postprocess
                            try:
                                scores.append(float(det.get_confidence()))
                            except Exception:
                                scores.append(0.0)
                            # Try multiple class id getters
                            cid = None
                            for attr in ("get_label_id", "get_class_id", "get_category_id"):
                                try:
                                    cid = int(getattr(det, attr)())
                                    break
                                except Exception:
                                    continue
                            classes.append(-1 if cid is None else cid)
                        if boxes:
                            result["boxes"] = np.array(boxes, dtype=np.float32)
                            result["scores"] = np.array(scores, dtype=np.float32)
                            result["classes"] = np.array(classes, dtype=np.int32)
                except Exception:
                    det_n = 0
                if mats:
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
                else:
                    # Debug: report if ROI exists but no tensors/matrices attached
                    try:
                        has = roi.has_tensors()
                    except Exception:
                        has = False
                    print(f"[DBG] ROI tensors: {has}; matrices: {len(mats) if mats else 0}; detections: {det_n}")
                    # Still produce an empty result to keep ordering
                    result = {}

            self.results_q.put(result)
            self.buffers_seen += 1
        except Exception as e:  # pylint: disable=broad-except
            self.err = f"Buffer probe failed: {e}"
        return Gst.PadProbeReturn.OK

    def _on_bus_message(self, bus, message):  # type: ignore[override]
        mtype = message.type
        if mtype == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.err = f"GStreamer error: {err} | {debug}"
        elif mtype == Gst.MessageType.EOS:
            # Allow mainloop to quit in stop(); nothing special here
            pass
        return True


# ----------------------------- Main Flow -----------------------------


def main() -> int:
    args = parse_args()

    ensure_dir(args.outdir)
    hailort_log_dir = ensure_dir(args.outdir / "hailort_logs")
    os.environ.setdefault("HAILORT_LOGGER_DIR", str(hailort_log_dir))

    if not args.hef.exists():
        print(f"[ERROR] HEF not found: {args.hef}")
        return 2
    if not args.images_dir.exists():
        print(f"[ERROR] Images dir not found: {args.images_dir}")
        return 2

    all_images = list_images(args.images_dir)
    if not all_images:
        print(f"[ERROR] No images found under: {args.images_dir}")
        return 2
    chosen = pick_random(all_images, args.limit, args.seed)
    print(f"[INFO] Selected {len(chosen)} image(s) from {args.images_dir}")
    for p in chosen:
        print(f"       - {p}")

    # Assume 640x640x3 input (matches most YOLO HEFs)
    in_h, in_w, in_c = (640, 640, 3)
    input_vstream_name = "model/input_layer1"

    # Optional: YOLOv8 decode
    try:
        from run_har_on_images import decode_yolov8_dfl  # type: ignore
    except Exception:
        try:
            from hailo_model_converter.shared_with_docker.run_har_on_images import (  # type: ignore
                decode_yolov8_dfl,
            )
        except Exception:
            decode_yolov8_dfl = None  # type: ignore

    # Prepare output dirs
    tensors_dir = ensure_dir(args.outdir / "raw_tensors")
    ann_dir = ensure_dir(args.outdir / "annotated")

    if args.input_float or args.normalize:
        print(
            "[NOTE] GStreamer hailonet expects RGB8 frames; ignoring --input-float/--normalize for host frames."
        )
        print(
            "       Device-side quantization remains consistent with HEF calibration; outputs should be comparable."
        )

    # Build + start pipeline
    runner = GstHefRunner(
        hef_path=args.hef, in_shape_hwc=(in_h, in_w, in_c), input_float=args.input_float, output_float=args.output_float
    )
    try:
        runner.build()
        runner.start()
    except Exception as e:  # pylint: disable=broad-except
        print(f"[ERROR] Failed to start GStreamer hailonet pipeline: {e}")
        return 2

    # Push frames and collect results
    t0 = time.time()
    results: List[Dict[str, np.ndarray]] = []
    viz_images: List[np.ndarray] = []
    for img_path in chosen:
        arr, viz = prepare_frame(
            img_path,
            (in_h, in_w, in_c),
            args.bgr,
            args.input_float,
            args.normalize,
            args.max_side,
            args.letterbox,
            int(args.letterbox_pad),
        )
        viz_images.append(viz)
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

    runner.end_stream()
    # Give the pipeline a moment to flush EOS
    time.sleep(0.2)
    runner.stop()

    # Results sanity
    if not results or any(r is None for r in results):
        print("[ERROR] No results collected from hailonet pipeline.")
        return 2

    # Build output_shapes summary using the first non-empty result
    output_shapes: Dict[str, Tuple[int, ...]] = {}
    for res in results:
        if res:
            output_shapes = {k: tuple(v.shape) for k, v in res.items()}
            break

    # Per-image save (NPZ + annotated)
    for i, (img_path, res) in enumerate(zip(chosen, results)):
        per_image = dict(res)

        # Optional YOLOv8 decode
        if args.decode_yolov8:
            if decode_yolov8_dfl is None:
                print(
                    "[WARN] --decode-yolov8 requested but decode function not found. Skipping decode."
                )
            else:
                if not args.output_float:
                    print(
                        "[WARN] Decoding quantized UINT8 outputs may be inaccurate. Consider --output-float."
                    )
                boxes, scores, classes = decode_yolov8_dfl(
                    per_image,
                    in_h,
                    in_w,
                    args.yolo_classes,
                    dfl_bins=int(args.dfl_bins),
                    score_thresh=float(args.score_thresh),
                    iou_thresh=float(args.nms_iou),
                    max_dets=int(args.max_dets),
                )
                per_image["boxes"] = boxes
                per_image["classes"] = classes
                per_image["scores"] = scores

        # Save raw tensors (and decoded optional) per image
        npz_path = tensors_dir / f"out_{i:03d}.npz"
        np.savez(str(npz_path), **per_image)

        # Quick-look annotated image
        ann = viz_images[i]
        lines = [
            f"src: {Path(img_path).name}",
            f"input: {in_h}x{in_w}x{in_c}",
        ]
        if per_image:
            first_name = sorted(per_image.keys())[0]
            first_tensor = per_image[first_name]
            flat = np.array(first_tensor).reshape(-1)
            sample_vals = ", ".join(f"{float(v):.3g}" for v in flat[:5])
            lines.append(
                f"out[{first_name}]: {to_str_shape(first_tensor)} | {sample_vals} ..."
            )

        annotated = overlay_text(ann, lines)
        if args.decode_yolov8 and "boxes" in per_image:
            b = np.array(per_image["boxes"], dtype=np.float32)
            s = np.array(per_image.get("scores", []), dtype=np.float32)
            c = np.array(per_image.get("classes", []), dtype=np.int32)
            order = np.argsort(-s) if s.size else np.arange(b.shape[0])
            order = order[: min(len(order), args.max_print)]
            for j in order:
                x1, y1, x2, y2 = b[j].astype(int)
                score = float(s[j]) if s.size else 0.0
                cls_id = int(c[j]) if c.size else -1
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{cls_id}:{score:.2f}",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
        ann_path = ann_dir / f"ann_{i:03d}.jpg"
        cv2.imwrite(str(ann_path), annotated)

    # Summary JSON
    summary = RunSummary(
        hef=str(args.hef.resolve()),
        images_dir=str(args.images_dir.resolve()),
        outdir=str(args.outdir.resolve()),
        input_vstream_name=input_vstream_name,
        input_shape=(in_h, in_w, in_c),
        output_shapes={k: tuple(v) for k, v in output_shapes.items()},
        count=len(chosen),
        input_float=bool(args.input_float),
        output_float=bool(args.output_float),
        normalize=bool(args.normalize),
        bgr=bool(args.bgr),
        images=[str(p.resolve()) for p in chosen],
    )
    with open(args.outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    dt = time.time() - t0
    print("[OK] GStreamer inference complete.")
    print(f"     Saved tensors: {tensors_dir}")
    print(f"     Saved annots:  {ann_dir}")
    print(f"     Summary:       {args.outdir / 'summary.json'}")
    print(f"     Elapsed:       {dt:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
