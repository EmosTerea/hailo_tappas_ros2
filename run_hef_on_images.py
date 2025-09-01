#!/usr/bin/env python3

"""
Run a Hailo HEF on a handful of images and save results.

This is a minimal, model-agnostic smoke test that:
 - loads a compiled `model.hef` with HailoRT (Python API)
 - selects N random images from a directory
 - resizes them to the network's input shape (from the HEF metadata)
 - runs synchronous inference via `InferVStreams`
 - saves raw output tensors per-image (NPZ) and a summary JSON to `--outdir`
 - saves quick-look annotated images (just textual overlays; no model-specific postprocess)

Notes
 - Without model-specific pre/post-process, the numeric results may not be meaningful.
   This script aims to verify the HEF runs end-to-end and produce reproducible outputs.
 - If your model expects normalization or letterbox, add `--normalize` and/or add your
   model's exact preprocessing to `prepare_batch`.

Reference: HailoRT User Guide 4.22.0, Python Inference Tutorial (InferVStreams).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore


# ----------------------------- CLI & Utils -----------------------------


def parse_args() -> argparse.Namespace:
    default_outdir = (
        Path("shared_with_docker/outputs")
        if Path("shared_with_docker").exists()
        else Path("outputs")
    )
    p = argparse.ArgumentParser(
        description="Run a Hailo HEF on random images and save outputs.",
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
    p.add_argument(
        "--interface",
        choices=["pcie", "auto"],
        default="pcie",
        help="Stream interface for ConfigureParams",
    )
    p.add_argument(
        "--input-float",
        action="store_true",
        help="Send FLOAT32 to input vstream (else UINT8)",
    )
    p.add_argument(
        "--output-float",
        action="store_true",
        help="Read outputs as FLOAT32 (else UINT8)",
    )
    p.add_argument(
        "--normalize",
        action="store_true",
        help="If --input-float: divide pixels by 255.0",
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
    # Preprocess options (match HAR runner)
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
    p.add_argument(
        "--dfl-bins", type=int, default=16, help="DFL bins per side (usually 16)"
    )
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
    interface: str
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


def prepare_batch(
    image_paths: Sequence[Path],
    hwc: Tuple[int, int, int],
    use_bgr: bool,
    input_float: bool,
    normalize: bool,
    max_side: int | None,
    letterbox: bool,
    lb_pad: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    h, w, c = hwc
    batch = np.empty(
        (len(image_paths), h, w, c), dtype=(np.float32 if input_float else np.uint8)
    )
    viz_images: List[np.ndarray] = []
    for i, path in enumerate(image_paths):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        # Optional pre-resize cap to save memory
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
            ih, iw = img.shape[:2]
            r = min(w / iw, h / ih)
            nw, nh = int(round(iw * r)), int(round(ih * r))
            resized_inner = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((h, w, c), int(lb_pad), dtype=resized_inner.dtype)
            top = (h - nh) // 2
            left = (w - nw) // 2
            canvas[top : top + nh, left : left + nw] = resized_inner
            resized = canvas
        else:
            resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        if input_float:
            arr = resized.astype(np.float32)
            if normalize:
                arr /= 255.0
        else:
            arr = resized.astype(np.uint8)
        batch[i] = arr
        viz = (
            resized
            if resized.dtype == np.uint8
            else np.clip(resized * (255.0 if normalize else 1.0), 0, 255).astype(
                np.uint8
            )
        )
        viz_images.append(viz)
    return batch, viz_images


def to_str_shape(arr: np.ndarray | Tuple[int, ...]) -> str:
    shape = arr.shape if isinstance(arr, np.ndarray) else arr
    return "x".join(str(x) for x in shape)


# ----------------------------- Main Flow -----------------------------


def main() -> int:
    args = parse_args()

    # Defer heavy imports until after args so we can direct logging to a writable folder
    ensure_dir(args.outdir)
    hailort_log_dir = ensure_dir(args.outdir / "hailort_logs")
    # Many environments require HailoRT log dir to exist and be writable
    os.environ.setdefault("HAILORT_LOGGER_DIR", str(hailort_log_dir))

    t_start = time.time()

    try:
        from hailo_platform import (
            HEF,
            VDevice,
            HailoStreamInterface,
            ConfigureParams,
            InputVStreamParams,
            OutputVStreamParams,
            InferVStreams,
            FormatType,
        )
    except Exception as e:  # pylint: disable=broad-except
        print("[ERROR] Failed to import hailo_platform Python API.")
        print("        Ensure HailoRT is installed in your current Python env.")
        print(f"        Import error: {e}")
        print(
            "        Tip: activate the provided virtualenv: source workspace/hailo_virtualenv/bin/activate"
        )
        return 2

    # Optional: reuse YOLOv8 decode from HAR runner if available
    try:  # local import; no hard dependency
        from run_har_on_images import decode_yolov8_dfl  # type: ignore
    except Exception:
        try:
            # Fallback: shared implementation under hailo_model_converter
            from hailo_model_converter.shared_with_docker.run_har_on_images import (  # type: ignore
                decode_yolov8_dfl,
            )
        except Exception:
            decode_yolov8_dfl = None  # type: ignore

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

    # Prepare output dirs
    tensors_dir = ensure_dir(args.outdir / "raw_tensors")
    ann_dir = ensure_dir(args.outdir / "annotated")

    # Load HEF and configure device and vstreams
    hef = HEF(str(args.hef))

    # Choose interface
    if args.interface == "pcie":
        iface = HailoStreamInterface.PCIe
    else:
        iface = HailoStreamInterface.PCIe  # default to PCIe for most setups

    configure_params = ConfigureParams.create_from_hef(hef=hef, interface=iface)

    # Query input/output vstream infos
    input_infos = hef.get_input_vstream_infos()
    output_infos = hef.get_output_vstream_infos()
    if len(input_infos) != 1:
        print(
            f"[ERROR] This script currently supports a single input vstream; found {len(input_infos)}: "
            + ", ".join(info.name for info in input_infos)
        )
        return 2
    input_info = input_infos[0]

    # input_info.shape is (H, W, C)
    try:
        in_h, in_w, in_c = input_info.shape  # type: ignore[attr-defined]
    except Exception:
        # Fallback in case of API difference
        shp = tuple(int(x) for x in input_info.shape)
        if len(shp) != 3:
            print(f"[ERROR] Unexpected input shape: {shp}")
            return 2
        in_h, in_w, in_c = shp  # type: ignore[misc]

    # Build params
    in_ftype = FormatType.FLOAT32 if args.input_float else FormatType.UINT8
    out_ftype = FormatType.FLOAT32 if args.output_float else FormatType.UINT8

    try:
        with VDevice() as vdevice:
            network_groups = vdevice.configure(hef, configure_params)
            if not network_groups:
                print("[ERROR] No network groups returned from configure().")
                return 2
            network_group = network_groups[0]
            network_group_params = network_group.create_params()

            input_vstreams_params = InputVStreamParams.make(
                network_group, format_type=in_ftype
            )
            output_vstreams_params = OutputVStreamParams.make(
                network_group, format_type=out_ftype
            )

            # Prepare input batch (NHWC)
            batch, viz_images = prepare_batch(
                chosen,
                (in_h, in_w, in_c),
                args.bgr,
                args.input_float,
                args.normalize,
                args.max_side,
                args.letterbox,
                int(args.letterbox_pad),
            )
            input_data = {input_info.name: batch}

            print(
                f"[INFO] Running inference | input={input_info.name} shape={batch.shape} dtype={batch.dtype}"
            )
            with InferVStreams(
                network_group, input_vstreams_params, output_vstreams_params
            ) as infer_pipeline:
                with network_group.activate(network_group_params):
                    infer_results: Mapping[str, np.ndarray] = infer_pipeline.infer(
                        input_data
                    )
    except Exception as e:  # pylint: disable=broad-except
        msg = str(e)
        if "HAILO_DRIVER_NOT_INSTALLED" in msg or "hailo" in msg.lower():
            print(
                "[ERROR] Failed to open Hailo device via HailoRT. Is the driver installed and device connected?"
            )
            print(f"        Details: {e}")
            print(
                "        Note: HEF inference requires HailoRT + Hailo device. Use HAR + emulator otherwise."
            )
            return 2
        raise

    # Save outputs per-image as NPZ and annotated image with brief stats
    out_shapes = {name: tuple(arr.shape) for name, arr in infer_results.items()}
    for i, img_path in enumerate(chosen):
        per_image = {name: arr[i] for name, arr in infer_results.items()}

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

        npz_path = tensors_dir / f"out_{i:03d}.npz"
        np.savez(str(npz_path), **per_image)

        # Quick-look annotated image (use preprocessed viz to match input geometry)
        ann = (
            viz_images[i].copy()
            if "viz_images" in locals()
            else cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        )
        if ann is None:
            continue
        lines = [
            f"src: {img_path.name}",
            f"input: {input_info.name} -> {in_h}x{in_w}x{in_c}",
        ]
        # Add first output's brief stats
        first_name, first_tensor = next(iter(per_image.items()))
        flat = np.array(first_tensor).reshape(-1)
        sample_vals = ", ".join(f"{float(v):.3g}" for v in flat[:5])
        lines.append(
            f"out[{first_name}]: {to_str_shape(first_tensor)} | {sample_vals} ..."
        )
        annotated = overlay_text(ann, lines)
        # Draw decoded boxes if available
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

    # Write summary JSON
    summary = RunSummary(
        hef=str(args.hef.resolve()),
        images_dir=str(args.images_dir.resolve()),
        outdir=str(args.outdir.resolve()),
        input_vstream_name=input_info.name,
        input_shape=(in_h, in_w, in_c),
        output_shapes={k: tuple(v) for k, v in out_shapes.items()},
        count=len(chosen),
        interface=args.interface,
        input_float=bool(args.input_float),
        output_float=bool(args.output_float),
        normalize=bool(args.normalize),
        bgr=bool(args.bgr),
        images=[str(p.resolve()) for p in chosen],
    )
    with open(args.outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    dt = time.time() - t_start
    print("[OK] Inference complete.")
    print(f"     Saved tensors: {tensors_dir}")
    print(f"     Saved annots:  {ann_dir}")
    print(f"     Summary:       {args.outdir / 'summary.json'}")
    print(f"     Elapsed:       {dt:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
