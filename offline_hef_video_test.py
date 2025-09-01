#!/usr/bin/env python3

"""
Offline HEF-on-MP4 smoke test for Hailo on RPi5.

Reads resources/face_recognition.mp4, runs resources/best.hef via GStreamer
Hailo elements, overlays detections using your custom labels, and writes:
 - an annotated AVI (MJPEG) to outputs/best_annotated.avi
 - a JSON summary to outputs/best_summary.json

This does not require ROS 2 and is safe for headless runs.
"""

import argparse
import json
import os
import signal
import sys
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # type: ignore

import cv2  # type: ignore
import numpy as np  # type: ignore
import hailo  # type: ignore


# Default custom labels provided by user
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


class OfflineHefVideoTest:
    def __init__(
        self,
        input_path: str,
        hef_path: str,
        output_dir: str,
        width: int = 640,
        height: int = 640,
        labels: Optional[List[str]] = None,
        nms_score: float = 0.3,
        nms_iou: float = 0.45,
        labels_mode: str = "custom",
        plugin_nms: bool = False,
        normalize_input: bool = False,
    ) -> None:
        self.input_path = input_path
        self.hef_path = hef_path
        self.output_dir = output_dir
        self.width = int(width)
        self.height = int(height)
        self.labels_mode = labels_mode  # "custom" or "standard"
        self.prefer_plugin_labels = self.labels_mode == "standard"
        self.labels = (labels or CUSTOM_LABELS) if self.labels_mode == "custom" else []
        self.nms_score = float(nms_score)
        self.nms_iou = float(nms_iou)
        # Only set hailonet NMS properties when explicitly requested and
        # when using plugin-provided postprocess. Many custom HEFs do not
        # include NMS outputs; passing these props causes CHECK_SUCCESS=6.
        self.plugin_nms = bool(plugin_nms and self.prefer_plugin_labels)
        # Debug knob: normalize frames to [0,1] float32 before hailonet
        self.normalize_input = bool(normalize_input)

        self.pipeline: Optional[Gst.Element] = None
        self.mainloop: Optional[GLib.MainLoop] = None
        self.mainloop_thread: Optional[threading.Thread] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.writer_fps: float = 30.0
        self.summary = Summary()
        # Image mode detection & state
        self.is_image: bool = self._looks_like_image(self.input_path)
        self.saved_image: bool = False
        self.image_out_path: str = os.path.join(self.output_dir, "best_annotated.jpg")
        self.orig_image_bgr: Optional[np.ndarray] = None
        self._stop_scheduled: bool = False

    # --------------- Pipeline setup ---------------
    def build_pipeline(self) -> None:
        Gst.init(None)

        # Try decodebin to handle MP4 variants; we convert to RGB and letterbox to 640x640
        # When using "standard" labels, we omit the hailofilter config-path so the plugin
        # provides its default label set (e.g., COCO for YOLO models).
        # Configure postprocess and optional config JSON
        cfg_line = (
            ""
            if self.prefer_plugin_labels
            else "config-path=resources/labels_custom.json"
        )
        # Optional NMS properties (only when explicitly requested and in standard mode)
        nms_props = (
            f"nms-score-threshold={self.nms_score} nms-iou-threshold={self.nms_iou} "
            if self.plugin_nms
            else ""
        )
        freeze = "imagefreeze !" if self.is_image else ""

        # When normalize_input is enabled, convert frames to RGBF32 and divide by 255
        # in a pad-probe (pre_norm_cb) before hailonet. Also ask hailonet to accept
        # float32 input so HailoRT quantizes according to calibration ranges.
        if self.normalize_input:
            pre_path = f"""
                videoconvert !
                video/x-raw,format=RGB !
                videoscale method=0 add-borders=true !
                video/x-raw,width={self.width},height={self.height},format=RGB !
                videoconvert !
                video/x-raw,format=RGBF32 !
                queue max-size-buffers=2 leaky=downstream !
                identity name=pre_norm_cb !
            """
            hailonet_extra = "input-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        else:
            pre_path = f"""
                videoconvert !
                videoscale method=0 add-borders=true !
                video/x-raw,format=RGB,width={self.width},height={self.height} !
                queue max-size-buffers=2 leaky=downstream !
            """
            hailonet_extra = ""

        # Choose postprocess library + function
        # Standard: TAPPAS YOLO postprocess on device tensors
        # Custom: keep legacy filter_letterbox path
        if self.prefer_plugin_labels:
            post_so_candidates = [
                "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so",
                "/usr/lib/x86_64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so",
                "/hailo-apps-infra/resources/libyolo_hailortpp_postprocess.so",
            ]
            post_fn = "yolov8"
        else:
            post_so_candidates = [
                "/hailo-apps-infra/resources/libyolo_hailortpp_postprocess.so",
                "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so",
                "/usr/lib/x86_64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so",
            ]
            post_fn = "filter_letterbox"

        # Pick the first existing .so path
        post_so = None
        for cand in post_so_candidates:
            try:
                if os.path.isfile(cand):
                    post_so = cand
                    break
            except Exception:
                continue
        if post_so is None:
            # Fallback to first candidate; pipeline creation may fail, but error will be clear
            post_so = post_so_candidates[0]

        # Try to build pipeline with a list of function-name candidates
        fn_candidates = (
            ["yolov8", "yolov5", "yolo", "yolov5_letterbox", "yolov8_letterbox"]
            if self.prefer_plugin_labels
            else [post_fn]
        )
        last_error: Optional[Exception] = None
        for fn in fn_candidates:
            pipeline_desc = f"""
                filesrc location={self.input_path} !
                decodebin !
                {freeze}
                {pre_path}
                hailonet hef-path={self.hef_path}
                         scheduling-algorithm=1
                         vdevice_group_id=1
                         batch-size=1
                         {hailonet_extra}
                         {nms_props}output-format-type=HAILO_FORMAT_TYPE_FLOAT32 !
                queue max-size-buffers=2 leaky=downstream !
                hailofilter so-path={post_so}
                            {cfg_line}
                            function-name={fn} !
                queue max-size-buffers=2 leaky=downstream !
                identity name=identity_cb !
                fakesink sync=false
            """
            try:
                self.pipeline = Gst.parse_launch(pipeline_desc)
                print(f"[INFO] Using postprocess function-name='{fn}' from '{post_so}'")
                last_error = None
                break
            except Exception as e:  # pylint: disable=broad-except
                last_error = e
                continue
        if self.pipeline is None:
            print(
                "[ERROR] Failed to create pipeline with available yolov* postprocess functions."
            )
            if last_error is not None:
                print(f"[HINT] Last error: {last_error}")
            print(
                "[HINT] Ensure TAPPAS post-process library is installed and exports yolov8/yolov5."
            )
            print(
                "[HINT] On RPi: sudo apt install hailo-tappas-post-processes or update to TAPPAS >= 4.28."
            )
            sys.exit(1)

        identity = self.pipeline.get_by_name("identity_cb")
        if identity is None:
            print("[ERROR] identity element not found in pipeline")
            sys.exit(1)

        srcpad = identity.get_static_pad("src")
        if srcpad is None:
            print("[ERROR] Could not get src pad from identity")
            sys.exit(1)

        srcpad.add_probe(Gst.PadProbeType.BUFFER, self.on_buffer)

        # Attach pre-normalization probe if requested
        if self.normalize_input:
            pre_identity = self.pipeline.get_by_name("pre_norm_cb")
            if pre_identity is None:
                print(
                    "[ERROR] pre_norm_cb element not found in pipeline (normalize_input)"
                )
                sys.exit(1)
            pre_srcpad = pre_identity.get_static_pad("src")
            if pre_srcpad is None:
                print("[ERROR] Could not get src pad from pre_norm_cb")
                sys.exit(1)
            pre_srcpad.add_probe(Gst.PadProbeType.BUFFER, self._pre_norm_buffer)

        # Handle bus messages
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)

    # --------------- Probing & drawing ---------------
    def _ensure_writer(self, caps: Gst.Caps) -> None:
        if self.writer is not None:
            return

        # In single-image mode we do not open a VideoWriter.
        if self.is_image:
            return

        structure = caps.get_structure(0)
        w = int(structure.get_value("width"))
        h = int(structure.get_value("height"))

        # FPS if provided by caps (be tolerant to ranges/lists)
        if structure.has_field("framerate"):
            # Try the typed getter first; on most builds returns (num, denom)
            try:
                num, denom = structure.get_fraction("framerate")
                if denom:
                    self.writer_fps = float(num) / float(denom)
            except Exception:
                # Fall back to duck-typing of get_value output which may be
                # a fraction-like object or simple tuple; ignore ranges/lists.
                try:
                    val = structure.get_value("framerate")
                    num = getattr(val, "numerator", getattr(val, "num", None))
                    den = getattr(val, "denominator", getattr(val, "denom", None))
                    if num is None and isinstance(val, tuple) and len(val) == 2:
                        num, den = val
                    if num is not None and den:
                        self.writer_fps = float(num) / float(den)
                except Exception:
                    pass  # keep default

        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, "best_annotated.avi")

        # MJPEG-in-AVI works on default GStreamer installs without extra codecs
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.writer = cv2.VideoWriter(out_path, fourcc, self.writer_fps, (w, h))
        if not self.writer.isOpened():
            print("[ERROR] Failed to open VideoWriter for output.")
            sys.exit(1)

        print(
            f"[INFO] Writing annotated video to: {out_path} @ {self.writer_fps:.2f} FPS"
        )

    def _label_from_detection(self, det: hailo.HailoObject) -> str:
        # Try to derive label from class id if available, fallback to det.get_label()
        if self.prefer_plugin_labels:
            try:
                return det.get_label()  # type: ignore[attr-defined]
            except Exception:
                return "unknown"

        label: Optional[str] = None
        class_id: Optional[int] = None
        # Many Hailo detection objects expose get_label_id(); we guard to be safe
        for attr in ("get_label_id", "get_class_id", "get_category_id"):
            try:
                method = getattr(det, attr)
                cid = method()  # type: ignore[call-arg]
                if isinstance(cid, (int, np.integer)):
                    class_id = int(cid)
                    break
            except Exception:
                continue

        if class_id is not None and 0 <= class_id < len(self.labels):
            label = self.labels[class_id]
        else:
            try:
                label = det.get_label()  # type: ignore[attr-defined]
            except Exception:
                label = "unknown"

        return label

    def on_buffer(self, pad: Gst.Pad, info: Gst.PadProbeInfo):  # type: ignore[override]
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK

        # Get detections from Hailo metadata
        try:
            roi = hailo.get_roi_from_buffer(buf)
            detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
            try:
                print(f"[DBG offline] has_tensors={roi.has_tensors()} tensors={len(roi.get_tensors())}")
            except Exception:
                pass
        except Exception as e:  # pylint: disable=broad-except
            print(f"[WARN] Could not get Hailo detections: {e}")
            detections = []

        # Prepare frame extraction
        caps = pad.get_current_caps()
        if caps is not None:
            # Only needed for video path; image path doesn't use writer
            self._ensure_writer(caps)
            try:
                structure = caps.get_structure(0)
                w = int(structure.get_value("width"))
                h = int(structure.get_value("height"))
            except Exception:
                w, h = self.width, self.height
        else:
            # Some sources (single images) may not expose caps on first/only buffer
            # Fall back to the configured network size (post-videoscale)
            w, h = self.width, self.height

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.PadProbeReturn.OK

        try:
            frame_rgb = np.frombuffer(map_info.data, dtype=np.uint8).reshape((h, w, 3))
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Draw detections (Hailo bbox coordinates are normalized [0,1])
            for det in detections:
                try:
                    bbox = det.get_bbox()
                    # Normalize to pixel coordinates
                    bxmin = float(bbox.xmin())
                    bymin = float(bbox.ymin())
                    if hasattr(bbox, "xmax") and hasattr(bbox, "ymax"):
                        bxmax = float(bbox.xmax())
                        bymax = float(bbox.ymax())
                    else:
                        bxmax = bxmin + float(bbox.width())
                        bymax = bymin + float(bbox.height())

                    # Scale to the current frame size
                    x_min = int(np.clip(round(bxmin * w), 0, w - 1))
                    y_min = int(np.clip(round(bymin * h), 0, h - 1))
                    x_max = int(np.clip(round(bxmax * w), 0, w - 1))
                    y_max = int(np.clip(round(bymax * h), 0, h - 1))

                    # Skip degenerate boxes
                    if x_max <= x_min or y_max <= y_min:
                        continue

                    conf = (
                        float(det.get_confidence())
                        if hasattr(det, "get_confidence")
                        else 0.0
                    )
                    label = self._label_from_detection(det)

                    cv2.rectangle(
                        frame_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame_bgr,
                        f"{label}:{conf:.2f}",
                        (x_min, max(0, y_min - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                    self.summary.bump(label)
                except Exception:
                    # Be robust to any single detection failure
                    continue

            # Save per input type
            if self.writer is not None:
                # Video path: stream frames to writer
                self.writer.write(frame_bgr)
            elif self.is_image and not self.saved_image:
                # Image path: save a single annotated JPEG
                os.makedirs(self.output_dir, exist_ok=True)
                try:
                    # If we have the original image, remove letterbox and resize back
                    save_frame = frame_bgr
                    if self.orig_image_bgr is not None:
                        src_h, src_w = self.orig_image_bgr.shape[:2]
                        # Compute letterbox mapping parameters used by videoscale add-borders=true
                        scale = min(w / float(src_w), h / float(src_h))
                        new_w = int(round(scale * src_w))
                        new_h = int(round(scale * src_h))
                        pad_x = int(round((w - new_w) / 2.0))
                        pad_y = int(round((h - new_h) / 2.0))
                        x0 = max(0, min(pad_x, w - 1))
                        y0 = max(0, min(pad_y, h - 1))
                        x1 = max(x0 + 1, min(x0 + new_w, w))
                        y1 = max(y0 + 1, min(y0 + new_h, h))
                        roi = frame_bgr[y0:y1, x0:x1]
                        try:
                            save_frame = cv2.resize(
                                roi, (src_w, src_h), interpolation=cv2.INTER_LINEAR
                            )
                        except Exception:
                            save_frame = frame_bgr
                    cv2.imwrite(self.image_out_path, save_frame)
                    print(f"[INFO] Wrote annotated image to: {self.image_out_path}")
                    self.saved_image = True
                    # Schedule stop from the main loop to avoid state changes
                    # from the streaming thread (prevents GStreamer warnings).
                    if not self._stop_scheduled:
                        self._stop_scheduled = True
                        try:
                            GLib.idle_add(self._idle_stop)
                        except Exception:
                            # Fallback: post EOS, handled by bus message callback
                            try:
                                self.pipeline.send_event(Gst.Event.new_eos())  # type: ignore[union-attr]
                            except Exception:
                                pass
                except Exception as e:  # pylint: disable=broad-except
                    print(f"[ERROR] Failed to save annotated image: {e}")

            # Count frames processed regardless of output mode
            self.summary.frames_processed += 1
        finally:
            buf.unmap(map_info)

        return Gst.PadProbeReturn.OK

    # --------------- Bus / lifecycle ---------------
    def on_bus_message(self, bus: Gst.Bus, message: Gst.Message):  # type: ignore[override]
        mtype = message.type
        if mtype == Gst.MessageType.EOS:
            print("[INFO] EOS received.")
            self.stop()
        elif mtype == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[ERROR] {err} | debug: {debug}")
            self.stop()
        return True

    def _pre_norm_buffer(self, pad: Gst.Pad, info: Gst.PadProbeInfo):  # type: ignore[override]
        """Divide RGBF32 pixels by 255 in-place before hailonet (debug-only)."""
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
        success, map_info = buf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE)
        if not success:
            return Gst.PadProbeReturn.OK
        try:
            caps = pad.get_current_caps()
            if caps is not None:
                s = caps.get_structure(0)
                w = int(s.get_value("width"))
                h = int(s.get_value("height"))
            else:
                w, h = self.width, self.height
            # Expect RGBF32 interleaved; divide in-place
            arr = np.frombuffer(map_info.data, dtype=np.float32)
            if arr.size != w * h * 3:
                return Gst.PadProbeReturn.OK
            arr /= 255.0
        except Exception:
            pass
        finally:
            buf.unmap(map_info)
        return Gst.PadProbeReturn.OK

    def run(self) -> None:
        if not os.path.isfile(self.input_path):
            print(f"[ERROR] Input not found: {self.input_path}")
            sys.exit(1)
        if not os.path.isfile(self.hef_path):
            print(f"[ERROR] HEF not found: {self.hef_path}")
            sys.exit(1)

        self.build_pipeline()
        assert self.pipeline is not None

        self.pipeline.set_state(Gst.State.PLAYING)
        self.mainloop = GLib.MainLoop()

        if self.is_image:
            print("[INFO] Detected image input; will save an annotated JPG.")
            # Load original image to preserve native resolution for annotation
            try:
                self.orig_image_bgr = cv2.imread(self.input_path, cv2.IMREAD_COLOR)
                if self.orig_image_bgr is None:
                    print(
                        f"[WARN] Could not read original image '{self.input_path}'. Will annotate resized frame."
                    )
            except Exception as e:
                print(f"[WARN] Failed to load original image: {e}")

        # Allow Ctrl+C to stop the mainloop gracefully
        def _sigint_handler(signum, frame):  # noqa: ARG001
            print("[INFO] SIGINT received, stopping...")
            self.stop()

        signal.signal(signal.SIGINT, _sigint_handler)

        self.mainloop_thread = threading.Thread(target=self.mainloop.run, daemon=True)
        self.mainloop_thread.start()

        # Block until mainloop quits
        self.mainloop_thread.join()

        # Write summary JSON
        os.makedirs(self.output_dir, exist_ok=True)
        summary_path = os.path.join(self.output_dir, "best_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "frames_processed": self.summary.frames_processed,
                    "label_counts": self.summary.label_counts,
                },
                f,
                indent=2,
            )
        print(f"[INFO] Wrote summary: {summary_path}")

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
        if self.mainloop is not None:
            try:
                self.mainloop.quit()
            except Exception:
                pass
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass

    # Ensure state changes happen on the main loop thread
    def _idle_stop(self) -> bool:
        try:
            self.stop()
        finally:
            # Returning False removes this idle source
            return False

    @staticmethod
    def _looks_like_image(path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline HEF-on-video or single-image test using Hailo GStreamer"
    )
    p.add_argument(
        "--input",
        default="resources/face_recognition.mp4",
        help="Input media path (MP4 video or single image: jpg/png/bmp)",
    )
    p.add_argument("--hef", default="resources/best.hef", help="HEF model path")
    p.add_argument("--outdir", default="outputs", help="Directory to write outputs")
    p.add_argument("--width", type=int, default=640, help="Resize width for inference")
    p.add_argument(
        "--height", type=int, default=640, help="Resize height for inference"
    )
    p.add_argument("--nms-score", type=float, default=0.3, help="NMS score threshold")
    p.add_argument("--nms-iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument(
        "--plugin-nms",
        action="store_true",
        help="Enable hailonet NMS properties (only for HEFs with on-chip NMS).",
    )
    p.add_argument(
        "--labels-json",
        default=None,
        help="Optional JSON file with label list (overrides built-in list)",
    )
    p.add_argument(
        "--labels-mode",
        choices=["custom", "standard"],
        default="custom",
        help="Use 'standard' to rely on plugin-provided labels (e.g., COCO); 'custom' uses built-in or --labels-json.",
    )
    p.add_argument(
        "--normalize-input",
        action="store_true",
        help="Debug: convert frames to RGBF32 and divide by 255 before hailonet; also set hailonet input-format-type=FLOAT32.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    labels: Optional[List[str]] = None
    if args.labels_mode == "custom" and args.labels_json:
        try:
            with open(args.labels_json, "r", encoding="utf-8") as f:
                labels = json.load(f)
            if not isinstance(labels, list) or not all(
                isinstance(x, str) for x in labels
            ):
                print(
                    "[WARN] labels-json must be a list of strings; falling back to defaults"
                )
                labels = None
        except Exception as e:  # pylint: disable=broad-except
            print(f"[WARN] Could not load labels JSON: {e}; using defaults")

    runner = OfflineHefVideoTest(
        input_path=args.input,
        hef_path=args.hef,
        output_dir=args.outdir,
        width=args.width,
        height=args.height,
        labels=labels,
        nms_score=args.nms_score,
        nms_iou=args.nms_iou,
        labels_mode=args.labels_mode,
        plugin_nms=args.plugin_nms,
        normalize_input=args.normalize_input,
    )
    runner.run()


if __name__ == "__main__":
    main()
