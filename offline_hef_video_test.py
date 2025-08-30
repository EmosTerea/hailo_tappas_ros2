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
    ) -> None:
        self.input_path = input_path
        self.hef_path = hef_path
        self.output_dir = output_dir
        self.width = int(width)
        self.height = int(height)
        self.labels = labels or CUSTOM_LABELS
        self.nms_score = float(nms_score)
        self.nms_iou = float(nms_iou)

        self.pipeline: Optional[Gst.Element] = None
        self.mainloop: Optional[GLib.MainLoop] = None
        self.mainloop_thread: Optional[threading.Thread] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.writer_fps: float = 30.0
        self.summary = Summary()

    # --------------- Pipeline setup ---------------
    def build_pipeline(self) -> None:
        Gst.init(None)

        # Try decodebin to handle MP4 variants; we convert to RGB and letterbox to 640x640
        pipeline_desc = f"""
            filesrc location={self.input_path} !
            decodebin !
            videoconvert !
            videoscale method=0 add-borders=true !
            video/x-raw,format=RGB,width={self.width},height={self.height} !
            queue max-size-buffers=2 leaky=downstream !
            hailonet hef-path={self.hef_path}
                     scheduling-algorithm=1
                     vdevice_group_id=1
                     batch-size=1
                     nms-score-threshold={self.nms_score}
                     nms-iou-threshold={self.nms_iou}
                     output-format-type=HAILO_FORMAT_TYPE_FLOAT32 !
            queue max-size-buffers=2 leaky=downstream !
            hailofilter so-path=/hailo-apps-infra/resources/libyolo_hailortpp_postprocess.so
                        config-path=resources/labels_custom.json
                        function-name=filter_letterbox !
            queue max-size-buffers=2 leaky=downstream !
            identity name=identity_cb !
            fakesink sync=false
        """

        try:
            self.pipeline = Gst.parse_launch(pipeline_desc)
        except Exception as e:  # pylint: disable=broad-except
            print(f"[ERROR] Failed to create pipeline: {e}")
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

        # Handle bus messages
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)

    # --------------- Probing & drawing ---------------
    def _ensure_writer(self, caps: Gst.Caps) -> None:
        if self.writer is not None:
            return

        structure = caps.get_structure(0)
        w = int(structure.get_value("width"))
        h = int(structure.get_value("height"))

        # FPS if provided by caps
        if structure.has_field("framerate"):
            frac = structure.get_value("framerate")
            try:
                self.writer_fps = float(frac.numerator) / float(frac.denominator)
            except Exception:  # pragma: no cover - defensive
                self.writer_fps = 30.0

        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, "best_annotated.avi")

        # MJPEG-in-AVI works on default GStreamer installs without extra codecs
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.writer = cv2.VideoWriter(out_path, fourcc, self.writer_fps, (w, h))
        if not self.writer.isOpened():
            print("[ERROR] Failed to open VideoWriter for output.")
            sys.exit(1)

        print(f"[INFO] Writing annotated video to: {out_path} @ {self.writer_fps:.2f} FPS")

    def _label_from_detection(self, det: hailo.HailoObject) -> str:
        # Try to derive label from class id if available, fallback to det.get_label()
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
        except Exception as e:  # pylint: disable=broad-except
            print(f"[WARN] Could not get Hailo detections: {e}")
            detections = []

        # Prepare frame extraction
        caps = pad.get_current_caps()
        if caps is None:
            return Gst.PadProbeReturn.OK

        self._ensure_writer(caps)
        structure = caps.get_structure(0)
        w = int(structure.get_value("width"))
        h = int(structure.get_value("height"))

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.PadProbeReturn.OK

        try:
            frame_rgb = np.frombuffer(map_info.data, dtype=np.uint8).reshape((h, w, 3))
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Draw detections
            for det in detections:
                try:
                    bbox = det.get_bbox()
                    x_min = int(bbox.xmin())
                    y_min = int(bbox.ymin())
                    x_max = int(bbox.xmax()) if hasattr(bbox, "xmax") else int(bbox.xmin() + bbox.width())
                    y_max = int(bbox.ymax()) if hasattr(bbox, "ymax") else int(bbox.ymin() + bbox.height())

                    conf = float(det.get_confidence()) if hasattr(det, "get_confidence") else 0.0
                    label = self._label_from_detection(det)

                    cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
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

            if self.writer is not None:
                self.writer.write(frame_bgr)
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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline HEF-on-video test using Hailo GStreamer")
    p.add_argument("--input", default="resources/face_recognition.mp4", help="Input MP4 path")
    p.add_argument("--hef", default="resources/best.hef", help="HEF model path")
    p.add_argument("--outdir", default="outputs", help="Directory to write outputs")
    p.add_argument("--width", type=int, default=640, help="Resize width for inference")
    p.add_argument("--height", type=int, default=640, help="Resize height for inference")
    p.add_argument("--nms-score", type=float, default=0.3, help="NMS score threshold")
    p.add_argument("--nms-iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument(
        "--labels-json",
        default=None,
        help="Optional JSON file with label list (overrides built-in list)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    labels: Optional[List[str]] = None
    if args.labels_json:
        try:
            with open(args.labels_json, "r", encoding="utf-8") as f:
                labels = json.load(f)
            if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
                print("[WARN] labels-json must be a list of strings; falling back to defaults")
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
    )
    runner.run()


if __name__ == "__main__":
    main()
