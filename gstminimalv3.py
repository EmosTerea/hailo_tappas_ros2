#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import hailo

from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import (
    Detection2DArray,
    Detection2D,
    ObjectHypothesisWithPose,
    BoundingBox2D,
)

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import threading
import cv2
import numpy as np
import signal
import sys
from hailo_apps_infra.hailo_rpi_common import get_caps_from_pad, get_numpy_from_buffer


class RosCameraNode(Node):
    def __init__(self):
        super().__init__("RosCameraNode")

        # Use this to store the frame_id from the first image message we receive.
        self._camera_frame_id = None

        # Create the GStreamer pipeline (with appsrc)
        self.pipeline = None
        self.appsrc = None
        self._create_gst_pipeline()

        # Start the GStreamer main loop in a dedicated thread
        self._start_gst_mainloop()

        # Subscribe to the ROS camera topic
        self.subscription = self.create_subscription(
            CompressedImage, "/camera/compressed", self.image_callback, 10
        )

        self.detections_pub = self.create_publisher(
            Detection2DArray, "object_detections_2d", 10
        )
        self.image_bbx_pub = self.create_publisher(
            CompressedImage, "image_bbx/compressed", 10
        )

        self.get_logger().info("Node initialized.")

    def _create_gst_pipeline(self):
        """Builds the GStreamer pipeline string with appsrc and sets up the bus handling."""
        Gst.init(None)

        pipeline_description = f"""
appsrc name=app_source is-live=true format=3 do-timestamp=true block=false !
image/jpeg, width=1920, height=1080, framerate=30/1 !
queue name=q1 max-size-buffers=2 leaky=downstream max-size-bytes=0 max-size-time=0 !
jpegdec !
queue name=q2 max-size-buffers=2 leaky=downstream max-size-bytes=0 max-size-time=0 !
videoconvert !
videoscale method=0 add-borders=true !
video/x-raw,format=RGB,width=640,height=640,framerate=30/1 !
queue name=q3 max-size-buffers=2 leaky=downstream max-size-bytes=0 max-size-time=0 !
hailonet hef-path=resources/yolov8m.hef
    scheduling-algorithm=1
    vdevice_group_id=1
    batch-size=1
    nms-score-threshold=0.3
    nms-iou-threshold=0.45
    output-format-type=HAILO_FORMAT_TYPE_FLOAT32 !
queue name=q4 max-size-buffers=2 leaky=downstream max-size-bytes=0 max-size-time=0 !
hailofilter so-path=/hailo-apps-infra/resources/libyolo_hailortpp_postprocess.so
    function-name=filter_letterbox !
queue name=q5 max-size-buffers=2 leaky=downstream max-size-bytes=0 max-size-time=0 !
hailooverlay name=hailo_overlay
    show-confidence=false
    line-thickness=3
    font-thickness=2 !
queue name=q6 max-size-buffers=2 leaky=downstream max-size-bytes=0 max-size-time=0 !
identity name=identity_callback !
fakevideosink sync=false
"""

        try:
            self.pipeline = Gst.parse_launch(pipeline_description)
        except Gst.Error as e:
            self.get_logger().error(f"Error creating pipeline: {e}")
            sys.exit(1)

        # Retrieve the appsrc element so we can push buffers from the subscription callback
        self.appsrc = self.pipeline.get_by_name("app_source")
        if not self.appsrc:
            self.get_logger().error("Could not find appsrc in pipeline.")
            sys.exit(1)

        identity = self.pipeline.get_by_name("identity_callback")
        if identity:
            srcpad = identity.get_static_pad("src")
            srcpad.add_probe(Gst.PadProbeType.BUFFER, self._my_detection_callback)
            self.get_logger().info("identity created")

        # Setup the bus to handle messages (errors, EOS, etc.)
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_gst_message)

    def _my_detection_callback(self, pad, info):
        """Pad probe callback to read Hailo detection metadata & frames."""
        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK

        # 1. Retrieve Hailo metadata (detections)
        roi = hailo.get_roi_from_buffer(buffer)
        hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        # 2. Convert hailo_detections -> vision_msgs/Detection2DArray
        detection_msg = self._map_hailo_detections(hailo_detections)

        # Use the stored camera frame_id if available, otherwise a fallback
        detection_msg.header.frame_id = self._camera_frame_id or "camera_frame"

        # Retrieve the caps and read out the image data
        caps = pad.get_current_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            self.get_logger().warn("Failed to map GstBuffer.")
            return Gst.PadProbeReturn.OK

        frame_data = map_info.data  # raw bytes
        frame_np = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))
        buffer.unmap(map_info)

        # Convert from RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        # 3. Optionally draw bounding boxes on frame_bgr
        for det in hailo_detections:
            bbox = det.get_bbox()
            label = det.get_label()
            conf = det.get_confidence()

            x_min = int(bbox.xmin())
            y_min = int(bbox.ymin())
            w = int(bbox.width())
            h = int(bbox.height())
            x_max = x_min + w
            y_max = y_min + h

            cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                frame_bgr,
                f"{label}:{conf:.2f}",
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        # 4. Publish detection array
        self.detections_pub.publish(detection_msg)

        # 5. Encode the BGR image as JPEG, publish CompressedImage
        success, encoded = cv2.imencode(".jpg", frame_bgr)
        if success:
            compressed_msg = CompressedImage()
            compressed_msg.header.stamp = self.get_clock().now().to_msg()
            compressed_msg.header.frame_id = self._camera_frame_id or "camera_frame"
            compressed_msg.format = "jpeg"
            compressed_msg.data = encoded.tobytes()
            self.image_bbx_pub.publish(compressed_msg)

        return Gst.PadProbeReturn.OK

    def _map_hailo_detections(self, hailo_detections):
        """Convert a list of hailo.HAILO_DETECTION to a Detection2DArray message."""
        detection_array_msg = Detection2DArray()
        detection_array_msg.header.stamp = self.get_clock().now().to_msg()
        # We'll set frame_id outside (in _my_detection_callback) since we need
        # the stored frame ID from the camera.

        for det in hailo_detections:
            label = det.get_label()
            conf = float(det.get_confidence())
            bbox = det.get_bbox()

            # Create a Detection2D
            detection_msg = Detection2D()
            detection_msg.header = detection_array_msg.header

            # bounding box center and size
            center_x = bbox.xmin() + bbox.width() / 2.0
            center_y = bbox.ymin() + bbox.height() / 2.0

            detection_msg.bbox.center.position.x = center_x * 1920
            detection_msg.bbox.center.position.y = center_y * 1080
            detection_msg.bbox.size_x = bbox.width() * 1920
            detection_msg.bbox.size_y = bbox.height() * 1080

            # Add the ObjectHypothesisWithPose
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = label  # string label
            hypothesis.hypothesis.score = conf
            detection_msg.results.append(hypothesis)

            # (Optional) If there's a track ID from HAILO_UNIQUE_ID:
            track_ids = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if track_ids:
                detection_msg.id = str(track_ids[0].get_id())

            # Append to array
            detection_array_msg.detections.append(detection_msg)

        return detection_array_msg

    def _on_gst_message(self, bus, message):
        """Called whenever a message arrives on the GStreamer bus."""
        msg_type = message.type

        if msg_type == Gst.MessageType.EOS:
            self.get_logger().info("GStreamer: End-of-stream reached.")
            self.pipeline.set_state(Gst.State.READY)

        elif msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.get_logger().error(f"GStreamer Error: {err}, debug={debug}")
            self.shutdown_pipeline()

        return True

    def _start_gst_mainloop(self):
        """Starts the GLib main loop in a separate thread, so GStreamer can process the pipeline."""
        self.pipeline.set_state(Gst.State.PAUSED)
        self.pipeline.set_state(Gst.State.PLAYING)

        self.mainloop = GLib.MainLoop()
        self.mainloop_thread = threading.Thread(target=self.mainloop.run, daemon=True)
        self.mainloop_thread.start()
        self.get_logger().info("Started GStreamer main loop in a background thread.")

    def shutdown_pipeline(self):
        """Gracefully shut down the GStreamer pipeline."""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if hasattr(self, "mainloop") and self.mainloop is not None:
            self.mainloop.quit()
        self.get_logger().info("GStreamer pipeline shut down.")

    def image_callback(self, msg: CompressedImage):
        """ROS subscription callback for camera topic (JPEG-compressed)."""
        # Save the frame_id from this topic once (assuming it never changes).
        if self._camera_frame_id is None:
            self._camera_frame_id = msg.header.frame_id
            self.get_logger().info(f"Storing camera frame_id: {self._camera_frame_id}")

        if not self.appsrc:
            self.get_logger().warn("appsrc not initialized yet")
            return

        # Create a new Gst.Buffer with the size of compressed data
        buffer = Gst.Buffer.new_allocate(None, len(msg.data), None)
        buffer.fill(0, msg.data)

        ret = self.appsrc.emit("push-buffer", buffer)

        if ret != Gst.FlowReturn.OK:
            self.get_logger().error(f"push-buffer failed: {ret}")

    def destroy_node(self):
        self.shutdown_pipeline()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = RosCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
