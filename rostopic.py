import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
from vision_msgs.msg import (
    Detection2DArray,
    Detection2D,
    ObjectHypothesisWithPose,
    BoundingBox2D,
)

import cv2
import numpy as np
import hailo_platform as hpf

from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy


def letterbox_image(image, target_size=(640, 640), color=(114, 114, 114)):
    """Resize and pad the image to match target_size while preserving aspect ratio."""
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Compute scale and resize while preserving aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute padding
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)

    # Add padding
    padded_img = cv2.copyMakeBorder(
        resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded_img, scale, (left, top)


class YoloInferenceNode(Node):
    def __init__(self):
        super().__init__("yolo_inference_node")
        self.get_logger().info("Initializing Hailo...")

        # Debug logs parameter
        self.declare_parameter("debug_logs", True)
        self.debug_logs = self.get_parameter("debug_logs").value

        # Toggle letterboxing parameter
        self.declare_parameter("use_letterbox", False)
        self.use_letterbox = self.get_parameter("use_letterbox").value

        # ---------------------------------------------------------------------
        # (A) Manually create & enter the VDevice context
        # ---------------------------------------------------------------------
        self.device_ctx = hpf.VDevice()
        self.device_ctx.__enter__()  # device stays open until we exit

        # Load the HEF
        self.hef_path = "yolov8m.hef"
        self.hef = hpf.HEF(self.hef_path)

        # Configure the device
        cfg = hpf.ConfigureParams.create_from_hef(
            self.hef, interface=hpf.HailoStreamInterface.PCIe
        )
        self.network_group = self.device_ctx.configure(self.hef, cfg)[0]
        self.network_params = self.network_group.create_params()

        # Get input/output info
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.output_info = self.hef.get_output_vstream_infos()[0]

        # Prepare vstreams
        self.input_vstreams = hpf.InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=hpf.FormatType.UINT8
        )
        self.output_vstreams = hpf.OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
        )

        # ---------------------------------------------------------------------
        # (B) Manually activate the network group
        # ---------------------------------------------------------------------
        self.network_group_ctx = self.network_group.activate(self.network_params)
        self.network_group_ctx.__enter__()

        # ---------------------------------------------------------------------
        # (C) Manually create & enter InferVStreams context
        # ---------------------------------------------------------------------
        self.pipeline_ctx = hpf.InferVStreams(
            self.network_group, self.input_vstreams, self.output_vstreams
        )
        self.pipeline_ctx.__enter__()  # pipeline is now open

        # Retrieve the target shape from input_info
        height, width = self.input_info.shape[:2]

        # Preallocate a buffer for inference input
        self.infer_input = np.zeros((1, height, width, 3), dtype=np.uint8)
        # ---------------------------------------------------------------------
        # (Warm-Up Inference)
        # ---------------------------------------------------------------------
        try:
            self.get_logger().info("Performing a warm-up inference...")
            for i in range(20):
                _ = self.pipeline_ctx.infer({self.input_info.name: self.infer_input})
            self.get_logger().info("Warm-up inference done.")
        except Exception as e:
            self.get_logger().error(f"Warm-up inference error: {e}")

        # ---------------------------------------------------------------------
        # (D) ROS Subscriptions
        # ---------------------------------------------------------------------
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        self.subscription = self.create_subscription(
            CompressedImage, "/camera/compressed", self.on_image, qos_profile
        )
        self.get_logger().info("Subscribed to /camera/compressed")

        # ---------------------------------------------------------------------
        # (E) ROS Publishers
        # ---------------------------------------------------------------------
        # 1) Publish bounding box–drawn frames as CompressedImage
        self.image_publisher_compressed = self.create_publisher(
            CompressedImage, "~/image_bbx/compressed", 10
        )

        # 2) Publish detections as Detection2DArray
        self.detections_publisher = self.create_publisher(
            Detection2DArray, "~/detections", 10
        )

    def on_image(self, msg: CompressedImage):
        """Callback: Process incoming CompressedImage, run inference, log results."""
        # 1) Decode the compressed image (BGR)
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn("Failed to decode CompressedImage.")
            return

        original_frame = frame.copy()  # Keep a copy for drawing later

        # 2) Preprocess (letterbox or direct resize)
        height, width = self.input_info.shape[:2]
        if self.use_letterbox:
            frame_resized, scale, (pad_left, pad_top) = letterbox_image(
                frame, target_size=(width, height)
            )
        else:
            frame_resized = cv2.resize(
                frame, (width, height), interpolation=cv2.INTER_LINEAR
            )
            scale = min(width / frame.shape[1], height / frame.shape[0])
            pad_left, pad_top = 0, 0

        # Convert to RGB if needed by Hailo
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        self.infer_input[0] = frame_rgb

        # 3) Inference using the persistent pipeline
        try:
            results = self.pipeline_ctx.infer({self.input_info.name: self.infer_input})
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
            return

        # 4) Process results
        #    Assume YOLOv8 w/NMS typically has one output key
        output_key = list(results.keys())[0]
        detections_per_class = results[output_key][0]  # shape: [num_classes, N, 5]

        # We'll collect all detections in a single Detection2DArray
        detection_array_msg = Detection2DArray()
        detection_array_msg.header.frame_id = "camera_frame"
        detection_array_msg.header.stamp = self.get_clock().now().to_msg()

        # For drawing bounding boxes
        for class_idx, class_detections in enumerate(detections_per_class):
            if class_detections.shape[0] == 0:
                continue
            for det in class_detections:
                x, y, w, h, conf = det[0], det[1], det[2], det[3], det[4]

                # Scale from [0,1]*640 to letterboxed coords
                x *= 640
                y *= 640
                w *= 640
                h *= 640

                # Remove letterbox
                x -= pad_left
                y -= pad_top

                # Undo the resizing
                x /= scale
                y /= scale
                w /= scale
                h /= scale

                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)

                # Build a Detection2D message
                detection_msg = Detection2D()
                detection_msg.header = detection_array_msg.header

                # Build bounding box in vision_msgs format
                bbox = BoundingBox2D()
                bbox.center.position.x = float(x + w / 2.0)
                bbox.center.position.y = float(y + h / 2.0)
                bbox.size_x = float(w)
                bbox.size_y = float(h)
                detection_msg.bbox = bbox

                # Build a hypothesis message for class and score
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(class_idx)
                hypothesis.hypothesis.score = float(conf)
                detection_msg.results.append(hypothesis)

                # Collect the detection
                detection_array_msg.detections.append(detection_msg)

                # Optionally draw bounding box on original_frame
                color = (0, 255, 0)
                cv2.rectangle(original_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    original_frame,
                    f"Cls{class_idx}:{conf:.2f}",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                if self.debug_logs:
                    self.get_logger().info(
                        f"Class={class_idx}, BBox=[{x}, {y}, {w}, {h}], Conf={conf:.2f}"
                    )

        # 5) Publish the detections
        self.detections_publisher.publish(detection_array_msg)

        # 6) Publish the bounding-box-drawn frame as CompressedImage
        #    Convert BGR → JPG → CompressedImage message
        ret, buffer = cv2.imencode(".jpg", original_frame)
        if ret:
            out_msg = CompressedImage()
            out_msg.header = detection_array_msg.header
            out_msg.format = "jpeg"
            out_msg.data = buffer.tobytes()
            self.image_publisher_compressed.publish(out_msg)
        else:
            self.get_logger().warn("Failed to compress annotated image.")

    def destroy_node(self):
        """Cleanly close the pipeline, network group, and device contexts."""
        # Close the pipeline context
        if hasattr(self, "pipeline_ctx"):
            self.pipeline_ctx.__exit__(None, None, None)

        # Deactivate the network group
        if hasattr(self, "network_group_ctx"):
            self.network_group_ctx.__exit__(None, None, None)

        # Close the VDevice context
        if hasattr(self, "device_ctx"):
            self.device_ctx.__exit__(None, None, None)

        self.get_logger().info("Destroying node...")
        super().destroy_node()
        self.get_logger().info("Destroyed...")


def main(args=None):
    rclpy.init(args=args)
    node = YoloInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
