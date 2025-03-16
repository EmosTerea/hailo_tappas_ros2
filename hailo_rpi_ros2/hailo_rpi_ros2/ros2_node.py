# Copyright 2025 Stefanos Kyrikakis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# !/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from hailo_rpi_ros2_interfaces.srv import (
    SaveFace,
    DeleteFace,
)
from hailo_rpi_ros2 import face_recognition
from hailo_rpi_ros2.face_gallery import (
    Gallery,
    GalleryAppendStatus,
    GalleryDeletionStatus,
)
import cv2
from rclpy import Parameter
from hailo_rpi_ros2.face_recognition_pipeline import GStreamerFaceRecognitionApp
from threading import Thread
import os


class HailoDetection(Node):
    def __init__(self):
        Node.__init__(self, "hailo_detection")

        self.image_publisher_compressed = self.create_publisher(
            CompressedImage, "/camera/image_raw/compressed", 10
        )
        self.image_publisher_ = self.create_publisher(Image, "/camera/image_raw", 10)

        self.create_service(SaveFace, "~/save_face", self.add_face_callback)

        self.create_service(DeleteFace, "~/delete_face", self.add_face_callback)

        self.declare_parameters(
            namespace="",
            parameters=[
                ("face_recognition.input", Parameter.Type.STRING),
                ("face_recognition.local_gallery_file", Parameter.Type.STRING),
                ("face_recognition.similarity_threshhold", Parameter.Type.DOUBLE),
                ("face_recognition.queue_size", Parameter.Type.INTEGER),
            ],
        )
        self.input = (
            self.get_parameter("face_recognition.input")
            .get_parameter_value()
            .string_value
        )
        self.local_gallery_file = (
            self.get_parameter("face_recognition.local_gallery_file")
            .get_parameter_value()
            .string_value
        )
        self.similarity_threshhold = (
            self.get_parameter("face_recognition.similarity_threshhold")
            .get_parameter_value()
            .double_value
        )
        self.queue_size = (
            self.get_parameter("face_recognition.queue_size")
            .get_parameter_value()
            .integer_value
        )

        gallery_file_path = self._get_absolute_file_path_in_build_dir(
            self.local_gallery_file
        )
        self.gallery = Gallery(
            json_file_path=gallery_file_path,
            similarity_thr=self.similarity_threshhold,
            queue_size=self.queue_size,
        )

        self.face_recognition = face_recognition.FaceRecognition(
            self.gallery, self.frame_callback
        )

        gstreamer_app = GStreamerFaceRecognitionApp(
            self.input, self.face_recognition.app_callback, self.face_recognition
        )

        self.detection_thread = Thread(target=gstreamer_app.run)
        self.detection_thread.start()

    def _get_absolute_file_path_in_build_dir(self, file: str) -> str:
        # Get the directory of the current Python file
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the absolute file path
        absolute_file_path = os.path.join(current_dir, "resources", file)

        try:
            # Read the file content
            with open(absolute_file_path, "r") as file:
                file.read()

            # Process the file content
            self.get_logger().info(f"File found: {absolute_file_path}")
            return absolute_file_path
        except FileNotFoundError as e:
            self.get_logger().error(f"File not found: {absolute_file_path}")
            raise e

    def add_face_callback(self, request: SaveFace.Request, response: SaveFace.Response):
        self.get_logger().info(f"Incoming request: Add face {request.name}")
        status = self.gallery.append_new_item(request.name, request.append)
        match status:
            case GalleryAppendStatus.SUCCESS:
                response.result = 0
                response.message = "Person added"
            case GalleryAppendStatus.FACE_EXISTS_WITH_IDENTICAL_NAME:
                response.result = 1
                response.message = (
                    "Similar embedding found with identical name. Aborted "
                    "Consider calling with append=true"
                )
            case GalleryAppendStatus.FACE_EXISTS_WITH_DIFFERENT_NAME:
                response.result = 1
                response.message = (
                    "Similar embedding found with different name. Aborted "
                    "Consider calling with append=true"
                )
            case GalleryAppendStatus.NAME_EXISTS_WITH_NON_SIMILAR_FACE:
                response.result = 1
                response.message = (
                    "The name exists but no similar embedding found. Aborted "
                    "Consider calling with append=true"
                )
            case GalleryAppendStatus.MULTIPLE_FACES_FOUND:
                response.success = 2
                response.message = "Error: Multiple faces found"
            case GalleryAppendStatus.NO_FACES_FOUND:
                response.result = 3
                response.message = "Error: No faces found"
            case _:
                response.success = 4
                response.message = "Failed, see the logs for more details"

        return response

    def delete_face_callback(
        self, request: DeleteFace.Request, response: DeleteFace.Response
    ):
        self.get_logger().info(f"Incoming request: Delete face {request.name}")
        status = self.gallery.delete_item_by_name(request.name, True)
        match status:
            case GalleryDeletionStatus.SUCCESS:
                response.result = 0
                response.message = "Face deleted"
            case GalleryDeletionStatus.NOT_FOUND:
                response.result = 1
                response.message = "Name not found"
            case _:
                response.success = 2
                response.message = "Failed, see the logs for more details"

        return response

    def frame_callback(self, frame: cv2.UMat):
        ret, buffer = cv2.imencode(".jpg", frame)
        msg = CompressedImage()
        msg.header.frame_id = "camera_frame"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = buffer.tobytes()

        self.image_publisher_compressed.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    detection = HailoDetection()

    rclpy.spin(detection)
    if hasattr(detection, "detection_thread") and detection.detection_thread.is_alive():
        detection.detection_thread.join()  # Wait for the thread to finish
        print("Detection thread joined.")
    detection.destroy_node()
    rclpy.shutdown()


# Main program logic follows:
if __name__ == "__main__":
    main()
