#!/usr/bin/env python3
"""
Real-time object detection node using YOLOv8 (ultralytics).
Subscribes to camera image, runs inference, publishes annotated image + stats.
"""

# Fix for Jetson: preload OpenMP TLS before torch/ultralytics imports
import ctypes
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
try:
    ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass

import json
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')

        # --- parameters ---
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('model_path', '')
        self.declare_parameter('enable', True)
        # Process every N frames (1 = every frame, 2 = every other, etc.)
        self.declare_parameter('inference_skip', 1)
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')

        self._conf = self.get_parameter('conf_threshold').value
        self._model_path = self.get_parameter('model_path').value
        self._enable = self.get_parameter('enable').value
        self._skip = self.get_parameter('inference_skip').value
        image_topic = self.get_parameter('image_topic').value

        self._bridge = CvBridge()
        self._frame_counter = 0
        self._lock = threading.Lock()

        # --- load model ---
        self._model = None
        self._load_model()

        # --- publishers ---
        self._annotated_pub = self.create_publisher(
            Image, '/camera/detection/annotated', 10)
        self._stats_pub = self.create_publisher(
            String, '/camera/detection/stats', 10)

        # --- subscriptions ---
        self._img_sub = self.create_subscription(
            Image, image_topic, self._img_cb, 10)

        self.get_logger().info(
            f'DetectionNode started — conf={self._conf}, skip={self._skip}, '
            f'enable={self._enable}, topic={image_topic}')

    def _load_model(self):
        try:
            from ultralytics import YOLO
            path = self._model_path if self._model_path else 'yolov8n.pt'
            self._model = YOLO(path)
            self.get_logger().info(f'YOLO model loaded: {path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {e}')
            self._model = None

    def _img_cb(self, msg: Image):
        """Main callback: convert, optionally run inference, publish results."""
        # --- skip frames for performance ---
        self._frame_counter += 1
        if self._frame_counter % self._skip != 0:
            return

        # --- convert ROS Image → OpenCV ---
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv bridge error: {e}')
            return

        annotated = frame.copy()
        stats = {'objects': [], 'count': 0, 'enabled': self._enable}

        # --- run inference if enabled ---
        if self._enable and self._model is not None:
            # Force ultralytics to use pure-PyTorch NMS (TorchNMS.nms)
            # instead of torchvision.ops.nms which has incompatible C++ ops on Jetson
            import sys
            sys.modules.pop('torchvision', None)
            try:
                results = self._model.predict(
                    frame,
                    conf=self._conf,
                    verbose=False,
                    device='cuda:0' if self._cuda_available() else 'cpu',
                )

                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    # draw bounding boxes
                    annotated = results[0].plot()

                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i])
                        label = results[0].names[cls_id]
                        conf = float(boxes.conf[i])
                        xyxy = boxes.xyxy[i].tolist()
                        stats['objects'].append({
                            'label': label,
                            'confidence': round(conf, 3),
                            'bbox': [round(v, 2) for v in xyxy],
                        })
                    stats['count'] = len(stats['objects'])

            except Exception as e:
                self.get_logger().error(f'Inference error: {e}')

        # --- publish annotated image ---
        try:
            annotated_msg = self._bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            annotated_msg.header = msg.header
            self._annotated_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().warn(f'publish annotated image error: {e}')

        # --- publish stats ---
        stats_msg = String()
        stats_msg.data = json.dumps(stats)
        self._stats_pub.publish(stats_msg)

    @staticmethod
    def _cuda_available():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
