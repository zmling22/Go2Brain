#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time

class VideoRecorder(Node):
    def __init__(self):
        super().__init__('video_recorder')

        self.declare_parameter('topic', '/camera/camera/color/image_raw')
        self.declare_parameter('output', 'camera.mp4')
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('fourcc', 'mp4v')  # 若打不开可改成 XVID 并输出 .avi

        self.topic = self.get_parameter('topic').value
        self.output = self.get_parameter('output').value
        self.fps = float(self.get_parameter('fps').value)
        self.fourcc = self.get_parameter('fourcc').value

        self.bridge = CvBridge()
        self.writer = None
        self.last_log = time.time()

        self.sub = self.create_subscription(Image, self.topic, self.cb, 10)
        self.get_logger().info(f"Subscribing: {self.topic}")
        self.get_logger().info(f"Writing to: {self.output}, fps={self.fps}, fourcc={self.fourcc}")

    def _init_writer(self, frame):
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        self.writer = cv2.VideoWriter(self.output, fourcc, self.fps, (w, h))
        if not self.writer.isOpened():
            raise RuntimeError("VideoWriter open failed. Try fourcc:=XVID with output .avi, or install extra codecs.")
        self.get_logger().info(f"Video opened: {w}x{h}")

    def cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if self.writer is None:
            self._init_writer(frame)

        self.writer.write(frame)

        now = time.time()
        if now - self.last_log > 2.0:
            self.get_logger().info("Recording...")
            self.last_log = now

    def destroy_node(self):
        if self.writer is not None:
            self.writer.release()
        super().destroy_node()

def main():
    rclpy.init()
    node = VideoRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

# python ros2_video_recorder.py --ros-args -p topic:=/camera/camera/color/image_raw -p output:=go2_color.avi -p fourcc:=XVID -p fps:=30.0