"""
Dashboard server -- entry point.

Creates the ROS2 node, assigns it to the routes module, starts Flask,
and (optionally) initialises the Unitree Go2 AudioHubClient for voice alerts.
"""
import threading

import rclpy

from ros.bridge import DashboardBridge
import web.routes as routes
from audio_sdk import sdk as audio_sdk


rclpy.init()
routes.bridge = DashboardBridge()

# Initialise Go2 audio (best-effort, in background to avoid blocking ROS)
audio_sdk.init_audio_client_async()


def ros_spin_thread() -> None:
    rclpy.spin(routes.bridge)


threading.Thread(target=ros_spin_thread, daemon=True).start()

if __name__ == "__main__":
    routes.app.run(host="0.0.0.0", port=5001, debug=False)
