from flask import Flask, Response, jsonify, render_template, request
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from cv_bridge import CvBridge

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowWaypoints
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rcl_interfaces.msg import Log
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import tf2_ros


app = Flask(__name__, template_folder="templates", static_folder="static")


def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def build_pose_stamped(frame_id: str, x: float, y: float, yaw: float, clock) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = clock.now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = 0.0
    pose.pose.orientation.z = math.sin(yaw / 2.0)
    pose.pose.orientation.w = math.cos(yaw / 2.0)
    return pose


def apply_2d_transform(x: float, y: float, yaw: float, transform) -> Dict[str, float]:
    tf_yaw = quaternion_to_yaw(transform.transform.rotation)
    cos_yaw = math.cos(tf_yaw)
    sin_yaw = math.sin(tf_yaw)
    tx = transform.transform.translation.x
    ty = transform.transform.translation.y

    return {
        "x": tx + cos_yaw * x - sin_yaw * y,
        "y": ty + sin_yaw * x + cos_yaw * y,
        "yaw": yaw + tf_yaw,
    }


@dataclass
class SharedState:
    pose: Optional[Dict[str, float]] = None
    speed: Dict[str, float] = field(default_factory=lambda: {"linear": 0.0, "angular": 0.0})
    trajectory: List[Dict[str, float]] = field(default_factory=list)
    plan: List[Dict[str, float]] = field(default_factory=list)
    map_msg: Optional[OccupancyGrid] = None
    map_seq: int = 0
    last_command: str = ""
    nav_status: str = "idle"
    nav_detail: str = ""
    last_nav_log: str = ""
    current_route: List[Dict[str, float]] = field(default_factory=list)
    current_waypoint_index: int = -1
    route_running: bool = False
    camera_jpeg: Optional[bytes] = None
    camera_stamp: float = 0.0
    camera_format: str = "jpeg"
    updated_at: float = 0.0


state = SharedState()
lock = threading.Lock()
camera_condition = threading.Condition()


class DashboardBridge(Node):
    def __init__(self):
        super().__init__("dashboard_bridge")

        self.command_pub = self.create_publisher(String, "/nl_command", 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_cb, 20)
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_cb, 1)
        self.plan_sub = self.create_subscription(Path, "/plan", self.plan_cb, 10)
        self.rosout_sub = self.create_subscription(Log, "/rosout", self.rosout_cb, 50)
        self.camera_subs = [
            self.create_subscription(
                CompressedImage,
                "/camera/camera/color/image_raw/compressed",
                self.camera_cb,
                10,
            ),
            self.create_subscription(
                CompressedImage,
                "/camera/image_raw/compressed",
                self.camera_cb,
                10,
            ),
            self.create_subscription(
                CompressedImage,
                "/camera/image/raw/compressed",
                self.camera_cb,
                10,
            ),
        ]
        self.raw_camera_sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.raw_camera_cb,
            10,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()

        self.follow_waypoints_client = ActionClient(self, FollowWaypoints, "/FollowWaypoints")
        self.current_goal_handle = None

    def publish_text_command(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.command_pub.publish(msg)
        with lock:
            state.last_command = text
            state.updated_at = time.time()

    def odom_cb(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        twist = msg.twist.twist
        odom_pose = {
            "x": pose.position.x,
            "y": pose.position.y,
            "yaw": quaternion_to_yaw(pose.orientation),
        }
        map_pose = odom_pose

        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                msg.header.frame_id or "odom",
                Time(),
            )
            map_pose = apply_2d_transform(
                odom_pose["x"],
                odom_pose["y"],
                odom_pose["yaw"],
                transform,
            )
        except Exception:
            pass

        with lock:
            state.pose = map_pose
            state.speed = {
                "linear": math.hypot(twist.linear.x, twist.linear.y),
                "angular": twist.angular.z,
            }

            append_point = False
            if not state.trajectory:
                append_point = True
            else:
                last = state.trajectory[-1]
                dx = map_pose["x"] - last["x"]
                dy = map_pose["y"] - last["y"]
                if math.hypot(dx, dy) >= 0.05:
                    append_point = True

            if append_point:
                state.trajectory.append(
                    {
                        "x": map_pose["x"],
                        "y": map_pose["y"],
                        "t": time.time(),
                    }
                )
                if len(state.trajectory) > 4000:
                    state.trajectory = state.trajectory[-4000:]

            state.updated_at = time.time()

    def map_cb(self, msg: OccupancyGrid) -> None:
        with lock:
            state.map_msg = msg
            state.map_seq += 1
            state.updated_at = time.time()

    def plan_cb(self, msg: Path) -> None:
        points = []
        for pose_stamped in msg.poses:
            points.append(
                {
                    "x": pose_stamped.pose.position.x,
                    "y": pose_stamped.pose.position.y,
                }
            )

        with lock:
            state.plan = points
            state.updated_at = time.time()

    def rosout_cb(self, msg: Log) -> None:
        text = msg.msg or ""
        name = msg.name or ""

        with lock:
            if "semantic_navigator" in name or "bt_navigator" in name or "controller_server" in name:
                state.last_nav_log = f"[{name}] {text}"

            if "Navigation succeeded" in text or "导航成功" in text or "Reached the goal" in text:
                state.nav_status = "succeeded"
                state.nav_detail = text
            elif "导航失败" in text or "failed" in text.lower():
                state.nav_status = "failed"
                state.nav_detail = text
            elif "距离目标点剩余" in text:
                state.nav_status = "navigating"
                state.nav_detail = text

            state.updated_at = time.time()

    def camera_cb(self, msg: CompressedImage) -> None:
        fmt = (msg.format or "").lower()
        if "jpeg" in fmt or "jpg" in fmt:
            image_format = "jpeg"
        elif "png" in fmt:
            image_format = "png"
        else:
            return

        with lock:
            state.camera_jpeg = bytes(msg.data)
            state.camera_stamp = time.time()
            state.camera_format = image_format
            state.updated_at = time.time()
        with camera_condition:
            camera_condition.notify_all()

    def raw_camera_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
            if not ok:
                return
        except Exception:
            return

        with lock:
            state.camera_jpeg = jpg.tobytes()
            state.camera_stamp = time.time()
            state.camera_format = "jpeg"
            state.updated_at = time.time()
        with camera_condition:
            camera_condition.notify_all()

    def send_route(self, waypoints: List[Dict[str, float]]) -> Dict[str, Any]:
        if not waypoints:
            return {"ok": False, "error": "waypoints required"}

        if not self.follow_waypoints_client.wait_for_server(timeout_sec=3.0):
            with lock:
                state.nav_status = "route_unavailable"
                state.nav_detail = "/FollowWaypoints action server unavailable"
                state.updated_at = time.time()
            return {"ok": False, "error": "FollowWaypoints action server unavailable"}

        goal = FollowWaypoints.Goal()
        poses: List[PoseStamped] = []

        for idx, point in enumerate(waypoints):
            x = float(point["x"])
            y = float(point["y"])

            if idx < len(waypoints) - 1:
                next_point = waypoints[idx + 1]
                yaw = math.atan2(float(next_point["y"]) - y, float(next_point["x"]) - x)
            elif idx > 0:
                prev_point = waypoints[idx - 1]
                yaw = math.atan2(y - float(prev_point["y"]), x - float(prev_point["x"]))
            else:
                yaw = 0.0

            poses.append(build_pose_stamped("map", x, y, yaw, self.get_clock()))

        goal.poses = poses

        with lock:
            state.current_route = [{"x": float(p["x"]), "y": float(p["y"])} for p in waypoints]
            state.current_waypoint_index = 0
            state.route_running = True
            state.nav_status = "route_sent"
            state.nav_detail = f"sent {len(waypoints)} waypoints"
            state.updated_at = time.time()

        send_goal_future = self.follow_waypoints_client.send_goal_async(
            goal,
            feedback_callback=self.route_feedback_cb,
        )
        send_goal_future.add_done_callback(self.route_goal_response_cb)
        return {"ok": True, "message": f"sent {len(waypoints)} waypoints"}

    def route_feedback_cb(self, feedback_msg) -> None:
        current_waypoint = int(feedback_msg.feedback.current_waypoint)
        with lock:
            state.current_waypoint_index = current_waypoint
            state.nav_status = "route_running"
            state.nav_detail = f"executing waypoint {current_waypoint + 1}"
            state.updated_at = time.time()

    def route_goal_response_cb(self, future) -> None:
        goal_handle = future.result()
        with lock:
            if not goal_handle.accepted:
                state.route_running = False
                state.nav_status = "route_rejected"
                state.nav_detail = "route rejected by action server"
                state.updated_at = time.time()
                return

        self.current_goal_handle = goal_handle
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.route_result_cb)

    def route_result_cb(self, future) -> None:
        result = future.result()
        missed_waypoints = list(result.result.missed_waypoints)
        self.current_goal_handle = None

        with lock:
            state.route_running = False
            if missed_waypoints:
                state.nav_status = "route_partial"
                state.nav_detail = f"missed waypoints: {missed_waypoints}"
            else:
                state.nav_status = "route_completed"
                state.nav_detail = "route completed"
            state.updated_at = time.time()

    def cancel_route(self) -> Dict[str, Any]:
        if self.current_goal_handle is None:
            with lock:
                state.route_running = False
            return {"ok": False, "error": "no active route"}

        cancel_future = self.current_goal_handle.cancel_goal_async()

        def _done(_future) -> None:
            self.current_goal_handle = None
            with lock:
                state.route_running = False
                state.nav_status = "route_cancelled"
                state.nav_detail = "route cancelled"
                state.updated_at = time.time()

        cancel_future.add_done_callback(_done)
        return {"ok": True, "message": "cancel requested"}


rclpy.init()
ros_node = DashboardBridge()


def ros_spin_thread() -> None:
    rclpy.spin(ros_node)


threading.Thread(target=ros_spin_thread, daemon=True).start()


@app.get("/")
def dashboard_index():
    return render_template("dashboard.html")


@app.get("/api/status")
def api_status():
    with lock:
        pose = state.pose
        response = {
            "ok": True,
            "pose": pose,
            "speed": state.speed,
            "trajectory_count": len(state.trajectory),
            "plan_count": len(state.plan),
            "map_available": state.map_msg is not None,
            "map_seq": state.map_seq,
            "last_command": state.last_command,
            "nav_status": state.nav_status,
            "nav_detail": state.nav_detail,
            "last_nav_log": state.last_nav_log,
            "route_running": state.route_running,
            "current_waypoint_index": state.current_waypoint_index,
            "route_waypoint_count": len(state.current_route),
            "camera_available": state.camera_jpeg is not None,
            "camera_stamp": state.camera_stamp,
            "updated_at": state.updated_at,
        }
    return jsonify(response)


@app.get("/api/map")
def api_map():
    with lock:
        map_msg = state.map_msg
        map_seq = state.map_seq

    if map_msg is None:
        return jsonify({"ok": False, "error": "map unavailable"}), 404

    return jsonify(
        {
            "ok": True,
            "seq": map_seq,
            "width": map_msg.info.width,
            "height": map_msg.info.height,
            "resolution": map_msg.info.resolution,
            "origin": {
                "x": map_msg.info.origin.position.x,
                "y": map_msg.info.origin.position.y,
                "yaw": quaternion_to_yaw(map_msg.info.origin.orientation),
            },
            "frame_id": map_msg.header.frame_id,
            "data": list(map_msg.data),
        }
    )


@app.get("/api/overlay")
def api_overlay():
    with lock:
        return jsonify(
            {
                "ok": True,
                "pose": state.pose,
                "trajectory": state.trajectory,
                "plan": state.plan,
                "current_route": state.current_route,
                "current_waypoint_index": state.current_waypoint_index,
                "route_running": state.route_running,
            }
        )


@app.get("/api/camera/frame.jpg")
def api_camera_frame():
    with lock:
        camera_jpeg = state.camera_jpeg
        camera_format = state.camera_format

    if camera_jpeg is None:
        return jsonify({"ok": False, "error": "camera frame unavailable"}), 404

    return Response(camera_jpeg, mimetype=f"image/{camera_format}")


@app.get("/api/camera/stream")
def api_camera_stream():
    def generate():
        last_stamp = 0.0
        while True:
            with camera_condition:
                camera_condition.wait(timeout=1.0)

            with lock:
                camera_jpeg = state.camera_jpeg
                camera_stamp = state.camera_stamp
                camera_format = state.camera_format

            if camera_jpeg is None or camera_stamp == last_stamp:
                continue

            last_stamp = camera_stamp
            header = (
                "--frame\r\n"
                f"Content-Type: image/{camera_format}\r\n"
                "Cache-Control: no-cache\r\n\r\n"
            ).encode("ascii")
            yield (
                header +
                camera_jpeg +
                b"\r\n"
            )

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        },
    )


@app.post("/api/command")
def api_command():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "text required"}), 400

    ros_node.publish_text_command(text)
    return jsonify({"ok": True, "message": f"published: {text}"})


@app.post("/api/route")
def api_route():
    data = request.get_json(force=True, silent=True) or {}
    waypoints = data.get("waypoints") or []
    try:
        result = ros_node.send_route(waypoints)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    if not result.get("ok"):
        return jsonify(result), 400
    return jsonify(result)


@app.post("/api/route/cancel")
def api_route_cancel():
    result = ros_node.cancel_route()
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.post("/api/trajectory/clear")
def api_trajectory_clear():
    with lock:
        state.trajectory = []
        state.updated_at = time.time()
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
