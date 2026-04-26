from flask import Flask, Response, jsonify, render_template, request
import math
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from cv_bridge import CvBridge

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowWaypoints
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rcl_interfaces.msg import Log
from sensor_msgs.msg import CameraInfo, Image
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_msgs.msg import Bool
import tf2_ros
import json

import patrol_core


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
    # detection fields
    detection_results: dict = field(default_factory=lambda: {"objects": [], "count": 0, "enabled": False})
    detection_stamp: float = 0.0
    show_detection: bool = False  # user toggle in dashboard

    # patrol fields
    patrol_active: bool = False
    patrol_id: str = ""
    patrol_status: str = "idle"  # idle|generating|running|stopping|completed|aborted
    patrol_waypoints: List[Dict[str, float]] = field(default_factory=list)
    patrol_current_index: int = -1
    patrol_person_detections: List[dict] = field(default_factory=list)
    patrol_photos_dir: str = ""
    patrol_start_time: float = 0.0
    patrol_report: Optional[dict] = None
    patrol_warnings: List[str] = field(default_factory=list)
    patrol_photo_seq: int = 0

    # depth / camera info (for person localisation)
    depth_frame: Optional[dict] = None
    camera_fx: float = 0.0
    camera_fy: float = 0.0
    camera_cx: float = 0.0
    camera_cy: float = 0.0
    camera_info_available: bool = False


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

        # --- detection subscription ---
        self.detection_stats_sub = self.create_subscription(
            String,
            "/camera/detection/stats",
            self.detection_stats_cb,
            10,
        )

        # --- depth + camera info (for person localisation) ---
        self.depth_sub = self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            self.depth_cb,
            10,
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.camera_info_cb,
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
        except Exception:
            return

        # draw detection boxes on frame if available
        with lock:
            if state.show_detection and state.detection_results.get("objects"):
                for obj in state.detection_results["objects"]:
                    bbox = obj.get("bbox")
                    if not bbox or len(bbox) < 4:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                    label = obj.get("label", "?")
                    conf = obj.get("confidence", 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, f"{label} {conf:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                    )

        try:
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

    # ---- detection callbacks ----

    def detection_stats_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        with lock:
            state.detection_results = data
            state.detection_stamp = time.time()
            state.updated_at = time.time()
            # patrol: check for persons
            if state.patrol_active and data.get("objects"):
                self._check_patrol_detections(data["objects"])

    # ---- depth / camera info callbacks ----

    def depth_cb(self, msg: Image) -> None:
        try:
            depth_arr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        except Exception:
            return
        with lock:
            state.depth_frame = {"data": depth_arr.copy(), "stamp": time.time()}

    def camera_info_cb(self, msg: CameraInfo) -> None:
        with lock:
            state.camera_fx = msg.k[0]
            state.camera_fy = msg.k[4]
            state.camera_cx = msg.k[2]
            state.camera_cy = msg.k[6]
            state.camera_info_available = True

    # ---- patrol methods ----

    def generate_patrol_route(self) -> Dict[str, Any]:
        """Auto-generate patrol route from current map."""
        with lock:
            state.patrol_status = "generating"
            map_msg = state.map_msg
            pose = state.pose

        if map_msg is None:
            with lock:
                state.patrol_status = "idle"
            return {"ok": False, "error": "map unavailable"}
        if pose is None:
            with lock:
                state.patrol_status = "idle"
            return {"ok": False, "error": "robot pose unknown"}

        resolution = map_msg.info.resolution
        print(f"[patrol] robot pose: ({pose['x']:.2f}, {pose['y']:.2f}), "
              f"map {map_msg.info.width}x{map_msg.info.height}, "
              f"res {resolution}")

        waypoints = patrol_core.generate_patrol_route(
            map_data=list(map_msg.data),
            width=map_msg.info.width,
            height=map_msg.info.height,
            resolution=resolution,
            origin_x=map_msg.info.origin.position.x,
            origin_y=map_msg.info.origin.position.y,
            robot_x=pose["x"],
            robot_y=pose["y"],
        )
        print(f"[patrol] generated {len(waypoints)} waypoints")
        if not waypoints:
            with lock:
                state.patrol_status = "idle"
            return {"ok": False, "error": "no navigable space found"}

        with lock:
            state.patrol_waypoints = waypoints
            state.patrol_status = "route_ready"
        return {"ok": True, "waypoints": waypoints}

    def start_patrol(self) -> Dict[str, Any]:
        """Start patrol execution."""
        with lock:
            waypoints = list(state.patrol_waypoints)
        if not waypoints:
            return {"ok": False, "error": "no patrol route, generate one first"}

        patrol_id = f"patrol_{time.strftime('%Y%m%d_%H%M%S')}"
        photos_dir = patrol_core.ensure_patrol_dirs(patrol_id)
        patrol_core.cleanup_old_patrol_sessions()

        with lock:
            state.patrol_active = True
            state.patrol_id = patrol_id
            state.patrol_status = "running"
            state.patrol_current_index = -1
            state.patrol_person_detections = []
            state.patrol_photos_dir = photos_dir
            state.patrol_start_time = time.time()
            state.patrol_report = None
            state.patrol_warnings = []
            state.patrol_photo_seq = 0

        result = self.send_route(waypoints)
        return {"ok": result.get("ok", False),
                "patrol_id": patrol_id,
                "message": f"patrol started with {len(waypoints)} waypoints"}

    def stop_patrol(self) -> Dict[str, Any]:
        """Stop patrol and save partial report."""
        cancel_result = self.cancel_route()
        self._finalise_patrol("aborted")
        return {"ok": True, "message": "patrol stopped"}

    def _check_patrol_detections(self, objects: List[dict]) -> None:
        """Check detection results for persons during active patrol."""
        for obj in objects:
            if obj.get("label") != "person" or obj.get("confidence", 0) < 0.5:
                continue
            self._record_person_detection(obj["bbox"], obj["confidence"])

    def _record_person_detection(self, bbox: List[float], confidence: float) -> None:
        """Localise, photograph, and record a person detection."""
        depth_info: Optional[dict] = None
        with lock:
            df = state.depth_frame
            fx, fy = state.camera_fx, state.camera_fy
            cpx, cpy = state.camera_cx, state.camera_cy
            cinfo_ok = state.camera_info_available
            idx = state.patrol_current_index
            photos_dir = state.patrol_photos_dir
            seq = state.patrol_photo_seq + 1
            state.patrol_photo_seq = seq

        # read latest frame from raw_camera_cb for photo
        bgr_frame: Optional[np.ndarray] = None
        with lock:
            if state.camera_jpeg is not None:
                try:
                    bgr_frame = cv2.imdecode(
                        np.frombuffer(state.camera_jpeg, np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                except Exception:
                    pass

        # project to map coordinates
        loc: Optional[Dict[str, Any]] = None
        if df is not None and cinfo_ok:
            try:
                tf_stamped = self.tf_buffer.lookup_transform(
                    "map", "camera_color_optical_frame", Time())
                tf_matrix = patrol_core.tf_to_matrix(tf_stamped.transform)
                loc = patrol_core.person_localize(
                    bbox, df["data"], fx, fy, cpx, cpy, tf_matrix)
            except Exception:
                with lock:
                    state.patrol_warnings.append(
                        "TF lookup failed for person detection")
                loc = patrol_core.person_localize(
                    bbox, df["data"], fx, fy, cpx, cpy, None)
        else:
            with lock:
                state.patrol_warnings.append(
                    "Depth/CameraInfo not available for person localisation")

        # check distance debounce: skip if within 1.5m of last detection
        if loc and loc.get("map_x") is not None:
            with lock:
                last_dets = state.patrol_person_detections
                if last_dets:
                    last = last_dets[-1]
                    lx = last.get("map_x")
                    ly = last.get("map_y")
                    if (lx is not None and ly is not None
                            and loc.get("map_x") is not None
                            and loc.get("map_y") is not None):
                        dist = math.hypot(
                            loc["map_x"] - lx, loc["map_y"] - ly)
                        if dist < 1.5:
                            return  # same person, skip

        # save photo
        photo_url: Optional[str] = None
        if bgr_frame is not None and photos_dir:
            photo_url = patrol_core.save_patrol_photo(
                bgr_frame, bbox, "person", confidence, photos_dir, seq)

        # record
        record = {
            "index": seq,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "waypoint_index": idx,
            "confidence": round(confidence, 3),
            "bounding_box": [round(v, 2) for v in bbox[:4]],
        }
        if loc:
            record["map_x"] = loc.get("map_x")
            record["map_y"] = loc.get("map_y")
            record["depth_mm"] = loc.get("depth_mm")
            record["warning"] = loc.get("warning", "")
        if photo_url:
            record["photo"] = photo_url

        with lock:
            state.patrol_person_detections.append(record)
            state.updated_at = time.time()

    def _on_patrol_waypoint_change(self, wp_index: int) -> None:
        """Called when the robot reaches a new waypoint during patrol."""
        with lock:
            state.patrol_current_index = wp_index

    def _finalise_patrol(self, status: str) -> None:
        """Generate patrol report and save to disk."""
        with lock:
            active = state.patrol_active
            if not active:
                return
            start = state.patrol_start_time
            waypoints = list(state.patrol_waypoints)
            detections = list(state.patrol_person_detections)
            warnings = list(state.patrol_warnings)
            pid = state.patrol_id
            # determine completed waypoints
            completed = list(range(state.patrol_current_index + 1))

        end = time.time()
        report = patrol_core.generate_report(
            patrol_id=pid,
            start_time=start,
            end_time=end,
            waypoints=waypoints,
            completed_indices=completed,
            person_detections=detections,
            warnings=warnings,
        )
        patrol_core.save_report_to_disk(report, pid)

        with lock:
            state.patrol_active = False
            state.patrol_status = status
            state.patrol_report = report
            state.updated_at = time.time()

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
        patrol_active = False
        with lock:
            state.current_waypoint_index = current_waypoint
            state.nav_status = "route_running"
            state.nav_detail = f"executing waypoint {current_waypoint + 1}"
            state.updated_at = time.time()
            patrol_active = state.patrol_active
        if patrol_active:
            self._on_patrol_waypoint_change(current_waypoint)

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
        patrol_active = False

        with lock:
            state.route_running = False
            if missed_waypoints:
                state.nav_status = "route_partial"
                state.nav_detail = f"missed waypoints: {missed_waypoints}"
            else:
                state.nav_status = "route_completed"
                state.nav_detail = "route completed"
            state.updated_at = time.time()
            patrol_active = state.patrol_active

        if patrol_active:
            status = "completed" if not missed_waypoints else "partial"
            self._finalise_patrol(status)

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
            "detection_count": state.detection_results.get("count", 0),
            "detection_enabled": state.show_detection,
            "patrol_status": state.patrol_status,
            "patrol_active": state.patrol_active,
            "patrol_waypoint_count": len(state.patrol_waypoints),
            "patrol_current_index": state.patrol_current_index,
            "patrol_person_count": len(state.patrol_person_detections),
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


# ---- detection APIs ----

@app.get("/api/detection")
def api_detection():
    with lock:
        return jsonify({
            "ok": True,
            "objects": state.detection_results.get("objects", []),
            "count": state.detection_results.get("count", 0),
            "enabled": state.show_detection,
        })


@app.post("/api/detection/toggle")
def api_detection_toggle():
    data = request.get_json(force=True, silent=True) or {}
    enabled = data.get("enable", not state.show_detection)
    with lock:
        state.show_detection = bool(enabled)
    return jsonify({"ok": True, "enabled": state.show_detection})


# ---- patrol APIs ----

@app.post("/api/patrol/generate")
def api_patrol_generate():
    try:
        result = ros_node.generate_patrol_route()
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.post("/api/patrol/start")
def api_patrol_start():
    try:
        result = ros_node.start_patrol()
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.post("/api/patrol/stop")
def api_patrol_stop():
    try:
        result = ros_node.stop_patrol()
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify(result)


@app.get("/api/patrol/state")
def api_patrol_state():
    with lock:
        return jsonify({
            "ok": True,
            "patrol_id": state.patrol_id,
            "status": state.patrol_status,
            "active": state.patrol_active,
            "waypoint_count": len(state.patrol_waypoints),
            "current_index": state.patrol_current_index,
            "person_count": len(state.patrol_person_detections),
            "detections": list(state.patrol_person_detections),
            "report": state.patrol_report,
        })


@app.get("/api/patrol/report/<patrol_id>")
def api_patrol_report(patrol_id: str):
    report = patrol_core.load_report(patrol_id)
    if report is None:
        return jsonify({"ok": False, "error": "report not found"}), 404
    return jsonify({"ok": True, "report": report})


@app.get("/api/patrol/reports")
def api_patrol_reports():
    return jsonify({"ok": True, "reports": patrol_core.list_reports()})


@app.route("/patrol/photos/<path:filename>")
def api_patrol_photos(filename: str):
    base = os.path.join(os.path.dirname(__file__), "static", "patrol_photos")
    return Response(
        open(os.path.join(base, filename), "rb").read(),
        mimetype="image/jpeg",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
