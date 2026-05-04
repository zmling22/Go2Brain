"""
DashboardBridge -- ROS2 node that bridges ROS topics to shared state.
"""
import json
import math
import time
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
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
from std_msgs.msg import String
import tf2_ros

from utils.state import (
    state, lock, camera_condition,
    build_pose_stamped, apply_2d_transform, quaternion_to_yaw,
)
from patrol import route as patrol_route
from patrol import photo as patrol_photo
from patrol import report as patrol_report
from patrol import rules as patrol_rules
from audio_sdk import player as voice_player
from utils import localize
from utils import map_analysis


class DashboardBridge(Node):
    """ROS2 node that subscribes to robot topics and updates shared state."""

    def __init__(self):
        super().__init__("dashboard_bridge")

        # publishers
        self.command_pub = self.create_publisher(String, "/nl_command", 10)

        # subscribers
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_cb, 20)
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_cb, 1)
        self.plan_sub = self.create_subscription(Path, "/plan", self.plan_cb, 10)
        self.rosout_sub = self.create_subscription(Log, "/rosout", self.rosout_cb, 50)
        self.camera_subs = [
            self.create_subscription(
                CompressedImage, "/camera/camera/color/image_raw/compressed",
                self.camera_cb, 10,
            ),
            self.create_subscription(
                CompressedImage, "/camera/image_raw/compressed",
                self.camera_cb, 10,
            ),
            self.create_subscription(
                CompressedImage, "/camera/image/raw/compressed",
                self.camera_cb, 10,
            ),
        ]
        self.raw_camera_sub = self.create_subscription(
            Image, "/camera/camera/color/image_raw",
            self.raw_camera_cb, 10,
        )

        self.detection_stats_sub = self.create_subscription(
            String, "/camera/detection/stats", self.detection_stats_cb, 10,
        )

        self.depth_sub = self.create_subscription(
            Image, "/camera/camera/aligned_depth_to_color/image_raw",
            self.depth_cb, 10,
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, "/camera/camera/color/camera_info",
            self.camera_info_cb, 10,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()

        self.follow_waypoints_client = ActionClient(self, FollowWaypoints, "/FollowWaypoints")
        self.current_goal_handle = None

        # FPS tracking counters (accessed only from spin thread, no lock needed)
        self._camera_fps_count = 0
        self._camera_fps_timer = time.time()
        self._map_fps_count = 0
        self._map_fps_timer = time.time()

    # ---- text command ----

    def publish_text_command(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.command_pub.publish(msg)
        with lock:
            state.last_command = text
            state.updated_at = time.time()

    # ---- ROS callbacks ----

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
                "map", msg.header.frame_id or "odom", Time(),
            )
            map_pose = apply_2d_transform(
                odom_pose["x"], odom_pose["y"], odom_pose["yaw"], transform,
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
                state.trajectory.append({"x": map_pose["x"], "y": map_pose["y"], "t": time.time()})
                if len(state.trajectory) > 4000:
                    state.trajectory = state.trajectory[-4000:]

            state.updated_at = time.time()

    def map_cb(self, msg: OccupancyGrid) -> None:
        now = time.time()
        self._map_fps_count += 1
        computed_fps = None
        if now - self._map_fps_timer >= 2.0:
            computed_fps = self._map_fps_count / (now - self._map_fps_timer)
            self._map_fps_count = 0
            self._map_fps_timer = now

        with lock:
            state.map_msg = msg
            state.map_seq += 1
            if computed_fps is not None:
                state.map_fps = round(computed_fps, 1)
            state.updated_at = time.time()

    def plan_cb(self, msg: Path) -> None:
        points = [{"x": p.pose.position.x, "y": p.pose.position.y} for p in msg.poses]
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

        now = time.time()
        self._camera_fps_count += 1
        computed_fps = None
        if now - self._camera_fps_timer >= 2.0:
            computed_fps = self._camera_fps_count / (now - self._camera_fps_timer)
            self._camera_fps_count = 0
            self._camera_fps_timer = now

        with lock:
            state.camera_jpeg = bytes(msg.data)
            state.camera_stamp = now
            state.camera_format = image_format
            if computed_fps is not None:
                state.camera_fps = round(computed_fps, 1)
            state.updated_at = now
        with camera_condition:
            camera_condition.notify_all()

    def raw_camera_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return

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

        now = time.time()
        self._camera_fps_count += 1
        computed_fps = None
        if now - self._camera_fps_timer >= 2.0:
            computed_fps = self._camera_fps_count / (now - self._camera_fps_timer)
            self._camera_fps_count = 0
            self._camera_fps_timer = now

        with lock:
            state.camera_jpeg = jpg.tobytes()
            state.camera_stamp = now
            state.camera_format = "jpeg"
            if computed_fps is not None:
                state.camera_fps = round(computed_fps, 1)
            state.updated_at = now
        with camera_condition:
            camera_condition.notify_all()

    # ---- detection callback ----

    def detection_stats_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        patrol_active = False
        objects = None
        with lock:
            state.detection_results = data
            state.detection_stamp = time.time()
            state.updated_at = time.time()
            patrol_active = state.patrol_active
            objects = list(data.get("objects", [])) if data.get("objects") else None

        if patrol_active and objects:
            self._check_patrol_detections(objects)
            self._check_patrol_rules(objects)

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

        waypoints = patrol_route.generate_patrol_route(
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

    # ---- region segmentation ----

    def segment_map(self, room_clearance_m: float = 0.7,
                    min_room_area_m2: float = 2.0) -> Dict[str, Any]:
        """Run room/corridor segmentation on the current map."""
        with lock:
            map_msg = state.map_msg
            pose = state.pose

        if map_msg is None:
            return {"ok": False, "error": "map unavailable"}
        if pose is None:
            return {"ok": False, "error": "robot pose unknown"}

        resolution = map_msg.info.resolution

        result = map_analysis.segment_map(
            map_data=list(map_msg.data),
            width=map_msg.info.width,
            height=map_msg.info.height,
            resolution=resolution,
            origin_x=map_msg.info.origin.position.x,
            origin_y=map_msg.info.origin.position.y,
            robot_x=pose["x"],
            robot_y=pose["y"],
            room_clearance_m=room_clearance_m,
            min_room_area_m2=min_room_area_m2,
        )

        with lock:
            state.regions = result.get("regions", [])
            state.robot_region = result.get("robot_region")

        print(f"[segment_map] {len(state.regions)} regions, "
              f"robot in {state.robot_region}")
        return {"ok": True, **result}

    def start_patrol(self) -> Dict[str, Any]:
        """Start patrol execution."""
        with lock:
            waypoints = list(state.patrol_waypoints)
        if not waypoints:
            return {"ok": False, "error": "no patrol route, generate one first"}

        patrol_id = f"patrol_{time.strftime('%Y%m%d_%H%M%S')}"
        photos_dir = patrol_photo.ensure_patrol_dirs(patrol_id)
        patrol_photo.cleanup_old_patrol_sessions()

        with lock:
            state.patrol_active = True
            state.patrol_id = patrol_id
            state.patrol_status = "running"
            state.patrol_current_index = -1
            state.patrol_person_detections = []
            state.patrol_violations = []
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
        self.cancel_route()
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
        with lock:
            df = state.depth_frame
            fx, fy = state.camera_fx, state.camera_fy
            cpx, cpy = state.camera_cx, state.camera_cy
            cinfo_ok = state.camera_info_available
            idx = state.patrol_current_index
            photos_dir = state.patrol_photos_dir
            seq = state.patrol_photo_seq + 1
            state.patrol_photo_seq = seq

        # read latest frame for photo
        bgr_frame: Optional[np.ndarray] = None
        jpeg_bytes: Optional[bytes] = None
        with lock:
            jpeg_bytes = state.camera_jpeg
        if jpeg_bytes is not None:
            try:
                bgr_frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                pass

        # project to map coordinates
        loc: Optional[Dict[str, Any]] = None
        if df is not None and cinfo_ok:
            try:
                tf_stamped = self.tf_buffer.lookup_transform(
                    "map", "camera_color_optical_frame", Time())
                tf_matrix = localize.tf_to_matrix(tf_stamped.transform)
                loc = localize.person_localize(bbox, df["data"], fx, fy, cpx, cpy, tf_matrix)
            except Exception:
                with lock:
                    state.patrol_warnings.append("TF lookup failed for person detection")
                loc = localize.person_localize(bbox, df["data"], fx, fy, cpx, cpy, None)
        else:
            with lock:
                state.patrol_warnings.append(
                    "Depth/CameraInfo not available for person localisation")

        # distance debounce: skip if within 1.5m of last detection
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
                        dist = math.hypot(loc["map_x"] - lx, loc["map_y"] - ly)
                        if dist < 1.5:
                            return

        # save photo
        photo_url: Optional[str] = None
        if bgr_frame is not None and photos_dir:
            photo_url = patrol_photo.save_patrol_photo(
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

    def _check_patrol_rules(self, objects: List[dict]) -> None:
        """Check detection objects against all enabled patrol rules."""
        with lock:
            rules = [r for r in state.patrol_rules if r.get("enabled", True)]
        if not rules:
            return

        for rule in rules:
            match = patrol_rules.evaluate_rule(rule, objects)
            if match is None:
                continue

            # 3-second debounce per rule
            with lock:
                last_for_rule = [
                    v for v in state.patrol_violations
                    if v.get("rule_id") == rule["id"]
                ]
            if last_for_rule:
                last_ts = last_for_rule[-1].get("timestamp_epoch", 0)
                if time.time() - last_ts < 3.0:
                    continue

            self._record_violation(rule, match, objects)

    def _record_violation(self, rule: dict, match: dict, all_objects: List[dict]) -> None:
        """Record a single rule violation: localise, photograph, trigger response."""
        with lock:
            df = state.depth_frame
            fx, fy = state.camera_fx, state.camera_fy
            cpx, cpy = state.camera_cx, state.camera_cy
            cinfo_ok = state.camera_info_available
            idx = state.patrol_current_index
            photos_dir = state.patrol_photos_dir
            pid = state.patrol_id

        # latest frame for photo
        bgr_frame = None
        jpeg_bytes = None
        with lock:
            jpeg_bytes = state.camera_jpeg
        if jpeg_bytes is not None:
            try:
                bgr_frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                pass

        # localisation: use person bbox if available, else first matched bbox
        person_bbox = None
        for obj in all_objects:
            if obj.get("label") == "person":
                person_bbox = obj.get("bbox")
                break
        target_bbox = person_bbox
        if target_bbox is None and match.get("bounding_boxes"):
            target_bbox = match["bounding_boxes"][0]

        loc = None
        if target_bbox and df is not None and cinfo_ok:
            try:
                tf_stamped = self.tf_buffer.lookup_transform(
                    "map", "camera_color_optical_frame", Time())
                tf_matrix = localize.tf_to_matrix(tf_stamped.transform)
                loc = localize.person_localize(
                    target_bbox, df["data"], fx, fy, cpx, cpy, tf_matrix)
            except Exception:
                pass

        # save photo with violation annotation
        photo_url = None
        seq = 0
        if bgr_frame is not None and photos_dir:
            with lock:
                seq = state.patrol_photo_seq + 1
                state.patrol_photo_seq = seq
            photo_url = self._save_violation_photo(
                bgr_frame, match["bounding_boxes"], rule["text"],
                match["detected_objects"], match["confidence"], photos_dir, seq)

        record = {
            "index": seq,
            "rule_id": rule["id"],
            "rule_text": rule["text"],
            "detected_objects": match["detected_objects"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_epoch": time.time(),
            "waypoint_index": idx,
            "confidence": match["confidence"],
            "bounding_boxes": match["bounding_boxes"],
            "person_related": rule.get("person_related", False),
            "voice_alert_played": False,
            "resolved": False,
        }
        if loc:
            record["map_x"] = loc.get("map_x")
            record["map_y"] = loc.get("map_y")
            record["depth_mm"] = loc.get("depth_mm")
        if photo_url:
            record["photo"] = photo_url

        with lock:
            state.patrol_violations.append(record)
            state.updated_at = time.time()

        # person-related: approach and voice
        if rule.get("person_related") and record.get("map_x") is not None:
            self._handle_person_violation(record)

    def _save_violation_photo(
        self, bgr_frame, bboxes, rule_text, labels, confidence, photos_dir, seq,
    ) -> Optional[str]:
        """Draw violation bounding boxes and rule text on frame, save as JPEG."""
        try:
            annotated = bgr_frame.copy()
            colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0)]
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                color = colors[i % len(colors)]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label_text = labels[i] if i < len(labels) else "?"
                cv2.putText(
                    annotated, label_text, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                )

            cv2.putText(
                annotated, f"VIOLATION: {rule_text}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                annotated, ts, (8, annotated.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

            fname = f"violation_{seq:04d}_{confidence:.2f}.jpg"
            path = os.path.join(photos_dir, fname)
            cv2.imwrite(path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return f"/patrol/photos/{os.path.basename(photos_dir)}/{fname}"
        except Exception:
            return None

    def _handle_person_violation(self, violation: dict) -> None:
        """Navigate to person location and play voice alert."""
        mx = violation.get("map_x")
        my = violation.get("map_y")
        if mx is None or my is None:
            return

        # cancel current patrol route
        self.cancel_route()

        # navigate to person
        self._navigate_to_pose(mx, my)

        # play voice alert (daemon thread, non-blocking)
        rule_text = violation.get("rule_text", "")
        if rule_text:
            voice_player.play_alert(rule_text)

        with lock:
            violation["voice_alert_played"] = True

    def _navigate_to_pose(self, x: float, y: float) -> None:
        """Send a single NavigateToPose goal (Nav2)."""
        try:
            from nav2_msgs.action import NavigateToPose
        except ImportError:
            self.get_logger().warning("nav2_msgs.action.NavigateToPose not available")
            return

        if not hasattr(self, '_navigate_to_pose_client'):
            self._navigate_to_pose_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")

        if not self._navigate_to_pose_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().warning("/navigate_to_pose action server unavailable")
            return

        goal = NavigateToPose.Goal()
        goal.pose = build_pose_stamped("map", x, y, 0.0, self.get_clock())

        send_future = self._navigate_to_pose_client.send_goal_async(goal)
        send_future.add_done_callback(self._navigate_to_pose_response_cb)

    def _navigate_to_pose_response_cb(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning("NavigateToPose goal rejected")

    def _on_patrol_waypoint_change(self, wp_index: int) -> None:
        with lock:
            state.patrol_current_index = wp_index

    def _finalise_patrol(self, status: str) -> None:
        """Generate patrol report and save to disk."""
        with lock:
            if not state.patrol_active:
                return
            start = state.patrol_start_time
            waypoints = list(state.patrol_waypoints)
            detections = list(state.patrol_person_detections)
            violations = list(state.patrol_violations)
            warnings = list(state.patrol_warnings)
            pid = state.patrol_id
            completed = list(range(state.patrol_current_index + 1))

        end = time.time()
        report = patrol_report.generate_report(
            patrol_id=pid,
            start_time=start,
            end_time=end,
            waypoints=waypoints,
            completed_indices=completed,
            person_detections=detections,
            warnings=warnings,
            violations=violations,
        )
        patrol_report.save_report_to_disk(report, pid)

        with lock:
            state.patrol_active = False
            state.patrol_status = status
            state.patrol_report = report
            state.updated_at = time.time()

    # ---- route action client ----

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
            goal, feedback_callback=self.route_feedback_cb,
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
