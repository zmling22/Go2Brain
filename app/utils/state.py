"""
Shared state and helpers for the dashboard server.
"""
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nav_msgs.msg import OccupancyGrid


def quaternion_to_yaw(q) -> float:
    """Convert a quaternion to a yaw angle (radians)."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def build_pose_stamped(frame_id: str, x: float, y: float, yaw: float, clock):
    """Build a PoseStamped message with the given frame, position, and yaw."""
    from geometry_msgs.msg import PoseStamped
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
    """Apply a 2D transform from a geometry_msgs TransformStamped."""
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
    """Thread-safe shared state between ROS callbacks and Flask routes."""

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
    detection_results: dict = field(
        default_factory=lambda: {"objects": [], "count": 0, "enabled": False}
    )
    detection_stamp: float = 0.0
    show_detection: bool = False

    # patrol fields
    patrol_active: bool = False
    patrol_id: str = ""
    patrol_status: str = "idle"
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

    # FPS monitoring
    camera_fps: float = 0.0
    map_fps: float = 0.0

    # patrol inspection rules
    patrol_rules: List[dict] = field(default_factory=list)
    patrol_violations: List[dict] = field(default_factory=list)

    # region segmentation fields
    regions: List[dict] = field(default_factory=list)
    robot_region: Optional[str] = None
    region_yaml_path: str = ""


# Global singleton instances
state = SharedState()
lock = threading.Lock()
camera_condition = threading.Condition()
