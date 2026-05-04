"""
Person localisation -- 2D bbox + depth -> map-frame (x, y)
"""
from typing import Any, Dict, List, Optional

import numpy as np


def person_localize(
    bbox: List[float],
    depth_16uc1: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    tf_matrix: Optional[np.ndarray],
    max_depth_m: float = 15.0,
) -> Optional[Dict[str, Any]]:
    """Project a person detection bounding box to a map-frame coordinate.

    *bbox*           ``[x1, y1, x2, y2]`` pixel coordinates.
    *depth_16uc1*    16-bit single-channel depth image (mm), shape (H, W).
    *fx, fy, cx, cy* Camera intrinsics (pixels).
    *tf_matrix*       Optional 4x4 homogeneous transform from camera-optical
                     frame to map frame (``map_T_cam``).  If *None*, the
                     camera-frame point is returned instead.
    *max_depth_m*     Maximum valid depth in metres (default 15 m).

    Returns a dict with keys ``map_x``, ``map_y``, ``depth_mm``, or *None*
    if localisation fails (invalid depth, TF missing, etc.).
    """
    h, w = depth_16uc1.shape[:2]

    # centroid slightly below the bbox centre (more likely to hit torso)
    bx1, by1, bx2, by2 = bbox
    bbox_h = by2 - by1
    px = (bx1 + bx2) / 2.0
    py = (by1 + by2) / 2.0 + 0.15 * bbox_h  # 15 % down from centre
    # clamp
    px = max(0, min(w - 1, int(round(px))))
    py = max(0, min(h - 1, int(round(py))))

    # read depth with 3x3 median fallback
    depth_mm = _read_depth(depth_16uc1, px, py)
    if depth_mm <= 0 or depth_mm > max_depth_m * 1000.0:
        return None

    # deproject  (pinhole model, camera-optical-frame: Z forward, X right, Y down)
    d_m = depth_mm / 1000.0
    x_cam = (px - cx) * d_m / fx
    y_cam = (py - cy) * d_m / fy
    z_cam = d_m

    if tf_matrix is not None:
        # 4x4 homogeneous transform: map_T_cam
        pt_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        pt_map = tf_matrix @ pt_cam
        return {
            "map_x": round(float(pt_map[0]), 3),
            "map_y": round(float(pt_map[1]), 3),
            "depth_mm": int(depth_mm),
        }
    else:
        # return camera-frame point as fallback
        return {
            "map_x": round(x_cam, 3),
            "map_y": round(z_cam, 3),  # use Z as a stand-in Y
            "depth_mm": int(depth_mm),
            "warning": "no TF available, coordinates in camera frame",
        }


def _read_depth(depth: np.ndarray, u: int, v: int) -> int:
    """Read depth in mm at (u, v), with 3x3 median fallback if center is 0."""
    d = int(depth[v, u])
    if d > 0:
        return d
    h, w = depth.shape
    u1, v1 = max(0, u - 1), max(0, v - 1)
    u2, v2 = min(w - 1, u + 1), min(h - 1, v + 1)
    patch = depth[v1:v2 + 1, u1:u2 + 1]
    valid = patch[patch > 0]
    if len(valid) > 0:
        return int(np.median(valid))
    return 0


def tf_to_matrix(transform) -> np.ndarray:
    """Convert a ``geometry_msgs.msg.Transform`` to a 4x4 homogeneous matrix."""
    q = transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    t = transform.translation
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T
