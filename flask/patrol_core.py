"""
Patrol logic module -- pure functions for route generation, person localization, report generation.
No ROS imports at module level; ROS-dependent data (map, TF, depth) is passed as arguments.
"""

import json
import math
import os
import time
import traceback
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
#  Route generation  --  Contour-based medial-axis patrol
# ---------------------------------------------------------------------------

def generate_patrol_route(
    map_data: List[int],
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    robot_x: float,
    robot_y: float,
    fov_radius: float = 3.0,
    max_waypoints: int = 200,
    robot_radius: float = 0.22,
) -> List[Dict[str, float]]:
    """Generate a patrol route using contour tracing of the corridor network.

    Algorithm:
      1. Parse occupancy grid → free mask, close small SLAM gaps.
      2. Erode by robot_radius → safe mask.
      3. BFS reachable region from robot.
      4. Morphological skeleton of the reachable region.
      5. Dilate skeleton to bridge gaps → largest corridor network.
      6. Trace the outer contour of the corridor network.
      7. Sample waypoints evenly along the contour path.
      8. Rotate so nearest waypoint to robot comes first, close loop.
    """
    if width <= 0 or height <= 0 or resolution <= 0:
        return []

    # --- 1. parse map ---
    arr = np.asarray(map_data, dtype=np.int8).reshape(height, width)
    free_mask = arr == 0
    if not free_mask.any():
        return []

    # --- 2. robot pixel (search for free cell if robot is in unknown/occupied) ---
    robot_c = int(round((robot_x - origin_x) / resolution))
    robot_r = int(round((robot_y - origin_y) / resolution))
    robot_c = max(0, min(width - 1, robot_c))
    robot_r = max(0, min(height - 1, robot_r))

    # --- 3. close small SLAM gaps ---
    close_radius = max(3, int(round(0.5 / resolution)))
    ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (2 * close_radius + 1, 2 * close_radius + 1))
    free_uint8 = free_mask.astype(np.uint8)
    closed = cv2.morphologyEx(free_uint8, cv2.MORPH_CLOSE, ck)
    # closed: 1 = free after close, 0 = occupied/unknown

    # --- 4. erode by robot_radius for safe navigation ---
    safe_cells = max(1, int(round(robot_radius / resolution)))
    sk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (2 * safe_cells + 1, 2 * safe_cells + 1))
    safe_mask = cv2.erode(closed, sk).astype(bool)

    # --- 5. BFS reachable region from robot ---
    # If robot cell is not safe, search for nearest safe cell
    if not safe_mask[robot_r, robot_c]:
        sy, sx = _find_nearest_safe(safe_mask, robot_r, robot_c)
        if sy is None:
            print(f"[patrol_core] no safe cell reachable from robot")
            return []
        robot_r, robot_c = sy, sx

    reachable = _bfs_mask(safe_mask, robot_r, robot_c)
    if not reachable.any():
        print(f"[patrol_core] no reachable area from robot position")
        return []

    # --- 6. mask narrow bottlenecks from reachable area ---
    # Distance transform of the closed free space → clearance from obstacles.
    free_for_dt = (closed > 0).astype(np.uint8)  # 1 = free, 0 = obstacle
    dt_free = cv2.distanceTransform(free_for_dt, cv2.DIST_L2, 5)
    # Only keep cells with ≥ 0.50 m clearance (~1 m corridor width)
    min_clearance = 10
    clearance_px = min_clearance / resolution
    
    wide_mask = reachable & (dt_free > clearance_px)
    if not wide_mask.any():
        wide_mask = reachable  # fallback

    # --- 7. morphological skeleton of the wide-area mask ---
    skel = _morphological_skeleton(wide_mask)

    # --- 8. dilate skeleton to bridge fragments, largest component ---
    bridge_radius = max(3, int(round(0.3 / resolution)))
    elem = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * bridge_radius + 1, 2 * bridge_radius + 1))
    dilated = cv2.dilate(skel.astype(np.uint8), elem)
    num_comp, labels = cv2.connectedComponents(dilated)
    if num_comp < 2:
        print(f"[patrol_core] dilated skeleton has no components")
        return []

    sizes = [(np.sum(labels == i), i) for i in range(1, num_comp)]
    sizes.sort(reverse=True)
    corridor_mask = labels == sizes[0][1]
    # Clip to the wide-clearance area so the contour doesn't hug walls.
    corridor_mask = corridor_mask & wide_mask

    # --- 9. erode corridor mask slightly to keep contour off the exact boundary ---
    wall_buffer = max(1, int(round(0.1 / resolution)))
    buf_elem = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * wall_buffer + 1, 2 * wall_buffer + 1))
    corridor_safe = cv2.erode(corridor_mask.astype(np.uint8), buf_elem).astype(bool)
    corridor_safe = corridor_safe & wide_mask
    if not corridor_safe.any():
        corridor_safe = corridor_mask  # fallback

    # --- 10. extract outer contour of corridor network ---
    ctr = _largest_outer_contour(corridor_safe)
    if ctr is None or len(ctr) < 4:
        print(f"[patrol_core] corridor contour empty, falling back to reachable contour")
        ctr = _largest_outer_contour(reachable)
        if ctr is None or len(ctr) < 4:
            return []

    # --- 11. convert contour pixels to world coordinates ---
    # Rotate so nearest waypoint to robot comes first, then close the loop.
    ctr_pts = ctr.squeeze(axis=1)  # (N, 2)  pixel coords (col, row)
    world = [(origin_x + c * resolution, origin_y + r * resolution)
             for (c, r) in ctr_pts]
    dists = [math.hypot(x - robot_x, y - robot_y) for (x, y) in world]
    ni = int(np.argmin(dists))
    ordered_world = world[ni:] + world[:ni]
    ordered_world.append(ordered_world[0])

    if not ordered_world:
        return []

    # --- 12. sample waypoints evenly along the path ---
    # Build cumulative distance
    cum = [0.0]
    for i in range(1, len(ordered_world)):
        dx = ordered_world[i][0] - ordered_world[i - 1][0]
        dy = ordered_world[i][1] - ordered_world[i - 1][1]
        cum.append(cum[-1] + math.hypot(dx, dy))
    total = cum[-1]

    if total < 0.5:
        return [{"x": round(robot_x, 3), "y": round(robot_y, 3)}]

    # Target spacing: fov_radius * 0.6 (e.g., 3.0 * 0.6 = 1.8m)
    spacing_world = fov_radius * 0.6
    num_pts = max(8, min(max_waypoints, int(total / spacing_world)))

    result_raw: List[Tuple[float, float]] = []
    for i in range(num_pts):
        target = i * total / num_pts
        idx = int(np.searchsorted(cum, target))
        if idx == 0:
            result_raw.append(ordered_world[0])
        elif idx >= len(ordered_world):
            result_raw.append(ordered_world[-1])
        else:
            t = (target - cum[idx - 1]) / (cum[idx] - cum[idx - 1])
            x = ordered_world[idx - 1][0] + t * (ordered_world[idx][0] - ordered_world[idx - 1][0])
            y = ordered_world[idx - 1][1] + t * (ordered_world[idx][1] - ordered_world[idx - 1][1])
            result_raw.append((x, y))

    # --- 13. subsample duplicates / close points ---
    result: List[Dict[str, float]] = []
    for x, y in result_raw:
        if result and math.hypot(x - result[-1]["x"], y - result[-1]["y"]) < 0.5:
            continue
        result.append({"x": round(x, 3), "y": round(y, 3)})
        if len(result) >= max_waypoints:
            break
    if not result:
        result.append({"x": round(robot_x, 3), "y": round(robot_y, 3)})
        return result

    # --- 14. snap any waypoints that fell into occupied/unknown cells ---
    # Snap to nearest original free cell for conservative navigation.
    for wp in result:
        c = int(round((wp["x"] - origin_x) / resolution))
        r = int(round((wp["y"] - origin_y) / resolution))
        if not (0 <= r < height and 0 <= c < width and free_mask[r, c]):
            best_sq = float("inf")
            best_px = (c, r)
            r0, r1 = max(0, r - 20), min(height, r + 21)
            c0, c1 = max(0, c - 20), min(width, c + 21)
            for dr in range(r0, r1):
                for dc in range(c0, c1):
                    if free_mask[dr, dc]:
                        d = (dr - r) * (dr - r) + (dc - c) * (dc - c)
                        if d < best_sq:
                            best_sq = d
                            best_px = (dc, dr)
            wp["x"] = round(origin_x + best_px[0] * resolution, 3)
            wp["y"] = round(origin_y + best_px[1] * resolution, 3)

    # --- 15. remove waypoints near black (occupied) or gray (unknown) cells ---
    # Dashboard renders occupied (≥65) as black and unknown (-1) as gray.
    # Waypoints within check_radius of any such cell are removed so the route
    # visibly stays away from walls and unexplored areas.
    check_radius = max(1, int(round(0.5 / resolution)))  # 0.5 m radius
    filtered: List[Dict[str, float]] = []
    for wp in result:
        cx = int(round((wp["x"] - origin_x) / resolution))
        ry = int(round((wp["y"] - origin_y) / resolution))
        r0, r1 = max(0, ry - check_radius), min(height, ry + check_radius + 1)
        c0, c1 = max(0, cx - check_radius), min(width, cx + check_radius + 1)
        window = arr[r0:r1, c0:c1]
        if not np.any((window >= 65) | (window == -1)):
            filtered.append(wp)
    if len(filtered) >= 4:
        result = filtered
    # else keep unfiltered (too few waypoints left)

    skel_px = int(np.sum(skel))
    print(f"[patrol_core] reachable={reachable.sum()} px  "
          f"skeleton={skel_px} px  "
          f"contour_path={len(ordered_world)} pts  "
          f"waypoints={len(result)}  "
          f"route_len={total:.1f} m")
    return result


def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    """Compute morphological skeleton using CROSS structuring element."""
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = np.zeros_like(mask, dtype=bool)
    temp = mask.astype(np.uint8)
    while cv2.countNonZero(temp) > 0:
        eroded = cv2.erode(temp, cross)
        opening = cv2.dilate(eroded, cross)
        skel = skel | ((temp > 0) & (opening == 0))
        temp = eroded
    return skel


def _find_nearest_safe(
    mask: np.ndarray, start_r: int, start_c: int
) -> Tuple[Optional[int], Optional[int]]:
    """BFS for nearest True cell in mask from (start_r, start_c)."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    q = deque([(start_r, start_c)])
    visited[start_r, start_c] = True
    while q:
        r, c = q.popleft()
        if mask[r, c]:
            return r, c
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                visited[nr, nc] = True
                q.append((nr, nc))
    return None, None


def _largest_outer_contour(
    mask: np.ndarray,
) -> Optional[np.ndarray]:
    """Find the largest outer contour of a binary mask.

    Returns an array of shape (N, 1, 2) — OpenCV contour format.
    """
    uint8 = mask.astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(uint8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


# ---------------------------------------------------------------------------
#  BFS  --  flood-fill for reachable-region detection
# ---------------------------------------------------------------------------

def _bfs_mask(mask: np.ndarray, start_r: int, start_c: int) -> np.ndarray:
    """Flood-fill from (start_r, start_c) over True cells in *mask*."""
    if not mask[start_r, start_c]:
        return np.zeros_like(mask)
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    queue = [(start_r, start_c)]
    visited[start_r, start_c] = True
    idx = 0
    while idx < len(queue):
        r, c = queue[idx]
        idx += 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))
    return visited


# ---------------------------------------------------------------------------
#  Person localisation  --  2D bbox + depth → map-frame (x, y)
# ---------------------------------------------------------------------------

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
    *tf_matrix*       Optional 4×4 homogeneous transform from camera-optical
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

    # read depth with 3×3 median fallback
    depth_mm = _read_depth(depth_16uc1, px, py)
    if depth_mm <= 0 or depth_mm > max_depth_m * 1000.0:
        return None

    # deproject  (pinhole model, camera-optical-frame: Z forward, X right, Y down)
    d_m = depth_mm / 1000.0
    x_cam = (px - cx) * d_m / fx
    y_cam = (py - cy) * d_m / fy
    z_cam = d_m

    if tf_matrix is not None:
        # 4×4 homogeneous transform: map_T_cam
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
    """Read depth in mm at (u, v), with 3×3 median fallback if center is 0."""
    d = int(depth[v, u])
    if d > 0:
        return d
    # 3×3 window
    h, w = depth.shape
    u1, v1 = max(0, u - 1), max(0, v - 1)
    u2, v2 = min(w - 1, u + 1), min(h - 1, v + 1)
    patch = depth[v1:v2 + 1, u1:u2 + 1]
    valid = patch[patch > 0]
    if len(valid) > 0:
        return int(np.median(valid))
    return 0


# ---------------------------------------------------------------------------
#  TF helper
# ---------------------------------------------------------------------------

def tf_to_matrix(transform) -> np.ndarray:
    """Convert a ``geometry_msgs.msg.Transform`` to a 4×4 homogeneous matrix."""
    q = transform.rotation
    # quaternion to rotation matrix (Hamilton convention)
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


# ---------------------------------------------------------------------------
#  Photo archival
# ---------------------------------------------------------------------------

PHOTOS_BASE = os.path.join(
    os.path.dirname(__file__), "static", "patrol_photos"
)
REPORTS_DIR = os.path.join(
    os.path.dirname(__file__), "static", "patrol_reports"
)


def ensure_patrol_dirs(patrol_id: str) -> str:
    """Create ``patrol_photos/<patrol_id>/`` and return its path."""
    d = os.path.join(PHOTOS_BASE, patrol_id)
    os.makedirs(d, exist_ok=True)
    return d


def save_patrol_photo(
    bgr_frame: np.ndarray,
    bbox: List[float],
    label: str,
    confidence: float,
    photos_dir: str,
    seq: int,
) -> Optional[str]:
    """Draw detection overlay on *bgr_frame* and save as JPEG.

    Returns the relative URL path or *None* on failure.
    """
    try:
        annotated = bgr_frame.copy()
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated, f"{label} {confidence:.2f}", (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
        )
        # timestamp watermark
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            annotated, ts, (8, annotated.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        fname = f"detection_{seq:04d}_{label}_{confidence:.2f}.jpg"
        path = os.path.join(photos_dir, fname)
        cv2.imwrite(path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return f"/patrol/photos/{os.path.basename(photos_dir)}/{fname}"
    except Exception:
        return None


def cleanup_old_patrol_sessions(max_keep: int = 10) -> None:
    """Remove oldest patrol photo sessions beyond *max_keep*."""
    if not os.path.isdir(PHOTOS_BASE):
        return
    sessions = sorted(
        d for d in os.listdir(PHOTOS_BASE)
        if os.path.isdir(os.path.join(PHOTOS_BASE, d))
    )
    while len(sessions) > max_keep:
        old = sessions.pop(0)
        import shutil
        shutil.rmtree(os.path.join(PHOTOS_BASE, old), ignore_errors=True)


# ---------------------------------------------------------------------------
#  Report generation
# ---------------------------------------------------------------------------

def generate_report(
    patrol_id: str,
    start_time: float,
    end_time: float,
    waypoints: List[Dict[str, float]],
    completed_indices: List[int],
    person_detections: List[Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, Any]:
    """Build a patrol report dict from the gathered data."""
    duration = end_time - start_time if end_time > start_time else 0.0
    total = len(waypoints)
    completed = len(completed_indices)
    missed = sorted(set(range(total)) - set(completed_indices))

    # detections per waypoint
    by_wp: Dict[int, int] = {}
    for d in person_detections:
        wp = d.get("waypoint_index", -1)
        by_wp[wp] = by_wp.get(wp, 0) + 1

    # unique locations (cluster within 1.5 m)
    positions = []
    for d in person_detections:
        mx = d.get("map_x")
        my = d.get("map_y")
        if mx is not None and my is not None:
            positions.append((mx, my))
    unique_locations = _cluster_positions(positions, 1.5)

    report: Dict[str, Any] = {
        "patrol_id": patrol_id,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
        "duration_seconds": round(duration, 1),
        "status": "completed" if not missed else "partial",
        "route_summary": {
            "total_waypoints": total,
            "completed_waypoints": completed,
            "missed_waypoints": missed,
        },
        "persons": person_detections,
        "summary": {
            "total_person_detections": len(person_detections),
            "unique_locations": unique_locations,
            "detections_by_waypoint": by_wp,
        },
        "warnings": warnings,
    }
    return report


def _cluster_positions(
    positions: List[Tuple[float, float]], radius: float
) -> int:
    """Count clusters of points within *radius* (simple greedy)."""
    if not positions:
        return 0
    pts = list(positions)
    clusters = 0
    while pts:
        ref = pts.pop(0)
        clusters += 1
        pts = [p for p in pts if math.hypot(p[0] - ref[0], p[1] - ref[1]) > radius]
    return clusters


def save_report_to_disk(report: Dict[str, Any], patrol_id: str) -> Optional[str]:
    """Write report JSON to ``patrol_reports/<patrol_id>.json``.

    Returns the file path or *None* on failure.
    """
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        path = os.path.join(REPORTS_DIR, f"{patrol_id}.json")
        with open(path, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return path
    except Exception:
        return None


def load_report(patrol_id: str) -> Optional[Dict[str, Any]]:
    """Load a saved report from disk."""
    path = os.path.join(REPORTS_DIR, f"{patrol_id}.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  Region segmentation — extract rooms/corridors from occupancy grid
# ---------------------------------------------------------------------------

def segment_map(
    map_data: List[int],
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    robot_x: float = 0.0,
    robot_y: float = 0.0,
    room_clearance_m: float = 0.7,
    min_room_area_m2: float = 2.0,
    robot_radius: float = 0.22,
) -> Dict[str, Any]:
    """Segment an occupancy grid into rooms and corridors.

    Algorithm:
      1. Distance transform on free space → each cell's clearance to obstacles.
      2. High-clearance cells (>= room_clearance_m) → room candidates.
      3. Connected-components on room candidates → individual rooms.
      4. Free cells with lower clearance → corridor candidates (narrow but traversable).
      5. For each region: extract polygon boundary (outer + holes), center, safe point.
      6. Detect doorways where rooms meet corridors (dilated room ∩ corridor).
      7. Determine which region the robot is currently in.

    Supports ring-shaped corridors (口字型) via hole detection in RETR_TREE contours.

    Returns a dict with keys:
      ``regions`` — list of region dicts
      ``robot_region`` — id of region containing the robot, or None
    """
    try:
        return _segment_map_impl(map_data, width, height, resolution,
                                 origin_x, origin_y, robot_x, robot_y,
                                 room_clearance_m, min_room_area_m2, robot_radius)
    except Exception:
        traceback.print_exc()
        raise


def _segment_map_impl(
    map_data, width, height, resolution, origin_x, origin_y,
    robot_x, robot_y, room_clearance_m, min_room_area_m2, robot_radius,
) -> Dict[str, Any]:
    """Internal implementation of segment_map with logging."""
    arr = np.asarray(map_data, dtype=np.int8).reshape(height, width)
    free_mask = arr == 0
    if not free_mask.any():
        return {"regions": [], "robot_region": None}

    # --- 1. distance transform on free space ---
    free_uint8 = free_mask.astype(np.uint8)
    dt = cv2.distanceTransform(free_uint8, cv2.DIST_L2, 5)
    dt_m = dt * resolution  # metres from nearest occupied/unknown cell
    print(f"[segment_map] dt range: {float(dt.min()):.2f}–{float(dt.max()):.2f} px, "
          f"resolution={resolution}, room_clearance={room_clearance_m}m")

    # --- 2. room mask: cells far from obstacles ---
    room_mask = dt_m > room_clearance_m
    print(f"[segment_map] room_mask: {np.sum(room_mask)}/{free_mask.sum()} free cells")

    # close small gaps (furniture gaps, open doorways up to 0.3 m)
    close_k = max(1, int(round(0.3 / resolution)))
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_k + 1, 2 * close_k + 1))
        room_mask = cv2.morphologyEx(room_mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)

    # --- 3. label room components, filter tiny ones ---
    min_px = max(4, int(min_room_area_m2 / (resolution * resolution)))
    num_rooms, room_labels = cv2.connectedComponents(room_mask.astype(np.uint8))

    valid_rooms = []
    room_masks = {}
    for i in range(1, num_rooms):
        m = (room_labels == i)
        if np.sum(m) >= min_px:
            valid_rooms.append(i)
            room_masks[i] = m

    # --- 4. corridor mask: free, not room, but traversable ---
    min_traversable = robot_radius * 0.4
    corr_raw = free_mask & (dt_m <= room_clearance_m) & (dt_m >= min_traversable)
    corr_clean = _clean_regions(corr_raw, min_px)

    if corr_clean.any():
        num_corr, corr_labels = cv2.connectedComponents(corr_clean.astype(np.uint8))
    else:
        num_corr = 0

    valid_corrs = []
    corr_masks = {}
    for i in range(1, num_corr):
        m = (corr_labels == i)
        if np.sum(m) >= min_px:
            valid_corrs.append(i)
            corr_masks[i] = m

    # If no separate corridor, treat remaining free space as one
    if not valid_corrs and valid_rooms:
        all_room = np.zeros_like(free_mask)
        for rm in room_masks.values():
            all_room |= rm
        remaining = free_mask & ~all_room
        remaining = _clean_regions(remaining, min_px)
        if remaining.any():
            corr_masks[-1] = remaining
            valid_corrs.append(-1)

    # --- 5. build region dicts ---
    regions = []
    all_masks: Dict[str, np.ndarray] = {}

    for rid in valid_rooms:
        m = room_masks[rid]
        reg = _build_room(rid, m, dt, arr, width, height, resolution, origin_x, origin_y)
        rid_str = reg["id"]
        regions.append(reg)
        all_masks[rid_str] = m

    for cid in valid_corrs:
        m = corr_masks[cid]
        reg = _build_corridor(cid, m, width, height, resolution, origin_x, origin_y)
        rid_str = reg["id"]
        regions.append(reg)
        all_masks[rid_str] = m

    # --- 6. detect doorways between rooms and corridors ---
    rooms = [r for r in regions if r["type"] == "room"]
    corrs = [r for r in regions if r["type"] == "corridor"]

    for rm in rooms:
        rm_mask = all_masks[rm["id"]]
        doorways = []
        for cr in corrs:
            cr_mask = all_masks[cr["id"]]
            pts = _find_doorways(rm_mask, cr_mask, resolution, width, height,
                                  origin_x, origin_y)
            for p in pts:
                p["connects_to"] = cr["id"]
                doorways.append(p)

            # also register on corridor side
            existing = {e.get("connects_to") for e in cr.get("entries", [])}
            if rm["id"] not in existing:
                for p in pts:
                    p_copy = dict(p, connects_to=rm["id"])
                    cr.setdefault("entries", []).append(p_copy)

        rm["entries"] = doorways
        rm["connects_to"] = sorted({e["connects_to"] for e in doorways})

    for cr in corrs:
        cr.setdefault("entries", [])
        cr.setdefault("connects_to", [])

    # --- 7. determine robot's current region ---
    robot_region = None
    rx = int(round((robot_x - origin_x) / resolution))
    ry = int(round((robot_y - origin_y) / resolution))
    if 0 <= rx < width and 0 <= ry < height:
        for reg in regions:
            m = all_masks[reg["id"]]
            if 0 <= ry < m.shape[0] and 0 <= rx < m.shape[1] and m[ry, rx]:
                robot_region = reg["id"]
                break

    return {"regions": regions, "robot_region": robot_region}


def _region_contours(mask: np.ndarray):
    """Extract outer contour and hole contours from binary mask using RETR_TREE.

    Returns ``(outer, holes)`` where *outer* is ``(N, 2)`` pixel coords
    and *holes* is a list of ``(M, 2)`` arrays.  Both may be None/empty.
    """
    uint8 = mask.astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None:
        return None, []
    h = hierarchy[0]

    # outer = largest-area contour with no parent
    parent_map: Dict[int, int] = {}
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 4:
            continue
        parent = int(h[i][3])
        if parent == -1:
            parent_map.setdefault(-1, []).append(i)
        else:
            parent_map.setdefault(parent, []).append(i)

    # If nothing found, fall back to largest overall
    if -1 not in parent_map:
        areas = [(cv2.contourArea(c), i) for i, c in enumerate(contours)]
        if not areas:
            return None, []
        _, outer_idx = max(areas)
        outer = contours[outer_idx].squeeze(1)
        return outer, []

    # Pick the outer contour with largest area
    outer_idx = max(parent_map[-1], key=lambda i: cv2.contourArea(contours[i]))
    outer = contours[outer_idx].squeeze(1)

    # Children of the outer contour = holes
    hole_idxs = parent_map.get(outer_idx, [])
    holes = [contours[i].squeeze(1) for i in hole_idxs if cv2.contourArea(contours[i]) >= 4]

    return outer, holes


def _simplify_contour_px(contour: np.ndarray, resolution: float) -> np.ndarray:
    """Douglas-Peucker simplification; epsilon ≈ 2 pixels."""
    if contour is None or len(contour) < 3:
        return contour
    eps = max(1.0, 2.0 / resolution)  # 2 cells at most
    simplified = cv2.approxPolyDP(contour, eps, True)
    # approxPolyDP returns (N, 1, 2); squeeze to (N, 2) for _contour_to_world
    return simplified.squeeze(1) if simplified.ndim == 3 else simplified


def _contour_to_world(contour: np.ndarray, origin_x: float, origin_y: float,
                       resolution: float) -> List[Dict[str, float]]:
    """Convert ``(N, 2)`` pixel coords to world-coordinate dict list."""
    if contour is None or len(contour) < 2:
        return []
    return [
        {"x": round(origin_x + float(p[0]) * resolution, 3),
         "y": round(origin_y + float(p[1]) * resolution, 3)}
        for p in contour
    ]


def _mask_centroid(mask: np.ndarray):
    """Return ``(row, col)`` of the mask centroid."""
    mom = cv2.moments(mask.astype(np.uint8))
    if mom["m00"] == 0:
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return 0, 0
        return int(np.mean(ys)), int(np.mean(xs))
    return int(mom["m01"] / mom["m00"]), int(mom["m10"] / mom["m00"])


def _clean_regions(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Remove connected components smaller than *min_area_px*."""
    num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    out = np.zeros_like(mask, dtype=bool)
    for i in range(1, num):
        if np.sum(labels == i) >= min_area_px:
            out |= (labels == i)
    return out


def _build_room(comp_label: int, mask: np.ndarray, dt: np.ndarray,
                occ: np.ndarray, width: int, height: int,
                resolution: float, origin_x: float, origin_y: float) -> Dict[str, Any]:
    """Build a region dict for a room component."""
    rid = f"room_{comp_label}"
    outer, holes = _region_contours(mask)
    outer_s = _simplify_contour_px(outer, resolution) if outer is not None else None
    holes_s = [_simplify_contour_px(h, resolution) for h in holes if len(h) >= 3]

    boundary = _contour_to_world(outer_s, origin_x, origin_y, resolution)
    hole_list = [_contour_to_world(h, origin_x, origin_y, resolution) for h in holes_s]

    cy, cx = _mask_centroid(mask)
    center = {"x": round(origin_x + cx * resolution, 3),
              "y": round(origin_y + cy * resolution, 3)}

    # safe point = cell furthest from obstacles within this room
    room_dt = dt.copy()
    room_dt[~mask] = 0
    max_loc = np.unravel_index(room_dt.argmax(), room_dt.shape)
    safe = {"x": round(origin_x + max_loc[1] * resolution, 3),
            "y": round(origin_y + max_loc[0] * resolution, 3)}

    area_sqm = round(float(np.sum(mask) * resolution * resolution), 2)

    return {
        "id": rid,
        "type": "room",
        "label": rid,
        "area_sqm": area_sqm,
        "boundary": boundary,
        "holes": hole_list,
        "nav_targets": {"center": center, "safe": safe},
        "entries": [],
        "connects_to": [],
    }


def _build_corridor(comp_label: int, mask: np.ndarray,
                     width: int, height: int,
                     resolution: float, origin_x: float,
                     origin_y: float) -> Dict[str, Any]:
    """Build a region dict for a corridor component."""
    rid = f"corridor_{comp_label}"
    outer, holes = _region_contours(mask)
    outer_s = _simplify_contour_px(outer, resolution) if outer is not None else None
    holes_s = [_simplify_contour_px(h, resolution) for h in holes if len(h) >= 3]

    boundary = _contour_to_world(outer_s, origin_x, origin_y, resolution)
    hole_list = [_contour_to_world(h, origin_x, origin_y, resolution) for h in holes_s]

    cy, cx = _mask_centroid(mask)
    center = {"x": round(origin_x + cx * resolution, 3),
              "y": round(origin_y + cy * resolution, 3)}

    area_sqm = round(float(np.sum(mask) * resolution * resolution), 2)

    return {
        "id": rid,
        "type": "corridor",
        "label": rid,
        "area_sqm": area_sqm,
        "boundary": boundary,
        "holes": hole_list,
        "nav_targets": {"center": center},
        "entries": [],
        "connects_to": [],
    }


def _find_doorways(room_mask: np.ndarray, corr_mask: np.ndarray,
                    resolution: float, width: int, height: int,
                    origin_x: float, origin_y: float) -> List[Dict[str, float]]:
    """Find doorway points between a room and a corridor.

    Dilates the room by 0.15 m, takes the intersection with the corridor,
    then clusters the overlap pixels.  Each cluster centroid is an entry point.
    """
    dk = max(1, int(round(0.15 / resolution)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dk + 1, 2 * dk + 1))
    dilated = cv2.dilate(room_mask.astype(np.uint8), k).astype(bool)

    contact = dilated & corr_mask
    if not contact.any():
        return []

    num_comps, labels = cv2.connectedComponents(contact.astype(np.uint8))
    results = []
    for i in range(1, num_comps):
        zone = (labels == i)
        if np.sum(zone) < 3:
            continue
        cy, cx = _mask_centroid(zone)
        results.append({
            "x": round(origin_x + cx * resolution, 3),
            "y": round(origin_y + cy * resolution, 3),
        })
    return results


# ---------------------------------------------------------------------------
#  YAML persistence for semantic regions
# ---------------------------------------------------------------------------

def save_semantic_map_yaml(regions: List[Dict[str, Any]],
                            filepath: str) -> bool:
    """Save region data + legacy location entries to a YAML file.

    The output includes three top-level sections:
      * ``regions``           — full polygon/hole data for the Dashboard
      * ``semantic_locations`` — point entries for ``region_center``,
                                ``region_safe``, ``region_entry`` (Nav2
                                backward-compatibility)
      * ``semantic_alias``    — auto-generated alias lists
    """
    try:
        import yaml
    except ImportError:
        return False

    sem_locations: Dict[str, Dict[str, Any]] = {}
    sem_alias: Dict[str, List[str]] = {}
    regions_out: List[Dict[str, Any]] = []

    for reg in regions:
        # strip raw numpy arrays; keep only serializable data
        region_entry = {
            "id": reg["id"],
            "label": reg["label"],
            "type": reg["type"],
            "area_sqm": reg.get("area_sqm", 0),
            "boundary": reg.get("boundary", []),
            "holes": reg.get("holes", []),
            "nav_targets": reg.get("nav_targets", {}),
            "entries": reg.get("entries", []),
            "connects_to": reg.get("connects_to", []),
        }
        regions_out.append(region_entry)

        label = reg["label"]

        # store nav_targets as legacy semantic_locations
        for target_name, pt in reg.get("nav_targets", {}).items():
            key = label if target_name == "center" else f"{label}_{target_name}"
            sem_locations[key] = {
                "x": pt["x"], "y": pt["y"],
                "yaw": 0.0, "frame_id": "map",
            }

        # store entry points
        for idx, e in enumerate(reg.get("entries", [])):
            e_key = f"{label}_entry"
            if idx > 0:
                e_key = f"{label}_entry_{idx + 1}"
            sem_locations[e_key] = {
                "x": e["x"], "y": e["y"],
                "yaw": 0.0, "frame_id": "map",
            }

        # aliases (auto-generated; user may edit later)
        sem_alias[label] = [label, f"去{label}", f"到{label}"]

    data: Dict[str, Any] = {
        "regions": regions_out,
        "semantic_locations": sem_locations,
        "semantic_alias": sem_alias,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True,
                  default_flow_style=None, sort_keys=False)

    return True


def load_semantic_map_yaml(filepath: str) -> Optional[Dict[str, Any]]:
    """Load a previously saved semantic regions YAML."""
    try:
        import yaml
    except ImportError:
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def list_saved_regions(dirpath: str) -> List[Dict[str, Any]]:
    """List all semantic region YAML files in *dirpath*.

    Returns dicts with ``filepath``, ``filename``, ``region_count``.
    """
    results = []
    if not os.path.isdir(dirpath):
        return results
    for fn in sorted(os.listdir(dirpath)):
        if fn.endswith((".yaml", ".yml")):
            fp = os.path.join(dirpath, fn)
            try:
                import yaml
                with open(fp) as f:
                    data = yaml.safe_load(f) or {}
                rglen = len(data.get("regions") or [])
                results.append({
                    "filepath": fp,
                    "filename": fn,
                    "region_count": rglen,
                })
            except Exception:
                results.append({"filepath": fp, "filename": fn, "region_count": 0})
    return results


def list_reports() -> List[Dict[str, Any]]:
    """List all saved patrol reports (id + start_time)."""
    if not os.path.isdir(REPORTS_DIR):
        return []
    reports = []
    for fn in sorted(os.listdir(REPORTS_DIR)):
        if fn.endswith(".json"):
            path = os.path.join(REPORTS_DIR, fn)
            try:
                with open(path) as f:
                    data = json.load(f)
                reports.append({
                    "patrol_id": data.get("patrol_id", fn[:-5]),
                    "start_time": data.get("start_time", ""),
                    "status": data.get("status", ""),
                    "total_persons": data.get("summary", {}).get("total_person_detections", 0),
                })
            except Exception:
                pass
    return list(reversed(reports))
