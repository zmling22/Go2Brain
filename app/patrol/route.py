"""
Patrol route generation -- contour-based medial-axis patrol.
"""
import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


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
      1. Parse occupancy grid -> free mask, close small SLAM gaps.
      2. Erode by robot_radius -> safe mask.
      3. BFS reachable region from robot.
      4. Morphological skeleton of the reachable region.
      5. Dilate skeleton to bridge gaps -> largest corridor network.
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

    # --- 4. erode by robot_radius for safe navigation ---
    safe_cells = max(1, int(round(robot_radius / resolution)))
    sk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (2 * safe_cells + 1, 2 * safe_cells + 1))
    safe_mask = cv2.erode(closed, sk).astype(bool)

    # --- 5. BFS reachable region from robot ---
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
    free_for_dt = (closed > 0).astype(np.uint8)
    dt_free = cv2.distanceTransform(free_for_dt, cv2.DIST_L2, 5)
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
    ctr_pts = ctr.squeeze(axis=1)
    world = [(origin_x + c * resolution, origin_y + r * resolution)
             for (c, r) in ctr_pts]
    dists = [math.hypot(x - robot_x, y - robot_y) for (x, y) in world]
    ni = int(np.argmin(dists))
    ordered_world = world[ni:] + world[:ni]
    ordered_world.append(ordered_world[0])

    if not ordered_world:
        return []

    # --- 12. sample waypoints evenly along the path ---
    cum = [0.0]
    for i in range(1, len(ordered_world)):
        dx = ordered_world[i][0] - ordered_world[i - 1][0]
        dy = ordered_world[i][1] - ordered_world[i - 1][1]
        cum.append(cum[-1] + math.hypot(dx, dy))
    total = cum[-1]

    if total < 0.5:
        return [{"x": round(robot_x, 3), "y": round(robot_y, 3)}]

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
    check_radius = max(1, int(round(0.5 / resolution)))
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


def _largest_outer_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Find the largest outer contour of a binary mask."""
    uint8 = mask.astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(uint8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


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
