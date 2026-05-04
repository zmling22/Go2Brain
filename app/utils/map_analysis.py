"""
Region segmentation -- extract rooms/corridors from occupancy grid.
"""
import os
import traceback
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


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
      1. Distance transform on free space -> each cell's clearance to obstacles.
      2. High-clearance cells (>= room_clearance_m) -> room candidates.
      3. Connected-components on room candidates -> individual rooms.
      4. Free cells with lower clearance -> corridor candidates (narrow but traversable).
      5. For each region: extract polygon boundary (outer + holes), center, safe point.
      6. Detect doorways where rooms meet corridors (dilated room & corridor).
      7. Determine which region the robot is currently in.

    Returns a dict with keys ``regions`` and ``robot_region``.
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
    dt_m = dt * resolution
    print(f"[segment_map] dt range: {float(dt.min()):.2f}-{float(dt.max()):.2f} px, "
          f"resolution={resolution}, room_clearance={room_clearance_m}m")

    # --- 2. room mask: cells far from obstacles ---
    room_mask = dt_m > room_clearance_m
    print(f"[segment_map] room_mask: {np.sum(room_mask)}/{free_mask.sum()} free cells")

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
        regions.append(reg)
        all_masks[reg["id"]] = m

    for cid in valid_corrs:
        m = corr_masks[cid]
        reg = _build_corridor(cid, m, width, height, resolution, origin_x, origin_y)
        regions.append(reg)
        all_masks[reg["id"]] = m

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

    parent_map: Dict[int, List[int]] = {}
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 4:
            continue
        parent = int(h[i][3])
        if parent == -1:
            parent_map.setdefault(-1, []).append(i)
        else:
            parent_map.setdefault(parent, []).append(i)

    if -1 not in parent_map:
        areas = [(cv2.contourArea(c), i) for i, c in enumerate(contours)]
        if not areas:
            return None, []
        _, outer_idx = max(areas)
        outer = contours[outer_idx].squeeze(1)
        return outer, []

    outer_idx = max(parent_map[-1], key=lambda i: cv2.contourArea(contours[i]))
    outer = contours[outer_idx].squeeze(1)

    hole_idxs = parent_map.get(outer_idx, [])
    holes = [contours[i].squeeze(1) for i in hole_idxs if cv2.contourArea(contours[i]) >= 4]

    return outer, holes


def _simplify_contour_px(contour: np.ndarray, resolution: float) -> np.ndarray:
    """Douglas-Peucker simplification; epsilon approx 2 pixels."""
    if contour is None or len(contour) < 3:
        return contour
    eps = max(1.0, 2.0 / resolution)
    simplified = cv2.approxPolyDP(contour, eps, True)
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

    Dilates the room by 0.15 m, takes the intersection with the corridor,
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
      * ``regions``           -- full polygon/hole data for the Dashboard
      * ``semantic_locations`` -- point entries for Nav2 backward-compatibility
      * ``semantic_alias``    -- auto-generated alias lists
    """
    try:
        import yaml
    except ImportError:
        return False

    sem_locations: Dict[str, Dict[str, Any]] = {}
    sem_alias: Dict[str, List[str]] = {}
    regions_out: List[Dict[str, Any]] = []

    for reg in regions:
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

        for target_name, pt in reg.get("nav_targets", {}).items():
            key = label if target_name == "center" else f"{label}_{target_name}"
            sem_locations[key] = {
                "x": pt["x"], "y": pt["y"],
                "yaw": 0.0, "frame_id": "map",
            }

        for idx, e in enumerate(reg.get("entries", [])):
            e_key = f"{label}_entry"
            if idx > 0:
                e_key = f"{label}_entry_{idx + 1}"
            sem_locations[e_key] = {
                "x": e["x"], "y": e["y"],
                "yaw": 0.0, "frame_id": "map",
            }

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
    """List all semantic region YAML files in *dirpath*."""
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
