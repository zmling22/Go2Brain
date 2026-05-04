"""
Report generation -- patrol session reports.
"""
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple


from utils import APP_ROOT

REPORTS_DIR = os.path.join(APP_ROOT, "static", "patrol_reports")


def generate_report(
    patrol_id: str,
    start_time: float,
    end_time: float,
    waypoints: List[Dict[str, float]],
    completed_indices: List[int],
    person_detections: List[Dict[str, Any]],
    warnings: List[str],
    violations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a patrol report dict from the gathered data."""
    duration = end_time - start_time if end_time > start_time else 0.0
    total = len(waypoints)
    completed = len(completed_indices)
    missed = sorted(set(range(total)) - set(completed_indices))

    by_wp: Dict[int, int] = {}
    for d in person_detections:
        wp = d.get("waypoint_index", -1)
        by_wp[wp] = by_wp.get(wp, 0) + 1

    positions = []
    for d in person_detections:
        mx = d.get("map_x")
        my = d.get("map_y")
        if mx is not None and my is not None:
            positions.append((mx, my))
    unique_locations = _cluster_positions(positions, 1.5)

    violations = violations or []
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
            "total_violations": len(violations),
        },
        "warnings": warnings,
        "violations": violations,
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
    """Write report JSON to ``patrol_reports/<patrol_id>.json``."""
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
