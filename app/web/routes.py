"""
Flask routes for the dashboard server.
"""
import os
import time
from typing import Optional

from flask import Flask, Response, jsonify, render_template, request

from utils import APP_ROOT
from utils.state import state, lock, camera_condition, quaternion_to_yaw
from patrol.report import load_report, list_reports
from patrol.rules import create_rule
from audio_sdk import player as voice_player
from utils.map_analysis import save_semantic_map_yaml, load_semantic_map_yaml

# Set after bridge creation in dashboard_launch.py
bridge: Optional['DashboardBridge'] = None

app = Flask(__name__,
    template_folder=os.path.join(APP_ROOT, "templates"),
    static_folder=os.path.join(APP_ROOT, "static"),
)


# ---- page ----

@app.get("/")
def dashboard_index():
    return render_template("dashboard.html")


# ---- status / data APIs ----

@app.get("/api/status")
def api_status():
    with lock:
        response = {
            "ok": True,
            "pose": state.pose,
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
            "camera_fps": state.camera_fps,
            "map_fps": state.map_fps,
            "region_count": len(state.regions),
            "robot_region": state.robot_region,
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

    client_seq = request.args.get("seq", type=int)
    if client_seq is not None and client_seq == map_seq:
        return jsonify({"ok": True, "seq": map_seq, "changed": False})

    return jsonify({
        "ok": True,
        "seq": map_seq,
        "changed": True,
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
    })


@app.get("/api/overlay")
def api_overlay():
    with lock:
        return jsonify({
            "ok": True,
            "pose": state.pose,
            "trajectory": state.trajectory[-200:] if len(state.trajectory) > 200 else state.trajectory,
            "plan": state.plan,
            "current_route": state.current_route,
            "current_waypoint_index": state.current_waypoint_index,
            "route_running": state.route_running,
        })


@app.get("/api/pose")
def api_pose():
    """Lightweight pose-only endpoint for similar to 200ms polling."""
    with lock:
        return jsonify({
            "ok": True,
            "pose": state.pose,
            "speed": state.speed,
            "updated_at": state.updated_at,
        })


# ---- camera APIs ----

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
            yield header + camera_jpeg + b"\r\n"

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        },
    )


# ---- command / route APIs ----

@app.post("/api/command")
def api_command():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "text required"}), 400

    bridge.publish_text_command(text)
    return jsonify({"ok": True, "message": f"published: {text}"})


@app.post("/api/route")
def api_route():
    data = request.get_json(force=True, silent=True) or {}
    waypoints = data.get("waypoints") or []
    try:
        result = bridge.send_route(waypoints)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    if not result.get("ok"):
        return jsonify(result), 400
    return jsonify(result)


@app.post("/api/route/cancel")
def api_route_cancel():
    result = bridge.cancel_route()
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


# ---- region segmentation APIs ----

@app.post("/api/map/segment")
def api_map_segment():
    data = request.get_json(force=True, silent=True) or {}
    try:
        result = bridge.segment_map(
            room_clearance_m=float(data.get("room_clearance_m", 0.7)),
            min_room_area_m2=float(data.get("min_room_area_m2", 2.0)),
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 500
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.get("/api/map/regions")
def api_map_regions():
    with lock:
        return jsonify({
            "ok": True,
            "regions": state.regions,
            "robot_region": state.robot_region,
            "region_count": len(state.regions),
        })


@app.post("/api/map/regions/rename")
def api_map_region_rename():
    data = request.get_json(force=True, silent=True) or {}
    region_id = (data.get("id") or "").strip()
    new_label = (data.get("label") or "").strip()
    if not region_id or not new_label:
        return jsonify({"ok": False, "error": "id and label required"}), 400

    with lock:
        found = False
        for reg in state.regions:
            if reg["id"] == region_id:
                reg["label"] = new_label
                found = True
                break
    if not found:
        return jsonify({"ok": False, "error": f"region '{region_id}' not found"}), 404
    return jsonify({"ok": True, "region_id": region_id, "label": new_label})


@app.post("/api/map/regions/save")
def api_map_regions_save():
    data = request.get_json(force=True, silent=True) or {}
    filepath = (data.get("filepath") or "").strip()

    if not filepath:
        script_dir = APP_ROOT
        filepath = os.path.join(script_dir, "semantic_regions.yaml")

    with lock:
        regions_copy = list(state.regions)

    ok = save_semantic_map_yaml(regions_copy, filepath)
    if not ok:
        return jsonify({"ok": False, "error": "save failed (missing yaml module?)"}), 500

    with lock:
        state.region_yaml_path = filepath
    return jsonify({"ok": True, "filepath": filepath, "region_count": len(regions_copy)})


@app.get("/api/map/regions/load")
def api_map_regions_load():
    filepath = request.args.get("filepath", "").strip()
    if not filepath:
        script_dir = APP_ROOT
        filepath = os.path.join(script_dir, "semantic_regions.yaml")

    data = load_semantic_map_yaml(filepath)
    if data is None:
        return jsonify({"ok": False, "error": "load failed or file not found"}), 404

    with lock:
        state.regions = data.get("regions", [])
        state.region_yaml_path = filepath
    return jsonify({"ok": True, "regions": state.regions,
                    "region_count": len(state.regions)})


@app.post("/api/map/regions/clear")
def api_map_regions_clear():
    with lock:
        state.regions = []
        state.robot_region = None
    return jsonify({"ok": True})


# ---- patrol APIs ----

@app.post("/api/patrol/generate")
def api_patrol_generate():
    try:
        result = bridge.generate_patrol_route()
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.post("/api/patrol/start")
def api_patrol_start():
    try:
        result = bridge.start_patrol()
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.post("/api/patrol/stop")
def api_patrol_stop():
    try:
        result = bridge.stop_patrol()
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify(result)


# ---- inspection rule APIs ----

@app.get("/api/patrol/rules")
def api_patrol_rules():
    with lock:
        return jsonify({
            "ok": True,
            "rules": list(state.patrol_rules),
        })


@app.post("/api/patrol/rules")
def api_patrol_rule_create():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "rule text required"}), 400

    rule = create_rule(text)
    with lock:
        state.patrol_rules.append(rule)
        state.updated_at = time.time()
    return jsonify({"ok": True, "rule": rule})


@app.delete("/api/patrol/rules/<rule_id>")
def api_patrol_rule_delete(rule_id: str):
    with lock:
        before = len(state.patrol_rules)
        state.patrol_rules = [r for r in state.patrol_rules if r["id"] != rule_id]
        if len(state.patrol_rules) == before:
            return jsonify({"ok": False, "error": "rule not found"}), 404
        state.updated_at = time.time()
    return jsonify({"ok": True})


@app.post("/api/patrol/rules/<rule_id>/toggle")
def api_patrol_rule_toggle(rule_id: str):
    data = request.get_json(force=True, silent=True) or {}
    enabled = data.get("enabled", None)
    with lock:
        for rule in state.patrol_rules:
            if rule["id"] == rule_id:
                if enabled is None:
                    rule["enabled"] = not rule["enabled"]
                else:
                    rule["enabled"] = bool(enabled)
                state.updated_at = time.time()
                return jsonify({"ok": True, "rule": rule})
    return jsonify({"ok": False, "error": "rule not found"}), 404


@app.get("/api/patrol/violations")
def api_patrol_violations():
    with lock:
        return jsonify({
            "ok": True,
            "violations": list(state.patrol_violations),
            "count": len(state.patrol_violations),
        })


@app.get("/api/patrol/voice/<rule_id>")
def api_patrol_voice(rule_id: str):
    with lock:
        rule = next((r for r in state.patrol_rules if r["id"] == rule_id), None)
    if rule is None:
        return jsonify({"ok": False, "error": "rule not found"}), 404
    url = voice_player.generate_mp3_to_static(rule["text"])
    if url is None:
        return jsonify({"ok": False, "error": "TTS generation failed"}), 500
    return jsonify({"ok": True, "url": url})


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
            "violations": list(state.patrol_violations),
            "violation_count": len(state.patrol_violations),
            "rule_count": len(state.patrol_rules),
        })


@app.get("/api/patrol/report/<patrol_id>")
def api_patrol_report(patrol_id: str):
    report = load_report(patrol_id)
    if report is None:
        return jsonify({"ok": False, "error": "report not found"}), 404
    return jsonify({"ok": True, "report": report})


@app.get("/api/patrol/reports")
def api_patrol_reports():
    return jsonify({"ok": True, "reports": list_reports()})


@app.route("/patrol/photos/<path:filename>")
def api_patrol_photos(filename: str):
    base = os.path.join(APP_ROOT, "static", "patrol_photos")
    return Response(
        open(os.path.join(base, filename), "rb").read(),
        mimetype="image/jpeg",
    )
