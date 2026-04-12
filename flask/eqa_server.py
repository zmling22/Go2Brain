from flask import Flask, request, jsonify
import threading
import time
import base64
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from rcl_interfaces.msg import Log  # 订阅 /rosout
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

app = Flask(__name__)

# --------- 共享状态 ---------
@dataclass
class SharedState:
    arrived: bool = False
    arrived_area: Optional[str] = None
    last_nav_log: str = ""
    last_target_text: str = ""

    last_image_jpeg: Optional[bytes] = None
    last_image_stamp: float = 0.0

state = SharedState()
lock = threading.Lock()

# 你可以把这个表拷到机器狗端，用于从“去客厅/去厨房”推断 arrived_area
AREA_SYNONYMS = {
    "living_room": ["客厅", "到客厅", "去客厅", "回客厅"],
    "kitchen": ["厨房", "到厨房", "去厨房"],
    "dock": ["dock", "停靠站", "对接站"],
    "office": ["办公室", "办公区", "到办公室"],
    "toilet": ["厕所", "洗手间", "卫生间", "廁所"],
    "bedroom": ["卧室", "睡觉的地方", "休息区", "休息區", "臥室"],
    "diningroom": ["餐厅", "吃饭的地方", "用餐区"],
    "door": ["门口", "大门", "出入口"],
}

def infer_area_from_text(text: str) -> Optional[str]:
    t = (text or "").strip()
    for area, syns in AREA_SYNONYMS.items():
        for w in syns:
            if w and w in t:
                return area
    return None


class NLCommandPublisher(Node):
    def __init__(self):
        super().__init__('nl_command_http_bridge')

        # 你现有的发布：保持不变
        self.pub = self.create_publisher(String, '/nl_command', 10)

        # 订阅 rosout：用日志判断导航是否成功
        self.rosout_sub = self.create_subscription(Log, '/rosout', self.on_rosout, 50)

        # 订阅相机：用你的真实 topic
        self.bridge = CvBridge()
        self.img_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.on_image,
            10
        )

    def publish_text(self, text: str):
        msg = String()
        msg.data = text
        self.pub.publish(msg)
        self.get_logger().info(f"Published to /nl_command: {text}")

        with lock:
            state.last_target_text = text
            state.arrived = False
            # 这里提前把“目标区域”记录下来（到达时直接返回这个 area）
            state.arrived_area = infer_area_from_text(text)
            state.last_nav_log = ""

    def on_rosout(self, msg: Log):
        # 只关心导航相关日志，避免误触发
        name = msg.name or ""
        text = msg.msg or ""

        # 你日志里出现的关键字（匹配任意一个就认为到达）
        hit = False

        # 1) 你的 semantic_navigator 成功日志（最准确）
        if "semantic_navigator" in name and ("导航成功到达目标点" in text or "导航成功" in text):
            hit = True

        # 2) Nav2 控制器/BT 的成功日志（也可用作兜底）
        if ("controller_server" in name and "Reached the goal" in text) or \
           ("bt_navigator" in name and "Navigation succeeded" in text):
            hit = True

        if hit:
            with lock:
                state.arrived = True
                state.last_nav_log = f"[{name}] {text}"
        else:
            # 可选：记录最近一条导航中的日志，用于 /status 展示
            if "semantic_navigator" in name and ("导航中" in text or "距离目标点剩余" in text):
                with lock:
                    state.last_nav_log = f"[{name}] {text}"

    def on_image(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            ok, jpg = cv2.imencode(".jpg", cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                return
            with lock:
                state.last_image_jpeg = jpg.tobytes()
                state.last_image_stamp = time.time()
        except Exception as e:
            self.get_logger().warn(f"on_image failed: {e}")


# --------- ROS2 spin thread ---------
rclpy.init()
ros_node = NLCommandPublisher()

def ros_spin_thread():
    rclpy.spin(ros_node)

threading.Thread(target=ros_spin_thread, daemon=True).start()


# --------- HTTP APIs ---------
@app.post("/nl_command")
def nl_command():
    data = request.get_json(force=True, silent=True) or {}
    target_text = (data.get("target_text") or "").strip()
    raw_text = (data.get("raw_text") or "").strip()

    if not target_text:
        return jsonify({"ok": False, "error": "target_text required", "raw_text": raw_text}), 400

    ros_node.publish_text(target_text)
    return jsonify({"ok": True, "published": target_text, "raw_text": raw_text})


@app.get("/status")
def status():
    with lock:
        return jsonify({
            "ok": True,
            "arrived": state.arrived,
            "arrived_area": state.arrived_area,  # 来自“命令推断”，足够给主机做匹配
            "last_nav_log": state.last_nav_log,
            "last_target_text": state.last_target_text,
            "has_image": state.last_image_jpeg is not None,
            "image_stamp": state.last_image_stamp,
        })


@app.post("/capture")
def capture():
    with lock:
        jpg = state.last_image_jpeg
        stamp = state.last_image_stamp
        arrived = state.arrived
        arrived_area = state.arrived_area

    if jpg is None:
        return jsonify({"ok": False, "error": "no camera frame received yet"}), 503

    return jsonify({
        "ok": True,
        "arrived": arrived,
        "arrived_area": arrived_area,
        "stamp": stamp,
        "jpeg_base64": base64.b64encode(jpg).decode("utf-8"),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
