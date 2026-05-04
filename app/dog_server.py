from flask import Flask, request, jsonify
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

app = Flask(__name__)

class NLCommandPublisher(Node):
    def __init__(self):
        super().__init__('nl_command_http_bridge')
        self.pub = self.create_publisher(String, '/nl_command', 10)

    def publish_text(self, text: str):
        msg = String()
        msg.data = text
        self.pub.publish(msg)
        self.get_logger().info(f"Published to /nl_command: {text}")

rclpy.init()
ros_node = NLCommandPublisher()

def ros_spin_thread():
    rclpy.spin(ros_node)

threading.Thread(target=ros_spin_thread, daemon=True).start()

@app.post("/nl_command")
def nl_command():
    data = request.get_json(force=True, silent=True) or {}
    target_text = (data.get("target_text") or "").strip()
    raw_text = (data.get("raw_text") or "").strip()

    if not target_text:
        return jsonify({"ok": False, "error": "target_text required", "raw_text": raw_text}), 400

    # 这里你可以选择发布 target_text，或发布更结构化的 JSON 字符串
    ros_node.publish_text(target_text)
    return jsonify({"ok": True, "published": target_text})

if __name__ == "__main__":
    # host=0.0.0.0 让局域网可访问
    app.run(host="0.0.0.0", port=5000, debug=False)
