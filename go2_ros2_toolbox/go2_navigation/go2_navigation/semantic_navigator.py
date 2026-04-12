#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import yaml

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
from ament_index_python.packages import get_package_share_directory


class SemanticNavigator(Node):
    def __init__(self):
        super().__init__('semantic_navigator')

        # ========== 读取语义地图配置 ==========
        # 优先用参数 semantic_map_path；如果没传，就用本包 config/semantic_map.yaml
        self.declare_parameter('semantic_map_path', '')
        map_path = self.get_parameter('semantic_map_path').get_parameter_value().string_value

        if not map_path:
            try:
                pkg_share = get_package_share_directory('go2_navigation')
                map_path = os.path.join(pkg_share, 'config', 'semantic_map.yaml')
            except Exception as e:
                self.get_logger().error(f'获取 go2_navigation 包路径失败: {e}')
                map_path = ''

        self.semantic_locations = {}
        self.semantic_alias = {}

        if map_path and os.path.exists(map_path):
            try:
                with open(map_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                self.semantic_locations = data.get('semantic_locations', {})
                self.semantic_alias = data.get('semantic_alias', {})
                self.get_logger().info(
                    f'语义地图加载成功: {map_path}, 位置数量: {len(self.semantic_locations)}'
                )
            except Exception as e:
                self.get_logger().error(f'加载语义地图失败: {e}')
        else:
            self.get_logger().warn(f'语义地图路径无效或不存在: "{map_path}"')

        # ========== Nav2 Action Client（完全复用 tcpbridge 的风格） ==========
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # 重试相关（复用你原来的逻辑）
        self.max_retries = 3
        self.current_goal = None
        self.retry_count = 0

        # ========== 订阅自然语言命令 ==========
        # 和你想要的 RViz2 输入配合：RViz Panel/工具只要往 /nl_command 发 std_msgs/String 即可
        self.cmd_sub = self.create_subscription(
            String,
            'nl_command',          # 话题名：/nl_command
            self.nl_command_cb,
            10
        )

        self.get_logger().info('SemanticNavigator 节点已启动，等待 /nl_command 文本命令...')

    # ===================== 自然语言命令回调 =====================

    def nl_command_cb(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        self.get_logger().info(f'收到语义命令: "{text}"')

        # 匹配语义位置
        key = self.match_location(text)
        if not key:
            self.get_logger().warn(f'无法从命令 "{text}" 中匹配到任何已知语义位置')
            return

        loc = self.semantic_locations.get(key)
        if not loc:
            self.get_logger().error(f'语义 key "{key}" 在 semantic_locations 中没有配置')
            return

        frame_id = loc.get('frame_id', 'map')
        x = float(loc.get('x', 0.0))
        y = float(loc.get('y', 0.0))
        yaw = float(loc.get('yaw', 0.0))

        self.get_logger().info(
            f'命令匹配到 [{key}] -> frame="{frame_id}", x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}'
        )

        # 组装 NavigateToPose.Goal（完全沿用 tcpbridge 的方式）
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = frame_id
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y

        # yaw -> quaternion（只用 z, w）
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        # 重置重试计数（复用原逻辑）
        self.retry_count = 0
        self.current_goal = goal_msg

        # 等待 action server
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('NavigateToPose action server 未启动')
            return

        # 发送导航目标（复用 send_navigation_goal / 回调）
        self.send_navigation_goal(goal_msg)

    # ===================== 匹配命令到语义位置 =====================

    def match_location(self, text: str):
        """
        根据 semantic_alias 做最简单的包含匹配：
        例如 semantic_alias:
          living_room: ["客厅", "去客厅", "到客厅"]
        命令中包含任意一个字符串就算匹配。
        """
        for key, alias_list in self.semantic_alias.items():
            for alias in alias_list:
                if alias and alias in text:
                    self.get_logger().info(
                        f'命令 "{text}" 通过别名 "{alias}" 匹配到位置 key="{key}"'
                    )
                    return key
        return None

    # ===================== 以下完全复用 Nav2TcpBridge 的发送/重试结构 =====================

    def send_navigation_goal(self, goal_msg):
        """发送导航目标点"""
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('目标点被拒绝')
            return
        self.get_logger().info('目标点已接受，等待结果...')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'导航中，距离目标点剩余: {feedback.distance_remaining:.2f} 米'
        )

    def get_result_callback(self, future):
        result = future.result().result
        status = future.result().status
        if status == 4:  # SUCCEEDED
            self.get_logger().info('导航成功到达目标点！')
            self.retry_count = 0
        else:
            self.get_logger().info(f'导航失败，状态码: {status}')
            if self.retry_count < self.max_retries and self.current_goal is not None:
                self.retry_count += 1
                self.get_logger().info(
                    f'尝试重新发送目标点，第 {self.retry_count} 次重试'
                )
                self.send_navigation_goal(self.current_goal)
            else:
                self.get_logger().error(
                    f'导航失败，已达到最大重试次数 {self.max_retries}'
                )
                self.retry_count = 0
                self.current_goal = None


def main(args=None):
    rclpy.init(args=args)
    node = SemanticNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
