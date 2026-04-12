# Go2Brain 工作区说明

本仓库用于统一管理 `ros2_ws/src` 下的 Go2 ROS2 相关代码，包括导航、感知、RViz 插件、Web Dashboard 和录像工具。

## 目录结构

- `go2_ros2_toolbox`：Go2 ROS2 核心工具箱，包含底盘控制、SLAM、导航、点云处理和 Unitree 相关消息/API
- `go2_rviz_plugins`：RViz 自定义插件，目前包含语义导航文本输入面板
- `flask`：Web 侧桥接与 Dashboard 服务
- `go2_record_toolbox`：录像与简单采集工具

## 快速开始

### 基础导航

环境安装教程参考 `go2_ros2_toolbox/README.md`。安装完成后执行下面指令可以启动可视化界面并进入建图导航模式：

```bash
source ros2_ws/install/setup.bash
ros2 launch go2_core go2_startup.launch.py use_slam:=true
```

### 保存地图

启动基础导航后，通过手动导航建图。建图完成后可以使用下面指令保存地图：

```bash
ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: '/home/your_name/maps/map.posegraph'}"
```

也可以直接使用 `slam_toolbox` 的可视化界面保存。

### 基于地图的导航

先在 `go2_ros2_toolbox/go2_slam/config/localization_slam_params.yaml` 中配置地图路径：

```yaml
map_file_name: /home/your_name/maps/map
map_start_pose: [dock pose]
```

其中 `[dock pose]` 可以通过 RViz2 中的 `2D Pose Estimate` 获取。

基于已经保存的地图进行导航：

```bash
ros2 launch go2_core go2_startup.launch.py use_slam:=false
```

### 预设点语义导航

使用 `2D Pose Estimate` 获取位置后，将语义位置信息配置到：

`go2_ros2_toolbox/go2_navigation/config/semantic_map.yaml`

启动方式：

```bash
# terminal 1
ros2 launch go2_core go2_startup.launch.py use_slam:=false

# terminal 2
ros2 topic pub /nl_command std_msgs/String "data: '导航目标文本'" -1
```

## Dashboard

当前仓库中的 `flask/dashboard_server.py` 提供 Web Dashboard，支持：

- 查看地图、轨迹和机器人实时状态
- 在地图上点选巡检路线并下发
- 输入文本导航指令
- 查看 RGB 相机画面

启动方式：

```bash
python3 flask/dashboard_server.py
```

默认访问地址：

```text
http://<robot-ip>:5001/
```

## 说明

- 顶层仓库用于统一版本管理整个 `src` 工作区
- `go2_ros2_toolbox/README-zml.md` 保留为原始快速说明，不在本次整理中改动
- `build/`、`install/`、`log/` 等构建产物不纳入版本管理
