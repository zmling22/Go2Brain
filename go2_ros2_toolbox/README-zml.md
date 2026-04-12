# 快速开始

## 基础导航
环境安装教程参考`README.md`，安装完成后执行下面指令可以开启可视化界面并导航
```
source ros2_ws/install/setup.bash
ros2 launch go2_core go2_startup.launch.py use_slam:=true
```

## 保存地图
启动基础导航后，通过手动导航建图，建图完成后可以使用下面指令保存地图：
```
ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: '/home/your_name/maps/map.posegraph'}"
```
或使用`slam_toolbox`的可视化界面保存

## 基于地图的导航
配置地图路径：
`go2_slam/config/localization_slam_params.yaml`中配置
```
map_file_name: /home/your_name/maps/map
map_start_pose: [dock pose]
```
`[dock pose]`使用rviz2中的`2d pose estimate`功能获得

基于已经保存的地图进行导航：
```
ros2 launch go2_core go2_startup.launch.py use_slam:=false
```

## 预设点的语义导航
使用`2d pose estimate`获取位置，将语义信息配置在`go2_navigation/config/semantic_map.yaml`中
```
# 启动预设点的语义导航
# terminal 1
ros2 launch go2_core go2_startup.launch.py use_slam:=false
# terminal 2 
ros2 topic pub /nl_command std_msgs/String "data: '导航目标文本'" -1
```

