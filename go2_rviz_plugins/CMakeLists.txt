cmake_minimum_required(VERSION 3.5)
project(go2_rviz_plugins)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Qt / MOC 相关设置
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(QT_DEFAULT_MAJOR_VERSION 5)
set(QT_MIN_VERSION "5.9.0")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rviz_common REQUIRED)
find_package(rviz_default_plugins REQUIRED)
find_package(pluginlib REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)

include_directories(
  include
  ${Qt5Widgets_INCLUDE_DIRS}
)

# ★ 关键：把需要 Q_OBJECT 的头文件加入到库的源里
set(SRC_FILES
  src/nl_command_panel.cpp
  include/go2_rviz_plugins/nl_command_panel.hpp
)

add_library(${PROJECT_NAME} SHARED
  ${SRC_FILES}
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  std_msgs
  rviz_common
  rviz_default_plugins
  pluginlib
)

target_link_libraries(${PROJECT_NAME}
  Qt5::Widgets
)

pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)

install(
  TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION lib
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  FILES plugin_description.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
