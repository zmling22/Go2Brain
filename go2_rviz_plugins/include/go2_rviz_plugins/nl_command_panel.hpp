#ifndef GO2_RVIZ_PLUGINS__NL_COMMAND_PANEL_HPP_
#define GO2_RVIZ_PLUGINS__NL_COMMAND_PANEL_HPP_

#include <memory>

#include <QLineEdit>
#include <QPushButton>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include <rviz_common/panel.hpp>

namespace go2_rviz_plugins
{

class NLCommandPanel : public rviz_common::Panel
{
  Q_OBJECT   // ★ 关键：之前你缺的就是这一行

public:
  explicit NLCommandPanel(QWidget * parent = nullptr);

  void onInitialize() override;

public Q_SLOTS:
  void sendCommand();

private:
  QLineEdit * input_;
  QPushButton * send_button_;

  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
};

}  // namespace go2_rviz_plugins

#endif  // GO2_RVIZ_PLUGINS__NL_COMMAND_PANEL_HPP_
