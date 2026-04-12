#include "go2_rviz_plugins/nl_command_panel.hpp"

#include <QHBoxLayout>
#include <QString>

#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/display_context.hpp>

namespace go2_rviz_plugins
{

NLCommandPanel::NLCommandPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  input_ = new QLineEdit(this);
  input_->setPlaceholderText("输入语义命令，例如：去客厅");

  send_button_ = new QPushButton("发送语义命令", this);

  auto * layout = new QHBoxLayout;
  layout->addWidget(input_);
  layout->addWidget(send_button_);
  setLayout(layout);

  connect(send_button_, SIGNAL(clicked()), this, SLOT(sendCommand()));
}

void NLCommandPanel::onInitialize()
{
  auto ros_node_abstraction = getDisplayContext()->getRosNodeAbstraction().lock();
  node_ = ros_node_abstraction->get_raw_node();
  pub_ = node_->create_publisher<std_msgs::msg::String>("nl_command", 10);
}

void NLCommandPanel::sendCommand()
{
  if (!pub_) {
    return;
  }

  QString text = input_->text().trimmed();
  if (text.isEmpty()) {
    return;
  }

  std_msgs::msg::String msg;
  msg.data = text.toStdString();

  pub_->publish(msg);
}

}  // namespace go2_rviz_plugins

// 注意：在命名空间块外面
PLUGINLIB_EXPORT_CLASS(go2_rviz_plugins::NLCommandPanel, rviz_common::Panel)
