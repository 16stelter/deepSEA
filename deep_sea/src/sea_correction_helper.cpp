#include "deep_sea/sea_correction_helper.h"

namespace deep_sea {

  SeaCorrectionHelper::SeaCorrectionHelper() : Node("sea_correction_helper") {
    lknee_node_ = rclcpp::Node::make_shared("lknee_pid");
    rknee_node_ = rclcpp::Node::make_shared("rknee_pid");

    lknee_node_->declare_parameter<double>("p", 0.0);
    lknee_node_->declare_parameter<double>("i", 0.0);
    lknee_node_->declare_parameter<double>("d", 0.0);
    lknee_node_->declare_parameter<double>("i_clamp_max", 0.0);
    lknee_node_->declare_parameter<double>("i_clamp_min", 0.0);
    lknee_node_->declare_parameter<bool>("antiwindup", false);
    rknee_node_->declare_parameter<double>("p", 0.0);
    rknee_node_->declare_parameter<double>("i", 0.0);
    rknee_node_->declare_parameter<double>("d", 0.0);
    rknee_node_->declare_parameter<double>("i_clamp_max", 0.0);
    rknee_node_->declare_parameter<double>("i_clamp_min", 0.0);
    rknee_node_->declare_parameter<bool>("antiwindup", false);

    lknee_pid_ = std::make_shared<control_toolbox::PidROS>(lknee_node_, "");
    rknee_pid_ = std::make_shared<control_toolbox::PidROS>(rknee_node_, "");
    lknee_pid_->initPid();
    rknee_pid_->initPid();

    latched_command_ = bitbots_msgs::msg::JointCommand();

    sub_command_ = nh_->create_subscription<bitbots_msgs::msg::JointCommand>(
      "/DynamixelController/command", 1, std::bind(&SeaCorrectionHelper::commandCb,
                                                   this, std::placeholders::_1));
    sub_hall_l_ = nh_->create_subscription<bitbots_msgs::msg::FloatStamped>(
      "/hall/left/filtered", 1, std::bind(&SeaCorrectionHelper::hallLCb,
                                          this, std::placeholders::_1));
    sub_hall_r_ = nh_->create_subscription<bitbots_msgs::msg::FloatStamped>(
      "/hall/right/filtered", 1, std::bind(&SeaCorrectionHelper::hallRCb,
                                           this, std::placeholders::_1));
    sub_state_ = nh_->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 1, std::bind(&SeaCorrectionHelper::stateCb,
                                    this, std::placeholders::_1));
    pub_ = nh_->create_publisher<bitbots_msgs::msg::JointCommand>("/DynamixelController/corrected", 1);
  }

  void SeaCorrectionHelper::hallLCb(const bitbots_msgs::msg::FloatStamped &msg) {
    hall_l_ = msg.value;
  }

  void SeaCorrectionHelper::hallRCb(const bitbots_msgs::msg::FloatStamped &msg) {
    hall_r_ = msg.value;
  }

  void SeaCorrectionHelper::commandCb(bitbots_msgs::msg::JointCommand msg) {
    latched_command_ = msg;
  }

  void SeaCorrectionHelper::stateCb(const sensor_msgs::msg::JointState &msg) {
    bitbots_msgs::msg::JointCommand out = bitbots_msgs::msg::JointCommand();
    if (latched_command_.joint_names.size() != 0) {
      out = latched_command_;
    } else {
      out.header.stamp = nh_->get_clock()->now();
      out.joint_names = msg.name;
      out.positions = msg.position;
      out.velocities = msg.velocity;
      std::vector<double> accs(msg.name.size(), -1.0);
      out.accelerations = accs;
      out.max_currents = msg.effort;
    }
    for (unsigned int i = 0; i < out.joint_names.size(); i++) {
      if (out.joint_names[i] == "LKnee") {
        out.positions[i] -= lknee_pid_->computeCommand(hall_l_ - out.positions[i],
                                                       rclcpp::Duration::from_nanoseconds(1e9 * 0.001));
      } else if (out.joint_names[i] == "RKnee") {
        out.positions[i] -= rknee_pid_->computeCommand(hall_r_ - out.positions[i],
                                                       rclcpp::Duration::from_nanoseconds(1e9 * 0.001));
      }
    }
    pub_->publish(out);
  }
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  // init node
  auto node = std::make_shared<deep_sea::SeaCorrectionHelper>();
  rclcpp::executors::EventsExecutor exec;
  exec.add_node(node);

  exec.spin();
  rclcpp::shutdown();
}
