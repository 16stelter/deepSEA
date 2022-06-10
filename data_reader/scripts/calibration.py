import time
import rclpy
from rclpy.node import Node
from bitbots_msgs.msg import FloatStamped, JointCommand
import yaml
import ament_index_python


class Calibration:
    def __init__(self):
        rclpy.init(args=None)
        self.node = Node("calibration_node")
        self.cfgpath = ament_index_python.get_package_share_directory("data_reader") + "/config/sensors.yaml"

        self.loffset = 0
        self.roffset = 0

        self.lsub = self.node.create_subscription(FloatStamped, "/hall/left", self.l_cb, 1)
        self.rsub = self.node.create_subscription(FloatStamped, "/hall/right", self.r_cb, 1)
        self.mgpub = self.node.create_publisher(JointCommand, "DynamixelController/command", 1)

    def l_cb(self, msg):
        self.loffset = 2048 - msg.value  # Dynamixel 2048 = 0 degrees
        return

    def r_cb(self, msg):
        self.roffset = 2048 - msg.value # Dynamixel 2048 = 0 degrees
        return

    def main(self):
        self.node.get_logger().info("Hall Sensor Calibration Tool \n \n")
        self.node.get_logger().info("Please make sure the robot is not touching the ground, "
                                    "then press any key to continue.\n")
        input()
        self.node.get_logger().info("Config path is " + self.cfgpath)
        self.node.get_logger().info("Setting all motors to zero.")
        msg = JointCommand()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.joint_names = ["RShoulderPitch", "LShoulderPitch", "RShoulderRoll", "LShoulderRoll", "RElbow",
                           "LElbow", "RHipYaw", "LHipYaw", "RHipRoll", "LHipRoll", "RHipPitch",
                           "LHipPitch", "RKnee", "LKnee", "RAnklePitch", "LAnklePitch", "RAnkleRoll",
                           "LAnkleRoll", "HeadPan", "HeadTilt"]
        msg.positions = [0.0] * 20
        msg.velocities = [5.0] * 20
        msg.accelerations = [-1.0] * 20
        msg.max_currents = [-1.0] * 20
        self.mgpub.publish(msg)
        rclpy.spin_once(self.node)
        self.node.get_logger().info("Waiting 3 seconds...")
        time.sleep(3)
        rclpy.spin_once(self.node)
        self.node.get_logger().info("Left offset is " + str(self.loffset))
        self.node.get_logger().info("Right offset is " + str(self.roffset))
        self.node.get_logger().info("Writing to config file...")
        output = {
            'reader': {
                'ros__parameters': {
                    'l_offset': self.loffset,
                    'r_offset': self.roffset
                }
            }
        }
        with open(self.cfgpath,'w') as f:
            yaml.dump(output, f)

        self.node.get_logger().info("Sensors calibrated successfully!")


if __name__ == "__main__":
    calibration = Calibration()
    calibration.main()






