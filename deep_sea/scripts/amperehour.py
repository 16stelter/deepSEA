#!/usr/bin/env python3

import time
from time import time_ns

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist


class EnergyNode(Node):
    def __init__(self):
        rclpy.init(args=None)
        super().__init__("energy_node")

        self.joint_state_sub = self.create_subscription(JointState, "/joint_states", self.joint_state_cb, 1)
        self.walk_pub = self.create_publisher(Twist, "/cmd_vel", 1)

        self.record = False
        self.first = True
        self.last_time = 0.0

        self.ah = 0.0

        self.main()


    def joint_state_cb(self, msg):
        if self.record:
            if self.first:
                self.last_time = time_ns()
                self.first = False
            else:
                for i in range(len(msg.name)):
                    if(msg.name[i] == "LKnee"):
                        current_time = time_ns()
                        # msg.effort * (torque to value) * (value to current) * duration * (nanosec to hour)
                        self.ah += msg.effort[i] * 149.795386991 * 0.00269 * (current_time - self.last_time) * 2.777777777778 * (10 ** -13)
                        self.last_time = current_time


    def main(self):
        self.get_logger().warning("Place the robot on the ground, hold it secure. Press any key to begin.")
        input()
        msg = Twist()
        msg.linear.x = 0.1
        self.walk_pub.publish(msg)
        start_time = time.time()
        self.record = True
        while time.time() - start_time < 30:
                rclpy.spin_once(self)
        self.record = False
        self.get_logger().warning("" + str(self.ah)+ " Ah")
        msg.linear.x = 0.0
        msg.angular.x = -1.0
        self.walk_pub.publish(msg)
        self.get_logger().warning("Done. Goodbye.")
        return

if __name__ == '__main__':
    node = EnergyNode()