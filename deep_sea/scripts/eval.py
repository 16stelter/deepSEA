#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from bitbots_msgs.msg import FootPressure, FloatStamped, JointCommand
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from bitbots_msgs.action import Dynup

class Evaluator(Node):
    def __init__(self):
        rclpy.init(args=None)
        super().__init__('evaluator')

        self.l_err_sub = self.create_subscription(FloatStamped, "/debug/l_hall_error", self.l_err_cb, 1)
        self.r_err_sub = self.create_subscription(FloatStamped, "/debug/r_hall_error", self.l_err_cb, 1)

        self.l_err_sum = 0.0
        self.r_err_sum = 0.0
        self.l_err_count = 0
        self.r_err_count = 0

        self.recording = False

        self.walk_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.dynup_action_client = ActionClient(self, Dynup, 'dynup')

        self.main()

    def l_err_cb(self, msg):
        if self.recording:
            self.l_err_sum += msg.value
            self.l_err_count += 1

    def r_err_cb(self, msg):
        if self.recording:
            self.r_err_sum += msg.value
            self.r_err_count += 1


    def main(self):
        self.get_logger().warning("Place the robot on the ground, hold it secure. Press any key to begin.")
        input()
        speeds = [0.05, 0.1, 0.2]
        for speed in speeds:
            self.get_logger().warning("Walking in each direction for 10 seconds with speed %f" % speed)
            directions = ["front", "back", "right", "rotate_right", "front_right", "back_right"]
            x = [1.0, -1.0, 0.0, 0.0, 1.0, -1.0]
            y = [0.0, 0.0, 0.5, 0.0, 0.5, 0.5]
            r = [0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
            for d in range(len(directions)):
                self.get_logger().info("Current direction: %s" % directions[d])
                msg = Twist()
                msg.linear.x = x[d] * speed
                msg.linear.y = y[d] * speed
                msg.angular.z = r[d] * speed
                self.walk_pub.publish(msg)
                start_time = time.time()
                self.recording = True
                while time.time() - start_time < 10:
                    rclpy.spin_once(self)
                self.recording = False
                self.get_logger().warn("Direction %s, speed %f. Average error left: %f, right: %f" %
                                       (directions[d], speed, self.l_err_sum / self.l_err_count,
                                        self.r_err_sum / self.r_err_count))
                self.l_err_sum = 0.0
                self.r_err_sum = 0.0
                self.l_err_count = 0
                self.r_err_count = 0
                msg = Twist()
                msg.angular.x = -1.0
                self.walk_pub.publish(msg)
                start_time = time.time()
                while time.time() - start_time < 5:
                    rclpy.spin_once(self)

        self.get_logger().warning("Walking is all done! Proceed with standing up now...")
        self.get_logger().warning("Moving on to standup. 3 trials per direction. Press any key to begin.")
        input()
        for i in range(3):
            self.get_logger().warning("Place robot on front, hold it secure. Press any key to begin.")
            input()
            goal = Dynup.Goal()
            goal.direction = "front"
            self.recording = True
            future = self.dynup_action_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self.action_client, future)
            self.recording = False
            self.get_logger().warn("Trial %d, direction front. Average error left: %f, right: %f" %
                                   (i, self.l_err_sum / self.l_err_count, self.r_err_sum / self.r_err_count))
            self.l_err_sum = 0.0
            self.r_err_sum = 0.0
            self.l_err_count = 0
            self.r_err_count = 0

            self.get_logger().warning("Place robot on back, hold it secure. Press any key to begin.")
            input()
            goal = Dynup.Goal()
            goal.direction = "back"
            self.recording = True
            future = self.dynup_action_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self.action_client, future)
            self.recording = False
            self.get_logger().warn("Trial %d, direction back. Average error left: %f, right: %f" %
                                   (i, self.l_err_sum / self.l_err_count, self.r_err_sum / self.r_err_count))
            self.l_err_sum = 0.0
            self.r_err_sum = 0.0
            self.l_err_count = 0
            self.r_err_count = 0
        self.get_logger().warning("All done! Congratulations!")
        self.get_logger().warning("Goodbye!")


if __name__ == '__main__':
    node = Evaluator()

