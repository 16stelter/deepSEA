#!/usr/bin/env python3
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.action import ActionClient
from bitbots_msgs.action import Dynup


class Record:
    def __init__(self):
        rclpy.init(args=None)
        self.node = Node("record_node")
        self.walk_pub = self.node.create_publisher(Twist, "/cmd_vel", 1)
        self.dynup_action_client = ActionClient(self.node, Dynup, 'dynup')

        self.main()
        rclpy.spin(self.node)

    def main(self):
        self.node.get_logger().warning("Place the robot on the ground, hold it secure. Press any key to begin.")
        input()
        speeds = [0.1, 0.2]
        for speed in speeds:
            self.node.get_logger().warning("Walking in each direction for 10 seconds with speed %f" % speed)
            directions = ["front", "left", "back", "right", "rotate_left", "rotate_right", "front_right", "front_left", "back_right", "back_left"]
            x = [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0 ,-1.0]
            y = [0.0, -0.5, 0.0, 0.5, 0.0, 0.0, 0.5, -0.5 ,0.5, -0.5]
            r = [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
            for d in range(len(directions)):
                self.node.get_logger().info("Current direction: %s" % directions[d])
                msg = Twist()
                msg.linear.x = x[d] * speed
                msg.linear.y = y[d] * speed
                msg.angular.z = r[d] * speed
                self.walk_pub.publish(msg)
                time.sleep(10)
                msg = Twist()
                msg.angular.x = -1.0
                self.walk_pub.publish(msg)
                time.sleep(5)

        self.node.get_logger().warning("Walking is all done! Proceed with standing up now...")
        self.node.get_logger().warning("Moving on to standup. 3 trials per direction.")
        for i in range(3):
            self.node.get_logger().warning("Place robot on front, hold it secure. Press any key to begin.")
            input()
            goal = Dynup.Goal()
            goal.direction = "front"
            self.dynup_action_client.send_goal_async(goal)
            self.node.get_logger().warning("Place robot on back, hold it secure. Press any key to begin.")
            input()
            goal = Dynup.Goal()
            goal.direction = "back"
            self.dynup_action_client.send_goal_async(goal)
        self.node.get_logger().warning("All done! Congratulations!")
        self.node.get_logger().warning("Goodbye!")



if __name__ == "__main__":
    r = Record()
