#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from bitbots_msgs.msg import FloatStamped
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from bitbots_msgs.action import Dynup
import pandas as pd

'''
Evaluation script to automatically perform a fixed set of motions and record the errors.
'''
class Evaluator(Node):
    def __init__(self):
        rclpy.init(args=None)
        super().__init__('evaluator')

        data_columns = ["direction", "speed", "avg_err_left", "avg_err_right", "peak_err_left_pos", "peak_err_left_neg",
                        "peak_err_right_pos", "peak_err_right_neg", "avg_err_left_pos", "avg_err_left_neg",
                        "avg_err_right_pos", "avg_err_right_neg"]
        self.df = pd.DataFrame(columns=data_columns)

        self.l_err_sub = self.create_subscription(FloatStamped, "/debug/l_hall_error", self.l_err_cb, 1)
        self.r_err_sub = self.create_subscription(FloatStamped, "/debug/r_hall_error", self.r_err_cb, 1)

        self.l_err_sum = self.r_err_sum = 0.0
        self.l_err_peak_pos = self.r_err_peak_pos = 0.0
        self.l_err_peak_neg = self.r_err_peak_neg = 0.0
        self.l_err_count = self.r_err_count = 0
        self.l_err_sum_pos = self.r_err_sum_pos = 0.0
        self.l_err_sum_neg = self.r_err_sum_neg = 0.0
        self.l_err_count_pos = self.r_err_count_pos = 0
        self.l_err_count_neg = self.r_err_count_neg = 0

        self.recording = False # toggles whether data should be stored right now

        self.walk_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.dynup_action_client = ActionClient(self, Dynup, 'dynup')

        self.main()

    '''
    Records the errors of the left leg.
    '''
    def l_err_cb(self, msg):
        if self.recording:
            self.l_err_sum += abs(msg.value)
            if msg.value > 0:
                if msg.value > self.l_err_peak_pos:
                    self.l_err_peak_pos = msg.value
                self.l_err_sum_pos += msg.value
                self.l_err_count_pos += 1
            elif msg.value < 0:
                if msg.value < self.l_err_peak_neg:
                    self.l_err_peak_neg = msg.value
                self.l_err_sum_neg += msg.value
                self.l_err_count_neg += 1

            self.l_err_count += 1

    '''
    Records the errors of the right leg.
    '''
    def r_err_cb(self, msg):
        if self.recording:
            self.r_err_sum += abs(msg.value)
            if msg.value > 0:
                if msg.value > self.r_err_peak_pos:
                    self.r_err_peak_pos = msg.value
                self.r_err_sum_pos += msg.value
                self.r_err_count_pos += 1
            elif msg.value < 0:
                if msg.value < self.l_err_peak_neg:
                    self.r_err_peak_neg = msg.value
                self.r_err_sum_neg += msg.value
                self.r_err_count_neg += 1
            self.r_err_count += 1

    '''
    Executes a row of actions in order. Walks in different directions and executes standup.
    '''
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
                self.get_logger().warn("Direction %s, speed %f.")
                self.logrow(directions[d], speed)
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
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            self.recording = False
            self.get_logger().warn("Trial %d, direction front." % i)
            self.logrow("dynup_front", 0.0)

            self.get_logger().warning("Place robot on back, hold it secure. Press any key to begin.")
            input()
            goal = Dynup.Goal()
            goal.direction = "back"
            self.recording = True
            future = self.dynup_action_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            self.recording = False
            self.get_logger().warn("Trial %d, direction back." % i)
            self.logrow("dynup_back", 0.0)
        self.get_logger().warning("All done! Congratulations!")
        self.get_logger().warning("Goodbye!")
        self.df.to_csv("eval.csv")

    '''
    Defines how data should be logged.
    '''
    def logrow(self, direction, speed=0.0):
        if self.l_err_count_neg == 0:  # avoid divide by zero. I'm sorry about this mess, I'm tired.
            self.l_err_count_neg = 1
        if self.l_err_count_pos == 0:
            self.l_err_count_pos = 1
        if self.r_err_count_neg == 0:
            self.r_err_count_neg = 1
        if self.r_err_count_pos == 0:
            self.r_err_count_pos = 1
        self.get_logger().warn("Average absolute error left: %f, right: %f" %
                               (self.l_err_sum / self.l_err_count, self.r_err_sum / self.r_err_count))
        self.get_logger().warn("Peak error positive left: %f, right: %f" % (self.l_err_peak_pos, self.r_err_peak_pos))
        self.get_logger().warn("Peak error negative left: %f, right: %f" % (self.l_err_peak_neg, self.r_err_peak_neg))
        self.get_logger().warn("Average error positive left: %f, right: %f" % (self.l_err_sum_pos / self.l_err_count_pos,
                                                                               self.r_err_sum_pos / self.r_err_count_pos))
        self.get_logger().warn("Average error negative left: %f, right: %f" % (self.l_err_sum_neg / self.l_err_count_neg,
                                                                               self.r_err_sum_neg / self.r_err_count_neg))
        data = [direction, speed, self.l_err_sum / self.l_err_count, self.r_err_sum / self.r_err_count,
                self.l_err_peak_pos, self.l_err_peak_neg, self.r_err_peak_pos, self.r_err_peak_neg,
                self.l_err_sum_pos / self.l_err_count_pos, self.l_err_sum_neg / self.l_err_count_neg,
                self.r_err_sum_pos / self.r_err_count_pos, self.r_err_sum_neg / self.r_err_count_neg]
        self.l_err_sum = self.r_err_sum = 0.0
        self.l_err_peak_pos = self.r_err_peak_pos = 0.0
        self.l_err_peak_neg = self.r_err_peak_neg = 0.0
        self.l_err_count = self.r_err_count = 0
        self.l_err_sum_pos = self.r_err_sum_pos = 0.0
        self.l_err_sum_neg = self.r_err_sum_neg = 0.0
        self.l_err_count_pos = self.r_err_count_pos = 0
        self.l_err_count_neg = self.r_err_count_neg = 0
        row = pd.Series(data, index=self.df.columns)
        self.df = self.df.append(row, ignore_index=True)

if __name__ == '__main__':
    node = Evaluator()

