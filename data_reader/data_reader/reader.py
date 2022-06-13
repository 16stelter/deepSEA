import math

import rclpy
from sensor_msgs.msg import Imu, JointState
from bitbots_msgs.msg import FootPressure, FloatStamped, JointCommand
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
import pandas as pd
import math
import copy


class Reader(Node):
    def __init__(self):
        super().__init__('reader')

        self.declare_parameter('l_offset', 0.0)
        self.declare_parameter('r_offset', 0.0)

        self.l_offset = self.get_parameter('l_offset').get_parameter_value().double_value
        self.r_offset = self.get_parameter('r_offset').get_parameter_value().double_value
        print("Offsets: l " + str(self.l_offset) + " r " + str(self.r_offset))

        data_columns = ['l_knee_pos', 'l_hall_pos', 'l_knee_vel', 'l_hall_vel', 'l_knee_eff',
                        'r_knee_pos', 'r_hall_pos', 'r_knee_vel', 'r_hall_vel', 'r_hall_eff',
                        'imu', 'l_pressure', 'r_pressure', 'command']
        self.df = pd.DataFrame(columns=data_columns)

        self.hist = [[0 for i in range(10)] for i in range(10)]
        self.command = None

        self.imu_sub = Subscriber(self, Imu, "/imu/data", qos_profile=1)
        self.pressure_left_sub = Subscriber(self, FootPressure, "/foot_pressure_left/filtered", qos_profile=1)
        self.pressure_right_sub = Subscriber(self, FootPressure, "/foot_pressure_right/filtered", qos_profile=1)
        self.lhall_sub = Subscriber(self, FloatStamped, "/hall/left", qos_profile=1)
        self.rhall_sub = Subscriber(self, FloatStamped, "/hall/right", qos_profile=1)
        self.joint_sub = Subscriber(self, JointState, "/joint_states", qos_profile=1)
        self.command_sub = self.create_subscription(JointCommand, "/DynamixelController/command", self.command_cb, 1)
        self.ts = ApproximateTimeSynchronizer([self.rhall_sub,
                                               self.lhall_sub,
                                               self.joint_sub,
                                               self.imu_sub,
                                               self.pressure_left_sub,
                                               self.pressure_right_sub], 10, 0.1)
        self.ts.registerCallback(self.sync_cb)

    def command_cb(self, msg):
        self.command = msg

    def sync_cb(self, hall_r_msg, hall_l_msg, joint_msg, imu_msg, pressure_l_msg, pressure_r_msg):
        for i in range(len(joint_msg.name)):
            if joint_msg.name[i] == 'LKnee':
                self.hist[0].append(joint_msg.position[i])
                self.hist[2].append(joint_msg.velocity[i])
                self.hist[4].append(joint_msg.effort[i])
            elif joint_msg.name[i] == 'RKnee':
                self.hist[5].append(joint_msg.position[i])
                self.hist[7].append(joint_msg.velocity[i])
                self.hist[9].append(joint_msg.effort[i])

        hall_l = (hall_l_msg.value + self.l_offset)
        if hall_l < 0:
            hall_l = 4096.0 - hall_l
        hall_l = hall_l % 4096.0
        if hall_l < 2048.0:
            hall_l = -math.pi + (hall_l / 2048.0) * math.pi
        else:
            hall_l = ((hall_l - 2048.0) / 2048.0) * math.pi

        hall_r = hall_r_msg.value + self.r_offset
        if hall_r < 0:
            hall_r = 4096.0 - hall_r
        hall_r = hall_r % 4096.0
        if hall_r < 2048.0:
            hall_r = -math.pi + (hall_r / 2048.0) * math.pi
        else:
            hall_r = ((hall_r - 2048.0) / 2048.0) * math.pi


        self.hist[1].append(hall_l)
        self.hist[6].append(hall_r)
        for i in self.hist:
            if len(i) > 10:
                i.pop(0)
        data = copy.deepcopy(self.hist)
        data.append(imu_msg)
        data.append(pressure_l_msg)
        data.append(pressure_r_msg)
        if self.command is not None and self.get_clock().now() - self.command.header.stamp < 0.1:
            data.append(self.command)
        else:
            data.append([])
        row = pd.Series(data, index=self.df.columns)
        self.df = self.df.append(row, ignore_index=True)


def main(args=None):
    rclpy.init(args=args)
    node = Reader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.df.to_csv('dataset.csv')
        print("done")
        pass
    node.destroy_node()
    rclpy.shutdown()
