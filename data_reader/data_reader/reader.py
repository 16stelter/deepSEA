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

        data_columns = ['l_knee_pos', 'l_hall_pos', 'l_knee_vel', 'l_hall_vel', 'l_knee_eff',
                        'r_knee_pos', 'r_hall_pos', 'r_knee_vel', 'r_hall_vel', 'r_knee_eff',
                        'imu', 'l_pressure', 'r_pressure', 'command', 'timestamp']
        self.df = pd.DataFrame(columns=data_columns)

        self.ptime = 0.0
        self.previous = [0.0, 0.0]

        self.hist = [0 for i in range(10)]
        self.command = None

        self.imu_sub = Subscriber(self, Imu, "/imu/data", qos_profile=1)
        self.pressure_left_sub = Subscriber(self, FootPressure, "/foot_pressure_left/filtered", qos_profile=1)
        self.pressure_right_sub = Subscriber(self, FootPressure, "/foot_pressure_right/filtered", qos_profile=1)
        self.lhall_sub = Subscriber(self, FloatStamped, "/hall/left/filtered", qos_profile=1)
        self.rhall_sub = Subscriber(self, FloatStamped, "/hall/right/filtered", qos_profile=1)
        self.joint_sub = Subscriber(self, JointState, "/joint_states", qos_profile=1)
        self.command_sub = self.create_subscription(JointCommand, "/DynamixelController/command", self.command_cb, 1)
        self.ts = ApproximateTimeSynchronizer([self.rhall_sub,
                                               self.lhall_sub,
                                               self.joint_sub,
                                               self.imu_sub,
                                               self.pressure_left_sub,
                                               self.pressure_right_sub], 10, (1/120))
        self.ts.registerCallback(self.sync_cb)

    def command_cb(self, msg):
        self.command = msg

    def sync_cb(self, hall_r_msg, hall_l_msg, joint_msg, imu_msg, pressure_l_msg, pressure_r_msg):
        time = self.get_clock().now().nanoseconds * 1e-9
        for i in range(len(joint_msg.name)):
            if joint_msg.name[i] == 'LKnee':
                self.hist[0] = joint_msg.position[i]
                self.hist[2] = joint_msg.velocity[i]
                self.hist[4] = joint_msg.effort[i]
            elif joint_msg.name[i] == 'RKnee':
                self.hist[5] = joint_msg.position[i]
                self.hist[7] = joint_msg.velocity[i]
                self.hist[9] = joint_msg.effort[i]

        self.hist[1] = hall_l_msg.value
        self.hist[6] = hall_r_msg.value

        self.hist[3] = (hall_l_msg.value - self.previous[0]) / (time - self.ptime)
        self.hist[8] = (hall_r_msg.value - self.previous[1]) / (time - self.ptime)
        self.previous[0] = hall_l_msg.value
        self.previous[1] = hall_r_msg.value

        data = copy.deepcopy(self.hist)
        data.append([imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w,
                     imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z,
                     imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z])
        data.append([pressure_l_msg.left_back, pressure_l_msg.left_front, pressure_l_msg.right_front, pressure_l_msg.right_back])
        data.append([pressure_r_msg.left_back, pressure_r_msg.left_front, pressure_r_msg.right_front, pressure_r_msg.right_back])

        l = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if self.command is not None and self.get_clock().now().nanoseconds * 1e-9 - \
                (self.command.header.stamp.sec + self.command.header.stamp.nanosec * 1e-9) < (1/120):
            for i in range(len(self.command.joint_names)):
                if self.command.joint_names[i] == "LKnee":
                    l[0] = self.command.positions[i]
                    l[1] = self.command.velocities[i]
                    l[2] = self.command.accelerations[i]
                    l[3] = self.command.max_currents[i]
                elif self.command.joint_names[i] == "RKnee":
                    l[4] = self.command.positions[i]
                    l[5] = self.command.velocities[i]
                    l[6] = self.command.accelerations[i]
                    l[7] = self.command.max_currents[i]
            data.append(l)
        else:
            data.append([])
        data.append(time)
        row = pd.Series(data, index=self.df.columns)
        self.df = self.df.append(row, ignore_index=True)
        self.ptime = time

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
