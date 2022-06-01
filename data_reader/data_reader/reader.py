import rclpy
from sensor_msgs.msg import Imu, JointState
from bitbots_msgs.msg import FootPressure, FloatStamped
from message_filters import TimeSynchronizer, Subscriber
from rclpy.node import Node
import pandas as pd
import copy


class Reader(Node):
    def __init__(self):
        super().__init__('reader')

        data_columns = ['l_knee_pos', 'l_hall_pos', 'l_knee_vel', 'l_hall_vel', 'l_knee_eff',
                        'r_knee_pos', 'r_hall_pos', 'r_knee_vel', 'r_hall_vel', 'r_hall_eff',
                        'imu', 'l_pressure', 'r_pressure']
        self.df = pd.DataFrame(columns=data_columns)

        self.hist = [[0 for i in range(10)] for i in range(13)]

        self.imu_sub = Subscriber(self, Imu, "/imu/data", qos_profile=1)
        self.pressure_left_sub = Subscriber(self, FootPressure, "/foot_pressure_left/raw", qos_profile=1)
        self.pressure_right_sub = Subscriber(self, FootPressure, "/foot_pressure_right/raw", qos_profile=1)
        self.lhall_sub = Subscriber(self, FloatStamped, "/hall/left", qos_profile=1)
        self.rhall_sub = Subscriber(self, FloatStamped, "/hall/right", qos_profile=1)
        self.joint_sub = Subscriber(self, JointState, "/joint_states", qos_profile=1)
        self.ts = TimeSynchronizer([
                                    self.rhall_sub,
                                    ], 10)
        self.ts.registerCallback(self.sync_cb)

    def sync_cb(self, hall_r_msg):
        # for i in range(len(joint_msg.name)):
        #     if joint_msg.name[i] == 'LKnee':
        #         self.hist[0].append(joint_msg.position[i])
        #         self.hist[2].append(joint_msg.velocity[i])
        #         self.hist[4].append(joint_msg.current[i])
        #     elif joint_msg.name[i] == 'Rknee':
        #         self.hist[5].append(joint_msg.position[i])
        #         self.hist[7].append(joint_msg.velocity[i])
        #         self.hist[9].append(joint_msg.current[i])
        #
        # self.hist[1].append(hall_l_msg.value)
        self.hist[6].append(hall_r_msg.value)
        print(hall_r_msg)
        for i in self.hist:
            if len(i) > 10:
                i.pop(0)
        data = copy.deepcopy(self.hist)
        # data.append(imu_msg)
        # data.append(pressure_l_msg)
        # data.append(pressure_r_msg)
        row = pd.Series(data, index=self.df.columns)
        self.df = self.df.append(row, ignore_index=True)


def main(args=None):
    rclpy.init(args=args)
    node = Reader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.df.to_csv('dataset.csv')
        pass
    node.destroy_node()
    rclpy.shutdown()
