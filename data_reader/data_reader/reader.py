import rclpy
from sensor_msgs.msg import Imu, JointState
from bitbots_msgs.msg import FootPressure, FloatStamped
from message_filters import TimeSynchronizer, Subscriber
from rclpy.node import Node


class Reader(Node):
    def __init__(self):
        super().__init__('reader')
        self.imu_sub = Subscriber(self, Imu, "/imu/data", qos_profile=1)
        self.pressure_left_sub = Subscriber(self, FootPressure, "/foot_pressure_left/raw", qos_profile=1)
        self.pressure_right_sub = Subscriber(self, FootPressure, "/foot_pressure_right/raw", qos_profile=1)
        self.lhall_sub = Subscriber(self, FloatStamped, "/hall/left", qos_profile=1)
        self.rhall_sub = Subscriber(self, FloatStamped, "/hall/right", qos_profile=1)
        self.joint_sub = Subscriber(self, JointState, "/joint_states", qos_profile=1)

        self.ts = TimeSynchronizer([self.imu_sub,
                                    self.pressure_right_sub,
                                    self.pressure_right_sub,
                                    self.lhall_sub,
                                    self.rhall_sub,
                                    self.joint_sub], 10)
        self.ts.registerCallback(self.syncCb)

    def sync_cb(self, imu_msg, pressure_r_msg, pressure_l_msg, hall_l_msg, hall_r_msg, joint_msg):

        return


def main(args=None):
    rclpy.init(args=args)
    node = Reader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
