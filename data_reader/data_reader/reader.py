import rclpy
from sensor_msgs.msg import Imu
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
        # self.motor_sub = self.create_subscription(Imu, "/imu/data", self.imu_cb, 1) TODO
        # self.target_sub = self.create_subscription(Imu, "/imu/data", self.imu_cb, 1)

        self.ts = TimeSynchronizer([self.imu_sub,
                                    self.pressure_right_sub,
                                    self.pressure_right_sub,
                                    self.lhall_sub,
                                    self.rhall_sub], 10)
        self.ts.registerCallback(self.syncCb)

    def syncCb(self, msg):
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
