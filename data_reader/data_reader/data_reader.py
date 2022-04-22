import rclpy
from sensor_msgs.msg import Imu
from bitbots_msgs.msg import FootPressure, FloatStamped
from message_filters import TimeSynchronizer


class DataReader(Node):
    def __init__(self):
        super().__init__('reader')
        self.imu_sub = self.create_subscription(Imu, "/imu/data", 1)
        self.pressure_left_sub = self.create_subscription(FootPressure, "/foot_pressure_left/raw", 1)
        self.pressure_right_sub = self.create_subscription(FootPressure, "/foot_pressure_right/raw", 1)
        self.lhall_sub = self.create_subscription(FloatStamped, "/hall/left", 1)
        self.rhall_sub = self.create_subscription(FloatStamped, "/hall/right", 1)
        # self.motor_sub = self.create_subscription(Imu, "/imu/data", self.imu_cb, 1) TODO
        # self.target_sub = self.create_subscription(Imu, "/imu/data", self.imu_cb, 1)

        self.ts = TimeSynchronizer([self.imu_sub,
                                    self.pressure_right_sub,
                                    self.pressure_right_sub,
                                    self.lhall_sub,
                                    self.rhall_sub], 10)
        self.ts.registerCallback(self.syncCb)
        rclpy.spin()

    def syncCb(self, msg):
        return

