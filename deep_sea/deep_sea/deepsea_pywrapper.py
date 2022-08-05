import math

import rclpy
from rclpy.node import Node
from models.simplemlp import SimpleMlp
from bitbots_msgs.srv import DeepSeaService
from sensor_msgs.msg import Imu, JointState
from bitbots_msgs.msg import FootPressure, FloatStamped, JointCommand


class DeepSea(Node):
    def __init__(self, model_type="mlp", input_size=0):
        super().__init__("deepsea")
        self.model_type = model_type
        if self.model_type == "mlp":
            self.model = SimpleMlp(input_size)
        else:
            raise Exception("Unknown model type")

        self.l_data = [0.0] * 5
        self.r_data = [0.0] * 5
        self.data = [0.0] * 15

        self.l_previous = 0.0
        self.r_previous = 0.0

        self.create_subscription(Imu, "/imu/data", self.imu_cb, 1)
        self.create_subscription(FootPressure, "/foot_pressure_left/filtered", self.l_pressure_cb, 1)
        self.create_subscription(FootPressure, "/foot_pressure_right/filtered", self.r_pressure_cb, 1)
        self.create_subscription(FloatStamped, "/hall/left/filtered", self.l_hall_cb, 1)
        self.create_subscription(FloatStamped, "/hall/right/filtered", self.r_hall_cb, 1)
        self.create_subscription(JointState, "/joint_states", self.joint_state_cb, 1)

        self.srv = self.create_service(DeepSeaService, "deepsea_service", self.service_cb)

    def service_cb(self, request, response):
        sample = [request.input]
        sample.extend(self.data)
        if self.request.leg == "left":
            sample.extend(self.l_data)
        elif self.request.leg == "right":
            sample.extend(self.r_data)
        response.result = self.model.predict(sample)
        return response

    def joint_state_cb(self, msg):
        for i in range(len(msg.name)):
            if msg.name[i] == "LKnee":
                self.l_data[0] = msg.position[i]
                self.l_data[2] = msg.velocity[i]
                self.l_data[4] = msg.effort[i]
            elif msg.name[i] == "RKnee":
                self.r_data[0] = msg.position[i]
                self.r_data[2] = msg.velocity[i]
                self.r_data[4] = msg.effort[i]

    def l_hall_cb(self, msg):
        self.l_data[1] = msg.data
        self.l_data[3] = msg.data - self.l_previous
        self.l_previous = msg.data

    def r_hall_cb(self, msg):
        self.r_data[1] = msg.data
        self.r_data[3] = msg.data - self.r_previous
        self.r_previous = msg.data

    def imu_cb(self, msg):
        imu = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
               msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
               msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        e = list(quaternion_to_euler_angle(imu[0], imu[1], imu[2], imu[3]))
        e.extend(imu[4:])
        self.data[0:8] = e

    def l_pressure_cb(self, msg):
        self.data[9:12] = [msg.left_front, msg.right_front, msg.right_back]

    def r_pressure_cb(self, msg):
        self.data[12:15] = [msg.left_back, msg.left_front, msg.right_front]


def quaternion_to_euler_angle(x, y, z, w):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z


def main(args=None):
    rclpy.init(args=None)
    node = DeepSea()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
