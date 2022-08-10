#!/usr/bin/env python3
import math

import rclpy
import torch
from rclpy.node import Node
from deep_sea.models.simplemlp import SimpleMlp
from sensor_msgs.msg import Imu, JointState
from bitbots_msgs.msg import FootPressure, FloatStamped, JointCommand


class DeepSea(Node):
    def __init__(self, model_type="mlp", input_size=21):
        super().__init__("deepsea")
        self.latched_command = None
        self.model_type = model_type
        if self.model_type == "mlp":
            self.model = SimpleMlp(input_size)
            checkpoint = torch.load("checkpoints/SimpleMlp_407.pt")
            self.model.eval()
            self.model.load_state_dict(checkpoint)
        else:
            raise Exception("Unknown model type")

        self.left_foot = [0.0] * 3
        self.right_foot = [0.0] * 3
        self.imu = [0.0] * 9
        self.left_hall_pos = 0.0
        self.right_hall_pos = 0.0
        self.left_hall_vel = 0.0
        self.right_hall_vel = 0.0

        self.l_previous = 0.0
        self.r_previous = 0.0

        self.create_subscription(Imu, "/imu/data", self.imu_cb, 1)
        self.create_subscription(FootPressure, "/foot_pressure_left/filtered", self.l_pressure_cb, 1)
        self.create_subscription(FootPressure, "/foot_pressure_right/filtered", self.r_pressure_cb, 1)
        self.create_subscription(FloatStamped, "/hall/left/filtered", self.l_hall_cb, 1)
        self.create_subscription(FloatStamped, "/hall/right/filtered", self.r_hall_cb, 1)
        self.create_subscription(JointState, "/joint_states", self.joint_state_cb, 1)
        self.create_subscription(JointCommand, "/DynamixelController/command", self.command_cb, 1)
        self.pub = self.create_publisher(JointCommand, "/DynamixelController/corrected", 1)

    def command_cb(self, msg):
        self.latched_command = msg

    def joint_state_cb(self, msg):
        out = JointCommand()
        if self.latched_command is not None:
            for j in range(len(self.latched_command.joint_names)):
                if self.latched_command.joint_names[j] == "LKnee":
                    for i in range(len(msg.name)):
                        if msg.name[i] == "LKnee":
                            sample = [self.latched_command.positions[j], msg.position[i], self.left_hall_pos,
                                      msg.velocity[i], self.left_hall_vel, msg.effort[i]]
                            sample.extend(self.imu)
                            sample.extend(self.left_foot)
                            sample.extend(self.right_foot)
                            break
                    prediction = self.model.forward(torch.tensor(sample)).item() * math.pi
                elif self.latched_command.joint_names[j] == "RKnee":
                    for i in range(len(msg.name)):
                        if msg.name[i] == "RKnee":
                            sample = [self.latched_command.positions[j], msg.position[i], self.left_hall_pos,
                                      msg.velocity[i], self.left_hall_vel, msg.effort[i]]
                            sample.extend(self.imu)
                            sample.extend(self.left_foot)
                            sample.extend(self.right_foot)
                            break
                    prediction = self.model.forward(torch.tensor(sample)).item() * math.pi
                if self.latched_command.joint_names[j] in ["RKnee", "LKnee"]:
                    out.positions.append(prediction)
                else:
                    out.positions.append(self.latched_command.positions[j])

                out.joint_names.append(self.latched_command.joint_names[j])
                out.velocities.append(self.latched_command.velocities[j])
                out.accelerations.append(self.latched_command.accelerations[j])
                out.max_currents.append(self.latched_command.max_currents[j])

                self.pub.publish(out)

    def l_hall_cb(self, msg):
        self.left_hall_pos = msg.value
        self.left_hall_vel = msg.value - self.l_previous
        self.l_previous = msg.value

    def r_hall_cb(self, msg):
        self.right_hall_pos = msg.value
        self.right_hall_vel = msg.value - self.r_previous
        self.r_previous = msg.value

    def imu_cb(self, msg):
        imu = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
               msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
               msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        self.imu = list(quaternion_to_euler_angle(imu[0], imu[1], imu[2], imu[3]))
        self.imu.extend(imu[4:])

    def l_pressure_cb(self, msg):
        self.left_foot = [msg.left_front, msg.right_front, msg.right_back]

    def r_pressure_cb(self, msg):
        self.right_foot = [msg.left_back, msg.left_front, msg.right_front]


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
