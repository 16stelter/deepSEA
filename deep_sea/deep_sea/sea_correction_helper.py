#!/usr/bin/env python3
import math

import rclpy
import torch
from rclpy.node import Node
from deep_sea.models.simplemlp import SimpleMlp
from deep_sea.ogma import Ogma
from sensor_msgs.msg import Imu, JointState
from bitbots_msgs.msg import FootPressure, FloatStamped, JointCommand

'''
Node for intercepting motor commands and applying corrections to them.
model_type defines, which network should be used.
input_size defines the size of the input vector.
'''
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
        elif self.model_type == "ogma":
            self.model = Ogma("./hierarchy")
        else:
            raise Exception("Unknown model type")

        self.left_foot = [0.0] * 3
        self.right_foot = [0.0] * 3
        self.imu = [0.0] * 9
        self.left_hall_pos = 0.0
        self.left_hall_hist = [0.0] * 10
        self.right_hall_pos = 0.0
        self.right_hall_hist = [0.0] * 10
        self.left_hall_vel = 0.0
        self.right_hall_vel = 0.0

        self.left_pos_hist = [0.0] * 10
        self.right_pos_hist = [0.0] * 10

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
        self.debug_pub_l = self.create_publisher(FloatStamped, "/debug/l_hall_error", 1)
        self.debug_pub_r = self.create_publisher(FloatStamped, "/debug/r_hall_error", 1)
        self.debug_torque_pub_l = self.create_publisher(FloatStamped, "/debug/l_hall_torque", 1)
        self.debug_torque_pub_r = self.create_publisher(FloatStamped, "/debug/r_hall_torque", 1)
        self.debug_command_pub_l = self.create_publisher(FloatStamped, "/debug/l_command", 1)
        self.debug_command_pub_r = self.create_publisher(FloatStamped, "/debug/r_command", 1)
        self.debug_motor_torque_pub_r = self.create_publisher(FloatStamped, "/debug/r_motor_torque", 1)
        self.debug_motor_torque_pub_l = self.create_publisher(FloatStamped, "/debug/l_motor_torque", 1)

    '''
    Latches the latest motor command.
    '''
    def command_cb(self, msg):
        self.latched_command = msg
        for i in range(len(msg.joint_names)):
            if msg.joint_names[i] == "LKnee":
                debug_msg = FloatStamped()
                debug_msg.value = msg.position[i]
                self.debug_command_pub_l.publish(debug_msg)
            elif msg.joint_names[i] == "RKnee":
                debug_msg = FloatStamped()
                debug_msg.value = msg.position[i]
                self.debug_command_pub_r.publish(debug_msg)

    '''
    Adds correction to the SEA error. Also calculates and publishes the current torque in the knee actuators.
    '''
    def joint_state_cb(self, msg):
        out = JointCommand()
        for i in range(len(msg.name)):
            if msg.name[i] == "LKnee":
                torque_msg = FloatStamped()
                torque_msg.value = (self.left_hall_pos - msg.position[i]) * 9.10564334645581
                self.debug_torque_pub_l.publish(torque_msg)
                debug_msg = FloatStamped()
                debug_msg.value = msg.effort[i]
                self.debug_motor_torque_pub_l.publish(debug_msg)
            elif msg.name[i] == "RKnee":
                torque_msg = FloatStamped()
                torque_msg.value = (self.right_hall_pos - msg.position[i]) * -9.10564334645581
                self.debug_torque_pub_r.publish(torque_msg)
                debug_msg = FloatStamped()
                debug_msg.value = msg.effort[i]
                self.debug_motor_torque_pub_r.publish(debug_msg)

        if self.latched_command is not None:
            for j in range(len(self.latched_command.joint_names)):
                if self.latched_command.joint_names[j] == "LKnee":
                    for i in range(len(msg.name)):
                        if msg.name[i] == "LKnee":
                            self.left_pos_hist.append(msg.position[i] / math.pi)
                            self.left_pos_hist.pop(0)
                            debug_msg = FloatStamped()
                            debug_msg.value = self.left_hall_pos - self.latched_command.positions[j]
                            self.debug_pub_l.publish(debug_msg)
                            sample = [self.latched_command.positions[j] / math.pi, self.left_pos_hist,
                                      self.left_hall_hist, msg.velocity[i], self.left_hall_vel,
                                      msg.effort[i] / 11.7]  # normalize this
                            sample.extend(self.imu)
                            sample.extend(self.left_foot)
                            sample.extend(self.right_foot)
                            break
                    prediction = self.model.forward(torch.tensor(sample)).item() * math.pi
                elif self.latched_command.joint_names[j] == "RKnee":
                    for i in range(len(msg.name)):
                        if msg.name[i] == "RKnee":
                            self.right_pos_hist.append(msg.position[i] / math.pi)
                            self.right_pos_hist.pop(0)
                            debug_msg = FloatStamped()
                            debug_msg.value = self.right_hall_pos - self.latched_command.positions[j]
                            self.debug_pub_r.publish(debug_msg)
                            sample = [self.latched_command.positions[j] / math.pi, self.right_pos_hist,
                                      self.right_hall_hist, msg.velocity[i], self.right_hall_vel,
                                      msg.effort[i] / 11.7]  # normalize this
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

    '''
    Collects data from the left Hall sensor.
    '''
    def l_hall_cb(self, msg):
        self.left_hall_pos = msg.value
        self.left_hall_hist.append(self.left_hall_pos / math.pi)
        self.left_hall_hist.pop(0)
        self.left_hall_vel = msg.value - self.l_previous
        self.l_previous = msg.value

    '''
    Collects data from the right Hall sensor.
    '''
    def r_hall_cb(self, msg):
        self.right_hall_pos = msg.value
        self.right_hall_hist.append(self.right_hall_pos / math.pi)
        self.right_hall_hist.pop(0)
        self.right_hall_vel = msg.value - self.r_previous
        self.r_previous = msg.value

    '''
    Collects data from the IMU.
    '''
    def imu_cb(self, msg):
        imu = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
               msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
               msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        self.imu = list(quaternion_to_euler_angle(imu[0], imu[1], imu[2], imu[3]))
        self.imu.extend(imu[4:])

    '''
    Collects data from the left foot pressure sensors.
    '''
    def l_pressure_cb(self, msg):
        self.left_foot = [msg.left_front, msg.right_front, msg.right_back]

    '''
    Collects data from the right foot pressure sensors.
    '''
    def r_pressure_cb(self, msg):
        self.right_foot = [msg.left_back, msg.left_front, msg.right_front]

'''
Helper function to convert quaternions to euler angles.
'''
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
