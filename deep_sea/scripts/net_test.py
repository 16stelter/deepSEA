#!/usr/bin/env python3
import math

import rclpy
import torch
from rclpy.node import Node
from deep_sea.models.simplemlp import SimpleMlp
from deep_sea.dataset import SeaDataset
from sensor_msgs.msg import Imu, JointState
from bitbots_msgs.msg import FootPressure, FloatStamped, JointCommand
from torch.utils.data import DataLoader
from tqdm import tqdm


class NetTest(Node):
    def __init__(self, input_size):
        super().__init__("nettest")
        self.model = SimpleMlp(input_size)
        checkpoint = torch.load("checkpoints/old/free_SimpleMlp_53.pt")
        self.model.eval()
        self.model.load_state_dict(checkpoint)

        self.pub_target = self.create_publisher(FloatStamped, "/debug/target", 1)
        self.pub_prediction = self.create_publisher(FloatStamped, "/debug/prediction", 1)

        ds = SeaDataset("data/dfreecn.csv", hist_len=1, vel=True, eff=True)

        self.loader = DataLoader(ds, batch_size=1, shuffle=False)

        self.test_loop()

    def test_loop(self):
        for i, (x, y) in enumerate(tqdm(self.loader)):
            #x, y = next(iter(self.loader))
            if i % 2 == 1:
                print(x)
                print(y)
                print("--")
                target_msg = FloatStamped()
                target_msg.value = y.item() * math.pi
                self.pub_target.publish(target_msg)

                pred_msg = FloatStamped()
                pred_msg.value = self.model.forward(x).item() * math.pi
                self.pub_prediction.publish(pred_msg)






def main(args=None):
    rclpy.init(args=None)
    node = NetTest(6)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
