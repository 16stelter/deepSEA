#!/usr/bin/env python3
import numpy as np
import torch
import getopt, sys

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from bitbots_msgs.msg import FloatStamped

from torch import optim
from torch.utils.data import DataLoader
from torchmetrics import R2Score, MeanSquaredLogError
from torchmetrics.functional import mean_absolute_percentage_error
from tqdm import tqdm

from models import simplemlp, siam
from dataset import SeaDataset

import pandas as pd

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TestEval(Node):
    def __init__(self, input_size):
        super().__init__("test_eval")
        self.model = siam.SiamNN(1)#simplemlp.SimpleMlp(1)
        self.modelname = "siam"
        checkpoint = torch.load("../../checkpoints/free_hanging/p_SiamNN_2500.pt", map_location=torch.device('cpu'))
        self.model.eval()
        self.model.load_state_dict(checkpoint)


        self.criterion = torch.nn.MSELoss().to(DEVICE)
        ds = SeaDataset("../../data/free/free_dtestcn.csv", siam=True, hall=False, vel=False, eff=False, imu=False, fp=False, hist_len=0)
        self.dl = DataLoader(ds, batch_size=1, shuffle=False)


        self.val_loss = 0
        self.val_targets = self.val_preds = torch.tensor([])

        self.losses = []

        self.pub_sample = self.create_publisher(Float64MultiArray, "/debug/sample", 1)
        self.pub_target = self.create_publisher(FloatStamped, "/debug/target", 1)
        self.pub_prediction = self.create_publisher(FloatStamped, "/debug/prediction", 1)

        self.main_loop()

    def main_loop(self):
        for step, s in enumerate(tqdm(self.dl)):
            x = s[0].to(DEVICE)
            if self.modelname == "mlp":
                y = s[1].unsqueeze(1).to(DEVICE)
                y_pred = self.model(x)
                if step % 2 == 0:
                    msg = Float64MultiArray()
                    msg.data = [i * math.pi for i in x.tolist()[0]]
                    self.pub_sample.publish(msg)
                    msg = FloatStamped()
                    msg.value = y.item() * math.pi
                    self.pub_target.publish(msg)
                    msg.value = y_pred.item() * math.pi
                    self.pub_prediction.publish(msg)
                loss = self.criterion(y_pred, y).item()
                self.val_loss += loss
                self.losses.append(loss)
                self.val_targets = torch.cat((self.val_targets, y))
                self.val_preds = torch.cat((self.val_preds, y_pred))
                #self.val_r2 += self.r2(y_pred, y)
            elif self.modelname == "siam":
                x1 = s[1].to(DEVICE)
                y = s[2].unsqueeze(1).to(DEVICE)
                x1_pred, y_pred = self.model(x, x1, y)
                if step % 2 == 0:
                    msg = Float64MultiArray()
                    msg.data = [i * math.pi for i in x.tolist()[0]]
                    self.pub_sample.publish(msg)
                    msg = FloatStamped()
                    msg.value = y.item() * math.pi
                    self.pub_target.publish(msg)
                    msg.value = y_pred.item() * math.pi
                    self.pub_prediction.publish(msg)
                y_loss = self.criterion(y_pred, y).item()
                x1_loss = self.criterion(x1_pred, x1).item()
                loss = y_loss + 0.1 * x1_loss
                self.val_loss += loss
                self.losses.append(loss)
                # val_r2 += r2(pred, target)
                # val_msle += msle(abs(pred), abs(target))
                self.val_targets = torch.cat((self.val_targets, y))
                self.val_preds = torch.cat((self.val_preds, y_pred))
        self.val_loss = self.val_loss / len(self.dl)
        #self.val_r2 = self.val_r2 / len(self.dl)
        print(self.val_preds)
        mape= mean_absolute_percentage_error(self.val_preds, self.val_targets)

        print("MAPE: " + str(mape.item()))
        print("MSE: " + str(self.val_loss))
        df = pd.DataFrame({'loss':self.losses})
        df.to_csv('./losses.csv')
        #print("R2: " + str(self.val_r2.item()))





if __name__ == '__main__':
    rclpy.init(args=None)
    node = TestEval(1)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()