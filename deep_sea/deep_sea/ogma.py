from cmath import inf
import math
import pyaogmaneo as neo
import numpy as np
from dataset import SeaDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import getopt, sys
import torch
import wandb


# helper function for squashing
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class Ogma:
    def __init__(self, mode):
        self.res = 129 #resolution
        self.se = ScalarEncoder(3, 9, self.res)
        neo.setNumThreads(8)

        self.input_size = 1
        self.column_size = 16
        self.epochs = 1000

        self.criterion = torch.nn.MSELoss()
        self.action = 0
        self.y_pred = 0.0

        self.lds = [] # generating model
        for i in range(4):
            ld = neo.LayerDesc()
            ld.hiddenSize = (5, 5, 16)
            self.lds.append(ld)

        self.h = neo.Hierarchy()
        if mode == "train":
            ds = SeaDataset("../../data/ground_with_support/datasetcn.csv", hall=False, vel=True, eff=True)
            train_size = int(0.8 * len(ds))
            test_size = len(ds) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, test_size])
            self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
            self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            self.h.initRandom([ neo.IODesc((3, 3, self.res), neo.none, eRadius=2, dRadius=2), neo.IODesc((1, 1, self.res), neo.action, eRadius=0, dRadius=2, historyCapacity=64) ], self.lds)
            self.h.setAVLR(1, 0.01)
            self.h.setAALR(1, 0.01)
            self.h.setADiscount(1, 0.99)
            self.h.setAMinSteps(1, 16)
            self.h.setAHistoryIters(1, 16)

            
            self.reward = 0.0
            self.best_val = -inf

            wandb.init(project="deepsea", config={"epochs": self.epochs, "model": "Ogma"})

            self.train_loop()
        else:
            self.h.initFromFile(mode)

    def action2motorgoal(self, position,  action):
        return position + action * (0.5 / self.res) - 0.5

    def train_loop(self):
        no_impr_count = 0
        for e in range(self.epochs):
            rsum = 0.0
            val_reward = 0.0
            for i, s in enumerate(tqdm(self.train_loader)):
                csdr = self.se.encode(sigmoid(np.matrix(s[0]).T * 4.0))
                self.h.step([ csdr, [ self.action ] ], True, self.reward)
                self.action = self.h.getPredictionCIs(1)[0]
                self.y_pred = self.action2motorgoal(s[0][0][0], self.action)
                self.reward = -self.criterion(torch.tensor([self.y_pred]), s[1]).item()
                rsum += self.reward
            print(rsum / len(self.train_loader))
            wandb.log({"train_loss": -rsum / len(self.train_loader)})
            wandb.log({"train_loss": -rsum / len(self.train_loader)})
            wandb.log({"train_loss": -rsum / len(self.train_loader)})
            for i, s in enumerate(tqdm(self.test_loader)):
                csdr = self.se.encode(sigmoid(np.matrix(s[0]).T * 4.0))
                self.h.step([ csdr, [ self.action ] ], False)
                self.action = self.h.getPredictionCIs(1)[0]
                self.y_pred = self.action2motorgoal(s[0][0][0], self.action)
                val_reward += -self.criterion(torch.tensor([self.y_pred]), s[1]).item()
            val_reward /= len(self.test_loader)
            wandb.log({"val_loss": -val_reward})
            wandb.log({"val_loss": -val_reward})
            wandb.log({"val_loss": -val_reward})
            print("val reward: " + str(val_reward))
            print("Epoch: " + str(e))
            if val_reward > self.best_val:
                self.best_val = val_reward
                self.h.saveToFile("./checkpoints/128hierarchy_{}".format(e))
                no_impr_count = 0
            else:
                no_impr_count += 1
            if no_impr_count > 100:
                return
        
    def forward(self, sample):
        csdr = self.se.encode(sigmoid(np.matrix(sample).T * 4.0))
        self.h.step([csdr, [self.action]], False)
        self.action = self.h.getPredictionCIs(1)[0]
        self.y_pred = self.action2motorgoal(sample[0][0], self.action)
        return torch.tensor([self.y_pred]).unsqueeze(0)


'''
Pre-encoder from the PyAOgmaNeo repository.

Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
'''
class ScalarEncoder:
    def __init__(self, num_scalars, num_columns, cells_per_column, lower_bound=0.0, upper_bound=1.0):
        self.num_scalars = num_scalars
        self.cells_per_column = cells_per_column

        self.protos = []

        for _ in range(num_columns):
            self.protos.append(np.random.rand(cells_per_column, num_scalars) * (upper_bound - lower_bound) + lower_bound)

    def encode(self, scalars):
        csdr = []

        for i in range(len(self.protos)):
            acts = -np.sum(np.square(np.repeat(scalars.T, self.cells_per_column, axis=0) - self.protos[i]), axis=1)

            csdr.append(np.argmax(acts).item())

        return csdr

    def decode(self, csdr):
        scalars = np.zeros(self.num_scalars)

        for i in range(len(self.protos)):
            scalars += self.protos[csdr[i]]

if __name__ == "__main__":
    ogma = Ogma("train")
