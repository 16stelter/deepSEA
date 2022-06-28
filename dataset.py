import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import ast

class SeaDataset(Dataset):
    def __init__(self, filename, hall=True, vel=True, imu=True, fp=True, use_poserr=False):
        self.data = pd.read_csv(filename)
        self.hall = hall
        self.vel = vel
        self.imu = imu
        self.fp = fp
        self.use_poserr = use_poserr

    def __getitem__(self, idx):
        # we can assume both joints are the same. thus we have twice the amount of samples.
        if idx % 2 == 0:  # left
            offset = 0
        else:  # right
            offset = 5
        idx = idx//2

        x = self.data.iloc[idx+1, 2+offset] # target position
        x = np.append(x, self.data.iloc[idx, 1+offset])  # current motor position
        if self.hall:
            x = np.append(x, self.data.iloc[idx, 2+offset])  # hall position current
        if self.use_poserr:
            if not self.hall:
                x = np.append(x, self.data.iloc[idx, 2+offset])
            x[0] = x[0] - x[2]
            x = np.delete(x, 2, axis=0)
        if self.vel:
            x = np.append(x, self.data.iloc[idx, 3+offset:5+offset].values)  # velocity
        if self.imu:
            x = np.concatenate((x, np.asarray(ast.literal_eval(self.data.iloc[idx, 11]))), axis=0)  # imu
        if self.fp:
            x = np.concatenate((x, np.asarray(ast.literal_eval(self.data.iloc[idx, 12]))), axis=0)  # left foot
            x = np.concatenate((x, np.asarray(ast.literal_eval(self.data.iloc[idx, 13]))), axis=0)  # right foot
        y = self.data.iloc[idx, 5+offset]
        return x, y

    def __len__(self):
        return len(self.data) * 2


if __name__ == "__main__":
    dataset = SeaDataset("data/d3.csv")
    print(dataset[474])
    print("---")
    print(dataset[1])

