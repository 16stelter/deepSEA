import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import ast

class SeaDataset(Dataset):
    def __init__(self, filename, vel=True, imu=False, fp=False, hist=2, use_poserr=True):
        ### Setting a history value forces poserr to be true!!!
        self.data = pd.read_csv(filename)
        self.vel = vel
        self.imu = imu
        self.fp = fp
        self.hist = hist
        self.use_poserr = use_poserr

    def __getitem__(self, idx):
        # we can assume both joints are the same. thus we have twice the amount of samples.
        if idx % 2 == 0:  # left
            offset = 0
        else:  # right
            offset = 5
        idx = idx//2

        deletes = [4]
        x = self.data.iloc[idx, 1+offset:6+offset].values
        if not self.vel:
            deletes.append(2)
            deletes.append(3)
        if self.use_poserr:
            x[0] = x[0] - x[1]
            deletes.append(1)
        x = np.delete(x, deletes, axis=0)
        if not self.hist == 0:
            if idx >= self.hist:
                if self.vel:
                    for i in range(self.hist):
                        x = np.insert(x, 1, self.data.iloc[idx-1-i, 3+offset], axis=0)
                        x = np.insert(x, 2, self.data.iloc[idx-1-i, 4+offset], axis=0)
                for i in range(self.hist):
                    x = np.insert(x, 0, (self.data.iloc[idx-1-i, 1+offset] - self.data.iloc[idx-1-i, 2+offset]), axis=0)


        if self.imu:
            x = np.concatenate((x, np.asarray(ast.literal_eval(self.data.iloc[idx, 11]))), axis=0)
        if self.fp:
            x = np.concatenate((x, np.asarray(ast.literal_eval(self.data.iloc[idx, 12]))), axis=0)
            x = np.concatenate((x, np.asarray(ast.literal_eval(self.data.iloc[idx, 13]))), axis=0)
        y = self.data.iloc[idx, 5+offset]
        return x, y


if __name__ == "__main__":
    dataset = SeaDataset("data/d3.csv")
    print(dataset[6])
    print("---")
    print(dataset[7])

