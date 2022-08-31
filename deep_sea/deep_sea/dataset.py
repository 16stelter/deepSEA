import pandas as pd
import torch
from numpy import double
from torch.utils.data import Dataset
import numpy as np
import ast

class SeaDataset(Dataset):
    def __init__(self, filename, hall=True, vel=False, imu=False, fp=False, eff=False, forcecontrol=False, hist_len=0, use_poserr=False):
        self.data = pd.read_csv(filename)
        self.hall = hall
        self.vel = vel
        self.imu = imu
        self.fp = fp
        self.eff = eff
        self.forcecontrol = forcecontrol
        self.use_poserr = use_poserr
        self.hist_len = hist_len

    def __getitem__(self, idx):
        # we can assume both joints are the same. thus we have twice the amount of samples.
        if idx % 2 == 0:  # left
            offset = 0
        else:  # right
            offset = 5
        idx = idx//2-1

        x = self.data.iloc[idx+1, 2+offset]  # target position
        x = np.append(x, self.data.iloc[idx, 1+offset])  # current motor position
        for i in range(self.hist_len):
            x = np.append(x, self.data.iloc[idx-i, 1+offset])  # motor pos history
        if self.hall:
            x = np.append(x, self.data.iloc[idx, 2+offset])  # hall position current
            for i in range(self.hist_len):
                x = np.append(x, self.data.iloc[idx-i, 2+offset])  # hall pos history
        if self.use_poserr:
            if not self.hall:
                x = np.append(x, self.data.iloc[idx, 2+offset])
            x[0] = x[0] - x[2]
            x = np.delete(x, 2, axis=0)
        if self.vel:
            x = np.append(x, self.data.iloc[idx, 3+offset])  # velocity
            if self.hall:
                x = np.append(x, self.data.iloc[idx, 4+offset])  # velocity
        if self.eff:
            x = np.append(x, self.data.iloc[idx, 5+offset])
        if self.imu:
            x = np.concatenate((x, np.asarray(ast.literal_eval(self.data.iloc[idx, 11]))), axis=0)  # imu
        if self.fp:
            x = np.concatenate((x, np.asarray(ast.literal_eval(self.data.iloc[idx, 12]))), axis=0)  # left foot
            x = np.concatenate((x, np.asarray(ast.literal_eval(self.data.iloc[idx, 13]))), axis=0)  # right foot
        if self.forcecontrol:
            y = np.array(self.data.iloc[idx, 5+offset])
        else:
            y = np.array(self.data.iloc[idx+1, 1+offset])

        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    def __len__(self):
        return len(self.data) * 2


if __name__ == "__main__":
    dataset = SeaDataset("data/pid_batt_d0c.csv")
    print(dataset[6])
    print("---")
    print(dataset[1])

