import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim

class SimpleMlp(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = F.softsign(self.fc1(x))
        x = F.softsign(self.fc2(x))
        x = F.softsign(self.fc3(x))
        y = F.sigmoid(self.out(x))

        return y
