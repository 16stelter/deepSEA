import torch.nn as nn
import torch.nn.functional as F

'''
Class defining a MLP network.
input_shape describes the length of the input vector.
'''
class SimpleMlp(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dropout(F.leaky_relu(self.fc1(x)))
        x = self.dropout(F.leaky_relu(self.fc2(x)))
        x = self.dropout(F.leaky_relu(self.fc3(x)))
        y = self.out(x)
        
        return y
