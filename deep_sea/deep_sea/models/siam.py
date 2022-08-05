import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim


class SiamNN(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.s_dense1 = nn.Linear(input_shape, 64)
        self.s_dense2 = nn.Linear(64, 64)
        self.s_dense3 = nn.Linear(64, 64)
        self.s_out = nn.Linear(64, 64)

        self.pred_xt1 = nn.Linear(64, input_shape)
        self.pred_ut = nn.Linear(64, 1)

    def forward(self, xt, xt1, ut):
        latent_xt = self.sister_forward(xt)
        latent_xt1 = self.sister_forward(xt1)

        pxt1 = self.pred_xt1(torch.cat((latent_xt, ut)))
        put = self.pred_ut(torch.cat((latent_xt, latent_xt1)))

        return torch.cat((pxt1, put))

    def predict_ut(self, xt, xt1):
        latent_xt = self.sister_forward(xt)
        latent_xt1 = self.sister_forward(xt1)
        return self.pred_ut(torch.cat((latent_xt, latent_xt1)))

    def predict_xt1(self, xt, ut):
        latent_xt = self.sister_forward(xt)
        return self.pred_xt1(torch.cat((latent_xt, ut)))

    def sister_forward(self, x):
        x = self.s_dense1(x)
        x = F.relu(x)
        x = self.s_dense2(x)
        x = F.relu(x)
        x = self.s_dense3(x)
        x = F.relu(x)
        x = self.s_dense1(x)
        latent = F.sigmoid(x)

        return latent


if __name__ == "__main__":
    net = SiamNN(16)
