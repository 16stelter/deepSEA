import numpy as np
import torch
import getopt, sys

from torch import optim
from torch.utils.data import DataLoader
from tqdm.asyncio import tqdm

from models import simplemlp, siam
from dataset import SeaDataset

learning_rate = 0.001
batch_size = 32
epochs = 100
input_shape = 23
model = None

use_hall, use_vel, use_imu, use_fp = True, True, True, True

argumentList = sys.argv[1:]
options = "hm:"
long_options = "help, model:"

try:
    # Parsing argument
    args, vals = getopt.getopt(argumentList, options, long_options)

    for arg, val in args:

        if arg in ("-e", "--epochs"):
            epochs = int(val)
        elif arg in ("-b", "--batch-size"):
            batch_size = int(val)
        elif arg in ("-l", "--learning-rate"):
            learning_rate = float(val)
        elif arg in ("-nh", "--no-hall"):
            input_shape -= 1
            use_hall = False
        elif arg in ("-nv", "--no-vel"):
            input_shape -= 2
            use_vel = False
        elif arg in ("-ni", "--no-imu"):
            input_shape -= 10
            use_imu = False
        elif arg in ("-nf", "--no-pressure"):
            input_shape -= 8
            use_pressure = False
        elif arg in ("-m", "--model"):
            if val == "mlp":
                model = simplemlp.SimpleMlp(input_shape)
            elif val == "siam":
                model = siam.SiamNN(input_shape)
            else:
                print("Model type not known. Valid models are: 'mlp', 'siam'.")
                raise ValueError
        else:
            print("Valid arguments: help, model, epochs, batch-size, learning-rate")
            exit(1)
    if model is None:
        print("Model type not known. Valid models are: 'mlp', 'siam'.")
        exit(1)

except getopt.error as err:
    print(str(err))

opt = optim.Adam(model.parameters(), lr=learning_rate)
min_val_loss = np.inf
best_val_epoch = -1
criterion = torch.nn.MSELoss()

ds = SeaDataset("data/d3.csv", hall=use_hall, vel=use_vel, imu=use_imu, fp=use_fp)
train_size = int(0.8 * len(ds))
test_size = len(ds) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

for e in range(epochs):
    model.train()
    for i, (x, y) in enumerate(tqdm(train_dataset)):
        opt.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        opt.step()
        if i % 100 == 0:
            print("Epoch: {}, batch: {}, loss: {}".format(e, i, loss.item()))
    model.eval()
    val_loss = 0

