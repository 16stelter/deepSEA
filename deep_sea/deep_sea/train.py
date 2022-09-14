import numpy as np
import torch
import getopt, sys

from torch import optim
from torch.utils.data import DataLoader
from torchmetrics import R2Score, MeanSquaredLogError
from tqdm import tqdm

from models import simplemlp, siam
from dataset import SeaDataset

import wandb

learning_rate = 0.01
batch_size = 32
epochs = 100
input_shape = 20
model = None


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

use_hall, use_vel, use_imu, use_fp, use_eff = True, True, True, True, True

argumentList = sys.argv[1:]
options = "hm:l:hvitfe:b:d:"
long_options = "help, model:, learning-rate:, no-hall, no-vel, no-imu, no-pressure, epochs:, batch-size:, hist-len:"

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
        elif arg in ("-h", "--no-hall"):
            input_shape -= 2
            use_hall = False
        elif arg in ("-v", "--no-vel"):
            input_shape -= 2
            use_vel = False
        elif arg in ("-i", "--no-imu"):
            input_shape -= 9
            use_imu = False
        elif arg in ("-t", "--no-effort"):
            input_shape -= 1
            use_eff = False
        elif arg in ("-f", "--no-pressure"):
            input_shape -= 6
            use_fp = False
        elif arg in ("-d", "--hist-len"):
            input_shape += int(val)
            if use_hall:
                input_shape += int(val)
            hist_len = int(val)
        elif arg in ("-m", "--model"):
            modelname = val
            if not (use_vel or use_hall):
                input_shape += 1
            if val == "mlp":
                model = simplemlp.SimpleMlp(input_shape).to(DEVICE)
                ds = SeaDataset("../../data/free/d0freecn.csv", siam=False, hall=use_hall, vel=use_vel, eff=use_eff, imu=use_imu, fp=use_fp, hist_len=hist_len)
            elif val == "siam":
                model = siam.SiamNN(input_shape).to(DEVICE)
                ds = SeaDataset("../../data/free/d0freecn.csv", siam=True, hall=use_hall, vel=use_vel, eff=use_eff, imu=use_imu, fp=use_fp, hist_len=hist_len)
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
criterion = torch.nn.MSELoss()  # RMSE

wandb.init(project="deepsea", config={"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                                      "use_hall": use_hall, "use_vel": use_vel, "use_imu": use_imu, "use_fp": use_fp,
                                      "model": model.__class__.__name__})

torch.set_flush_denormal(True)

train_size = int(0.8 * len(ds))
test_size = len(ds) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

r2 = R2Score().to(DEVICE)
msle = MeanSquaredLogError().to(DEVICE)

no_impr_count = 0
for e in range(epochs):
    model.train()
    for i, s in enumerate(tqdm(train_loader)):
        x = s[0].to(DEVICE)
        if modelname == "mlp":
            y = s[1].unsqueeze(1).to(DEVICE)
            y_pred = model(x)
            loss = criterion(y_pred, y)
        elif modelname == "siam":
            x1 = s[1].to(DEVICE)
            y = s[2].unsqueeze(1).to(DEVICE)
            x1_pred, y_pred = model(x, x1, y)
            y_loss = criterion(y_pred, y)
            x1_loss = criterion(x1_pred, x1)
            loss = y_loss + 0.1 * x1_loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if i % 500 == 0:
            wandb.log({"train_loss": loss.item()})
            wandb.log({"train_y_loss": y_loss.item()})
            wandb.log({"train_x1_loss": x1_loss.item()})
            # wandb.log({"train_r2": r2(y_pred, target)})
            # wandb.log({"train_msle": msle(abs(y_pred), abs(target))})
    model.eval()
    val_loss = 0
    val_r2 = 0
    val_msle = 0
    for step, s in enumerate(tqdm(test_loader)):
        x = s[0].to(DEVICE)
        if modelname == "mlp":
            y = s[1].unsqueeze(1).to(DEVICE)
            pred = model(x)
            target = y
            val_loss += criterion(y_pred, y)
        elif modelname == "siam":
            x1 = s[1].to(DEVICE)
            y = s[2].unsqueeze(1).to(DEVICE)
            x1_pred, y_pred = model(x, x1, y)
            y_loss = criterion(y_pred, y)
            x1_loss = criterion(x1_pred, x1)
            val_loss += y_loss + 0.1 * x1_loss
        # val_r2 += r2(pred, target)
        # val_msle += msle(abs(pred), abs(target))
    val_loss = val_loss / len(test_loader)
    val_r2 = val_r2 / len(test_loader)
    val_msle = val_msle / len(test_loader)
    no_impr_count += 1 if val_loss > min_val_loss else 0
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), "checkpoints/p_{}_{}.pt".format(model.__class__.__name__, e))
        no_impr_count = 0
    wandb.log({"val_loss": val_loss})
    # wandb.log({"val_r2": val_r2})
    # wandb.log({"val_msle": val_msle})
    print("Validation. Epoch: {}, val_loss: {}".format(e, val_loss))
    # wandb.watch(model, log_freq=100)
    if no_impr_count > 100:
        print("Early stopping")
        torch.save(model.state_dict(), "checkpoints/p_{}_{}.pt".format(model.__class__.__name__, e))
        break


