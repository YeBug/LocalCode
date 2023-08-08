import torch
import torch.optim
import torch.nn as nn
import data
from tqdm import tqdm
from model import SineModel

train_loader, valid_loader, test_loader = data.get_data_loader(batch_size=8)
model = SineModel()
model.cuda()
optmiz = torch.optim.Adam(model.parameters(), lr=0.0003)
loss = nn.MSELoss()
for i in tqdm(range(1000)):
    for x_train, y_train in train_loader:
        x_train =  x_train.view(8,1).to(torch.float32).cuda()
        y_train = y_train.view(8,1).to(torch.float32).cuda()
        # for x, y in zip(x_train, y_train):
        y_pred = model(x_train)
        loss_val = loss(y_pred, y_train)
        optmiz.zero_grad()
        loss_val.backward()
        optmiz.step()

for x_valid, y_valid in tqdm(valid_loader):
    x_valid =  x_valid.view(8,1).to(torch.float32).cuda()
    y_valid = y_valid.view(8,1).to(torch.float32).cuda()
    with torch.no_grad():
        y_pred = model(x_valid)
        loss_val = loss(y_pred, y_train)
        print(f"x = {x_valid}, \n y_pred = {y_pred}, \n y_valid = {y_valid}")

