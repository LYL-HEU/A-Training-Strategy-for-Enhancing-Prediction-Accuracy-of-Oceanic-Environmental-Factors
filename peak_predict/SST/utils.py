import sys
import numpy as np
import xarray as xr
import torch
from tqdm import tqdm
import scipy.io


num = 6
day = 6


def read_split_data(path_in: str, train_rate: float = 0.85):
    ssta = scipy.io.loadmat('./data/SSTA.mat')['ssta']
    ssta = ssta.transpose(2, 1, 0)
    ssta = np.expand_dims(ssta[:, ::2, ::2], axis=1)
    ssta = ssta[:, :, :-1, :-1]
    in_data = ssta
    out_data = ssta
    data_input = []
    for j in range(num):
        data_input.append(in_data[j:in_data.shape[0] - num - day + j])
    data_input = torch.tensor(np.concatenate(data_input, axis=1))
    data_out = []
    for j in range(day):
        data_out.append(out_data[j + num:out_data.shape[0] - day + j])
    data_out = torch.tensor(np.concatenate(data_out, axis=1))
    # #################################################################################
    num_disorder = 1
    np.random.seed(num_disorder)
    data_num = data_input.shape[0]
    index = np.arange(data_num)  # 生成下标
    np.random.shuffle(index)
    data_input = data_input[index]
    data_out = data_out[index]
    data_input = torch.nan_to_num(data_input)
    data_out = torch.nan_to_num(data_out)
    data_train_number = int(len(data_input) * train_rate)
    data_valid_number = int(len(data_input) - data_train_number)
    #################################################################################
    data_train_input, data_valid_input = torch.split(data_input, [data_train_number, data_valid_number])
    data_train_out, data_valid_out = torch.split(data_out, [data_train_number, data_valid_number])
    print("{} data for training.".format(data_train_number))
    print("{} data for validation.".format(data_valid_number))
    return data_train_input, data_valid_input, data_train_out, data_valid_out


def mae(actual, predicted):
    if len(predicted.shape) == 3:
        # 在最前面添加一个维度，大小为1
        predicted = predicted.unsqueeze(0)
    mask = ~torch.isnan(actual)
    return torch.mean(torch.abs(actual[mask] - predicted[mask]))


import torch.nn as nn


class WeightedMSELoss(nn.Module):
    def __init__(self, alpha=0.8):
        self.alpha = alpha
        super(WeightedMSELoss, self).__init__()

    def forward(self, prediction, target):
        weight = torch.ones_like(prediction)
        weight[torch.abs(target) > self.alpha] = weight[torch.abs(target) > self.alpha] * 9
        loss = torch.mean((prediction - target) ** 2 * weight)
        # loss = torch.sqrt(torch.mean(torch.pow(prediction[:, :, :, 1:-1, 1:-1] - target[:, :, :, 1:-1, 1:-1], 2)))
        # loss_function = torch.nn.L1Loss()
        # loss = loss_function(prediction, target)
        return loss, prediction


def train_one_epoch(model, optimizer, data, device, epoch):
    model.train()
    loss_function = nn.MSELoss()
    accu_loss = torch.zeros(1).to(device)  #
    optimizer.zero_grad()
    data_train = tqdm(data, file=sys.stdout)
    for step, data_ws in enumerate(data_train):
        input, out = data_ws
        input = input.type(torch.FloatTensor)
        out = out.type(torch.FloatTensor)
        pred = model(input.to(device))
        pred = torch.where(out == 0, out, pred.cpu())
        loss = loss_function(pred.to(device), out.to(device))
        loss.backward()
        accu_loss += loss.detach()
        data_train.desc = "[train epoch {}] loss: {:.2f} ".format(epoch, accu_loss.item() / (step + 1))

        if not torch.isfinite(loss):
            print('\nWARNING: non-finite loss, ending training, check your dataset. \n', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, data, device, epoch):
    loss_function = WeightedMSELoss()
    loss_function1 = torch.nn.L1Loss()
    day = 6
    model.eval()
    accu_Loss = torch.zeros(1).to(device)  # 累计损失
    accu_loss = torch.zeros(1).to(device)
    accu_Loss1 = torch.zeros(day).to(device)
    data_train = tqdm(data, file=sys.stdout)
    for step, data_ws in enumerate(data_train):
        input, out = data_ws
        input = input.type(torch.FloatTensor)
        out = out.type(torch.FloatTensor)
        pred = model(input.to(device))

        Data_out = out.clone()
        Data_out[Data_out == 0] = np.nan
        Mae = mae(Data_out.to(device), pred)
        pred = torch.where(out == 0, out, pred.cpu())
        loss, pred1 = loss_function(pred.to(device), out.to(device))
        Loss = torch.zeros(day).to(device)
        for i in range(day):
            Loss[i] = loss_function1(pred1[:, i, :, :].to(device), out[:, i, :, :].to(device))
        accu_Loss1 += Loss
        s = ""
        for i in range(day):
            s += "loss" + str(i + 1) + ": {:.4f} ".format(accu_Loss1[i].item() / (step + 1))
        accu_loss += loss
        accu_Loss += Mae
        data_train.desc = "[valid epoch {}] loss: {:.4f} loss_nan: {:.4f} ".format(epoch,
                                                                                   accu_loss.item() / (step + 1),
                                                                                   accu_Loss.item() / (step + 1)) + s
    return accu_loss.item() / (step + 1)
