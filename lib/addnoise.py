import numpy as np
import torch
import random


def add_noise2(xx, yy, ratio, device):
    x_num = xx.shape[0]
    n_num = int(x_num * ratio)
    if n_num == 0:
        return xx, yy
    x_len = xx.shape[1]
    y_len = yy.shape[1]

    noiseData = np.random.normal(0.5, 0.8 ** 2, (n_num, x_len + y_len))
    choice_num = np.random.choice(x_num, n_num)
    noise_stamp_x = xx[choice_num, :, 1:]
    noise_stamp_y = yy[choice_num, :, 1:]
    noise_data_xy = torch.hstack((noise_stamp_x, noise_stamp_y))
    noiseData = noiseData.reshape(n_num, x_len + y_len, 1)
    noiseData = torch.tensor(noiseData)
    noiseData = torch.cat((noiseData, noise_data_xy), 2)

    noiseData = torch.tensor(noiseData).to(device)
    noiseData = noiseData.float()
    noiseData_X = noiseData[:, :x_len]
    noiseData_Y = noiseData[:, -y_len:]

    xx = torch.vstack((xx.to(device), noiseData_X))
    yy = torch.vstack((yy.to(device), noiseData_Y))

    rid = torch.randperm(xx.size(0))
    xx = xx[rid, :]
    yy = yy[rid, :]

    return xx, yy, rid

