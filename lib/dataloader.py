#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@desc:
"""
import torch
import torch.utils.data
import pandas as pd
import numpy as np
from lib.add_window import Add_Window_Horizon, Add_Window_Horizon_stamp
from lib.load_dataset import load_st_dataset
from lib.normalization import NScalar, MinMax01Scaler, MinMax11Scaler, StandardScalar, ColumnMinMaxScaler
from lib.timefeatures import time_features

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum.values, maximum.values)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScalar(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScalar()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        # column min max, to be depressed
        # note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler


def split_data_by_ratio_stamp(data, dTime, args):
    train_data = []
    test_data = []
    val_data = []
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    data_len = len(data)

    timeenc = args.timeenc
    freq = args.freq
    dTime['date'] = pd.to_datetime(dTime.date)
    data_stamp = time_features(dTime, timeenc=timeenc, freq=freq)
    data_stamp = torch.tensor(data_stamp)


    if data_len > 12 * 30 * 24 + 8 * 30 * 24:
            #border1s = [4 * 30 * 24,     12 * 30 * 24,               12 * 30 * 24 + 6 * 30 * 24]
            #border2s = [12 * 30 * 24,    12 * 30 * 24 + 2 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
            border1s = [0,            12 * 30 * 24,               12 * 30 * 24 + 4 * 30 * 24]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

            train_data = data[border1s[0]:border2s[0]]
            test_data = data[border1s[1]:border2s[1]]
            val_data = data[border1s[2]:border2s[2]]

            train_data_mark = data_stamp[border1s[0]:border2s[0]]
            test_data_mark = data_stamp[border1s[1]:border2s[1]]
            val_data_mark = data_stamp[border1s[2]:border2s[2]]

    else:
            test_data = data[-int(data_len*test_ratio):]
            val_data = data[-int(data_len*(test_ratio + val_ratio)): -int(data_len*test_ratio)]
            train_data = data[:-int(data_len*(test_ratio+val_ratio))]

            test_data_mark = data_stamp[-int(data_len*test_ratio):]
            val_data_mark = data_stamp[-int(data_len*(test_ratio + val_ratio)): -int(data_len*test_ratio)]
            train_data_mark = data_stamp[:-int(data_len*(test_ratio+val_ratio))]

    # test_data = data[-int(data_len*test_ratio):]
    # val_data = data[-int(data_len*(test_ratio + val_ratio)): -int(data_len*test_ratio)]
    # train_data = data[:-int(data_len*(test_ratio+val_ratio))]

    # test_data_mark = data_stamp[-int(data_len*test_ratio):]
    # val_data_mark = data_stamp[-int(data_len*(test_ratio + val_ratio)): -int(data_len*test_ratio)]
    # train_data_mark = data_stamp[:-int(data_len*(test_ratio+val_ratio))]

    return train_data, val_data, test_data, train_data_mark, test_data_mark, val_data_mark

def split_data_by_ratio(data, args):
    train_data = []
    test_data = []
    val_data = []
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    data_len = len(data)

    if data_len > 12 * 30 * 24 + 8 * 30 * 24:
            #border1s = [4 * 30 * 24,     12 * 30 * 24,               12 * 30 * 24 + 6 * 30 * 24]
            #border2s = [12 * 30 * 24,    12 * 30 * 24 + 2 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
            border1s = [0,            12 * 30 * 24,               12 * 30 * 24 + 4 * 30 * 24]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

            train_data = data[border1s[0]:border2s[0]]
            test_data = data[border1s[1]:border2s[1]]
            val_data = data[border1s[2]:border2s[2]]
    else:
            test_data = data[-int(data_len*test_ratio):]
            val_data = data[-int(data_len*(test_ratio + val_ratio)): -int(data_len*test_ratio)]
            train_data = data[:-int(data_len*(test_ratio+val_ratio))]



    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):

    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def data_loader_stamp(X, Y, X_stamp, Y_stamp, batch_size, shuffle=True, drop_last=True):
    # cuda = True if torch.cuda.is_available() else False
    # TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # X = X.float()
    # Y = Y.float()
    # X_stamp = X_stamp.float()
    # Y_stamp = Y_stamp.float()
    #
    # X, Y ,X_stamp, Y_stamp = TensorFloat(X), TensorFloat(Y), TensorFloat(X_stamp), TensorFloat(Y_stamp)
    if len(X.shape) == 2:
        X = X.unsqueeze(2)
        Y = Y.unsqueeze(2)
    X_and_stamp = torch.cat((X, X_stamp), dim=2)
    Y_and_stamp = torch.cat((Y, Y_stamp), dim=2)

    data = torch.utils.data.TensorDataset(X_and_stamp, Y_and_stamp)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def delete_outlier(array_x, array_y):
    '''
    :param array_x: Samples*Lag*Dim
    :param array_y: Samples*Horizon
    :return: arrays without outliers
    '''
    index = np.argwhere(array_x == -99)
    index_ = np.unique(index[:, 0])
    x = np.delete(array_x, index_, axis=0)
    y = np.delete(array_y, index_, axis=0)
    index__ = np.unique(np.argwhere(y == -99)[:, 0])
    x_ = np.delete(x, index__, axis=0)
    y_ = np.delete(y, index__, axis=0)

    return x_, y_

def get_dataloader_stamp(args, normalizer='std',feature='M'):
    data, dTime = load_st_dataset(args.dataset, feature)
    # data = data.to(args.device)
    data, scaler = normalize_dataset(data.float(), normalizer, args.column_wise)

    data_train, data_val, data_test, dtime_train, dtime_val, dtime_test = split_data_by_ratio_stamp(data, dTime, args)
    x_tra, y_tra, x_tra_stamp, y_tra_stamp = Add_Window_Horizon_stamp(data_train, dtime_train, args, 1)
    # x_tra, y_tra = delete_outlier(x_tra_, y_tra_)
    x_val, y_val, x_val_stamp, y_val_stamp = Add_Window_Horizon_stamp(data_val, dtime_train, args, 0)
    # x_val, y_val = delete_outlier(x_val_, y_val_)
    x_test, y_test, x_test_stamp, y_test_stamp = Add_Window_Horizon_stamp(data_test, dtime_train, args, 0)
    # x_test, y_test = delete_outlier(x_test_, y_test_)
    print('Train:', x_tra.shape, y_tra.shape)
    print('Val:', x_val.shape, y_val.shape)
    print('Test:', x_test.shape, y_test.shape)

    train_dataloader = data_loader_stamp(x_tra, y_tra, x_tra_stamp, y_tra_stamp, args.batch_size, shuffle=True,
                                         drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader_stamp(x_val, y_val, x_val_stamp, y_val_stamp, args.batch_size, shuffle=False,
                                           drop_last=False)
    test_dataloader = data_loader_stamp(x_test, y_test, x_test_stamp, y_test_stamp, args.batch_size, shuffle=False,
                                        drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader, scaler



def get_dataloader(args, normalizer='max', feature='M'):

    data, _ = load_st_dataset(args.dataset, feature)
    data, scaler = normalize_dataset(data.float(), normalizer, args.column_wise)

    data_train, data_val, data_test = split_data_by_ratio(data, args)

    x_tra, y_tra = Add_Window_Horizon(data_train, args)

    x_val, y_val = Add_Window_Horizon(data_val, args)
    x_test, y_test = Add_Window_Horizon(data_test, args)
    print('Train:', x_tra.shape, y_tra.shape)
    print('Val:', x_val.shape, y_val.shape)
    print('Test:', x_test.shape, y_test.shape)

    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=True, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, 32, shuffle=True, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader, scaler

