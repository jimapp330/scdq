# -*- coding: utf-8 -*-
"""
@author: jimapp
@desc:
"""
import os
import sys
import torch
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)
import pandas as pd
import numpy as np
import argparse
import configparser
from datetime import datetime
import torch.nn as nn
from others.attn_lstm import Attn_LSTM
from lib.dataloader import get_dataloader_stamp
import time
from lib.BasicTrainer_sw import Trainer
from torch.utils.data.dataloader import DataLoader
from lib.dataloader import data_loader
import random
import pickle
from lib.addnoise import add_noise2

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True




today = time.time()
seed_num = 2024
for tt in range(3):
    setup_seed(seed_num+tt*100)
    print(f'--------------{tt}--------------------inter----')
    for DATASET in ['powerLoad', 'PEMS04', 'ETTh1', 'electricity', 'ETTh2', 'exchange_rate']:
        print(DATASET)      

        Mode = 'Train'  # Train or test
        DEBUG = 'True'
        optim = 'sgd'
        DEVICE = 'cuda:0'
        MODEL = 'lstm-att'
        ktype = 'sadc' # sadc
        noise_ratio = 0  # 0 0.3
        feature = 'S'
        task_name = 'long_term_forecast'
        finish_time = 2024
        # config_file
        config_file = 'configs/{}/{}.conf'.format(DATASET, MODEL)
        config = configparser.ConfigParser()
        config.read(config_file)

        from lib.metrics import MAE_torch


        def masked_mae_loss(scaler, mask_value):
            def loss(preds, labels):
                if scaler:
                    preds = scaler.inverse_transform(preds)
                    labels = scaler.inverse_transform(labels)
                mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
                return mae

            return loss


        # parser
        args = argparse.ArgumentParser(description='arguments')

        # basic config
        args.add_argument('--task_name', type=str, default=task_name,
                          help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
        args.add_argument('--is_training', type=int, default=1, help='status')
        args.add_argument('--model_id', type=str, default='test', help='model id')
        args.add_argument('--model', type=str, default=MODEL,
                          help='model name, options: [Autoformer, Transformer, TimesNet]')
        args.add_argument('--dataset', default=DATASET, type=str)
        args.add_argument('--mode', default=Mode, type=str)
        args.add_argument('--optim', default=optim, type=str)
        args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
        # args.add_argument('--debug', default=DEBUG, type=eval)
        # args.add_argument('--model', default=MODEL, type=str)
        # args.add_argument('--cuda', default=True, type=bool)

        # data
        args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
        args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
        args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
        args.add_argument('--stamp', default=config['data']['stamp'], type=bool)
        args.add_argument('--freq', type=str, default=config['data']['freq'],
                          help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        # model
        args.add_argument('--top_k', type=int, default=config['model']['top_k'], help='for TimesBlock')
        args.add_argument('--num_kernels', type=int, default=config['model']['num_kernels'], help='for Inception')
        args.add_argument('--dec_in', default=config['model']['dec_in'], type=int)
        args.add_argument('--enc_in', default=config['model']['enc_in'], type=int)
        args.add_argument('--c_out', type=int, default=config['model']['c_out'], help='output size')
        args.add_argument('--d_model', default=config['model']['d_model'], type=int)
        args.add_argument('--n_heads', type=int, default=config['model']['n_heads'], help='num of heads')
        args.add_argument('--d_ff', type=int, default=config['model']['d_ff'], help='dimension of fcn')
        args.add_argument('--moving_avg', type=int, default=config['model']['moving_avg'],
                          help='window size of moving average')
        args.add_argument('--factor', type=int, default=config['model']['factor'], help='attn factor')
        args.add_argument('--embed', type=str, default=config['model']['embed'],
                          help='time features encoding, options:[timeF, fixed, learned]')
        args.add_argument('--dropout', type=float, default=config['model']['dropout'], help='dropout')
        args.add_argument('--timeenc', type=int, default=config['model']['timeenc'], help='dropout')

        # args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
        # args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
        args.add_argument('--e_layers', type=int, default=config['model']['e_layers'], help='num of encoder layers')
        args.add_argument('--d_layers', type=int, default=config['model']['d_layers'], help='num of decoder layers')

        # args.add_argument('--layer_size', default=config['model']['layer_size'], type=int)
        # args.add_argument('--res_channels', default=config['model']['res_channels'], type=int)
        # args.add_argument('--skip_channels', default=config['model']['skip_channels'], type=int)
        args.add_argument('--column_wise', default=config['model']['column_wise'], type=bool)
        args.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        args.add_argument('--activation', type=str, default='gelu', help='activation')
        args.add_argument('--distil', action='store_false',
                          help='whether to use distilling in encoder, using this argument means not using distilling',
                          default=True)
        # train
        args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
        args.add_argument('--seed', default=config['train']['seed'], type=int)
        args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
        args.add_argument('--epochs', default=config['train']['epochs'], type=int)
        args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
        args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
        args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
        args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
        args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
        args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
        args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
        args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
        args.add_argument('--teacher_forcing', default=config['train']['teacher_forcing'], type=eval)
        args.add_argument('--tf_decay_steps', default=config['train']['tf_decay_steps'], type=int,
                          help='teacher forcing decay steps')
        args.add_argument('--real_value', default=config['train']['real_value'], type=eval,
                          help='use real value for loss calculation')
        # test
        args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
        args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
        # log
        # args.add_argument('--log_dir', default='./', type=str)
        args.add_argument('--log_step', default=config['log']['log_step'], type=int)
        # args.add_argument('--plot', default=config['log']['plot'], type=eval)

        # forecasting task
        args.add_argument('--lag', default=config['data']['lag'], type=int)
        args.add_argument('--step', default=config['data']['step'], type=int)
        args.add_argument('--window', default=config['data']['window'], type=int)
        args.add_argument('--interval', default=config['data']['interval'], type=int)
        args.add_argument('--horizon', default=config['data']['horizon'], type=int)
        args.add_argument('--label_len', type=int, default=config['data']['label_len'], help='start token length')

        # GPU
        args.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        args.add_argument('--gpu', type=int, default=0, help='gpu')
        args.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        args.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

        args = args.parse_args()


        ######################GPU#################################
        if args.device == 'cpu':
            args.device = 'cpu'
            args.use_gpu = False
        elif args.device == 'cuda:0':
            if torch.cuda.is_available():
                torch.cuda.set_device(int(args.device[5]))
                args.use_gpu = True
            else:
                args.device = 'cpu'
                args.use_gpu = False


        args.noise_ratio = noise_ratio



        if os.path.exists(f'./data/{DATASET}/train_loader_orig_stamp.pkl'):
            train_loader_orig = pickle.load(open(f'./data/{DATASET}/train_loader_orig_stamp.pkl', "rb"))
            val_loader = pickle.load(open(f'./data/{DATASET}/val_loader_stamp.pkl', "rb"))
            test_loader = pickle.load(open(f'./data/{DATASET}/test_loader_stamp.pkl', "rb"))
            scaler = pickle.load(open(f'./data/{DATASET}/scaler_stamp.pkl', "rb"))
        else:
            train_loader_orig, val_loader, test_loader, scaler = get_dataloader_stamp(args,
                                                                                          normalizer=args.normalizer,
                                                                                          feature=feature)
            with open(f'./data/{DATASET}/train_loader_orig_stamp.pkl', 'wb') as fh:
                pickle.dump(train_loader_orig, fh)
            with open(f'./data/{DATASET}/val_loader_stamp.pkl', 'wb') as fh:
                pickle.dump(val_loader, fh)
            with open(f'./data/{DATASET}/test_loader_stamp.pkl', 'wb') as fh:
                pickle.dump(test_loader, fh)
            with open(f'./data/{DATASET}/scaler_stamp.pkl', 'wb') as fh:
                pickle.dump(scaler, fh)
        # init model
        args.stamp = True

        ###############add nosie samples##########################

        if noise_ratio == 0:
            train_loader = train_loader_orig
        elif noise_ratio != 0:
            has_noise_flag = 0
            if args.stamp == True and os.path.exists(
                    f'./data/{DATASET}/train_loader_orig_stamp_{str(noise_ratio)}.pkl'):
                has_noise_flag = 1
                train_loader = pickle.load(
                    open(f'./data/{DATASET}/train_loader_orig_stamp_{str(noise_ratio)}.pkl', "rb"))
            elif args.stamp == False and os.path.exists(f'./data/{DATASET}/train_loader_orig_{str(noise_ratio)}.pkl'):
                has_noise_flag = 1
                train_loader = pickle.load(open(f'./data/{DATASET}/train_loader_orig_{str(noise_ratio)}.pkl', "rb"))

            if has_noise_flag == 0:
                train_set = train_loader_orig.dataset
                train_set = DataLoader(dataset=train_set, batch_size=1, shuffle=False)
                trainX = []
                trainY = []
                for index, (X, y_true) in enumerate(train_set):
                    trainX.append(X.cpu().numpy())
                    trainY.append(y_true.cpu().numpy())

                newX, newY, bidx = add_noise2(torch.tensor(trainX).squeeze(), torch.tensor(trainY).squeeze(),
                                              args.noise_ratio, args.batch_size)

                train_loader = data_loader(newX, newY, args.batch_size, shuffle=False, drop_last=True)

                # save
                if args.stamp == True:
                    with open(f'./data/{DATASET}/train_loader_orig_stamp_{str(noise_ratio)}.pkl', 'wb') as fh:
                        pickle.dump(train_loader, fh)
                elif args.stamp == False:
                    with open(f'./data/{DATASET}/train_loader_orig_{str(noise_ratio)}.pkl', 'wb') as fh:
                        pickle.dump(train_loader, fh)


        args.ktype = ktype
        ##############  run  ######################################
        for hh in [4]:
            if hh == 0:
                args.horizon = 1
                args.window = 1
            else:
                args.horizon = hh
                args.window = 6

            ##############变化######################################
            args.seq_len = args.interval * args.lag  # 144
            args.pred_len = args.window * args.horizon  # 18
            args.noise_ratio = noise_ratio

            #######################################################
            # config log path
            current_time = datetime.now().strftime('%Y%m%d%H%M%S')
            current_dir = os.path.dirname(os.path.realpath(__file__))
            log_dir = os.path.join(current_dir, 'experiments', args.dataset, current_time)
            args.log_dir = log_dir

            if feature == 'S':
                # args.en_input_dim = 1
                args.enc_in = 1
                args.dec_in = 1
                args.c_out = 1
            #######################################################

            model = Attn_LSTM(args)

            for p in model.parameters():
                if p.dim() > 1:
                    # nn.init.xavier_uniform_(p)
                    nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.uniform_(p)

            # init loss function, optimizer
            if args.loss_func == 'mask_mae':
                loss = masked_mae_loss(scaler, mask_value=0.0)
            elif args.loss_func == 'mae':
                loss = torch.nn.L1Loss().to(args.device)
            elif args.loss_func == 'mse':
                loss = torch.nn.MSELoss().to(args.device)
            else:
                raise ValueError


            quality = torch.tensor(1)
            if ktype in ['normal', 'sadc']:
                # model setup
                if args.optim == 'sgd':
                    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init)
                elif args.optim == 'adam':
                    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                                                 weight_decay=0, amsgrad=False)

                # optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr_init, weight_decay=0)
                # learning rate decay
                lr_scheduler = None
                if args.lr_decay:
                    print('Applying learning rate decay.')
                    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
                    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                                        milestones=lr_decay_steps,
                                                                        gamma=args.lr_decay_rate)
                # start training
                trainer = Trainer(ktype, model, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                                  args, lr_scheduler=lr_scheduler)

                if args.mode == 'Train':
                    trainer.train()


                    output_path = './ns_results/%02d' % (seed_num)
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    filename = f'{args.dataset}_{args.model}_{ktype}_{optim}_{args.early_stop_patience}_ns_{args.noise_ratio}_{args.horizon * args.window}_{today}.csv'

                    if tt == 0:
                        trainer.results.to_csv(
                            f'{output_path}/{filename}',
                            mode='a',
                            header=True
                        )
                    else:
                        trainer.results.to_csv(
                            f'{output_path}/{filename}',
                            mode='a',
                            header=False
                        )

                elif args.mode == 'test':                   

                    finish_time = 1704960048.2267923
                    args.dataset = 'electricity'
                    model.load_state_dict(torch.load('./experiments/best_model_{}_{}_{}_{}_zc.pth'.format(args.model,
                                                                                                          args.dataset,
                                                                                                          args.horizon,
                                                                                                          finish_time)))

                    train_loader_orig, val_loader, test_loader, scaler = get_dataloader_stamp(args,
                                                                                              normalizer=args.normalizer,
                                                                                              feature=feature)
                    if not os.path.exists(f'./data/{args.dataset}/test/'):
                        os.makedirs(f'./data/{args.dataset}/test/')
                    with open(f'./data/{args.dataset}/test/val_loader_stamp.pkl', 'wb') as fh:
                        pickle.dump(val_loader, fh)
                    with open(f'./data/{args.dataset}/test/test_loader_stamp.pkl', 'wb') as fh:
                        pickle.dump(test_loader, fh)
                    with open(f'./data/{args.dataset}/test/scaler_stamp.pkl', 'wb') as fh:
                        pickle.dump(scaler, fh)
                    # init model
                    args.stamp = True
                    
                    mae, rmse, mape = trainer.test(model, trainer.args, test_loader, scaler, finish_time=finish_time)

                    path = f'{args.dataset}_{args.model}_4_6_{finish_time}_zc1'
                    dataset = args.dataset

                    wind_pred = np.load('./results/{}/{}_pred.npy'.format(path, args.dataset))
                    wind_true = np.load('./results/{}/{}_true.npy'.format(path, args.dataset))
                    
                    pd.DataFrame(np.concatenate([wind_true.reshape(-1, 1), wind_pred.reshape(-1, 1)], axis=1),
                                 columns=['true', 'pred']).to_csv(f'./results/{path}/scsq_{dataset}_true_pred.csv')
                    wind_pred[:, :1] = wind_pred[:, :1] / 2
                   



                del quality, model
                torch.cuda.empty_cache()

        del train_loader, val_loader, test_loader, scaler
        torch.cuda.empty_cache()