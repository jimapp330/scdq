#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@desc:
"""
import copy
import torch
import math
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
from model.admm import proxADMM
from lib.metrics import All_Metrics
from lib.simple_Weight import entropyValue2, perform_bernoulli_trials, store_grad, overwrite_grad
import pandas as pd


class Trainer(object):
    def __init__(self, ktype, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.ktype = ktype
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.device = args.device
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.grad_dims = []
        self.confict_num = 0
        self.num_iterations = 10
        self.model = model.to(args.device)

        self.batach_atten_loss = np.zeros((1,2, self.args.batch_size,))
        if self.args.dataset == 'traffic':
            self.batach_attweight = np.zeros((1,self.args.batch_size, 576))
        else:
            self.batach_attweight = np.zeros((1, self.args.batch_size, 288))
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')

        self.results = pd.DataFrame(columns=['dataset', 'MAE', 'RMSE', 'MAPE', 'steps', 'bestModel'])

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (x_train, y_true) in enumerate(val_dataloader):

                x_train = x_train.to(self.device).float()
                y_true = y_true.to(self.device).float()

                y_true = y_true[:,
                             self.args.label_len:self.args.label_len + self.args.window * self.args.horizon]  # 有label长度
                label = y_true[:, :, 0]

                x_enc = x_train[:, :, 0].unsqueeze(2)
                x_mark_enc = x_train[:, :, 1:]
                x_dec = y_true[:, :, 0].unsqueeze(2)
                x_mark_dec = y_true[:, :, 1:]

                # seq_last = x_enc[:, -1, :].unsqueeze(1)
                # x_enc = x_enc - seq_last

                if self.ktype in ['sadc']:
                    output, decoder_attentions = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                else:
                    output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)

                # output = output + seq_last
                output = output.squeeze(2)


                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)

                loss = self.loss(output, label)
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        if epoch % self.args.log_step == 0:
            # self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
            print('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def check_symmetric(self, a, rtol=1e-05, atol=1e-08):
        return torch.allclose(a, a.T, rtol=rtol, atol=atol)


    def train_epoch(self, epoch):
        # global optimizer
        self.model.train()
        total_loss = 0
        train_epoch_loss = 0

        alo_time = []
        scsq_time = []
        conf_time = []

        if self.args.optim == 'sgd':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=self.args.lr_init)
        elif self.args.optim == 'adam':
            optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=self.args.lr_init, eps=1.0e-8,
                                         weight_decay=0, amsgrad=False)


        for batch_idx, (x_train, y_true) in enumerate(self.train_loader):
            sid = batch_idx*x_train.shape[0]
            x_train = x_train.to(self.device).float()
            y_true = y_true.to(self.device).float()

            y_true = y_true[:,
                         self.args.label_len:self.args.label_len + self.args.window * self.args.horizon]  # 有label长度
            label = y_true[:, :, 0]

            x_enc = x_train[:, :, 0].unsqueeze(2)
            x_mark_enc = x_train[:, :, 1:]
            x_dec = y_true[:, :, 0].unsqueeze(2)
            x_mark_dec = y_true[:, :, 1:]

            # seq_last = x_enc[:,-1,:].unsqueeze(1)
            # x_enc = x_enc - seq_last

            if self.ktype in ['sadc']:
                output, decoder_attentions = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            else:
                output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            # output = output + seq_last
            output = output.squeeze(2)


            self.grad_dims.clear()
            for param in self.model.parameters():
                self.grad_dims.append(param.data.numel())

            dim = x_train.shape
            self.grads = torch.Tensor(sum(self.grad_dims), dim[0])

            self.optimizer.zero_grad()

            #teacher_forcing for RNN encoder-decoder model
            #if teacher_forcing_ratio = 1: use label as input in the decoder for all steps
            if self.args.teacher_forcing:
                global_step = (epoch - 1) * self.train_per_epoch + batch_idx
                teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.args.tf_decay_steps)
            else:
                teacher_forcing_ratio = 0

            if self.args.real_value:
                label = self.scaler.inverse_transform(label)

            if self.ktype != 'normal':
                simple_ev = entropyValue2(decoder_attentions) #[2,64]
                smax = simple_ev.max()
                smin = simple_ev.min()
                sev_scale = (simple_ev - smin)/(smax - smin)
                if sid == 0:
                    self.simple_ev_list = list(simple_ev.detach().cpu().numpy())
                    self.simple_ev_list_scale = list(sev_scale.detach().cpu().numpy())
                ### noise identification
                # sorted_vals = torch.argsort(sev_scale)
                # prop_info = []
                # for prop in [5,10,20,40]:
                #     oridx = []
                #     for ii in range(prop):
                #         oridx.append(bidx[sorted_vals[ii].item()])
                #     oridxnp = np.array(oridx)
                #     prop_info.append(sum(oridxnp > 64))
                # print(f'5%,{prop_info[0]},10%,{prop_info[1]},15%,{prop_info[2]},20%,{prop_info[3]}')
                ###########################
                r_list = perform_bernoulli_trials(sev_scale)

                batch_loss = []
                for i in range(dim[0]):
                    optimizer.zero_grad()
                    b_item_loss = self.loss(output[i,:].cuda(), label[i,:])
                    b_item_loss.backward(retain_graph=True)
                    batch_loss.append(b_item_loss.detach().cpu().item())

                    store_grad(self.model.parameters, self.grads, self.grad_dims, r_list, i)

                if self.ktype == 'sadc':
                    tempG = self.grads.transpose(0, 1).detach().cpu().numpy()
                    alo_time_start = time.time()
                    self.admm = proxADMM(tempG)
                    last_x = np.zeros_like(np.mean(tempG, axis=0))
                    for i in range(0, self.num_iterations):
                        self.admm.step_iterative()
                        #conf = np.dot(tempG, self.admm.getParams())
                        # print('{}Val:'.format(i), conf)
                        # print(-conf.sum())
                        current_x = self.admm.getParams()
                        diff = np.abs(np.sum(last_x - current_x))
                        last_x = current_x
                        if diff < 1e-5:
                            scsq_time.append(i)
                            #print(f'early stop at {i}-th iterations')
                            break
                    alo_time_end = time.time()
                    alo_time.append(alo_time_end - alo_time_start)
                    #print(alo_time_end - alo_time_start)
                    #print(f'current x = {admm.getParams()}')
                    self.grads = torch.tensor(self.admm.getParams().reshape(-1,1))

                # copy gradients back
                overwrite_grad(self.model.parameters, self.grads, self.grad_dims)

                self.optimizer.step()
                total_loss += torch.tensor(batch_loss).sum()

            else:
                # normal approach
                b_loss = self.loss(output, label)
                b_loss.backward()

                self.optimizer.step()
                total_loss += b_loss.item()

        #log information
        if epoch % self.args.log_step == 0:
            print(f"conflict num {self.confict_num}")
            #self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
            #    epoch, batch_idx, self.train_per_epoch, loss.item()))
            train_epoch_loss = total_loss/self.train_per_epoch
            # self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}, tf_ratio: {:.6f}'.format(epoch, train_epoch_loss, teacher_forcing_ratio))
            print(
                '**********Train Epoch {}: averaged Loss: {:.6f}, tf_ratio: {:.6f}'.format(epoch, train_epoch_loss,
                                                                                       teacher_forcing_ratio))

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss, alo_time, scsq_time


    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        alotimes = []
        scsqtimes = []

        ev_List_0 = []
        ev_list_scale = []

        for epoch in range(1, self.args.epochs + 1):
            # print(epoch)
            #epoch_time = time.time()
            train_epoch_loss, alo_time, scsq_time = self.train_epoch(epoch)
            # ev_List_0.append(self.simple_ev_list)
            # ev_list_scale.append(self.simple_ev_list_scale)
            if epoch < 50:
                alotimes.extend(alo_time)
                scsqtimes.extend(scsq_time)
            #print(time.time()-epoch_time)
            #exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            #if self.val_loader == None:
            #val_epoch_loss = train_epoch_loss
            if best_loss - val_epoch_loss > 0.0001:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    # self.logger.info("Validation performance didn\'t improve for {} epochs. "
                    #                 "Training stops.".format(self.args.early_stop_patience))
                    print("Validation performance didn\'t improve for {} epochs. "
                          "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                '''
                if epoch % self.args.log_step == 0:
                    self.logger.info('*********************************Current best model saved!')
                '''
                best_model = copy.deepcopy(self.model.state_dict())

        finish_time = time.time()
        training_time = finish_time - start_time
        print("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        torch.save(best_model, './experiments/best_model_{}_{}_{}_{}_zc.pth'.format(self.args.model, self.args.dataset, self.args.horizon,finish_time))

        self.alotimes = pd.DataFrame(alotimes)
        self.scsqtimes = pd.DataFrame(scsqtimes)

        #test
        self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        mae, rmse, mape = self.test(self.model, self.args, self.test_loader, self.scaler, finish_time=finish_time)

        pf = pd.DataFrame({
            'dataset': [self.args.dataset],
            'MAE': [mae],
            'RMSE': [rmse],
            'MAPE': [mape],
            'steps': [epoch],
            'bestModel': [
                './experiments/best_model_{}_{}_{}_{}.pth'.format(self.args.model, self.args.dataset, self.args.horizon,
                                                                  finish_time)],
        })

        self.results = pd.concat([self.results, pf])

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        print("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, path=None, finish_time=0):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred_t = []
        y_true_t = []
        total_attentions = []
        total_datas = []

        with torch.no_grad():
            for batch_idx, (x_train, y_true) in enumerate(data_loader):

                x_train = x_train.to(args.device).float()
                y_true = y_true.to(args.device).float()

                y_true = y_true[:,
                             args.label_len:args.label_len + args.window * args.horizon]  # 有label长度
                label = y_true[:, :, 0]

                x_enc = x_train[:, :, 0].unsqueeze(2)
                x_mark_enc = x_train[:, :, 1:]
                x_dec = y_true[:, :, 0].unsqueeze(2)
                x_mark_dec = y_true[:, :, 1:]

                # seq_last = x_enc[:, -1, :].unsqueeze(1)
                # x_enc = x_enc - seq_last

                if args.ktype in ['sadc']:
                    output, decoder_attentions = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                else:
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

                # output = output + seq_last
                output = output.squeeze(2)


                y_true_t.append(label)
                y_pred_t.append(output)

        y_pred_t = torch.cat(y_pred_t, dim=0)
        y_true_t = torch.cat(y_true_t, dim=0)

        y_true_t1 = scaler.inverse_transform(y_true_t.cpu())
        y_pred_t1 = scaler.inverse_transform(y_pred_t.cpu())

        print_time = finish_time
        print(print_time)
        f_path = './results/{}_{}_{}_{}_{}_zc1'.format(args.dataset, args.model, args.horizon,
                                                     args.window,
                                                     print_time)
        if not os.path.exists(f_path):
            os.makedirs(f_path)

        np.save(f'{f_path}/{args.dataset}_true1.npy', y_true_t1.cpu().numpy())
        np.save(f'{f_path}/{args.dataset}_pred1.npy', y_pred_t1.cpu().numpy())
        np.save(f'{f_path}/{args.dataset}_true.npy', y_true_t.cpu().numpy())
        np.save(f'{f_path}/{args.dataset}_pred.npy', y_pred_t.cpu().numpy())



        r_metrics = []
        for t in range(y_true.shape[1]):
            mae, rmse, mape = All_Metrics(y_pred_t[:, t], y_true_t[:, t],
                                          args.mae_thresh, args.mape_thresh)
            print("Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape * 100))
            r_metrics.append([t, mae.cpu().item(), rmse.cpu().item(), mape.cpu().item()])


        mae, rmse, mape = All_Metrics(y_pred_t, y_true_t, args.mae_thresh, args.mape_thresh)

        print("Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
            mae, rmse, mape * 100))

        return mae.cpu().item(), rmse.cpu().item(), mape.cpu().item()


    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))