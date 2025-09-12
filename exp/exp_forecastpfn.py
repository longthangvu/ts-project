import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import datetime
import time
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import visual
from utils.dtw_metric import accelerated_dtw
from sklearn.preprocessing import StandardScaler, MinMaxScaler
 

from models.ForecastPFN import ForecastPFN, ForecastPFN_PAttn, ForecastPFN_TSMixer, ForecastPFN_Linear

warnings.filterwarnings('ignore')


class Exp_ForecastPFN(Exp_Basic):
    def __init__(self, args):
        super(Exp_ForecastPFN, self).__init__(args)

    def _build_model(self):
        print('loading model')
        pretrained = None
        if self.args.model == 'ForecastPFN':
            pretrained = ForecastPFN()
            pretrained.load_state_dict(torch.load('./checkpoints/forecastpfn_pytorch.pth'))
            # pretrained.load_state_dict(torch.load('./synthetic-data/models/model.pt'))
            # pretrained.load_state_dict(torch.load('./synthetic-data/models/model_f2.pt'))
        elif self.args.model == 'ForecastPFN-PAttn':
            pretrained = ForecastPFN_PAttn()
            pretrained.load_state_dict(torch.load('./synthetic-data/models/model2.pt'))
        elif self.args.model == 'ForecastPFN-TSMixer':
            pretrained = ForecastPFN_TSMixer()
            pretrained.load_state_dict(torch.load('./synthetic-data/models/model2.pt'))
        elif self.args.model == 'ForecastPFN-Linear':
            pretrained = ForecastPFN_Linear()
            pretrained.load_state_dict(torch.load('./synthetic-data/models/model2.pt'))
        return pretrained

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def train(self, setting):
        return 
    
    def _ForecastPFN_time_features(self, ts: np.ndarray):
        if type(ts[0]) == datetime.datetime:
            year = [x.year for x in ts]
            month = [x.month for x in ts]
            day = [x.day for x in ts]
            day_of_week = [x.weekday()+1 for x in ts]
            day_of_year = [x.timetuple().tm_yday for x in ts]
            return np.stack([year, month, day, day_of_week, day_of_year], axis=-1)
        ts = pd.to_datetime(ts)
        return np.stack([ts.year, ts.month, ts.day, ts.day_of_week + 1, ts.day_of_year], axis=-1)

    def _process_tuple(self,x,x_mark,y_mark, 
                       model, horizon):
        """
        x: tensor of shape (n, 1)
        x_mark: tensor of shape (n, d)
        y_mark: tensor of shape (horizon, d)

        where
        n       is the input  sequence length
        horizon is the output sequence length
        d is the dimensionality of the time_stamp (5 for ForecastPFN)
        """
        if torch.all(x == x[0]):
            x[-1] += 1

        history = x.cpu().numpy().reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(history)
        history = scaler.transform(history)

        history_mean = np.nanmean(history[-6:])
        history_std = np.nanstd(history[-6:])
        # local_scale = history_mean + history_std + 1e-4
        local_scale = max(history_mean + history_std, 1e-4)
        history = np.clip(history / local_scale, a_min=0, a_max=1)
        history = np.nan_to_num(history, nan=0.0, posinf=1.0, neginf=-1.0)

        n, d = x_mark.shape
        if n != 100:
            if n > 100:
                target = x_mark[-100:, :]
                history = torch.tensor(history[-100:], dtype=torch.float32)
            else:
                pad_len = 100 - n
                target = torch.cat([torch.zeros(pad_len, d, dtype=x_mark.dtype, device=self.device), x_mark], dim=0)
                history = torch.cat([
                    torch.zeros(pad_len, 1, dtype=torch.float32, device=self.device),
                    torch.tensor(history, dtype=torch.float32, device=self.device)
                ], dim=0)

            history = history.unsqueeze(0).repeat(horizon, 1, 1).squeeze(-1)
            ts = target.unsqueeze(0).repeat(horizon, 1, 1)
        else:
            ts = x_mark.unsqueeze(0).repeat(horizon, 1, 1)
            history = torch.tensor(history, dtype=torch.float32)

        task = torch.ones(horizon, dtype=torch.long, device=self.device)
        target_ts = y_mark[-horizon:, :].unsqueeze(1)
        ts = ts.long()
        target_ts = target_ts.long()


        model_input = {
            'ts': ts,
            'history': history,
            'target_ts': target_ts,
            'task': task
        }

        t1 = time.time()
        pred_vals = model(model_input)
        # print('pred_vals:', pred_vals)
        time_diff = time.time() - t1

        scaled_vals = pred_vals['result'].detach().cpu().numpy().T.reshape(-1)
        scales = pred_vals['scale'].detach().cpu().numpy().reshape(-1)
        scaled_vals = scaler.inverse_transform([scaled_vals * scales])

        return scaled_vals, time_diff
    
    def _ForecastPFN_process_batch(self, model, batch_x, batch_y, batch_x_mark, batch_y_mark):
        preds = []
        trues = []
        for idx, (x, y, x_mark, y_mark) in enumerate(zip(batch_x, batch_y, batch_x_mark, batch_y_mark)):

            pred, time_diff = self._process_tuple(
                x, x_mark, y_mark, model, self.args.pred_len)

            y = y.unsqueeze(-1)[-self.args.pred_len:, :].to(self.device)
            true = y.detach().cpu().numpy()
            
            preds += [pred]
            trues += [true]
        return preds, trues, time_diff

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        test_data.data_stamp = self._ForecastPFN_time_features(
            list(test_data.data_stamp_original['date']))
        self.model.eval()
        preds = []
        trues = []

        # torch.autograd.set_detect_anomaly(True)

        # def nan_hook(module, input, output):
        #     if isinstance(output, torch.Tensor) and torch.isnan(output).any():
        #         print(f"NaN in {module.__class__.__name__}")

        # for name, module in pretrained.named_modules():
        #     module.register_forward_hook(nan_hook)
        
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # print(folder_path)
        target_dim = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # Move to device
                batch_x = batch_x.float().to(self.device)              # [B, seq_len, D]
                batch_y = batch_y.float().to(self.device)              # [B, label_len + pred_len, D]
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Select target dimension only: shape [B, T]
                x = batch_x[:, :, target_dim]                          # [B, seq_len]
                y = batch_y[:, :, target_dim]                          # [B, label_len + pred_len]

                # ForecastPFN expects input shape [B, T], not multivariate
                pred, true, _ = self._ForecastPFN_process_batch(
                    self.model, x, y, batch_x_mark, batch_y_mark
                )

                preds.append(pred)
                trues.append(true)
                input = batch_x.detach().cpu().numpy()
                # if test_data.scale and self.args.inverse:
                    # shape = input.shape
                    # input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                true = np.array(true)
                pred = np.array(pred)
                # print('input shape:', input.shape)
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, 0, :]), axis=0)
                # print()
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, dtw:{}'.format(mse, mae, rmse, mape, mspe, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, dtw:{}'.format(mse, mae, rmse, mape, mspe, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
