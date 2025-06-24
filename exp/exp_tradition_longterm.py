import warnings, os
import torch
import numpy as np
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.traditional_models import (naive_last, naive_mean, naive_seasonal, 
                                       ets_forecast, dhr_arima_forecast, naive2_forecast)

from utils.dtw_metric import accelerated_dtw
from utils.metrics import metric
from utils.tools import visual
warnings.filterwarnings("ignore")

class Exp_Tradition_Longterm(Exp_Basic):
    def __init__(self, args):
        super(Exp_Tradition_Longterm, self).__init__(args)


    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _build_model(self):
        model_map = {
            'naive_last': naive_last,
            'naive_mean': naive_mean,
            'naive_seasonal': naive_seasonal,
            'naive2': naive2_forecast,
            'dhr_arima': dhr_arima_forecast,
            'ets': ets_forecast,
        }
        if self.args.model not in model_map:
            raise ValueError(f"Unknown model: {self.args.model}")
        return model_map[self.args.model]

    def _get_season_len(self):
        season_len_map = {
            'ETTh1':        24 * 7,   # Weekly seasonality
            'ETTh2':        24 * 7,   # Weekly seasonality
            'ETTm1':        96,       # 1 day of 15 min data
            'ETTm2':        96,       # 1 day of 15 min data
            'exchange':     7,        # 1 week of daily data
            'traffic':      24 * 7,   # Weekly seasonality
            'electricity':  24 * 7,   # Weekly seasonality
            'ili':          52,       # Weekly seasonality
            'weather':      24 * 7,   # Weekly seasonality
            'custom':       24 * 7,   # Weekly seasonality
        }
        if self.args.data not in season_len_map:
            raise ValueError(f"Unknown dataset: {self.args.data}")
        return season_len_map[self.args.data]
    
    def train(self):
        return 
    
    def test(self, setting, test=0):
        print(self.args)
        # Load data
        test_data, test_loader = self._get_data(flag='test')
        
        preds, trues = [], []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)        
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(test_loader):
                batch_x = batch_x.float() # [B, seq_len, D]
                batch_y = batch_y.float() # [B, label_len + pred_len, D]
                
                # true = batch_y[:, -self.args.pred_len:, :]  # [B, pred_len]
                target_dim = 0  # or infer from args.target
                true = batch_y[:, -self.args.pred_len:, target_dim].unsqueeze(-1)  # [B, pred_len, 1]

                B, pred_len, D = true.shape

                batch_preds = []
                for b in range(B):
                    series_preds = []
                    for d in [target_dim]:
                        x = batch_x[b, :, d].clone()  # [seq_len]
                        
                        try:
                            if self.args.model in ['naive_last', 'naive_mean']:
                                y_pred = self.model(x.unsqueeze(0), pred_len)[0]
                            else:
                                season_len = self._get_season_len()
                                y_pred = self.model(x.unsqueeze(0), pred_len, season_len)[0]
                        except Exception as e:
                            print(f"Error processing series {i}: {e}")
                            y_pred = torch.full((pred_len,), x[-1])
                        # print("x", x)
                        # print("x mean:", x.mean().item())
                        # print("y_pred:", y_pred)
                        # print("y_true:", true[b, :, 0])

                        series_preds.append(y_pred.unsqueeze(-1))
                    series_preds = torch.cat(series_preds, dim=-1)  # [pred_len, D]
                    batch_preds.append(series_preds.unsqueeze(0))  # [1, pred_len, D]
                batch_preds = torch.cat(batch_preds, dim=0)  # [B, pred_len, D]
                preds.append(batch_preds.numpy())  # [B, pred_len, D]
                trues.append(true.numpy())  # [B, pred_len, D]
                
                # if i % 20 == 0:
                input = batch_x.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = input.shape
                    input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], batch_preds[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    
        preds = np.concatenate(preds, axis=0)  # [N, pred_len]
        trues = np.concatenate(trues, axis=0)  # [N, pred_len]
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
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
        return 
        
    
    def test_full_series(self, setting):
        # Load data
        train_data = self._get_data(flag='train')[0]
        val_data = self._get_data(flag='val')[0]
        test_data = self._get_data(flag='test')[0]
        
        # Build input series: train + val (transpose to shape [D, T])
        full_x = np.concatenate([train_data.data_y, val_data.data_y], axis=0)
        full_x = torch.tensor(full_x.T, dtype=torch.float32) # [D, T]
        
        # Get test series: test (transpose to shape [D, T])
        test_y = torch.tensor(test_data.data_y.T, dtype=torch.float32)  # [D, T_test]
        pred_len = test_y.shape[1] - self.args.seq_len
        
        preds, trues = [], []
        
        for i in range(full_x.shape[0]):
            x = full_x[i].clone()   # [T]
            try:
                if self.args.model in ['naive_last', 'naive_mean']:
                    y_pred = self.model(x.unsqueeze(0), pred_len)[0]
                else:
                    season_len = self._get_season_len()
                    y_pred = self.model(x.unsqueeze(0), pred_len, season_len)[0]
            except Exception as e:
                print(f"Error processing series {i}: {e}")
                continue
            
            y_true = test_y[i, self.args.seq_len:]
            
            if y_pred.shape != y_true.shape:
                print(f"Shape mismatch for series {i}: predicted {y_pred.shape}, true {y_true.shape}")
                continue
            
            preds.append(y_pred.numpy())
            trues.append(y_true.numpy())
            
        preds = np.stack(preds)
        trues = np.stack(trues)
        print('test shape:', preds.shape, trues.shape)
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

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
        
        return