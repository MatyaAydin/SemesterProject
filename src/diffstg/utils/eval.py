# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from timeit import default_timer as timer


def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(array)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')

def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError

class Metric(object):
    """Computes and stores the average and current value,调用时先reset"""

    def __init__(self, T_p):
        self.time_start = timer()
        self.T_p = T_p
        self.best_metrics = {'mae': np.inf, 'rmse': np.inf, 'mape': np.inf, 'crps':np.inf, 'mis': np.inf, 'epoch': np.inf}
        self.metrics = {}
        # self.step_metrics_epoch = {'mae': {}, 'rmse': {}, 'mape': {}}


    def update_metrics(self, y_true, y_pred):
        """
        both y_true and y_pred should be numpy
        :param y_true: (B, T_p, V, D)
        :param y_pred: (B, n_samples, T_p, V, D) or (B, T_p, V, D)
        :return:
        """
        assert  isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray), \
            f"y_true and y_pred should be np.ndarray, now its type is y_true:{type(y_true)}, y_pred:{type(y_pred)}"
        self.metrics = {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0, 'crps': 0, 'mis': 0.0, 'time': 0.0}

        if y_pred.shape == y_true.shape: #  y_true: (B, T_p, V, D) and y_pred: (B, T_p, V, D)
            y_pred = np.expand_dims(y_pred, axis = 1) # (B, 1, T_p, V, D)

        # probabilistic metric
        eval_points = np.ones_like(y_true)
        self.metrics['crps'] = calc_quantile_CRPS(torch.from_numpy(y_true), torch.from_numpy(y_pred), torch.from_numpy(eval_points))
        self.metrics['mis'] = calc_mis(torch.from_numpy(y_true), torch.from_numpy(y_pred))


        # deterministic metric
        y_pred = np.mean(y_pred, axis=1) # # (B, T_p, V, D)

        self.metrics['mae'], self.metrics['rmse'], self.metrics['mape'], mse = self.get_metric(y_true, y_pred)
        # self.metrics['time'] = time_to_str((timer() - self.time_start))
        self.metrics['time'] = timer() - self.time_start



    def update_best_metrics(self, epoch=0):
        self.best_metrics['mae'], mae_state = self.get_best_metric(self.best_metrics['mae'], self.metrics['mae'])
        self.best_metrics['rmse'], rmse_state = self.get_best_metric(self.best_metrics['rmse'], self.metrics['rmse'])
        self.best_metrics['mape'], mape_state = self.get_best_metric(self.best_metrics['mape'], self.metrics['mape'])
        self.best_metrics['crps'], crps_state = self.get_best_metric(self.best_metrics['crps'], self.metrics['crps'])
        self.best_metrics['mis'], mis_state = self.get_best_metric(self.best_metrics['mis'], self.metrics['mis'])

        if mae_state:
            self.best_metrics['epoch'] = int(epoch)

    @staticmethod
    def get_metric(y_true, y_pred):
        mae = masked_mae_np(y_true, y_pred, 0)
        mse = masked_mse_np(y_true, y_pred, 0)
        mape = masked_mape_np(y_true, y_pred, 0)
        rmse = mse ** 0.5
        return mae, rmse, mape, mse

    @staticmethod
    def get_best_metric(best, candidate):
        state = False
        if candidate < best:

            best = candidate
            state = True
        return best, state

    def __str__(self):
        """For print"""
        return f"{self.metrics['mae']:<7.2f}{self.metrics['rmse']:<7.2f}{self.metrics['mape']:<7.2f}{self.metrics['crps']:<7.2f} {self.metrics['mis']:<7.2f} | {self.best_metrics['epoch'] + 1:<4} "

    def best_str(self):
        """For save"""
        return f"{self.best_metrics['epoch']},{self.best_metrics['mae']:.2f},{self.best_metrics['rmse']:.2f},{self.best_metrics['mape']:.2f}"


    def log_lst(self, epoch=None, sep=','):
        message_lst = []
        index = ['mae', 'rmse', 'mape']

        for i in index:
            message_lst.append(f"{i},{self.multi_step_str(obj=i, sep=sep, epoch=epoch)}")
        return message_lst

    def to_dict(self):
        return self.metrics


# metric for Probabilistic evaluation
# import torch
# def quantile_loss(target, forecast, q: float, eval_points) -> float:
#     return 2 * torch.sum(
#         torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
#     )

# def calc_denominator(target, eval_points):
#     return torch.sum(torch.abs(target * eval_points))


# def calc_quantile_CRPS(target, forecast, eval_points):
#     """
#     target: (B, T, V), torch.Tensor
#     forecast: (B, n_sample, T, V), torch.Tensor
#     eval_points: (B, T, V): which values should be evaluated,
#     """

#     # target = target * scaler + mean_scaler
#     # forecast = forecast * scaler + mean_scaler
#     quantiles = np.arange(0.05, 1.0, 0.05)
#     denom = calc_denominator(target, eval_points)
#     CRPS = 0
#     for i in range(len(quantiles)):
#         q_pred = []
#         for j in range(len(forecast)):
#             q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
#         q_pred = torch.cat(q_pred, 0)
#         q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
#         CRPS += q_loss / denom
#     return CRPS.item() / len(quantiles)


def MIS(
        target: np.ndarray,
        lower_quantile: np.ndarray,
        upper_quantile: np.ndarray,
        alpha: float,


) -> float:
    r"""
    mean interval score
    Implementation comes form glounts.evalution metrics
    .. math::
    msis = mean(U - L + 2/alpha * (L-Y) * I[Y<L] + 2/alpha * (Y-U) * I[Y>U])
    """
    numerator = np.mean(
        upper_quantile
        - lower_quantile
        + 2.0 / alpha * (lower_quantile - target) * (target < lower_quantile)
        + 2.0 / alpha * (target - upper_quantile) * (target > upper_quantile)
    )
    return numerator


# def calc_mis(target, forecast, alpha = 0.05):
#     """
#        target: (B, T, V),
#        forecast: (B, n_sample, T, V)
#     """
#     return MIS(
#         target = target.cpu().numpy(),
#         lower_quantile = torch.quantile(forecast, alpha / 2, dim=1).cpu().numpy(),
#         upper_quantile = torch.quantile(forecast, 1.0 - alpha / 2, dim=1).cpu().numpy(),
#         alpha = alpha,
#     )


if __name__ == "__main__":

    def run_eval(dataset):

        methods = [
            # 'diffconv_fixed', 'vanilla_fixed_gru', 'vanilla_fixed',
                   'vanilla_learnable', 'vanilla_learnable_conv_8_neighbor', 'vanilla_learnable_conv_16_neighbor',
                   'vanilla_learnable_static_conv_4_neighbor', 'vanilla_learnable_static_conv_8_neighbor', 'vanilla_learnable_static_conv_16_neighbor',
                   'vanilla_learnable_batch_conv_4_neighbor', 'vanilla_learnable_batch_conv_8_neighbor', 'vanilla_learnable_batch_conv_16_neighbor']
        if dataset == 'electricity_benchmark':
            methods = ['vanilla_learnable', 'vanilla_fixed']
        metrics = {}



        for m in methods:
            
            y_true = np.load(f'../preds/true_{dataset}_{m}.npy')
            y_pred = np.load(f'../preds/pred_{dataset}_{m}.npy')
            y_pred = np.mean(y_pred, axis=1)

            # print(y_true.shape, y_pred.shape)


            metric = Metric(T_p=1)

            mae, rmse, mape, mse = metric.get_metric(y_true, y_pred)

            metrics[m] = {'mae': mae, 'rmse': rmse, 'mape': mape, 'mse': mse}


        y_true_chronos = np.load(f'../preds/{d}_true_chronos.npy')
        y_pred_chronos = np.load(f'../preds/{d}_pred_chronos.npy')
        mae, rmse, mape, mse = metric.get_metric(y_true_chronos, y_pred_chronos)

        metrics["chronos"] = {'mae': mae, 'rmse': rmse, 'mape': mape, 'mse': mse}

        # computed in other folder

        rmse = 0.5576574046571184
        mse = rmse ** 2
        mae = 0.08680029203277909
        mape = 56.32928901284428
        metrics["CSDI"] = {'mae': mae, 'rmse': rmse, 'mape': mape, 'mse': mse}


        df = pd.DataFrame(metrics)

        df.rename(columns={
            # 'vanilla_fixed':'baseline', 'diffconv_fixed':'diffconv',
                           'vanilla_learnable': '4_k',
                           'vanilla_learnable_conv_8_neighbor': '8_k',
                           'vanilla_learnable_conv_16_neighbor': '16_k',
                            'vanilla_learnable_static_conv_4_neighbor': 'static_4_k',
                            'vanilla_learnable_static_conv_8_neighbor': 'static_8_k',
                            'vanilla_learnable_static_conv_16_neighbor': 'static_16_k',

                            'vanilla_learnable_batch_conv_4_neighbor': 'batch_4_k',
                            'vanilla_learnable_batch_conv_8_neighbor': 'batch_8_k',
                            'vanilla_learnable_batch_conv_16_neighbor': 'batch_16_k',
                        #    'vanilla_fixed_gru': 'gru',
                           }, inplace=True)
        
        return df
        

    datasets = ['ewz_daily']
    for d in datasets:
        df = run_eval(d)
        df['best_method'] = df.apply(lambda row: row.idxmin(), axis=1)


        print("="*30 + d + "="*30)
        print(df)
        print("\n")





