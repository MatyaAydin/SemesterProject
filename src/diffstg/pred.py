# %%
import os, sys
import torch
import argparse
import numpy as np
import torch.utils.data
from easydict import EasyDict as edict
from timeit import default_timer as timer
from tqdm import tqdm

from utils.eval import Metric
from utils.gpu_dispatch import GPU
from utils.common_utils import dir_check, to_device, ws, unfold_dict, dict_merge, GpuId2CudaId, Logger

from algorithm.dataset import CleanDataset, TrafficDataset
from algorithm.diffstg.model import DiffSTG, save2file
import pickle

parser = argparse.ArgumentParser(description='Entry point of the code')
parser.add_argument("--graph_method", type=str, default='fixed') # fixed, learnable, learnable_static, dagg
parser.add_argument("--gc_type", type=str, default='vanilla') # diffconv, gatconv or vanilla
parser.add_argument("--temporal_type", type=str, default='conv') # conv, lstm, gru or transformer
parser.add_argument("--k", type=int, default=4)
parser.add_argument("--data", type=str, default='PEMS08')

args, _ = parser.parse_known_args()
args_dict = vars(args)

GC_TYPE = args_dict['gc_type']
GRAPH_METHOD = args_dict['graph_method']
DATASET_NAME = args_dict['data']
TEMPORAL_TYPE = args_dict['temporal_type']
K_NEIGHBORS = f'_{args_dict["k"]}_neighbor'


# %%
trained_model_path = f'./output/model/{DATASET_NAME}_{GRAPH_METHOD}_{GC_TYPE}_{TEMPORAL_TYPE}{K_NEIGHBORS}.dm4stg'
DATA_path = f'./data/dataset/{DATASET_NAME}/'
flow_path = os.path.join(DATA_path, 'flow.npy')
adj_path = os.path.join(DATA_path, 'adj.npy')

# %%
flow = np.load(flow_path)
adj = np.load(adj_path)
T = flow.shape[0]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load(trained_model_path, map_location=device, weights_only=False)

with open(f'./configs/config_{DATASET_NAME}_{GC_TYPE}_{GRAPH_METHOD}_{TEMPORAL_TYPE}{K_NEIGHBORS}.pkl', 'rb') as f:
    config = edict(pickle.load(f))

clean_data = CleanDataset(config)

# %%
test_dataset = TrafficDataset(clean_data, (config.data.test_start_idx + config.model.T_p, -1), config)

batch_size = 8 if DATASET_NAME == 'electricity_benchmark' else 64
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

# %%
def predict(model, data_loader, config, clean_data, mode='Test'):

    y_pred, y_true = [], []
    metrics_future = Metric(T_p=config.model.T_p)
    metrics_history = Metric(T_p=config.model.T_h)
    model.eval()

    samples, targets = [], []
    for i, batch in enumerate(data_loader):

        future, history, pos_w, pos_d = to_device(batch, config.device) # target:(B,T,V,1), history:(B,T,V,1), pos_w: (B,1), pos_d:(B,T,1)

        x = torch.cat((history, future), dim=1).to(config.device)  # in cpu (B, T, V, F), T =  T_h + T_p
        x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(config.device)  # (B, T, V, F)
        targets.append(x.cpu())
        x = x.transpose(1, 3)  # (B, F, V, T)
        x_masked = x_masked.transpose(1, 3)  # (B, F, V, T)

        # n_samples = 1 if mode == 'Val' else config.n_samples
        n_samples = config.n_samples
        x_hat = model((x_masked, pos_w, pos_d), n_samples) # (B, n_samples, F, V, T)
        samples.append(x_hat.transpose(2,4).cpu())

        if x_hat.shape[-1] != (config.model.T_h + config.model.T_p): x_hat = x_hat.transpose(2,4)
        # assert x.shape == x_hat.shape, f"shape of x ({x.shape}) does not equal to shape of x_hat ({x_hat.shape})"
        x, x_hat= clean_data.reverse_normalization(x), clean_data.reverse_normalization(x_hat)
        x_hat = x_hat.detach()
        f_x, f_x_hat = x[:,:,:,-config.model.T_p:], x_hat[:,:,:,:,-config.model.T_p:] # future

        _y_true_ = f_x.transpose(1, 3).cpu().numpy()  # y_true: (B, T_p, V, D)
        _y_pred_ = f_x_hat.transpose(2, 4).cpu().numpy() # y_pred: (B, n_samples, T_p, V, D)
        _y_pred_ = np.clip(_y_pred_, 0, np.inf)
        metrics_future.update_metrics(_y_true_, _y_pred_)

        y_pred.append(_y_pred_)
        y_true.append(_y_true_)

        h_x, h_x_hat = x[:, :, :, :config.model.T_h], x_hat[:, :, :, :,  :config.model.T_h]
        _y_true_ = h_x.transpose(1, 3).cpu().numpy()  # y_true: (B, T_p, V, D)
        _y_pred_ = h_x_hat.transpose(2, 4).cpu().numpy()
        _y_pred_ = np.clip(_y_pred_, 0, np.inf)
        metrics_history.update_metrics(_y_true_, _y_pred_)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)


    if mode == 'test': # save the prediction result to file
        samples = torch.cat(samples, dim=0)[:50]
        targets = torch.cat(targets, dim=0)[:50]


    torch.cuda.empty_cache()
    return y_true, y_pred

# %%
y_true, y_pred = predict(model, test_loader, config, clean_data, mode='Val')
print('computed predictions')
np.save(f'./preds/pred_{DATASET_NAME}_{GC_TYPE}_{GRAPH_METHOD}_{TEMPORAL_TYPE}{K_NEIGHBORS}.npy', y_pred)
np.save(f'./preds/true_{DATASET_NAME}_{GC_TYPE}_{GRAPH_METHOD}_{TEMPORAL_TYPE}{K_NEIGHBORS}.npy', y_true)



