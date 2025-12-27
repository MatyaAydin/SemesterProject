import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

model_path = "./output/model/electricity_benchmark_dagg_I_vanilla_conv_0_neighbor.dm4stg"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load(model_path, map_location=device, weights_only=False)

A = torch.mm(model.eps_model.node_embed, model.eps_model.node_embed.transpose(0, 1))
a1 = F.softmax(F.relu(A), dim=1)

adj_numpy = a1.cpu().detach().numpy()

np.save("./dagg_I_electricity_benchmark_adj.npy", adj_numpy)