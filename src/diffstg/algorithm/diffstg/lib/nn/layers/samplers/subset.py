import torch
from torch import nn

import tsl

def nucleus_mask(scores, p):
    """Select top neighbors until cumulative probability >= p"""
    probs = torch.softmax(scores, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    cutoff_mask = cumsum_probs <= p
    # Always include at least the first token
    cutoff_mask[..., 0] = True
    
    # Create output mask
    mask = torch.zeros_like(scores)
    mask.scatter_(-1, sorted_indices, cutoff_mask.float())
    return mask

def relaxed_gumbel_nucleus(scores, p, tau, max_k=None):
    """Continuous relaxation of nucleus sampling"""
    # Sample gumbel noise
    g = -torch.log(-torch.log(torch.rand_like(scores)))
    noisy_scores = scores + g
    
    # Get nucleus mask from original scores to know target k per row
    mask = nucleus_mask(scores, p)
    k_per_row = mask.sum(dim=-1, keepdim=True)  # [batch, 1]
    
    if max_k is None:
        max_k = int(k_per_row.max().item())
    
    # Use continuous top-k with per-row early stopping
    relaxed_khot = torch.zeros_like(scores)
    onehot_approx = torch.zeros_like(scores)
    working_scores = noisy_scores.clone()
    
    for i in range(max_k):
        # Mask for rows that still need more neighbors
        active_rows = (i < k_per_row).float()  # [batch, 1]
        
        khot_mask = torch.clip(1.0 - onehot_approx, min=tsl.epsilon)
        working_scores = working_scores + torch.log(khot_mask)
        onehot_approx = torch.nn.functional.softmax(working_scores / tau, dim=-1)
        
        # Only accumulate for active rows
        relaxed_khot = relaxed_khot + onehot_approx * active_rows
        
    return relaxed_khot

def k_hot_topk(scores, k):
    khot = torch.zeros_like(scores)
    _, ind = torch.topk(scores, k, dim=-1)
    khot.scatter_(-1, ind, 1.)
    return khot

def relaxed_gumbel_top_k(scores, k, tau):
    # sample a gumbel for each node
    g = - torch.log(-torch.log(torch.rand_like(scores)))
    scores = scores + g

    # continuous top k
    relaxed_khot = torch.zeros_like(scores)
    onehot_approx = torch.zeros_like(scores)
    for i in range(k):
        khot_mask = torch.clip(1.0 - onehot_approx, min=tsl.epsilon)
        scores = scores + torch.log(khot_mask)
        onehot_approx = torch.nn.functional.softmax(scores / tau, dim=-1)
        relaxed_khot = relaxed_khot + onehot_approx
    return relaxed_khot


class StraightThroughSubsetSampler(nn.Module):
    """"""
    def __init__(self, k, tau, mode='learnable'):
        super(StraightThroughSubsetSampler, self).__init__()
        self.k = k
        self.tau = tau
        # self.mode = mode

    def forward(self, scores, inference_mode=False):
        if self.mode == 'learnable':
            if self.training and not inference_mode:
                sample = relaxed_gumbel_top_k(scores=scores,
                                            k=self.k,
                                            tau=self.tau)
                # top k
                khot = k_hot_topk(sample, self.k)
                return (khot - sample).detach() + sample
            return k_hot_topk(scores, self.k)

        else:
            p = self.k/100
            if self.training and not inference_mode:
                sample = relaxed_gumbel_nucleus(scores=scores,
                                            p=p,
                                            tau=self.tau)
                # Get hard mask
                mask = nucleus_mask(scores, p)
                return (mask - sample).detach() + sample
            return nucleus_mask(scores, p)