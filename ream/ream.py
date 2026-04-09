# Copyright (c) 2026. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""

Pseudo grouping function of REAM.

"""

import torch
import numpy as np
import torch.nn.functional as F
from typing import Union
from .utils import normalize_rows, dist2sim


@torch.no_grad()
def pseudo_group(saliency: Union[torch.Tensor, np.ndarray],
                 expert_logits: torch.Tensor,
                 k: int,
                 gate_logits: torch.Tensor = None,
                 group_size: int = 16):
    """
    Groups experts based on saliency, expert logits and, optionally, gate_logits.
    :param saliency: 1d array of length N
    :param expert_logits: 3d tensor (num_experts, batch*seq_len, hidden_size)
    :param k: number of expert groups (k < N)
    :param gate_logits: (batch*seq_len, num_experts)
    :param group_size: REAM's hyperparameter C (0 means using more simple MC-SMoE's procedure)
    :return: cluster_labels (of length N), centroid_inds (of length k)
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if gate_logits is not None:
        assert gate_logits.dim() == 2, gate_logits.shape  # (B*S, E)
        x = gate_logits.to(device).float()  # (B*S, E)
        x = normalize_rows(x.transpose(0, 1))  # (E, B*S)
        d_gate = torch.cdist(x, x).data.cpu().numpy()
        sim_gate = dist2sim(d_gate, remove_self_loops=False)  # similarity in [0,1]

    assert expert_logits.dim() == 3, expert_logits.shape  # (E, B*S, H)
    n_experts = expert_logits.shape[0]
    assert len(saliency) == n_experts, (len(saliency), expert_logits.shape)
    if isinstance(saliency, torch.Tensor):
        saliency = saliency.cpu().numpy()

    # first compute distances between all experts
    d = np.zeros((n_experts, n_experts), dtype=np.float32)
    for i in range(n_experts):
        for j in range(i + 1, n_experts):
            sim_out = expert_similarity(logits1=expert_logits[i].to(device),
                                        logits2=expert_logits[j].to(device))
            if gate_logits is not None:
                sim_out = (sim_out + sim_gate[i, j]) / 2
            dist = 1 - sim_out
            # check if dist is not nan or inf
            assert np.isfinite(dist), (i, j, dist)
            d[i, j] = d[j, i] = dist

    # get indices of k most salient experts
    centroid_inds = np.argsort(saliency)[::-1].copy()[:k]

    # assign each point to the closest center based on the distance between experts
    cluster_labels = np.zeros(n_experts) - 1
    cluster_labels[centroid_inds] = np.arange(k)  # first, assign centroids

    if group_size > 0:
        if k < n_experts // group_size:
            raise ValueError('the combination of group size (potentially too small), '
                             'k and n_experts is not valid')
        for centroid in centroid_inds:  # start from the most salient centroids (order is important)
            top_close = np.argsort(d[centroid])  # sort ascending because it's distance
            group = [centroid]
            for ind in top_close:
                # if ind is not the group center and not already assigned
                if ind != centroid and cluster_labels[ind] == -1:
                    group.append(ind)
                if len(group) >= group_size:  # stop when group_size reached
                    break
            # assign all experts in the group to the current centroid
            for ind in group:
                cluster_labels[ind] = cluster_labels[centroid]
            if (cluster_labels >= 0).all():  # stop if all experts assigned
                break
    else:
        # MC-SMoE style
        for j in range(n_experts):
            # this loop can be done in any order or in parallel
            if cluster_labels[j] >= 0:
                continue  # already assigned
            cluster_labels[j] = np.argmin(d[centroid_inds, j])  # get the centroid closest to j

    assert (cluster_labels >= 0).all(), cluster_labels  # confirm all experts assigned
    return cluster_labels, centroid_inds


@torch.no_grad()
def expert_similarity(logits1=None, logits2=None, metric='cosine'):
    """

    :param logits1: (n,d)
    :param logits2: (m,d)
    :param metric: cosine or euclidean
    :return: scalar from 0 to 1
    """
    # compare logits between two experts
    if metric == 'cosine':
        return ((F.cosine_similarity(logits1, logits2) + 1) / 2).mean().item()  # [0,1]
    elif metric == 'euclidean':
        dist = torch.norm(logits1 - logits2, p=2, dim=1)  # (n,)
        dist = dist.mean().item()
        return 1 / (1 + dist)  # [0,1]
    else:
        raise ValueError(metric)
