# Copyright (c) 2026. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""

Weight processing utils.

"""

import torch
from scipy.optimize import linear_sum_assignment
from typing import Union
from copy import deepcopy
from .utils import to_cpu_float


def ffn_weight_matrix(ffn: torch.nn.Module) -> torch.Tensor:
    """
    Obtains expert weight matrix given its FFN module.
    Assumes Qwen3 style FFN.
    :param ffn: torch.nn.Module
    ffn dict keys: 'gate_proj' (768,2048), 'up_proj' (768,2048), 'down_proj' (2048,768)
    Returns matrix: (hidden=768, feat=6144) per-neuron concatenated vectors:
        [ gate[i,:] || up[i,:] || down[:,i] ]
    """
    gate = to_cpu_float(ffn.gate_proj.weight)
    up = to_cpu_float(ffn.up_proj.weight)
    down = to_cpu_float(ffn.down_proj.weight)

    assert ffn.gate_proj.bias is None
    assert ffn.up_proj.bias is None
    assert ffn.down_proj.bias is None

    # gate/up: rows index hidden; down: columns index hidden
    #     print(gate.shape, up.shape, down.shape)
    matrix = torch.cat([gate, up, down.t()], dim=1)  # (768, 2048+2048+2048=6144)
    return matrix

def apply_perm_to_ffn(ffn, perm, in_place=True):
    """
    Permutes the hidden neurons of the expert FFN given the permutation matrix.
    :param ffn: expert FFN module (with gate_proj, up_proj, down_proj)
    :param perm: permutation of hidden neurons (of length hidden_size, e.g. 768)
    :param in_place:
    :return: expert FFN with permuted neurons
    """
    # perm maps A_i -> B_perm[i]; to align B to A order
    if in_place:
        ffn_aligned = ffn
    else:
        ffn_aligned = deepcopy(ffn)

    assert ffn.gate_proj.bias is None
    assert ffn.up_proj.bias is None
    assert ffn.down_proj.bias is None

    ffn_aligned.gate_proj.weight.data = ffn.gate_proj.weight.data[perm, :]            # rows permuted
    ffn_aligned.up_proj.weight.data   = ffn.up_proj.weight.data[perm, :]
    ffn_aligned.down_proj.weight.data = ffn.down_proj.weight.data[:, perm]            # columns permuted
    return ffn_aligned

def pca_reduce(features: Union[torch.Tensor, list], r: int, verbose: bool = False) -> Union[torch.Tensor, list]:
    """
    Computes PCA for features (like expert weights). Uses torch.pca_lowrank (fast, randomized).
    :param features: list of n tensors of shape (H, D) or a single tensor of shape (n*H, D)
    :param r: dimensionality of PCA
    :param verbose:
    :return: projected (list of n tensors of shape (H,r)) or a single tensor of shape (n*H, r)
    """
    if isinstance(features, list):
        x = torch.cat(features, dim=0)  # (n*H, D) or use torch.stack(features, dim=0) to get (n, H, D)
    else:
        x = features
        assert x.dim() == 2, x.shape  # (n*H, D)
    if verbose:
        print('PCA reduce: features', len(features), 'total shape', x.shape, 'to r=', r, flush=True)
    mn = x.mean(dim=-2, keepdim=True)
    x = x - mn  # center
    if verbose:
        print('PCA x centered', x.shape, x.dtype, flush=True)
    torch.manual_seed(0)
    u, s, v = torch.pca_lowrank(x , q=r)  # V: (D,r)
    if verbose:
        print('PCA U, S, V', u.shape, s.shape, v.shape, flush=True)
    p = v[:, :r]                         # projection
    if isinstance(features, list):
        projected = []
        for fp in features:
            fp_centered = fp - mn.to(fp)  # (H, D)
            projected.append(fp_centered @ (p.to(fp) * s[:r].to(fp).reshape(1, r)))  # (H, r)
    else:
        projected = x @ (p.to(x) * s[:r].to(x).reshape(1, r))
    return projected
