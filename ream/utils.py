# Copyright (c) 2026. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""

General utils.

"""

import numpy as np
import torch
import os
import psutil


process = psutil.Process(os.getpid())

def mem(device):
    return (torch.cuda.max_memory_reserved(device) if device != 'cpu' else process.memory_info().rss) / 10 ** 9

def to_cpu_float(t: torch.Tensor) -> torch.Tensor:
    if isinstance(t, torch.Tensor):
        return t.data.to('cpu', dtype=torch.float32)
    else:
        raise NotImplementedError(type(t))

def casted_mul(a, b):
    # convert both a and b to float16, then convert back to a's dtype
    # for FP-8 models
    if (a.dtype in [torch.float32, torch.float16, torch.bfloat16] and
            b.dtype in [torch.float32, torch.float16, torch.bfloat16]):
        return a * b
    a_fp32 = a.to(torch.float32)
    b_fp32 = b.to(torch.float32)
    return (a_fp32 * b_fp32).to(a.dtype)

def num_parameters(module):
    # count params in a module (not counting same memory pointers)
    # e.g. layer1 = layer2 shouldn't count twice
    return sum({p_.data_ptr(): p_.numel() for p_ in module.parameters()}.values())

def dist2sim(a, remove_self_loops=True):
    # distance to similarity
    a = a / a.max()  # [0, 1]
    a = 1 - a
    if remove_self_loops:
        a[np.diag_indices_from(a)] = 0  # no self-loops
    return a

def normalize_rows(x, dim=-1, keepdim=True, eps=1e-8):
    # x: (n, d)
    nrm = torch.linalg.norm(x, dim=dim, keepdim=keepdim)
    return x / (nrm + eps)
