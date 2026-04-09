# Copyright (c) 2026. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""

Expert saliency calculation.

"""

import torch
import torch.nn.functional as F
from collections import Counter


def freq(gate_logits: torch.Tensor,
         top_k: int = 8) -> torch.Tensor:
    """
    Computes frequency-based saliency using gate logits.
    :param gate_logits: tensor of shape (batch, seq_len, num_experts) with the gate logits for each token and expert.
    :param top_k: the number of top experts to consider for frequency counting.
    :return: a tensor of shape (num_experts,) containing the frequency-based saliency for each expert
    """
    n_experts = gate_logits.shape[-1]
    saliency = gate_logits.view(-1, n_experts).data.topk(top_k, dim=1).indices.cpu().numpy()  # (B*S, top_k)
    cnt = Counter()
    cnt.update(saliency.flatten().tolist())
    saliency = torch.tensor([cnt.get(m, 0) for m in range(n_experts)], dtype=torch.float32)
    return saliency


def reap(gate_logits: torch.Tensor,
         expert_activations: torch.Tensor,
         top_k: int = 8) -> torch.Tensor:
    """
    Computes REAP (https://www.arxiv.org/abs/2510.13999) saliency using gate logits and expert activations.
    See Eq. 9 in the paper.
    :param gate_logits: tensor of shape (batch, seq_len, num_experts) with the gate logits for each token and expert.
    :param expert_activations: tensor of shape (num_experts, batch*seq_len, expert output dim)
    :param top_k: number of top experts to consider for choosing tokens.
    None or <=0 means use all experts for all tokens (undocumented variant that worked well in some of our cases).
    :return: a tensor of shape (num_experts,) containing the REAP saliency for each expert
    """
    n_experts = gate_logits.shape[-1]
    assert expert_activations.dim() == 3 and expert_activations.shape[0] == n_experts, (
        expert_activations.shape, gate_logits.shape)

    # copied from the qwen3 moe forward pass
    gate = F.softmax(gate_logits.view(-1, n_experts), dim=-1, dtype=torch.float)  # (B*S, n_experts)
    if top_k is None or top_k <= 0:
        expert_mask = None
        expert_hitted = torch.arange(n_experts).unsqueeze(1)  # all experts
    else:
        # get topk experts by gate and multiply them by the corresponding expert activations
        gate, selected_experts = torch.topk(gate, k=top_k, dim=-1)
        expert_mask = F.one_hot(selected_experts, num_classes=n_experts).permute(2, 1, 0)  # (n_experts, B*S, top_k)
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    saliency = torch.zeros(n_experts)
    # Loop over all available experts in the model and perform the computation on each expert
    for exp_idx in expert_hitted:
        exp_idx = exp_idx.item()
        current_state = expert_activations[exp_idx]  # (B*S, H)
        if expert_mask is None:
            # use all tokens and simply multiply by gates
            saliency[exp_idx] += (current_state.norm(dim=-1) * gate[:, exp_idx]).mean().item()
        else:
            # use only tokens routed to expert exp_idx and multiply by gates on those tokens
            idx, top_x = torch.where(expert_mask[exp_idx])
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1, top-2, etc.)
            current_state = current_state[None, top_x].reshape(-1, current_state.shape[-1])
            saliency[exp_idx] += (current_state.norm(dim=-1) * gate[top_x, idx]).mean().item()
    return saliency
