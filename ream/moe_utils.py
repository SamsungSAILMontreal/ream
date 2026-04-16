# Copyright (c) 2026. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""

MoE utils.

"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Callable, Optional
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


def get_num_experts(moe):
    """
    Gets the number of experts in a MoE layer supporting Qwen3.5.
    :param moe: MoE layer
    :return: number of experts
    """
    return len(moe.experts) if isinstance(moe.experts, torch.nn.ModuleList) else moe.experts.num_experts

def get_moe_input(
        model,
        device,
        input_ids,
        attention_mask,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
):
    """
    Computes Qwen3-style MoE input (based on official hugginface implementation).
    :param model: Qwen3 MoE model
    :param device: cuda, cpu, etc.
    :param input_ids: from the loaded batch or tokenizer
    :param attention_mask: from the loaded batch or tokenizer
    :param position_ids: ignored in this implementation
    :param past_key_values: ignored in this implementation
    :param inputs_embeds: ignored in this implementation
    :param use_cache: ignored in this implementation
    :param cache_position: ignored in this implementation
    :return: dict with computed states
    """
    if inputs_embeds is None:
        model.model.embed_tokens.to(device)
        inputs_embeds = model.model.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    mask_function = create_causal_mask if getattr(model.model.config, 'sliding_window', None) is None else create_sliding_window_causal_mask
    causal_mask = mask_function(
        config=model.model.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    model.model.rotary_emb.to(device)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    return {
        'hidden_states': hidden_states,
        'position_embeddings': position_embeddings,
        'attention_mask': causal_mask,
        'position_ids': position_ids,
        'past_key_values': past_key_values,
        'use_cache': use_cache,
        'cache_position': cache_position
    }

def moe_forward(decoder_layer, inputs, i=None, chunk_size=None, device=None):
    """
    Single MoE layer forward pass.
    :param decoder_layer: Qwen3 style MoE decoder layer
    :param inputs: same dict as returned by get_moe_input
    :param i:
    :param chunk_size:
    :param device:
    :return: updated hidden_states
    """
    # for decoder_layer in self.layers[: self.config.num_hidden_layers]:
    # hidden_states torch.Size([64, 128, 2048]) torch.bfloat16
    # input attention_mask torch.Size([64, 1, 128, 128]) torch.bool
    # input position_ids torch.Size([1, 128]) torch.int64
    # input cache_position torch.Size([128]) torch.int64
    if device is not None:
        decoder_layer.to(device)
    hidden_states = decoder_layer(
        hidden_states=inputs['hidden_states'] if i is None else inputs['hidden_states'][i:i+chunk_size],
        position_embeddings=inputs['position_embeddings'],
        attention_mask=inputs['attention_mask'] if i is None or inputs['attention_mask'] is None else inputs['attention_mask'][i:i+chunk_size],
        position_ids=inputs['position_ids'],
        past_key_values=inputs['past_key_values'],
        use_cache=inputs['use_cache'],
        cache_position=inputs['cache_position'],
    )
    return hidden_states

def run_all_experts(moe_layer,
                    hidden_states,
                    only_gates=False,
                    final_reduce=False,
                    act_samples=0,
                    gated_sim=True):
    """
    Runs all experts in the MoE layer on the given hidden_states with/without router masking (gated_sim).
    Compared to moe_forward, this function returns the outputs of experts, their hidden features and gate logits.
    Returns tensors of shape:
        - router_logits (batch B, seq_len S, num_experts E)
        - outputs (E, B*S or 1 if final_reduce, model_hidden_dim H)
        - outputs_act (E, < B*S, expert_hidden_dim D)

        or router_logits if only_gates is True.
    """
    # Ensure shape: (B, S, H)
    B, S, H = hidden_states.shape

    flat_input = hidden_states.view(B * S, H)

    n_experts = get_num_experts(moe_layer)
    # get gate outputs
    if isinstance(moe_layer.gate, torch.nn.Linear):
        router_logits = moe_layer.gate(flat_input).view(B, S, n_experts)  # shape: (B, S, E)
    else:
        router_logits = F.linear(flat_input, moe_layer.gate.weight).view(B, S, n_experts)
    if only_gates:
        return router_logits

    if gated_sim:
        softmax_logits = F.softmax(router_logits.view(-1, n_experts), dim=-1, dtype=torch.float)  # (B*S, E)

    outputs_act = None
    outputs = None
    for i in range(n_experts):

        if isinstance(moe_layer.experts, torch.nn.ModuleList):
            expert = moe_layer.experts[i]
            # Each expert is Qwen3MoeMLP
            assert expert.__class__.__name__.find('MLP') >= 0, f'Unexpected expert class {expert.__class__.__name__}'
            # expert: gate_proj, up_proj, down_proj, act_fn
            # flat_input: (B*S, H)
            # out_act: (B*S, D)
            # out: (B*S, H)
            # implements original function: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            out_act = expert.act_fn(expert.gate_proj(flat_input)) * expert.up_proj(flat_input)
            out = expert.down_proj(out_act)  # shape: (B*S, H)
        else:
            # Each expert is a slice of the params in Qwen3_5
            gate, up = F.linear(flat_input, moe_layer.experts.gate_up_proj[i]).chunk(2, dim=-1)
            out_act = moe_layer.experts.act_fn(gate) * up
            out = F.linear(out_act, moe_layer.experts.down_proj[i])

        if gated_sim:
            out = out * softmax_logits[:, i].view(-1, 1)  # (B*S, H)

        if act_samples > 0:
            # sampling to improve efficiency
            ind = np.random.permutation(out_act.shape[0])[:min(act_samples, out_act.shape[0])]
            out_act = out_act[ind]

        if outputs_act is None:
            outputs_act = torch.zeros(n_experts, out_act.shape[0], out_act.shape[1],
                                      dtype=out_act.dtype, device=out_act.device)
            outputs = torch.zeros(n_experts, 1 if final_reduce else out.shape[0], out.shape[1],
                                  dtype=out.dtype, device=out.device)

        outputs_act[i] = out_act
        outputs[i] = out.mean(dim=0) if final_reduce else out

    return router_logits, outputs, outputs_act
