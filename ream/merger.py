# Copyright (c) 2026. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""

Merger.

"""

import torch
import numpy as np
import os.path
import torch.nn as nn
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from .hc import hcsmoe
from .ream import pseudo_group
from .utils import mem, normalize_rows, num_parameters
from .moe_utils import run_all_experts, moe_forward, get_moe_input
from .weight_utils import ffn_weight_matrix, pca_reduce, apply_perm_to_ffn
from .saliency import reap, freq
from data.calibration_data import print_seq_stats


class Merger:
    def __init__(self,
                 model: nn.Module,
                 mtp_state_dict: dict = None,
                 merge_size: int = 96,
                 grouping: str = 'ream',
                 merging: str = 'logits+weights',
                 saliency: str = 'reap',
                 dataset: str = 'c4+math+code',
                 mix_ratio: str = '0.0,0.3,0.7',
                 tokenizer_name: str = 'qwen3',
                 batch_size: int = 8,
                 group_size: int = 16,
                 sequential: bool = True,
                 use_gate_output: bool = True,
                 gated_sim: bool = True,
                 calibration_data_size: int = 3072,
                 calibration_data_seq_len: int = 512,
                 seed: int = 42,
                 verbose: bool = False
                 ):
        """
        Default parameters are good for merging Qwen3 models like Qwen3-Coder-Next.
        :param model: Qwen-MoE or other MoE model
        :param mtp_state_dict: MTP layer state dict (optional, for some models)
        :param merge_size: number of experts after merging (per each layer)
        :param grouping: hcsmoe or ream
        :param merging: approach to merge a group expert
        :param saliency: freq or reap
        :param dataset: calibration data
        :param mix_ratio: calibration data mix
        :param tokenizer_name: qwen3 or glm
        :param batch_size: chunk size for processing the calibration data
        :param group_size: hyperparameter C (for pseudo-pruning) from the REAM paper
        :param sequential: whether to update hidden states given the merged layer
        :param use_gate_output: add gate outputs to improve expert similarity metric
        :param gated_sim: whether to apply softmax to gate outputs before using them for clustering
        :param calibration_data_size: default value as in the REAM paper
        :param calibration_data_seq_len: default value as in the REAM paper
        :param seed: random seed for data shuffling (as always, this is the most important tuning hyperparameter)
        :param verbose: verbose print
        """
        self.model = model
        self.mtp_state_dict = mtp_state_dict
        self.merge_size = merge_size
        self.first_moe_layer = getattr(model.config, 'first_k_dense_replace', 0)  # e.g. for GLM models
        first_moe = model.model.layers[self.first_moe_layer].mlp
        if not hasattr(first_moe, 'top_k'):
            for idx in range(self.first_moe_layer, len(model.model.layers)):
                moe = model.model.layers[idx].mlp
                moe.top_k = moe.gate.top_k
                moe.num_experts = len(moe.experts)

        self.top_k = first_moe.top_k
        assert self.top_k == model.config.num_experts_per_tok, (self.top_k, model.config.num_experts_per_tok)
        self.n_experts = len(first_moe.experts)
        self.grouping = grouping
        self.merging = merging
        self.saliency = saliency
        assert self.saliency in ['freq', 'reap'], self.saliency
        self.sequential = sequential
        self.pca_dim = 64  # heuristic
        self.dataset = dataset
        self.mix_ratio = [float(x) for x in mix_ratio.split(',')]
        assert len(self.mix_ratio) >= len(dataset.split('+')), (self.mix_ratio, dataset)
        self.group_size = group_size
        self.batch_size = batch_size
        self.use_gate_output = use_gate_output
        self.gated_sim = gated_sim
        self.verbose = verbose

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        def get_batch_file(dset_):
            return (f'data/{dset_}_b{calibration_data_size}_'
                    f'seq{calibration_data_seq_len}_{tokenizer_name}_seed{seed}.pt')

        if '+' in dataset:
            dsets = dataset.split('+')
            batch = {'input_ids': [], 'attention_mask': []}
            for dset, ratio in zip(dsets, self.mix_ratio):

                batch_file = get_batch_file(dset)
                if os.path.exists(batch_file):
                    print(f'loading batch from {batch_file}', flush=True)
                    batch_dset = torch.load(batch_file)
                    print_seq_stats(batch_dset)
                    n_samples = int(calibration_data_size * ratio)
                    if n_samples == 0:
                        print(f'skipping dataset {dset} with ratio {ratio}')
                        continue
                    np.random.seed(seed)
                    ind = np.random.permutation(batch_dset['input_ids'].shape[0])[:n_samples]
                    batch_dset['input_ids'] = batch_dset['input_ids'][ind]
                    batch_dset['attention_mask'] = batch_dset['attention_mask'][ind]
                    assert batch_dset['input_ids'].shape[1] >= calibration_data_seq_len, (
                        f'seq len={batch_dset["input_ids"].shape[1]},'
                        f'make sure the dataset is properly prepared using calibration_data.py')
                    batch['input_ids'].append(batch_dset['input_ids'])
                    batch['attention_mask'].append(batch_dset['attention_mask'])
                    print_seq_stats(batch_dset)
                else:
                    raise FileNotFoundError(f'File {batch_file} not found')
            batch['input_ids'] = torch.cat(batch['input_ids'])
            batch['attention_mask'] = torch.cat(batch['attention_mask'])
            print(f'dataset {dataset} loaded', flush=True)
        else:
            batch_file = get_batch_file(dataset)
            if os.path.exists(batch_file):
                print(f'loading batch from {batch_file}', flush=True)
                batch = torch.load(batch_file)
                print_seq_stats(batch)
            else:
                raise ValueError(f'file {batch_file} not found, '
                                 f'please run calibration_data.py in the data folder first')

        print('final batch',
              'input_ids', batch['input_ids'].shape,
              'attention_mask', batch['attention_mask'].shape,
              'attention mask sum', batch['attention_mask'].sum().item(), flush=True)
        self.batch = {k: v.to(self.device) for k, v in batch.items()}

        self.num_layers = len(self.model.model.layers)
        if self.mtp_state_dict is None:
            self.mtp_layer = None
        else:
            # only qwen3 mtp layers have been checked
            from .qwen3_mtp import build_mtp_layer
            self.num_layers += 1
            # assuming qwen3 naming
            num_experts_orig = len(self.mtp_state_dict['mtp.layers.0.mlp.gate.weight'])
            self.mtp_layer = build_mtp_layer(self.mtp_state_dict, self.model, num_experts=num_experts_orig)
            self.mtp_layer.eval()
        for layer_ind in range(self.num_layers):
            moe_layer = (self.model.model.layers[layer_ind] if layer_ind < len(self.model.model.layers)
                         else self.mtp_layer.layer).mlp
            moe_layer._layer_ind = layer_ind

    @torch.no_grad()
    def _forward_pass(self, states, layer_ind, collect_outputs=True, upd_hid=False, inputs_embeds=None, verbose=False):
        """
        Computes expert activations and other necessary data for pruning/merging.
        Propagates activations to next layers by returning updated states.
        :param states:
        :param layer_ind:
        :param collect_outputs: populate expert_outs values or not
        :param upd_hid: update states['hidden_states'] or not
        :param inputs_embeds: for mtp layers
        :param verbose:
        :return:
        """
        expert_outs = {'gate': [], 'hid_act': [], 'final': 0, 'saliency': 0}
        handles = []
        hidden_states = []
        is_mtp = layer_ind >= len(self.model.model.layers)  # assuming mtp going after all MoE layers

        if collect_outputs:
            n_batches = max(1, self.batch['input_ids'].shape[0] // self.batch_size)

            # first forward pass with hooks to collect outputs from all the experts before merging
            def hook_fn(module, inputs, output):
                x = inputs[0]  # input features to the MoE layer
                # More efficient computation of all expert outputs with sampling and reduce when possible
                gate, final, act = run_all_experts(module,
                                                   x,
                                                   final_reduce=self.saliency != 'reap',
                                                   act_samples=max(1, 1024 * 32 // n_batches),
                                                   gated_sim=self.gated_sim)
                # gate: (B, S, E)
                # final: (E, B*S -> 1, H)
                # act: (E, <B*S, D)
                assert gate.dim() == 3, gate.shape
                assert final.dim() == 3, final.shape
                assert act.dim() == 3, act.shape

                if self.use_gate_output:
                    expert_outs['gate'].append(gate.data.cpu())
                expert_outs['hid_act'].append(act.data.cpu())

                if self.saliency == 'reap':
                    expert_outs['final'] += (final.data.mean(1, keepdim=True) / n_batches).cpu()  # (E,1,H)
                    expert_outs['saliency'] += (reap(gate, final, top_k=self.top_k) / n_batches)
                else:
                    assert final.shape[1] == 1, final.shape
                    expert_outs['final'] += (final.data / n_batches).cpu()  # (E,1,H)
                    # compute saliency based on gate_logits
                    expert_outs['saliency'] += (freq(gate, top_k=self.top_k) / n_batches)


            mlp = (self.mtp_layer.layer if is_mtp else self.model.model.layers[layer_ind]).mlp
            if verbose:
                print('adding hook to layer', layer_ind, mlp._layer_ind)
            handles.append(mlp.register_forward_hook(hook_fn))

        # run for small chunks of data to avoid OOM
        if verbose:
            print('moe forward input', 'mtp', is_mtp,
                  states['hidden_states'].shape, states['attention_mask'].shape)
        for i_ in tqdm(range(0, len(states['hidden_states']), self.batch_size),
                       desc=f"{'MTP' if is_mtp else 'MoE'} Running forward pass to "
                            f"{'collect expert activations' if collect_outputs else 'propagate hidden states'}..."):
            if not is_mtp:
                hid_states = moe_forward(self.model.model.layers[layer_ind],
                                         states,
                                         i=i_,
                                         chunk_size=self.batch_size,
                                         device=self.device).data
            else:
                if i_ == 0:
                    self.mtp_layer.to(self.device).to(states['hidden_states'].dtype)
                hid_states = self.mtp_layer(hidden_states=states['hidden_states'][i_:i_ + self.batch_size],
                                            position_ids=states['position_ids'],
                                            inputs_embeds=inputs_embeds[i_:i_ + self.batch_size],
                                            attention_mask=states['attention_mask'][i_:i_ + self.batch_size],
                                            shift=False)

            if i_ == 0 and verbose:
                n_exp = len(self.model.model.layers[layer_ind].mlp.experts) if hasattr(
                    self.model.model.layers[layer_ind].mlp, 'experts') and not is_mtp else \
                    ((len(self.mtp_layer.layer.mlp.experts)) if is_mtp else 'dense')
                print('moe forward, hid_states', hid_states.shape, hid_states.dtype, hid_states.device,
                      'experts', n_exp,
                      flush=True)

            if upd_hid:
                # update hid states given already merged experts in previous layers
                hidden_states.append(hid_states)

        if upd_hid:
            states['hidden_states'] = torch.cat(hidden_states).data
            if verbose:
                print('after moe forward, hidden_states', states['hidden_states'].shape, flush=True)

        # Remove hook when done
        for handle in handles:
            handle.remove()

        return expert_outs, states

    @torch.no_grad()
    def fit(self):
        """
        Main function pruning/merging experts in all MoE layers of the model.
        :return: merged model
        """

        total_experts = 0
        top_k = []
        for layer_ind, layer in enumerate(self.model.model.layers):
            if layer_ind < self.first_moe_layer:
                top_k.append(0)  # dense layer, no MoE
                continue
            total_experts += len(layer.mlp.experts)
            top_k.append(layer.mlp.top_k)
        print(f'\nTotal experts before merging: {total_experts}', flush=True)
        print(
            f'Number of activated experts per token (top_k min-max): '
            f'{np.min([t for t in top_k if t > 0])}-{np.max(top_k)}\n',
            flush=True)

        print(f'running merging for {self.num_layers} layers', flush=True)
        states = get_moe_input(self.model, self.device, **self.batch)
        inputs_embeds = None if self.mtp_layer is None else states['hidden_states'].clone()

        top_k = []
        total_experts = 0
        # main loop over layers
        for layer_ind in range(self.num_layers):
            verbose = self.verbose and layer_ind == 0

            if layer_ind < self.first_moe_layer:
                # first layer(s) can be dense (like in GLM), so just propagate hidden states
                if self.sequential:
                    _, states = self._forward_pass(states,
                                                   layer_ind,
                                                   collect_outputs=False,
                                                   upd_hid=True)
                continue

            is_mtp = layer_ind >= len(self.model.model.layers)  # assuming mtp going after all MoE layers
            moe_layer = (self.mtp_layer.layer if is_mtp else self.model.model.layers[layer_ind]).mlp  # expert ffn
            assert getattr(moe_layer.gate, 'bias', None) is None, ('no bias expected', moe_layer.gate)
            num_params = num_parameters(moe_layer)

            n_experts = len(moe_layer.experts)
            if self.merge_size < n_experts:
                # fills expert_outs for a single layer
                # for sequential merging, just get expert features at this step
                expert_outs, states = self._forward_pass(states,
                                                         layer_ind,
                                                         collect_outputs=True,
                                                         upd_hid=not self.sequential,
                                                         inputs_embeds=inputs_embeds,
                                                         verbose=verbose)

                if self.use_gate_output:
                    gate_logits = torch.cat(expert_outs['gate']).view(-1, n_experts)  # (B,S,E) -> (B*S, E)
                else:
                    gate_logits = None

                expert_logits = expert_outs['final']  # shape: (E, 1, H)
                expert_act = torch.cat(expert_outs['hid_act'], dim=1)  # shape: (E, len(ind)*n_chunks, D)
                saliency = expert_outs['saliency']

                if verbose:
                    print(f'layer {layer_ind}: expert_logits', expert_logits.shape, expert_logits.dtype,
                          'expert_act', expert_act.shape, expert_act.dtype,
                          'gate_logits', (gate_logits.shape, gate_logits.dtype) if self.use_gate_output else 'None',
                          f'{self.saliency} saliency scores', saliency.shape,
                          [str('%.8f' % f) for f in saliency], flush=True)

                zeros = saliency == 0
                n_zeros = zeros.sum().item()
                # replace zeros in saliency with min non-zero value to allow unused experts be merged
                if n_zeros > 0:
                    assert n_zeros < len(saliency), saliency
                    saliency[zeros] = min(0.5, saliency[saliency > 0].min().item())

                # get all the groups for a given layer
                print(f'\nrunning {self.grouping} grouping for {self.group_size} clusters on {self.device}')

                expert_weights = None
                if 'weight' in self.merging:
                    # mlps: list of length N=128; each element is a dict with 'gate_proj','up_proj','down_proj'
                    # use pca for building low-dimensional weight representations and faster processing
                    expert_weights = torch.stack([ffn_weight_matrix(mlp) for mlp in moe_layer.experts])  # (n, 768, 6144)
                    weights_sz = expert_weights.shape
                    expert_weights = pca_reduce(normalize_rows(expert_weights).flatten(0, 1),
                                                r=self.pca_dim,
                                                verbose=verbose)  # (n*768, 64)
                    expert_weights = expert_weights.reshape(n_experts, -1, self.pca_dim)  # (n, 768, 64)
                    if verbose:
                        print(f'layer {layer_ind}: expert_weights', weights_sz,
                                'expert_weights low dim', expert_weights.shape, expert_weights.dtype,
                                flush=True)

                if self.grouping == 'hcsmoe':
                    # hierarchical clustering
                    cluster_lbl, centroid_inds = hcsmoe(expert_logits, self.merge_size)
                elif self.grouping == 'ream':
                    # implements the baseline mc-smoe if self.group_size=0
                    cluster_lbl, centroid_inds = pseudo_group(saliency,
                                                              expert_logits=expert_logits,
                                                              k=self.merge_size,
                                                              gate_logits=gate_logits,
                                                              group_size=self.group_size)
                else:
                    raise NotImplementedError(self.grouping)

                groups = []
                group_sizes_stat = {}
                for j in range(self.merge_size):
                    group = list(map(int, np.where(cluster_lbl == j)[0]))  # expert indices
                    if centroid_inds is not None and len(group) > 1:
                        # make sure that the cluster centers are the first indices in a group (g)
                        group = list(sorted(group, key=lambda g: g not in centroid_inds))
                    groups.append(np.asarray(group))
                    s = len(group)
                    if s not in group_sizes_stat:
                        group_sizes_stat[s] = 0
                    group_sizes_stat[s] += 1

                print(f'grouping into {self.merge_size} groups is done', flush=True)
                if verbose:
                    for s in sorted(group_sizes_stat.keys()):
                        print(f'{group_sizes_stat[s]} groups with {s} experts', flush=True)

                experts_to_delete = []
                experts_to_keep = []
                for group_ind, group in enumerate(groups):
                    if verbose:
                        print(f'layer {layer_ind}: group_ind:', group_ind, 'size:', len(group),
                              'group:', group,
                              'saliency:', [float(saliency[ind]) for ind in group], flush=True)

                    # all the experts in the group point to the same memory (equivalent to removing experts)
                    if len(group) > 1:
                        merged_exp = self._merge(experts=moe_layer.experts,
                                                 group=group,
                                                 saliency=saliency,
                                                 expert_weights=expert_weights,
                                                 expert_act=expert_act
                                                 )
                        moe_layer.experts[group[0]] = merged_exp
                        experts_to_delete.extend(group[1:])  # keep the first expert in each group
                    experts_to_keep.append(group[0])

                experts_to_delete.sort(reverse=True)
                experts_to_keep.sort()
                experts_to_keep = np.array(experts_to_keep)
                for idx in experts_to_delete:
                    del moe_layer.experts[idx]
                moe_layer.num_experts = len(moe_layer.experts)
                moe_layer.gate.weight.data = moe_layer.gate.weight.data[experts_to_keep]
                moe_layer.gate.out_features = len(moe_layer.experts)
                # Update TopkRouter-specific attributes if present (e.g. GLM); no-op for nn.Linear (Qwen)
                if hasattr(moe_layer.gate, 'n_routed_experts'):
                    moe_layer.gate.n_routed_experts = len(moe_layer.experts)
                if hasattr(moe_layer.gate, 'e_score_correction_bias'):
                    moe_layer.gate.e_score_correction_bias = moe_layer.gate.e_score_correction_bias[experts_to_keep]

                if is_mtp:
                    self.mtp_layer.layer.mlp = moe_layer
                    self.mtp_state_dict = self.mtp_layer.state_dict()

            elif self.merge_size > n_experts:
                raise ValueError(f'merge_size {self.merge_size} > n_experts {n_experts}')
            else:
                print(f'\nlayer {layer_ind}: k={self.merge_size}, '
                      f'so model is already compressed, skipping compression', flush=True)

            assert moe_layer.num_experts == self.merge_size, (moe_layer.num_experts, self.merge_size)

            total_experts += len(moe_layer.experts)
            top_k.append(moe_layer.top_k)

            if self.sequential:
                # upd hid states given merged experts in the current layer
                _, states = self._forward_pass(states,
                                               layer_ind,
                                               collect_outputs=False,
                                               upd_hid=True,
                                               verbose=verbose)

            if is_mtp:
                self.mtp_layer.to('cpu')
            else:
                self.model.model.layers[layer_ind].to('cpu')

            print('finished layer', layer_ind,
                  'num experts orig', n_experts,
                  'num experts merge', len(self.mtp_layer.layer.mlp.experts
                                           if is_mtp else self.model.model.layers[layer_ind].mlp.experts),
                  'num params orig', num_params,
                  'num params merged', num_parameters(self.mtp_layer.layer.mlp
                                                      if is_mtp else self.model.model.layers[layer_ind].mlp),
                  'cpu mem=%.2f' % mem('cpu'),
                  'gpu mem=%.2f' % mem(0),
                  '\n\n',
                  flush=True)
            torch.cuda.empty_cache()  # some annoying gpu/cpu memory leaks happen

        if self.verbose:
            print(self.model.model.layers[self.first_moe_layer].mlp, flush=True)
            print(self.model.model.layers[self.first_moe_layer + 1].mlp, flush=True)
            print(self.model.model.layers[-1].mlp, flush=True)

        # update config for the merged model so that loading/saving works properly
        moe_layer = self.model.model.layers[self.first_moe_layer].mlp
        self.model.config.num_experts = moe_layer.num_experts
        self.model.config.num_experts_per_tok = moe_layer.top_k
        # GLM compatibility: also update n_routed_experts if present
        if hasattr(self.model.config, 'n_routed_experts'):
            self.model.config.n_routed_experts = moe_layer.num_experts

        print(f'\nTotal experts after merging: {total_experts}', flush=True)
        if len(top_k) > 0:
            print(f'Number of activated experts per token (top_k min-max): '
                  f'{np.min(top_k)}-{np.max(top_k)}\n',
                  flush=True)

        return self.model

    @torch.no_grad()
    def _merge(self,
               experts,
               group,
               saliency=None,
               expert_weights: torch.Tensor = None,
               expert_act: torch.Tensor = None) -> nn.Module:
        """
        Merges (or, as a special case, prunes) experts in a group.
        :param experts: all the experts in the MoE layer (list of N nn.Modules)
        :param group: indices of experts in the group to be merged
        :param saliency: expert saliency like freq or reap (of length N)
        :param expert_weights: tensor of shape (num_experts, 768, 6144) or (num_experts, 768, pca_dim)
        :param expert_act: tensor of shape (num_experts, sampled seq_len*batch, expert hidden dim)
        :return: merged expert (same type as experts)
        """

        assert len(group) > 0, group
        merged = experts[group[0]]

        if len(group) == 1 or self.merging == 'none':  # drop/prune other experts in case of none like in REAP
            return merged

        if self.merging == 'avg':
            # For each param, average over experts in group
            for name, param in merged.named_parameters():
                stacked = torch.stack([experts[i].state_dict()[name] for i in group], dim=0)
                avg_param = stacked.mean(dim=0).to(param)
                param.copy_(avg_param)
        else:
            # assume indices are already sorted by saliency
            assert len(saliency) == len(experts), (len(saliency), len(experts))
            if isinstance(saliency, torch.Tensor):
                saliency = saliency.cpu().numpy()
            w = saliency[group]  # use sorted saliency as weights
            w_sum = w.sum()
            assert np.all(w >= 0) and w_sum > 0, (w, w_sum)  # all non-negative and at least one non-zero
            w = w / w_sum

            use_logits = 'logits' in self.merging
            use_weights = 'weights' in self.merging

            if use_logits or use_weights:
                if use_logits:
                    expert_act = expert_act.permute(0, 2, 1).float()  # (E, D, B*S)
                    act_centroid = expert_act[group[0]].to(self.device)
                    act_centroid = normalize_rows(act_centroid) if act_centroid.shape[-1] > 1 else act_centroid
                if use_weights:
                    # expert_weights: (n, 768, 6144) or (n, 768, pca_dim)
                    expert_weights = normalize_rows(expert_weights)  # normalize last dim for all experts

                perm_lst = []  # permutations from other experts to the first expert in the group
                for j, idx in enumerate(group[1:]):
                    cost = 0
                    if use_logits:
                        act_j = expert_act[idx].to(self.device)
                        # normalize in the loop for better memory efficiency
                        act_j = normalize_rows(act_j) if act_j.shape[-1] > 1 else act_j
                        cost += torch.cdist(act_centroid, act_j).cpu().numpy()
                    if use_weights:
                        cost += torch.cdist(expert_weights[group[0]].to(self.device),
                                           expert_weights[idx].to(self.device)).cpu().numpy()
                    _, col_ind = linear_sum_assignment(cost)
                    perm_lst.append(col_ind)  # permutation of B aligned to A
            else:
                assert self.merging == 'avg_freq', f'{self.merging} is an unsupported merging method'

            for j, idx in enumerate(group[1:]):
                if self.merging == 'avg_freq':
                    aligned_b = experts[idx]
                else:
                    aligned_b = apply_perm_to_ffn(experts[idx], perm_lst[j])

                for name, param in merged.named_parameters():
                    # compute weighted average of params
                    if j == 0:
                        param.data = (param.data.float() * w[0]).to(param.data)  # centroid
                    param.data = (param.data.float() +
                                  w[j + 1] * aligned_b.state_dict()[name].data.float()).to(param.data)

        return merged
