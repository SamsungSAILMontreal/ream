# Copyright (c) 2026. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""

Merge experts in a MoE model into fewer experts,
with various options for saliency, grouping, and merging methods as defined in the REAM paper.

Usage:
    python merge.py --model <model_name_or_path> --merge_size <number_of_experts> --save_path <path_to_save_merged_model> [other options, see below]

"""

import os
import time
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from ream import Merger
from config import init_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--merge_size',
        default=None,
        type=int,
        required=True,
        help='number of experts after merging; None means no merging'
    )
    parser.add_argument(
        '--saliency',
        default='reap',
        choices=['freq', 'reap'],
        type=str,
        help='method to choose dominant (most salient) experts'
    )
    parser.add_argument(
        '--merging',
        default='logits+weights',
        choices=['avg', 'weights', 'logits', 'logits+weights', 'none'],
        type=str,
        help='method to merge a group of experts into one; '
             'none means keep only one (most dominant) expert in the group and drop/prune others'
    )
    parser.add_argument(
        '--grouping',
        default='ream',
        choices=['ream', 'hcsmoe'],
        type=str,
        help='method to group experts (ignored for pruning, i.e., when --merging is none)'
    )
    parser.add_argument(
        '--dataset',
        default='c4+math+code',
        type=str,
        help='calibration dataset(s) for dataset-based clustering, separated by +'
    )
    parser.add_argument(
        '--mix_ratio',
        default='0.0,0.3,0.7',
        type=str,
        help='data mix ratio for in the calibration data, should sum to 1 and match the datasets defined by --dataset'
    )
    parser.add_argument(
        '--group_size',
        default=16,
        type=int,
        help='group size in pseudo-pruning as defined in REAM; 0 means use simple MC-SMoE-style grouping'
    )
    parser.add_argument(
        '--no_gate_output',
        action='store_true',
        help='disable using gate outputs for grouping as extra information; enabled by default'
    )
    parser.add_argument(
        '--no_sequential',
        action='store_true',
        help='when set to True it will precompute expert input/outputs for all layers before merging,'
             'when set to False if will recompute expert input/outputs after merging each layer as defined in REAM'
    )
    parser.add_argument(
        '--no_gated_sim',
        action='store_true',
        help='disable applying softmax to the gate outputs; enabled by default'
    )
    parser.add_argument(
        '--save_path',
        default=None,
        required=True,
        type=str,
        help='path for saving merged models'
    )
    parser.add_argument(
        '--mtp_safe_tensors',
        default=None,
        type=str,
        help='path to the safe tensors file where MTP weights are stored')

    args = init_config(mode='merge', parser=parser, verbose=True)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        print('Setting pad_token_id to eos_token_id', flush=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype='auto',
        device_map='cpu' ,
        cache_dir=None if os.path.exists(args.model) else args.cache_dir,
        local_files_only=not args.download,  # download the model manually first (see README for examples)
        low_cpu_mem_usage=False,  # to avoid loading on meta device during merging
    ).eval()

    print(f'Number of parameters before merging: {model.num_parameters()}')
    for name, p in model.named_parameters():
        if not str(p.device).lower().startswith(('cuda', 'cpu')):
            raise ValueError(f'all parameters should be on cuda or cpu, '
                             f'but param {name} of shape {p.shape} and type {p.dtype} is on {p.device}')

    mtp_state_dict = None
    if args.mtp_safe_tensors not in [None, '', 'none', 'None']:
        from safetensors.torch import load_file, save_file
        try:
            mtp_state_dict = load_file(args.mtp_safe_tensors)
        except Exception as e:
            print(f'error loading mtp weights from {args.mtp_safe_tensors}', flush=True)

    tokenizer_name = None
    for sfx in ['qwen3', 'glm']:
        if sfx in args.model.lower():
            tokenizer_name = sfx
            break
    if tokenizer_name is None:
        raise ValueError(f'model name {args.model} is not recognized or not supported')

    merger = Merger(model,
                    mtp_state_dict=mtp_state_dict,
                    merge_size=args.merge_size,
                    grouping=args.grouping,
                    merging=args.merging,
                    saliency=args.saliency,
                    dataset=args.dataset,
                    mix_ratio=args.mix_ratio,
                    tokenizer_name=tokenizer_name,
                    batch_size=args.batch_size,
                    group_size=args.group_size,
                    sequential=not args.no_sequential,
                    use_gate_output=not args.no_gate_output,
                    gated_sim=not args.no_gated_sim)
    model = merger.fit()
    print(f'Number of parameters after merging: {model.num_parameters()}')

    # Saving the merged model and tokenizer, and the mtp state dict if exists.
    # add all the args to model.config for future reference
    model.config.merge_args = vars(args)
    print('\nmodel.config.merge_args:', model.config.merge_args, flush=True)

    try:
        print('saving the model with dtype', next(model.parameters()).dtype, flush=True)
    except Exception as e:
        print(e, flush=True)
    try:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
    except Exception as e:
        print(e, flush=True)
    try:
        tokenizer.save_pretrained(args.save_path)
    except Exception as e:
        print(e, flush=True)
        print('failed to save tokenizer to', args.save_path, flush=True)

    if merger.mtp_state_dict is not None:
        # this file needs to be then manually renamed in the folder and in model.safetensors.index.json
        try:
            save_file(merger.mtp_state_dict, args.save_path + '/mtp.safetensors')
            print('saved mtp state dict to', args.save_path + '/mtp.safetensors', flush=True)
        except Exception as e:
            print(e, flush=True)
            print('failed to save the mtp layer to', args.save_path + '/mtp.safetensors', flush=True)

    try:
        model.save_pretrained(args.save_path,
                              safe_serialization=True,
                              max_shard_size='10GB' if '-FP8' in args.model else '4GB')
        print('merged model saved to', args.save_path, flush=True)
    except Exception as e:
        print(e, flush=True)
        print('failed to save the merged model to', args.save_path, flush=True)
        raise

    print('done at:', time.strftime('%Y%m%d-%H%M%S'), flush=True)
