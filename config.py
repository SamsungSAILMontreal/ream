# Copyright (c) 2026. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""

Shared config for MC evaluation and merging.

"""

import argparse
import subprocess
import platform
import time
import os
import torch
import torch.backends.cudnn as cudnn
import lm_eval
import vllm
import transformers
import datasets
import numpy as np


def init_config(mode='eval', parser=None, verbose=True):
    if verbose:
        print('\nEnvironment:')
    env = {}
    try:
        # print git commit to ease code reproducibility
        env['git commit'] = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception as e:
        print(e, flush=True)
        env['git commit'] = 'no git'

    env['hostname'] = platform.node()
    env['torch'] = torch.__version__
    env['lm_eval'] = lm_eval.__version__
    env['vllm'] = vllm.__version__
    env['transformers'] = transformers.__version__
    env['datasets'] = datasets.__version__
    env['numpy'] = np.__version__
    env['cuda available'] = torch.cuda.is_available()
    env['cudnn enabled'] = cudnn.enabled
    env['cuda version'] = torch.version.cuda
    env['start time'] = time.strftime('%Y%m%d-%H%M%S')
    for x, y in env.items():
        env[x] = str(y)  # to enable easy serialization
        if verbose:
            print('{:20s}: {}'.format(x[:20], y), flush=True)

    if parser is None:
        # create parser or use the existing one if provided
        parser = argparse.ArgumentParser(description='REAM: Merging Improves Pruning of Experts in LLMs')

    parser.add_argument(
        '--model',
        default='Qwen/Qwen3-30B-A3B-Instruct-2507',
        type=str,
        help='model name or path'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help='cache dir for huggingface models (ignored if --model is full path)'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='allow to download the model from HuggingFace if not found locally'
    )
    parser.add_argument(
        '--batch_size',
        default=8,
        type=int,
        help='number of sequences used during single forward pass for merging or evaluation'
    )
    if mode == 'eval':
        parser.add_argument(
            '--task',
            type=str,
            default='winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte',
            help='a single task or a list of tasks; default is 8 MC tasks'
        )

    args = parser.parse_args()

    if verbose:
        def print_args(args_, name):
            print('\n%s:' % name)
            args_var = vars(args_)
            for x in sorted(args_var.keys()):
                print('{:20s}: {}'.format(x[:20], args_var[x]))
            print('\n', flush=True)

        print_args(args, 'Script Arguments ({} mode)'.format(mode))

        if torch.cuda.is_available():
            print('GPU Info:')
            for device in range(torch.cuda.device_count()):
                print(torch.cuda.get_device_properties(device).name,
                      'mem=%.2f GB' % (torch.cuda.max_memory_allocated(device) / (1024 ** 3)), flush=True)

    return args
