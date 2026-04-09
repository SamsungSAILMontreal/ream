# Copyright (c) 2026. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""

Evaluation on 8 MC tasks:

    python eval_mc.py --model Qwen/Qwen3-30B-A3B-Instruct-2507

"""

import os
import time
import numpy as np
import torch
import lm_eval
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from config import init_config
logging.getLogger('lm_eval').setLevel(logging.ERROR)


if __name__ == '__main__':
    args = init_config(mode='eval', verbose=True)

    # Load tokenizer and model
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        print('Setting pad_token_id to eos_token_id', flush=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    device_map = 'balanced'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype='auto',
        device_map=device_map,
        cache_dir=None if os.path.exists(model_name) else args.cache_dir,
        local_files_only=not args.download,  # download the model manually first (see README for examples)
        low_cpu_mem_usage=True,  # allow loading on meta device for eval
    ).eval()

    assert args.task not in [None, 'none'], 'Please specify the task(s) for evaluation'
    tasks = [t.replace(' ', '') for t in args.task.split(',') if len(t.replace(' ', '')) > 0]
    print('tasks:', len(tasks), tasks, flush=True)
    scores = []
    for task in tasks:

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()  # Reset peak memory before timing

        start = time.time()
        print(f'Evaluating {task}...', flush=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for all kernels to finish before starting timer

        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            trust_remote_code=True,
            device_map=device_map
        )
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=task,
            num_fewshot=None,
            batch_size=args.batch_size,
            random_seed=0,
            numpy_random_seed=1234,
            torch_random_seed=1234
        )
        try:
            print(make_table(results))
            if "groups" in results:
                print(make_table(results, "groups"))
        except Exception as e:
            print(e)

        try:
            acc = results['results'][task]['acc,none']
        except KeyError as e:
            print(f'Accuracy not found in results for task {task}, another metric needs to be chosen.')
            raise
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for all kernels to finish before stopping timer
        end = time.time()

        elapsed = end - start
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            peak_mem = -1
        print(f'Task={task}, acc={acc:.4f}, time: {elapsed:.2f} seconds, '
              f'Peak GPU memory: {peak_mem:.2f} GB', flush=True)
        scores.append(acc)
    for task, score in zip(tasks, scores):
        print(f'{task}: {score:.4f}', flush=True)
    print(f'\nAvg +- std score over {len(tasks)} tasks: '
          f'{np.mean(scores):.4f} +- {np.std(scores):.4f}', flush=True)
