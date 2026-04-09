# Copyright (c) 2026. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""

Pre-tokenize calibration data with a given model's tokenizer.
Creates batch files compatible with the merger's expected format.
To reproduce our results more closely, please use precomputed batches available in the ./data folder.

Usage:
    python create_data_glm.py [--model zai-org/GLM-4.5-Air] [--batch_size 3072] [--seq_len 512]

"""

import os
import argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

C4_LOCATION = '/datasets/c4/c4/en'

def print_seq_stats(batch_, tokenizer_=None):
    seq_len_stats = batch_['attention_mask'].sum(1).float()
    print('batch', batch_['input_ids'].shape,
          'seq len avg = %.2f' % (seq_len_stats.mean().item()),
          'min = %d' % (seq_len_stats.min().item()),
          'max = %d' % (seq_len_stats.max().item()),
          'total pad tokens = ',
          (batch_['input_ids'] == tokenizer_.pad_token_id).sum().item() if tokenizer_ else 'N/A',
          'total end of seq tokens = ',
          (batch_['input_ids'] == tokenizer_.eos_token_id).sum().item() if tokenizer_ else 'N/A'
          )


def create_batch(tokenizer, dataset, batch_size, seq_len, seed=42):
    min_seq_len_dset = int(0.9 * seq_len)  # use longer sequences for better expert freq calculation
    print('min_seq_len_dset', min_seq_len_dset)
    if dataset == 'c4':
        dset = load_dataset(C4_LOCATION, split='validation', streaming=True)
        dset = dset.shuffle(seed=seed)
        batch = {'input_ids': [], 'attention_mask': []}
        for example in dset:
            if len(example['text']) < min_seq_len_dset:
                continue
            token_ids = tokenizer(example['text'], return_tensors='pt',
                                  padding=True, truncation=True, max_length=seq_len)
            input_inds = token_ids['input_ids']
            attention_mask = token_ids['attention_mask']
            input_inds = F.pad(input_inds, (0, seq_len - input_inds.shape[1]),
                               value=tokenizer.pad_token_id)
            attention_mask = F.pad(attention_mask, (0, seq_len - attention_mask.shape[1]),
                                   value=0)
            batch['input_ids'].append(input_inds[0])
            batch['attention_mask'].append(attention_mask[0])
            if len(batch['input_ids']) >= batch_size * 3:
                break
    elif dataset == 'math':
        dset = load_dataset('AI-MO/NuminaMath-1.5', split='train')
        # filter the dset by source=='cn_k12' and length of solution > min_seq_len_dset
        # use cn_k12 and olympiads to avoid overlap with eval sets (though still not guaranteed)
        dset = dset.filter(lambda x: x['source'] in ['cn_k12', 'olympiads']
                                     and len(x['solution']) > min_seq_len_dset)
        print('dset size after filtering:', len(dset))
        dset = dset.shuffle(seed=seed)
        batch = tokenizer(dset[:batch_size * 100]['solution'],
                          return_tensors='pt',
                          padding=True, truncation=True, max_length=seq_len)
    elif dataset == 'code':
        dset = load_dataset('bigcode/the-stack-smol', split='train')
        dset = dset.filter(lambda x: len(x['content']) > min_seq_len_dset)
        print('dset size after filtering:', len(dset))
        dset = dset.shuffle(seed=seed)
        batch = tokenizer(dset[:batch_size * 3]['content'],
                          return_tensors='pt',
                          padding=True, truncation=True, max_length=seq_len)
    else:
        raise NotImplementedError(f'Dataset {dataset} not implemented')

    # Filter short sequences
    batch_filter = {'input_ids': [], 'attention_mask': []}
    for b in range(len(batch['input_ids'])):
        if (batch['input_ids'][b] == tokenizer.pad_token_id).sum() > (seq_len - min_seq_len_dset) and dataset != 's1':
            # skip too short sequences or sequences with too few non-pad tokens
            continue
        batch_filter['input_ids'].append(batch['input_ids'][b])
        batch_filter['attention_mask'].append(batch['attention_mask'][b])

        if len(batch_filter['input_ids']) >= batch_size:
            break

    batch_filter['input_ids'] = torch.stack(batch_filter['input_ids'], dim=0)
    batch_filter['attention_mask'] = torch.stack(batch_filter['attention_mask'], dim=0)
    print_seq_stats(batch_filter, tokenizer)

    return batch_filter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-30B-A3B-Instruct-2507', type=str)
    parser.add_argument('--batch_size', default=3072, type=int)
    parser.add_argument('--seq_len', default=512, type=int)
    parser.add_argument('--sfx', default='qwen3', type=str)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f'Tokenizer: pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for dataset in ['c4', 'math', 'code']:
        batch_file = f'{dataset}_b{args.batch_size}_seq{args.seq_len}_{args.sfx}_seed{args.seed}.pt'
        if os.path.exists(batch_file):
            print(f'{batch_file} already exists, skipping')
            continue

        print(f'\nCreating {batch_file}...')
        batch = create_batch(tokenizer, dataset, args.batch_size, args.seq_len, seed=args.seed)
        torch.save(batch, batch_file)
        print(f'Saved {batch_file}: {batch["input_ids"].shape}')

    print('\nDone! All batch files created.')
