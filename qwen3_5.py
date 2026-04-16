# Copyright (c) 2026. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""

Post-process the Qwen3.5 model to save it together with the original visual model.

1. Apply merging to all layers but MTP:
    python merge.py --model Qwen/Qwen3.5-122B-A10B --save_path /your-path/Qwen3.5-122B-A10B-REAM --merge_size 192 --group_size 32

2. Apply merging to the MTP layer (fix safetensors path based on your setup):

    python merge.py --model /your-path/Qwen3.5-122B-A10B-REAM --save_path /your-path/Qwen3.5-122B-A10B-REAM-mtp --merge_size 192 --group_size 32 \
    --mtp_safe_tensors Qwen/Qwen3.5-122B-A10B-REAM/model.safetensors-00037-of-00039.safetensors,Qwen/Qwen3.5-122B-A10B-REAM/model.safetensors-00038-of-00039.safetensors,Qwen/Qwen3.5-122B-A10B-REAM/model.safetensors-00039-of-00039.safetensors

3. Run this script

    python qwen3_5.py

 4. Rename files

    for file in *-00033.safetensors; do
        mv "$file" "${file/-00033.safetensors/-00034.safetensors}"
    done

"""

import os
import json
from transformers import AutoModelForCausalLM, Qwen3_5MoeForConditionalGeneration, AutoProcessor
from safetensors.torch import load_file, save_file


model_name = 'Qwen/Qwen3.5-122B-A10B'
merged_path = '/your-path/Qwen3.5-122B-A10B-REAM-mtp'
save_path = merged_path + '-full'
device = 'cpu'
original_vlm = Qwen3_5MoeForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype='auto',
    device_map=device,
)
llm_merged = AutoModelForCausalLM.from_pretrained(
    merged_path,
    torch_dtype='auto',
    device_map=device,
)
original_vlm.model.language_model = llm_merged.model
original_vlm.config.text_config.num_experts = llm_merged.model.config.num_experts
original_vlm.config.text_config.merge_args = llm_merged.model.config.merge_args
original_vlm.save_pretrained(
    save_path,
    safe_serialization=True,
    max_shard_size='7GB')

processor = AutoProcessor.from_pretrained(model_name)
processor.save_pretrained(save_path)

# renaming to fix the transformers bug for qwen3.5 and to add the mtp layer

# count the number of files ending with .safetensors
n_files = sum([1 for f in os.listdir(save_path) if f.endswith('.safetensors')])
model_safetensors = save_path + '/model.safetensors.index.json'
model_dict = json.load(open(model_safetensors, 'r'))
assert len(model_dict) == 2, f'expected 2 keys in the model dict, but got {list(model_dict.keys())}'
model_dict_new = {'metadata': model_dict['metadata'], 'weight_map': {}}
c = 0
for k, v in model_dict['weight_map'].items():
    new_v = v.replace(f'-{n_files:05d}.safetensors', f'-{n_files+1:05d}.safetensors')  # +1 because of mtp to be added
    if k.startswith('model.language_model.visual'):
        model_dict_new['weight_map'][k.replace('model.language_model.visual', 'model.visual')] = new_v
        c += 1
    else:
        model_dict_new['weight_map'][k] = new_v
print(f'renamed {c} keys')

# mtp fix
mtp_merged_path = merged_path + '/mtp.safetensors'
mtp_state = load_file(mtp_merged_path, device=device)
# copy shared expert
orig_mtp = load_file('/your-cache/hub/models--Qwen--Qwen3.5-122B-A10B/model.safetensors-00039-of-00039.safetensors', device=device)
for k, v in orig_mtp.items():
    if k.startswith('mtp.layers.0.mlp.shared_expert'):
        mtp_state[k] = v
# add mtp to model.safetensors.index.json
mtp_state_new = {}
for k in mtp_state:
    if not k.startswith('mtp.'):
        new_k = 'mtp.' + k.replace('layer.', 'layers.0.')
        print(new_k)
    else:
        new_k = k
    model_dict_new['weight_map'][new_k] = f'model-{n_files + 1:05d}-of-{n_files + 1:05d}.safetensors'
    mtp_state_new[new_k] = mtp_state[k]
save_file(mtp_state_new, os.path.join(save_path, f'model-{n_files + 1:05d}-of-{n_files + 1:05d}.safetensors'))
json.dump(model_dict_new, open(model_safetensors, 'w'), indent=4)

# load each file and rename model.language_model.visual to model.visual as in model.safetensors.index.json
for ind in range(1, n_files + 1):
    fpath = os.path.join(save_path, f'model-{ind:05d}-of-{n_files:05d}.safetensors')
    state = load_file(fpath, device=device)
    state_new = {}
    c = 0
    for k, v in state.items():
        if k.startswith('model.language_model.visual'):
            state_new[k.replace('model.language_model.visual', 'model.visual')] = v
            c += 1
        else:
            state_new[k] = v
    if c > 0:
        print(f'renamed {c} keys in {fpath}')
        save_file(state_new, fpath)

# check correctness
model_dict = json.load(open(model_safetensors, 'r'))
n_files = sum([1 for f in os.listdir(save_path) if f.endswith('.safetensors')])
states = {}
c = 0
for ind in range(1, n_files + 1):
    fpath = os.path.join(save_path,
                         f'model-{ind:05d}-of-{n_files:05d}.safetensors')
    print(fpath)
    state = load_file(fpath, device=device)
    for k, v in state.items():
        assert k not in states, f'key {k} is duplicated across files'
        assert model_dict['weight_map'][k] == f'model-{ind:05d}-of-{n_files:05d}.safetensors', \
            f'mismatch for key {k} between model dict and file name: {model_dict["weight_map"][k]} vs model-{ind:05d}-of-{n_files:05d}.safetensors'
        states[k] = v
        c += 1
print(f'checked {c} keys')

# check that can be loaded
# merged_vlm = Qwen3_5MoeForConditionalGeneration.from_pretrained(
#     save_path,
#     torch_dtype='auto',
#     device_map=device,
# )

print('done')
