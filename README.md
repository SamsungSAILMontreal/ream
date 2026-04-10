# REAM: Merging Improves Pruning of Experts in LLMs

Authors: [Saurav Jha](https://sauravjha.com.np/)\*, Maryam Hashemzadeh, Ali Saheb Pasand, Ali Parviz, Min-Joong Lee, [Boris Knyazev](https://bknyaz.github.io)\*

*equal contribution

[![arXiv](https://img.shields.io/badge/arXiv-2403.12143-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2604.04356) 

[blogpost](https://bknyaz.github.io/blog/2026/moe/)

REAM-ed Qwen3 and GLM-4.5-Air and models 🤗:
  - https://huggingface.co/collections/SamsungSAILMontreal/ream
  - https://huggingface.co/bknyaz

[Baseline REAP repo](https://github.com/CerebrasResearch/reap)

## Requirements

See requirements.txt for the necessary packages and their recommended versions.
These are the versions we used for our experiments, but you may be able to use other versions as well.

`pip install -r requirements.txt`

## Merging

Obtaining a merged model from the original one requires running the merge.py script with appropriate arguments, e.g.:

`python merge.py --model <> --merge_size <> --save_path <> ...`

See merge.py and config.py for the hyperparameters and options.
Default arguments correspond to our full REAM model.
`--merge_size` should be chosen appropriately, e.g. 25% or 50% of the original number of experts.

MTP (Multi-token Prediction) layers, present in some LLMs for more efficient decoding, 
are supported in our code (treated as an additional MoE layer), 
but it was checked only for Qwen3 and requires additional steps:
- setting the `--mtp_safe_tensors` file path
- renaming the merged model's safe tensors of the MTP and other layers and the corresponding model.safetensors.index.json file 

See [Qwen3-Next-80B-A3B-Instruct-REAM](https://huggingface.co/bknyaz/Qwen3-Next-80B-A3B-Instruct-REAM) for our example of the merged model with MTP layers.


## Evaluation

We evaluate the original and compressed (merged/pruned) models on the MC and GEN tasks.

### MC tasks

Run `python eval_mc.py` to evaluate the merged model on 8 multiple-choice tasks.
The main options are `--model` and `--batch_size`. 
See eval_mc.py and config.py for more options.

### GEN tasks

We evaluate on 6 generative tasks: IFEval, AIME25, GSM8K, GPQA-Diamond, HumanEval and LiveCodeBench
We use the following tools as described on [our main huggingface page](https://huggingface.co/SamsungSAILMontreal/Qwen3-30B-A3B-Instruct-2507-REAM) and in the paper:

- lm-eval https://github.com/EleutherAI/lm-evaluation-harness
- LiveCodeBench https://github.com/LiveCodeBench/LiveCodeBench
- For GLM-4.5-Air, HumanEval and LiveCodeBench tasks: https://github.com/zai-org/glm-simple-evals (see [GLM-4.5-Air-REAM](https://huggingface.co/bknyaz/GLM-4.5-Air-REAM) for details)


## License

MIT, see the [LICENSE](LICENSE) file.


## Citation

If you find this work useful, please consider citing:

```
@article{jha2026ream,
  title={REAM: Merging Improves Pruning of Experts in LLMs},
  author={Jha, Saurav and Hashemzadeh, Maryam and Pasand, Ali Saheb and Parviz, Ali and Lee, Min-Joong and Knyazev, Boris},
  journal={arXiv preprint arXiv:2604.04356},
  year={2026}
}
```