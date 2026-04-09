# REAM: Merging Improves Pruning of Experts in LLMs

Authors: [Saurav Jha](https://sauravjha.com.np/)\*, Maryam Hashemzadeh, Ali Saheb Pasand, Ali Parviz, Min-Joong Lee, [Boris Knyazev](https://bknyaz.github.io)\*

*equal contribution

[![arXiv](https://img.shields.io/badge/arXiv-2403.12143-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2604.04356) 

[blogpost](https://bknyaz.github.io/blog/2026/moe/)

REAM-ed models 🤗: 
- Qwen3:
  - [x] https://huggingface.co/collections/SamsungSAILMontreal/ream
  - [x] https://huggingface.co/bknyaz
- GLM:
- [ ] GLM-4.5-Air coming soon

[Baseline REAP](https://github.com/CerebrasResearch/reap)

## Merging

Run `python merge.py --model <> --merge_size <> --save_path <>` to merge (or prune) the experts.
See merge.py and config.py for the hyperparameters and options.
Default arguments correspond to our full REAM model.

MTP layers are supported, but checked only for qwen3 and requires additional steps:
- setting the `--mtp_safe_tensors` file path
- renaming the saved model safe tensors of the MTP and other layers and corresponding model.safetensors.index.json
- See https://huggingface.co/bknyaz/Qwen3-Next-80B-A3B-Instruct-REAM for our example of the resulted model.


## Evaluation

The original and compressed (merged/pruned) models are evaluated on MC and GEN tasks.

### MC tasks

Run `python eval_mc.py` to evaluate the merged model on multiple-choice tasks.
The main options are `--model` and `--batch_size`. 
See eval_mc.py and config.py for more options.

### GEN tasks

We use the following tools as described on our huggingface page and in the paper:

- lmeval https://github.com/EleutherAI/lm-evaluation-harness
- LiveCodeBench https://github.com/LiveCodeBench/LiveCodeBench
- GLM (for humaneval and lcb): https://github.com/zai-org/glm-simple-evals


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