---
configs:
- config_name: default
  data_files:
  - split: 'dharma_1_full'
    path: dharma_1_full*
  - split: 'dharma_1_mini'
    path: dharma_1_mini*
  - split: 'dharma_1_micro'
    path: dharma_1_micro*
  - split: 'dharma_1_unshuffled'
    path: dharma_eval_unshuffled*
---
# "Dharma Dataset"

A new carefully curated benchmark set, designed for a new era where the true end user uses LLM's for zero-shot and one-shot tasks, for a vast majority of the time.
Stop training your models on mindless targets (eval_loss, train_loss), start training your LLM on lightweight Dharma as an eval target. 
A mix of all the top benchmarks.

Formed to have an equal distribution of some of the most trusted benchmarks used by those developing SOTA LLMs, comprised of only 3,000 examples for the largest size, as well as 450 and 90 for Dharma-mini and Dharma-micro respectively.

The current version of Dharma is comprised of a curated sampling of the following benchmarks:

 - AGIEval 
 - Bigbench 
 - MMLU  
 - Winogrande 
 - Arc-C 
 - Arc- E 
 - OBQA 
 - TruthfulQA 
 - Bool-q 

Each of these original benchmark datasets have their own subsections, careful work has gone into also choosing an equal distribution of the important subsections of each these, to have the best representation of the original benchmark creators goals.

Dharma-1 is now integrated into Axolotl as well!, so you can focus on optimizing the other aspects of your training pipeline, model architecture and/or dataset, as opposed to worrying about what is the best evaluation measurement or optimization target that will best represent capabilities for the end user.

Benchmarking for top base model will be listed here when completed and verified.

Special thanks to @LDJnr for their contributions. Check out their Puffin dataset here: https://huggingface.co/LDJnr


[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)