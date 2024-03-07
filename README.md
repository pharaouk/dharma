
## dharma: build your own tiny benchmark datasets

<p align="center">
🤗 <a href="https://huggingface.co/pharaouk" target="_blank">HF Repo</a> • 🐦 <a href="https://twitter.com/far__el" target="_blank">Twitter</a> <br>
</p>

<img src='img/image.png' width=1000 >

use dharma to craft small or large benchmarking datasets that can be used during training or for fast evals. 
these serve as good indicators on the benchmarks you care about. make sure to craft a benchmark dataset appropriate for your use cases. 
more benchmarks and features are in the works to give you even more control over your bench datasets.
dharma's core value is the idea of 'eval through time' during a training run. it sheds light to on your model's performance as it processes and is optimized on your training data. this can be useful to train more powerful models that do exactly what you intend them to. 
of course, MCQ based benches do not inform us much on performance beyond this format, therefore dharma will expand to include non MCQ based benches as well. stay tuned.


## Quickstart
Setup:
```
pip install -r requirements.txt
```

Configs:
```
output: #(string) destination output path + dataset name, leave blank to use default

hf_namespace: #(string)  hf username/namespace

hf_upload: false  #(bool) hf username/namespace

prompt_format: "Question: {questions}. {options} Answer:"  #(string) prompt format to use for the eval datasets

dataset_size: 2000  #(int) total target dataset size

data_seed: 42  #(int) dataset seed

force_dist: true  #(bool) force even distribution for answers (i.e. A-25 B-25 C-25 D-25)

benchmarks: #this determines which benchmarks and counts/distirbutions for the target dataset. enter 0 if you don't want that dataset included.

  mmlu: 
    count: 1
  arc_c:
    count: 1
  arc_e:
    count: 1
  agieval:
    count: 1
  boolq:
    count: 1
  obqa:
    count: 1
  truthfulqa:
    count: 1
  winogrande:
    count: 1

```

Run:
```
python dharma.py
```
or
```
python dharma.py --config <CONFIG_PATH>
```



**How is Dharma used?**
Example dharma-1 dataset: https://huggingface.co/datasets/pharaouk/dharma-1
Example axolotl implementation: [https://github.com/OpenAccess-AI-Collective/axolotl/blob/638c2dafb54f1c7c61a5f7ad40f8cf6965bec896/src/axolotl/core/trainer_builder.py#L152](https://github.com/OpenAccess-AI-Collective/axolotl/blob/638c2dafb54f1c7c61a5f7ad40f8cf6965bec896/src/axolotl/utils/callbacks/__init__.py#L253)

Example wandb:
<img width="1290" alt="Wandb" src="https://github.com/pharaouk/dharma/assets/36641995/cd9cb5d4-c3d7-444b-b83a-1c70c3c83183">



TODOS


0. bigbench compatibility. **[in progress]** (currently not optimal)
1. Custom prompt formats (to replace standard one we've set)
2. Add upload to HF option with template MD as readme, and to custom namespace (check for HF token) 
3. standardize dataset cleaning funcs (add sim search and subject based segmentation)
4. Add a testing/eval script with local llm w local lb
5. Add a Callback example for training
6. Add axolotl example
7. Upload cleaned and corrected copies of all benchmrk datasets to HF
8. Fix uneven distributions
9. CLIx updates (tqdm + cleanup)
10. pip package
11. New benchmarks, non MCQ
12. HF Compatible Custom Callback library with customization options
13. better selection algo for the benchmarks
14. Randomize answers options  (could be useful to evaluate/minimize bias in model)
