
## dharma: build your own tiny benchmark datasets

<p align="center">
🤗 <a href="https://huggingface.co/pharaouk" target="_blank">HF Repo</a> • 🐦 <a href="https://twitter.com/far__el" target="_blank">Twitter</a> <br>
</p>

<img src='img/image.png' width=1000 >



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