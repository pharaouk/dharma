
## dharma: build your own tiny benchmark datasets

<p align="center">
🤗 <a href="https://huggingface.co/pharaouk" target="_blank">HF Repo</a> • 🐦 <a href="https://twitter.com/far__el" target="_blank">Twitter</a> <br>
</p>

<img src='img/image.png' width=1000 height=200 >



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
    count: 100
  arc_c:
    count: 100
  arc_e:
    count: 100
  agieval:
    count: 100
  boolq:
    count: 100
  obqa:
    count: 100
  bigbench:
    count: 100
  truthfulqa:
    count: 100
  winogrande:
    count: 100

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
1. Connect count, seed and force_dist to all benchmarks
2. Custom prompt formats 
3. Add upload to HF with template MD
4. 