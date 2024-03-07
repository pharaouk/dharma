import json
import random
import argparse
import yaml
import string
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
from utils import *
from benchmarks.mmlu import *
from benchmarks.agieval import *
from benchmarks.boolq import *
from benchmarks.obqa import *
from benchmarks.winogrande import *
from benchmarks.truthfulqa import *
from benchmarks.bigbench import *
from benchmarks.arc import *


processor = DataProcessor()

def compute_dataset_distribution(config):
    active_datasets = {k: v['count'] for k, v in config.benchmarks.items() if v['count'] > 0}
    total_count = sum(active_datasets.values())
    dataset_samples = {k: (v / total_count) * config.dataset_size for k, v in active_datasets.items()}
    dataset_samples = {k: round(v) for k, v in dataset_samples.items()}
    return dataset_samples

def is_dataset_active(dataset_samples, dataset_name):
    return dataset_samples.get(dataset_name, 0) > 0
def get_dataset_count(dataset_samples, dataset_name):
    return dataset_samples.get(dataset_name, 0)

def runner(output, config):
    path_final = f'{output}/final_eval.json'
    make_dir(path_final)
    data_seed = config.data_seed
    force_dist = config.force_dist

    dataset_samples = compute_dataset_distribution(config)
    print(f"DISTRIBUTIONS - {dataset_samples}")

    #MMLU
    if is_dataset_active(dataset_samples, 'mmlu'):
        mmlu_count = get_dataset_count(dataset_samples, 'mmlu')
        mmlu_path = f'{output}/benchmarks/mmlu.json'
        craft_mmlu(processor, output, mmlu_path, path_final, mmlu_count, data_seed, force_dist)
        check(output)
        mmlu_data = processor.load_json_data(mmlu_path)
        chunk_size = len(mmlu_data)
        count_answer_options(mmlu_path)

    #ARC-C and ARC-E
    if is_dataset_active(dataset_samples, 'arc_c'):
        arc_c_count = get_dataset_count(dataset_samples, 'arc_c')
        arc_c_path = f'{output}/benchmarks/arc_c.json'
        craft_arc_c(processor, arc_c_path, path_final, arc_c_count, data_seed, force_dist)
        check(output)
        count_answer_options(arc_c_path)

    #ARC-C and ARC-E
    if is_dataset_active(dataset_samples, 'arc_e'):
        arc_e_count = get_dataset_count(dataset_samples, 'arc_e')
        arc_e_path = f'{output}/benchmarks/arc_e.json'
        craft_arc_e(processor, arc_e_path, path_final, arc_e_count, data_seed, force_dist)
        check(output)
        count_answer_options(arc_e_path)



    # #BIGBENCH
    if is_dataset_active(dataset_samples, 'bigbench'):
        bigbench_count = get_dataset_count(dataset_samples, 'bigbench')
        bigbench_path = f'{output}/benchmarks/bigbench.json'
        craft_bigbench(chunk_size, processor, bigbench_path, path_final)
        check(output)

    #BOOLQ
    if is_dataset_active(dataset_samples, 'boolq'):
        boolq_count = get_dataset_count(dataset_samples, 'boolq')

        boolq_path = f'{output}/benchmarks/boolq.json'
        craft_boolq(chunk_size, processor, boolq_path, path_final)
        check(output)
        count_answer_options(boolq_path)

    #WINOGRANDE
    if is_dataset_active(dataset_samples, 'winogrande'):
        winogrande_count = get_dataset_count(dataset_samples, 'winogrande')
        wino_path = f'{output}/benchmarks/winogrande.json'
        craft_winogrande(chunk_size, processor, wino_path, path_final)
        check(output)
        count_answer_options(wino_path)

    #OBQA
    if is_dataset_active(dataset_samples, 'obqa'):
        obqa_count = get_dataset_count(dataset_samples, 'obqa')
        openbookqa_path = f'{output}/benchmarks/openbookqa.json'
        craft_obqa(chunk_size, processor, openbookqa_path, path_final)
        check(output)
        count_answer_options(openbookqa_path)

    #TRUTHFULQA
    if is_dataset_active(dataset_samples, 'truthfulqa'):
        truthfulqa_count = get_dataset_count(dataset_samples, 'truthfulqa')
        truthful_qa_path = f'{output}/benchmarks/truthful_qa.json'
        craft_truthfulqa(chunk_size, processor, truthful_qa_path, path_final)
        check(output)
        count_answer_options(truthful_qa_path)

    #AGIEVAL
    if is_dataset_active(dataset_samples, 'agieval'):
        agieval_count = get_dataset_count(dataset_samples, 'agieval')
        agieval_path = f'{output}/benchmarks/agieval.json'
        craft_agieval(chunk_size, processor, agieval_path, path_final)
        check(output)
        count_answer_options(agieval_path)


    filter_data(output)
    #Make datasets
    make_unshuffled(output)
    make_shuffled(output)
    make_datasets(output)

    #upload to huggingface Hub with each dataset as a config
    





def main():
    parser = argparse.ArgumentParser(description="dharma")
    parser.add_argument("--config", type=str, required=False, help="config path")
    args = parser.parse_args()
    if args.config:
        config_path = args.config
    else:
        config_path = "config.yml"

    config = load_config(config_path)
    if config.output:
        output = config.output
    else:
        random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
        output = f"dharma_{random_suffix}"

    runner(output, config)




if __name__ == "__main__":
    main()

