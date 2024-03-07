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


def runner(output, config):
    path_final = f'{output}/final_eval.json'


    #MMLU
    mmlu_path = f'{output}/benchmarks/mmlu.json'

    craft_mmlu(processor, output, mmlu_path, path_final)
    check(output)
    mmlu_data = processor.load_json_data(mmlu_path)
    chunk_size = len(mmlu_data)

    count_answer_options(mmlu_path)


    #ARC-C and ARC-E
    arc_c_path = f'{output}/benchmarks/arc_c.json'
    arc_e_path = f'{output}/benchmarks/arc_e.json'

    craft_arc(chunk_size, processor, arc_c_path, arc_e_path, path_final)
    check(output)
    count_answer_options(arc_c_path)
    count_answer_options(arc_e_path)

    # #BIGBENCH
    bigbench_path = f'{output}/benchmarks/bigbench.json'

    craft_bigbench(chunk_size, processor, bigbench_path, path_final)
    check(output)

    #BOOLQ
    boolq_path = f'{output}/benchmarks/boolq.json'

    craft_boolq(chunk_size, processor, boolq_path, path_final)
    check(output)
    count_answer_options(boolq_path)


    #WINOGRANDE
    wino_path = f'{output}/benchmarks/winogrande.json'

    craft_winogrande(chunk_size, processor, wino_path, path_final)
    check(output)
    count_answer_options(wino_path)


    #OBQA
    openbookqa_path = f'{output}/benchmarks/openbookqa.json'

    craft_obqa(chunk_size, processor, openbookqa_path, path_final)
    check(output)
    count_answer_options(openbookqa_path)


    #TRUTHFULQA
    truthful_qa_path = f'{output}/benchmarks/truthful_qa.json'

    craft_truthfulqa(chunk_size, processor, truthful_qa_path, path_final)
    check(output)
    count_answer_options(truthful_qa_path)


    #AGIEVAL
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

