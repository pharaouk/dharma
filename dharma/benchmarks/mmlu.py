import json
import random
import string
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
from dharma.utils import *
import random
import numpy as np

def craft_mmlu(processor, output_path, mmlu_path, path_final, count, seed=None, force=False):
    data = processor.load_json_data('dharma/benchmarks/seed_mmlu.jsonl')
    if seed is not None:
        random.seed(seed)
    if force:
        answers = set(row['output'] for row in data)
        data_by_answer = {answer: [row for row in data if row['output'] == answer] for answer in answers}
        samples_per_answer = count // len(answers)
        sampled_data = [random.sample(rows, min(samples_per_answer, len(rows))) for rows in data_by_answer.values()]
        sampled_data = [row for rows in sampled_data for row in rows]
        remaining_samples = count - len(sampled_data)
        if remaining_samples > 0:
            remaining_data = [row for row in data if row not in sampled_data]
            sampled_data.extend(random.sample(remaining_data, remaining_samples))
    else:
        sampled_data = random.sample(data, count)
    
    for row in sampled_data:
        row['subject'] = 'MMLU'
    make_dir(path_final)
    make_dir(mmlu_path)
    processor.write_json_data(path_final, sampled_data)
    processor.write_json_data(mmlu_path, sampled_data)