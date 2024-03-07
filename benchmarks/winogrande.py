import json
import random
import string
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
from utils import *

def craft_winogrande(processor, wino_path, path_final, count=None, seed=None, force=False):
    ds = load_dataset('winogrande', 'winogrande_debiased')
    ds = ds['validation']

    lines = []
    for doc in ds:
        options = {"1": doc["option1"], "2": doc["option2"]}
        doc["answer"] = "A" if doc["answer"] == "1" else "B"
        out_doc = {
            "input": doc["sentence"]
            + '\nChoices:\n' + '\n'.join([l + ': ' + options[l] for l in ['1', '2']])
            + "\nAnswer:",
            "output": doc["answer"],
            "subject": "winogrande"  
        }
        lines.append(out_doc)

    if force:
        answers = set(row['output'] for row in lines)
        data_by_answer = {answer: [row for row in lines if row['output'] == answer] for answer in answers}
        samples_per_answer = count // len(answers)
        sampled_data = [random.sample(rows, min(samples_per_answer, len(rows))) for rows in data_by_answer.values()]
        sampled_data = [row for rows in sampled_data for row in rows]
        remaining_samples = count - len(sampled_data)
        if remaining_samples > 0:
            remaining_data = [row for row in lines if row not in sampled_data]
            sampled_data.extend(random.sample(remaining_data, remaining_samples))
        lines = sampled_data

    else:
        lines = lines[:count]

    processor.write_json_data(wino_path, lines)
    processor.append_json_data(path_final, lines)