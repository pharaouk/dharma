import json
import string
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
from utils import *
import random
from collections import defaultdict


def craft_obqa(processor, obqa_path, path_final, count=None, seed=None, force=False):
    ds = load_dataset('openbookqa')
    ds = concatenate_datasets([ds['test'], ds['validation']])
    ds = ds.shuffle()
    lines = []
    for doc in ds:
        out_doc = {
            "input": "Question: " + doc["question_stem"] + "\nAnswer:"
            + '\nChoices:\n' + '\n'.join([l + ': ' + choice for l, choice in zip(doc['choices']['label'], doc['choices']['text'])]),
            "output": doc["answerKey"],
            "subject": "openbookqa" 
        }
        lines.append(out_doc)

    if force:
        answers = set(line['output'] for line in lines)
        lines_by_answer = {answer: [line for line in lines if line['output'] == answer] for answer in answers}
        samples_per_answer = count // len(answers)
        sampled_lines = [random.sample(lines, min(samples_per_answer, len(lines))) for lines in lines_by_answer.values()]
        sampled_lines = [line for lines in sampled_lines for line in lines]
        remaining_samples = count - len(sampled_lines)
        if remaining_samples > 0:
            remaining_lines = [line for line in lines if line not in sampled_lines]
            sampled_lines.extend(random.sample(remaining_lines, remaining_samples))
    else:
        sampled_lines = lines[:count]

    processor.write_json_data(obqa_path, sampled_lines)
    processor.append_json_data(path_final, sampled_lines)