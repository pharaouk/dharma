import json
import random
import string
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
from dharma.utils import *


def craft_agieval(processor, agieval_path, path_final, count=None, seed=None, force=False):
    datasets = [
        'agieval-sat-math',
        'agieval-sat-en-without-passage',
        'agieval-sat-en',
        'agieval-logiqa-en',
        'agieval-lsat-rc',
        'agieval-lsat-lr',
        'agieval-lsat-ar',
        'agieval-aqua-rat'
    ]

    lines = []

    for dataset in datasets:
        ds = load_dataset('dmayhem93/' + dataset)
        ds = concatenate_datasets([ ds['test']])

        for doc in ds:
            num_to_letter = {idx: letter for idx, letter in enumerate(string.ascii_uppercase[:len(doc['choices'])])}
            doc["answerKey"] = num_to_letter.get(doc["gold"][0], doc["gold"][0])
            out_doc = {
                "input": doc["query"] + '\n' + "\Choices:" +'\n'.join(doc['choices'])  + "\nAnswer:",
                "output": doc["answerKey"],
                "subject": 'agieval' 
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

    processor.write_json_data(agieval_path, lines)
    processor.append_json_data(path_final, lines)