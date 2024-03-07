import json
import random
import string
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
from utils import *


def craft_agieval(chunk_size, processor, agieval_path, path_final, count=None, seed=None, force=False):

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

    samples_per_dataset = 50

    for dataset in datasets:
        ds = load_dataset('dmayhem93/' + dataset)

        ds = concatenate_datasets([ ds['test']])
        
        lines = []
        for doc in ds:
            num_to_letter = {idx: letter for idx, letter in enumerate(string.ascii_uppercase[:len(doc['choices'])])}
            doc["answerKey"] = num_to_letter.get(doc["gold"][0], doc["gold"][0])
            out_doc = {
                "input": doc["query"] + '\n' + "\Choices:" +'\n'.join(doc['choices'])  + "\nAnswer:",
                "output": doc["answerKey"],
                "subject": 'agieval' 
            }
            lines.append(out_doc)

        lines = lines[:samples_per_dataset]

        processor.write_json_data(agieval_path, lines)
        processor.append_json_data(path_final, lines)



