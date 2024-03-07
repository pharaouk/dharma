import json
import random
import string
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
from utils import *


         

def craft_obqa(chunk_size, processor, obqa_path, path_final, count=None, seed=None, force=False):
    ds = load_dataset('openbookqa')
    ds = concatenate_datasets([ds['train'], ds['test'], ds['validation']])
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

    lines = lines[:chunk_size]

    processor.write_json_data(obqa_path, lines)
    processor.append_json_data(path_final, lines)
