import json
import random
import string
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
from utils import *

def craft_boolq(chunk_size, processor, boolq_path, path_final, count=None, seed=None, force=False):

    ds = load_dataset('boolq')
    # ds = concatenate_datasets([ds['train'], ds['validation']])
    ds = ds['validation']
    # Shuffle the dataset
    ds = ds.shuffle()
    lines = []
    for doc in ds:
        answer_key = 'A' if doc["answer"] else 'B'
        out_doc = {
            "input": "Passage: " + doc["passage"] + "\nQuestion: " + doc["question"] + "\nChoices:\nA: True\nB: False" + "\nAnswer:",
            "output": answer_key,
            "subject": 'BoolQ'  
        }
        lines.append(out_doc)

    lines = lines[:chunk_size]

    processor.write_json_data(boolq_path, lines)
    processor.append_json_data(path_final, lines)

