import json
import random
import string
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
from utils import *


def craft_winogrande(chunk_size, processor, wino_path, path_final, count=None, seed=None, force=False):
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

    lines = lines[:chunk_size]

    processor.write_json_data(wino_path, lines)
    processor.append_json_data(path_final, lines)
