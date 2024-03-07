import json
import random
import string
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
from utils import *



def craft_arc(chunk_size, processor, arc_c_path, arc_e_path, path_final):
    datasets = ['ARC-Challenge', 'ARC-Easy']
    file_names = [arc_c_path, arc_e_path]

    for dataset, file_name in zip(datasets, file_names):
        ds = load_dataset('ai2_arc', dataset)

        # ds = concatenate_datasets([ds['train'], ds['test'], ds['validation']])
        ds = ds['validation']

        lines = []
        for doc in ds:
            num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
            doc["answerKey"] = num_to_letter.get(doc["answerKey"], doc["answerKey"])
            out_doc = {
                "input": "Question: " + doc["question"]
                + '\nChoices:\n' + '\n'.join([l + ': ' + choice for l, choice in zip(['A', 'B', 'C', 'D', 'E'], doc['choices']['text'])])
                + "\nAnswer:",
                "output": doc["answerKey"],
                "subject": dataset
            }
            lines.append(out_doc)

        lines = lines[:chunk_size]

        processor.write_json_data(file_name, lines)
        processor.append_json_data(path_final, lines)

