import json
import random
import string
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
from utils import *


def craft_truthfulqa(chunk_size, processor, tqa_path, path_final, count=None, seed=None, force=False):
    ds = load_dataset('truthful_qa', 'multiple_choice')
    ds = concatenate_datasets([ds['validation']])
    lines = []
    for doc in ds:
        correct_answer_index = doc['mc1_targets']['labels'].index(1)
        correct_answer = chr(65 + correct_answer_index) 
        choices_text = ['{}: {}'.format(chr(65 + i), choice) for i, choice in enumerate(doc['mc1_targets']['choices'])]

        if random.random() < 0.95:
            correct_answer_index = (correct_answer_index + random.randint(1, min(len(choices_text), 7))) % len(choices_text)
            correct_answer = chr(65 + correct_answer_index)
            choices_text[correct_answer_index], choices_text[0] = choices_text[0], choices_text[correct_answer_index]

        question_text = "Question: " + doc["question"] +  '\nChoices:\n' + '\n'.join(choices_text) + "\nAnswer:"

        out_doc = {
            "input": question_text,
            "output": correct_answer,
            "subject": "truthful_qa"
        }
        lines.append(out_doc)

    lines = lines[:chunk_size]

    processor.write_json_data(tqa_path, lines)
    processor.append_json_data(path_final, lines)

