import json
import random
import string
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
from dharma.utils import *

def craft_truthfulqa(processor, tqa_path, path_final, count=None, seed=None, force=False):
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

    processor.write_json_data(tqa_path, lines)
    processor.append_json_data(path_final, lines)