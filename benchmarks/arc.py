import json
import random
import string
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
from utils import *


import random

# def craft_arc(chunk_size, processor, arc_c_path, arc_e_path, path_final, count=None, seed=None, force=False):
#     datasets = ['ARC-Challenge', 'ARC-Easy']
#     file_names = [arc_c_path, arc_e_path]

#     for dataset, file_name in zip(datasets, file_names):
#         ds = load_dataset('ai2_arc', dataset)
#         ds = ds['validation']
#         data = [doc for doc in ds]
#         if seed is not None:
#             random.seed(seed)
#         if count is not None:
#             if force:
#                 answers = set(doc['answerKey'] for doc in data)
#                 data_by_answer = {answer: [doc for doc in data if doc['answerKey'] == answer] for answer in answers}
#                 samples_per_answer = count // len(answers)
#                 sampled_data = [random.sample(docs, min(samples_per_answer, len(docs))) for docs in data_by_answer.values()]
#                 sampled_data = [doc for docs in sampled_data for doc in docs]
#                 remaining_samples = count - len(sampled_data)
#                 if remaining_samples > 0:
#                     remaining_data = [doc for doc in data if doc not in sampled_data]
#                     sampled_data.extend(random.sample(remaining_data, remaining_samples))
#             else:
#                 sampled_data = random.sample(data, min(count, len(data)))
#         else:
#             sampled_data = data

#         lines = []
#         for doc in sampled_data:
#             num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
#             doc["answerKey"] = num_to_letter.get(doc["answerKey"], doc["answerKey"])
#             out_doc = {
#                 "input": "Question: " + doc["question"]
#                 + '\nChoices:\n' + '\n'.join([l + ': ' + choice for l, choice in zip(['A', 'B', 'C', 'D', 'E'], doc['choices']['text'])])
#                 + "\nAnswer:",
#                 "output": doc["answerKey"],
#                 "subject": dataset
#             }
#             lines.append(out_doc)

#         processor.write_json_data(file_name, lines)
#         processor.append_json_data(path_final, lines)
# def craft_arc(chunk_size, processor, arc_c_path, arc_e_path, path_final, count=None, seed=None, force=False):
#     datasets = ['ARC-Challenge', 'ARC-Easy']
#     file_names = [arc_c_path, arc_e_path]

#     for dataset, file_name in zip(datasets, file_names):
#         ds = load_dataset('ai2_arc', dataset)

#         # ds = concatenate_datasets([ds['train'], ds['test'], ds['validation']])
#         ds = ds['validation']

#         lines = []
#         for doc in ds:
#             num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
#             doc["answerKey"] = num_to_letter.get(doc["answerKey"], doc["answerKey"])
#             out_doc = {
#                 "input": "Question: " + doc["question"]
#                 + '\nChoices:\n' + '\n'.join([l + ': ' + choice for l, choice in zip(['A', 'B', 'C', 'D', 'E'], doc['choices']['text'])])
#                 + "\nAnswer:",
#                 "output": doc["answerKey"],
#                 "subject": dataset
#             }
#             lines.append(out_doc)

#         lines = lines[:chunk_size]

#         processor.write_json_data(file_name, lines)
#         processor.append_json_data(path_final, lines)




import random

def craft_arc_c(processor, arc_c_path, path_final, count=None, seed=None, force=False):
    dataset = 'ARC-Challenge'
    craft_arc(processor, arc_c_path, path_final, dataset, count, seed, force)

def craft_arc_e(processor, arc_e_path, path_final, count=None, seed=None, force=False):
    dataset = 'ARC-Easy'
    craft_arc(processor, arc_e_path, path_final, dataset, count, seed, force)

def craft_arc(processor, file_name, path_final, dataset, count=None, seed=None, force=False):
    ds = load_dataset('ai2_arc', dataset)
    ds = ds['validation']

    # Convert the dataset into a list
    data = [doc for doc in ds]

    # Set the seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # If count is not None, sample a specific number of items from the data
    if count is not None:
        if force:
            # Get the unique answers
            answers = set(doc['answerKey'] for doc in data)

            # Group the data by answer
            data_by_answer = {answer: [doc for doc in data if doc['answerKey'] == answer] for answer in answers}

            # Calculate the number of samples for each answer
            samples_per_answer = count // len(answers)

            # Sample the data
            sampled_data = [random.sample(docs, min(samples_per_answer, len(docs))) for docs in data_by_answer.values()]

            # Flatten the list of lists
            sampled_data = [doc for docs in sampled_data for doc in docs]

            # If there are still remaining samples, sample randomly from all data
            remaining_samples = count - len(sampled_data)
            if remaining_samples > 0:
                remaining_data = [doc for doc in data if doc not in sampled_data]
                sampled_data.extend(random.sample(remaining_data, remaining_samples))
        else:
            # Sample a specific number of items from the data
            sampled_data = random.sample(data, min(count, len(data)))
    else:
        sampled_data = data

    lines = []
    for doc in sampled_data:
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

    processor.write_json_data(file_name, lines)
    processor.append_json_data(path_final, lines)