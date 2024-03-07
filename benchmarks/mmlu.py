import json
import random
import string
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
from utils import *



def craft_mmlu(processor, output_path, mmlu_path, path_final):

    data = processor.load_json_data('seed_mmlu.json')

    total_samples = len(data)
    percentage = 0.8
    subject_counts = {}

    for row in data:
        subject = row['subject']
        if subject not in subject_counts:
            subject_counts[subject] = 0
        subject_counts[subject] += 1

    subject_counts_to_delete = {}
    for subject, count in subject_counts.items():
        subject_counts_to_delete[subject] = int(count * percentage)

    data_after_deletion = []
    for row in data:
        subject = row['subject']
        if subject_counts_to_delete[subject] > 0:
            subject_counts_to_delete[subject] -= 1
        else:
            row['subject'] = 'MMLU' 
            data_after_deletion.append(row)
    make_dir(path_final)
    make_dir(mmlu_path)
    processor.write_json_data(path_final, data_after_deletion)
    processor.write_json_data(mmlu_path, data_after_deletion)

