import json
import random
import string
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
from utils import *



def craft_bigbench(chunk_size, processor, bigbench_path, path_final, count=None, seed=None, force=False):
    config_names = get_dataset_config_names("tasksource/bigbench")
    min_limit = 2  
    max_limit = 50 
    bigbench_total = 0
    subset_sizes = {}
    non_english_languages = ['russian', 'french', 'italian', 'spanish', 'language_identification', 'german', 'chinese', 'mandarin', 'hindi', 'arabic', 'hindu', 'hinglish_toxicity']
    non_english_configs = [config for config in config_names if any(language in config for language in non_english_languages)]
    config_names = [config for config in config_names if config not in non_english_configs]
    for config in config_names:
        try:
            ds = load_dataset('tasksource/bigbench', config)
            ds_concat = concatenate_datasets([ds[split] for split in ['validation']])
            subset_sizes[config] = len(ds_concat)
            bigbench_total += subset_sizes[config]
        except Exception as e:
            print(f"Skipping {config} due to DatasetGenerationError")
            continue

        ds_concat = concatenate_datasets([ds[split] for split in ['validation']])

        proportion = subset_sizes[config] / bigbench_total
        rows_to_take = int(chunk_size * proportion)

        final_size = max(min(rows_to_take, max_limit), min_limit)

        ds_concat = ds_concat.shuffle() 
        lines = []
        while len(lines) < final_size:
            for idx, doc in enumerate(ds_concat):
                if idx >= final_size:
                    break
                choice_letters = string.ascii_uppercase[:len(doc['multiple_choice_targets'])]
                choice_dict = {letter: choice for letter, choice in zip(choice_letters, doc['multiple_choice_targets'])}  # Replace the target with the appropriate letter

                if not doc['multiple_choice_targets'] or len(doc['multiple_choice_targets']) == 0  or len(doc['multiple_choice_targets']) >= 10:
                        final_size = 0
                        break

                try:
                    correct_index = doc['multiple_choice_scores'].index(1)
                    output1 = list(choice_dict.keys())[correct_index]
                    output_candidates = [k for k, v in choice_dict.items() if v == doc['targets'][0]]
                    if not output_candidates:
                        print("Error: No matching output found")
                    output2 = output_candidates[0]
                except Exception as e:
                    print('error')
                    output1 = None
                if output1 == output2:
                    output = output1  
                elif output1 is None or output1 != output2:
                    output = output2
                out_doc = {
                    "input": "Question: " + doc["inputs"]
                    + '\nChoices:\n' + '\n'.join([l + ': ' + choice for l, choice in choice_dict.items()])
                    + "\nAnswer:",
                    "output": output,
                    "subject": 'bigbench'  
                }
                lines.append(out_doc)
                if len(lines) >= final_size:
                    break

        processor.write_json_data(bigbench_path, lines)
        processor.append_json_data(path_final, lines)





import random
import string
from datasets import load_dataset, concatenate_datasets

def craft_bigbench(processor, bigbench_path, path_final, count=None, seed=None, force=False):
    config_names = get_dataset_config_names("tasksource/bigbench")
    bigbench_total = 0
    subset_sizes = {}
    non_english_languages = ['russian', 'french', 'italian', 'spanish', 'language_identification', 'german', 'chinese', 'mandarin', 'hindi', 'arabic', 'hindu', 'hinglish_toxicity']
    non_english_configs = [config for config in config_names if any(language in config for language in non_english_languages)]
    config_names = [config for config in config_names if config not in non_english_configs]

    lines = []

    for config in config_names:
        try:
            ds = load_dataset('tasksource/bigbench', config)
            ds_concat = concatenate_datasets([ds[split] for split in ['validation']])
            subset_sizes[config] = len(ds_concat)
            bigbench_total += subset_sizes[config]
        except Exception as e:
            print(f"Skipping {config} due to DatasetGenerationError")
            continue

        if force:
            # Calculate the number of samples for this config
            samples_for_config = count // len(config_names)

            # Sample the data
            sampled_data = random.sample(list(ds_concat), min(samples_for_config, len(ds_concat)))

            # If there are still remaining samples, sample randomly from all data
            remaining_samples = samples_for_config - len(sampled_data)
            if remaining_samples > 0:
                remaining_data = [row for row in ds_concat if row not in sampled_data]
                sampled_data.extend(random.sample(remaining_data, remaining_samples))

            data_to_process = sampled_data
        else:
            data_to_process = ds_concat

        # Process the data
        for idx, doc in enumerate(data_to_process):
            # if idx >= count:
            #     break
            choice_letters = string.ascii_uppercase[:len(doc['multiple_choice_targets'])]
            choice_dict = {letter: choice for letter, choice in zip(choice_letters, doc['multiple_choice_targets'])}  # Replace the target with the appropriate letter

            if not doc['multiple_choice_targets'] or len(doc['multiple_choice_targets']) == 0  or len(doc['multiple_choice_targets']) >= 10:
                    # count = 0
                    break

            try:
                correct_index = doc['multiple_choice_scores'].index(1)
                output1 = list(choice_dict.keys())[correct_index]
                output_candidates = [k for k, v in choice_dict.items() if v == doc['targets'][0]]
                if not output_candidates:
                    print("Error: No matching output found")
                output2 = output_candidates[0]
            except Exception as e:
                print('error')
                output1 = None
            if output1 == output2:
                output = output1  
            elif output1 is None or output1 != output2:
                output = output2
            out_doc = {
                "input": "Question: " + doc["inputs"]
                + '\nChoices:\n' + '\n'.join([l + ': ' + choice for l, choice in choice_dict.items()])
                + "\nAnswer:",
                "output": output,
                "subject": 'bigbench'  
            }
            print(out_doc)
            lines.append(out_doc)
            if len(lines) >= count:
                break

    processor.write_json_data(bigbench_path, lines)
    processor.append_json_data(path_final, lines)