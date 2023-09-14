import json
import random
import string
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names

def craft_mmlu():
    with open('seed_mmlu.json', 'r') as f:
        data = []
        for line in f:
            data.append(json.loads(line))

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

    with open('final_eval.json', 'w') as f:
        for item in data_after_deletion:
            f.write(json.dumps(item) + '\n')

    with open('mmlu.json', 'w') as f:
        for item in data_after_deletion:
            f.write(json.dumps(item) + '\n')


with open('mmlu.json', 'r') as f:
    mmlu_data = [json.loads(line) for line in f]
mmlu_size = len(mmlu_data)


def craft_arc():
    datasets = ['ARC-Challenge', 'ARC-Easy']
    file_names = ['arc_c.json', 'arc_e.json']

    for dataset, file_name in zip(datasets, file_names):
        ds = load_dataset('ai2_arc', dataset)

        ds = concatenate_datasets([ds['train'], ds['test'], ds['validation']])

        lines = []
        for doc in ds:
            num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
            doc["answerKey"] = num_to_letter.get(doc["answerKey"], doc["answerKey"])
            out_doc = {
                "input": "Question: " + doc["question"]
                + '\nChoices:\n' + '\n'.join([l + ': ' + choice for l, choice in zip(['A', 'B', 'C', 'D', 'E'], doc['choices']['text'])])
                + "\nAnswer:",
                "output": doc["answerKey"],
                "subject": dataset  # Add new "subject" key
            }
            lines.append(out_doc)

        lines = lines[:mmlu_size]

        with open(file_name, 'w') as f:
            for line in lines:
                f.write(json.dumps(line)  + '\n')

        with open('final_eval.json', 'a') as f:
            for line in lines:
                f.write(json.dumps(line)  + '\n')

def craft_bigbench():


    config_names = get_dataset_config_names("tasksource/bigbench")
    print(config_names)
    print(type(config_names))
    min_limit = 2  
    max_limit = 50 
    bigbench_total = 0
    subset_sizes = {}
    for config in config_names:
        #DO NOT DOWNLOAD IF russian, french, italian, german, chinese, mandarin, hindi is contained in the config name....

        try:
            ds = load_dataset('tasksource/bigbench', config)
            ds_concat = concatenate_datasets([ds[split] for split in ['train', 'validation']])
            subset_sizes[config] = len(ds_concat)
            bigbench_total += subset_sizes[config]
            print(bigbench_total)
        except Exception as e:
            print(f"Skipping {config} due to DatasetGenerationError")

        ds_concat = concatenate_datasets([ds[split] for split in ['train', 'validation']])

        proportion = subset_sizes[config] / bigbench_total
        rows_to_take = int(mmlu_size * proportion)

        final_size = max(min(rows_to_take, max_limit), min_limit)

        ds_concat = ds_concat.shuffle() 
        lines = []
        while len(lines) < final_size:
        print(final_size)
        print(len(lines))
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
            print(out_doc)
            if len(lines) >= final_size:
                break

        with open('final_eval.json', 'a') as f:
            for line in lines:
                f.write(json.dumps(line)  + '\n')



def craft_boolq():

    ds = load_dataset('boolq')
    ds = concatenate_datasets([ds['train'], ds['validation']])

    lines = []
    for doc in ds:
        # Convert bool answer to 'A' or 'B'
        answer_key = 'A' if doc["answer"] else 'B'
        out_doc = {
            "input": "Passage: " + doc["passage"] + "\nQuestion: " + doc["question"] + "\nChoices:\nA: True\nB: False" + "\nAnswer:",
            "output": answer_key,
            "subject": 'BoolQ'  # Add new "subject" key
        }
        lines.append(out_doc)

    lines = lines[:mmlu_size]

    with open('boolq.json', 'w') as f:
        for line in lines:
            f.write(json.dumps(line)  + '\n')

    with open('final_eval.json', 'a') as f:
        for line in lines:
            f.write(json.dumps(line)  + '\n')



def craft_winogrande()
    ds = load_dataset('winogrande', 'winogrande_debiased')

    ds = concatenate_datasets([ds['train'], ds['test'], ds['validation']])

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

    lines = lines[:mmlu_size]

    with open('winogrande.json', 'w') as f:
        for line in lines:
            f.write(json.dumps(line)  + '\n')

    with open('final_eval.json', 'a') as f:
        for line in lines:
            f.write(json.dumps(line)  + '\n')

def craft_obqa():
    ds = load_dataset('openbookqa')
    ds = concatenate_datasets([ds['train'], ds['test'], ds['validation']])
    lines = []
    for doc in ds:
        out_doc = {
            "input": "Question: " + doc["question_stem"] + "\nAnswer:"
            + '\nChoices:\n' + '\n'.join([l + ': ' + choice for l, choice in zip(doc['choices']['label'], doc['choices']['text'])]),
            "output": doc["answerKey"],
            "subject": "openbookqa" 
        }
        lines.append(out_doc)

    lines = lines[:mmlu_size]

    with open('openbookqa.json', 'w') as f:
        for line in lines:
            f.write(json.dumps(line)  + '\n')

    with open('final_eval.json', 'a') as f:
        for line in lines:
            f.write(json.dumps(line)  + '\n')


def craft_truthfulqa():


    ds = load_dataset('truthful_qa', 'multiple_choice')

    ds = concatenate_datasets([ds['validation']])

    lines = []
    for doc in ds:
        correct_answer_index = doc['mc1_targets']['labels'].index(1)
        correct_answer = chr(65 + correct_answer_index) 
        print(correct_answer_index)
        print(correct_answer)
        choices_text = '\n'.join([chr(65 + i) + ': ' + choice for i, choice in enumerate(doc['mc1_targets']['choices'])])
        question_text = "Question: " + doc["question"] +  '\nChoices:\n' + choices_text + "\nAnswer:"

        out_doc = {
            "input": question_text,
            "output": correct_answer,
            "subject": "truthful_qa"
        }
        lines.append(out_doc)

    lines = lines[:mmlu_size]

    with open('final_eval.json', 'a') as f:
        for line in lines:
            f.write(json.dumps(line)  + '\n')



def craft_agieval():

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

        with open('final_eval.json', 'a') as f:
            for line in lines:
                f.write(json.dumps(line)  + '\n')



def make_shuffled():
    with open('dharma_eval_unshuffled.json', 'r') as f:
        data = [json.loads(line) for line in f]

    random.shuffle(data)

    with open('dharma_eval_shuffled.json', 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def make_unshuffled():
    with open('final_eval.json', 'r') as f:
        data = [json.loads(line) for line in f]

    with open('dharma_eval_unshuffled.json', 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def make_datasets():

    with open('dharma_eval_unshuffled.json', 'r') as f:
        data = [json.loads(line) for line in f]

    subjects = {'MMLU': 336, 'ARC-Challenge': 336, 'ARC-Easy': 336, 'bigbench': 381,
                'BoolQ': 336, 'winogrande': 336, 'openbookqa': 336, 'truthful_qa': 336, 'agieval': 368}

    total_samples = sum(subjects.values())

    subject_proportions = {subject: size/total_samples for subject, size in subjects.items()}

    dataset_sizes = {'dharma-micro': 120, 'dharma-mini': 500, 'dharma-full': 3000}

    datasets = {name: [] for name in dataset_sizes.keys()}

    for subject, proportion in subject_proportions.items():
        subject_data = [item for item in data if item['subject'] == subject]
        random.shuffle(subject_data)

        for dataset, size in dataset_sizes.items():
            num_samples = round(proportion * size)
            datasets[dataset].extend(subject_data[:num_samples])
            subject_data = subject_data[num_samples:]

    for dataset, items in datasets.items():
        random.shuffle(items) 
        with open(f'{dataset}.json', 'w') as f:
            for item in items:
                f.write(json.dumps(item) + '\n')


def check_size():
    with open('final_eval.json', 'r') as f:
        data = []
        for line in f:
            data.append(json.loads(line))

    total_samples = len(data)
    print(total_samples)

def check_subjects():
    with open('final_eval.json', 'r') as f:
        data = [json.loads(line) for line in f]

    subject_counts = {}
    for row in data:
        subject = row['subject']
        if subject not in subject_counts:
            subject_counts[subject] = 0
        subject_counts[subject] += 1

    for subject, count in subject_counts.items():
        print(f'Subject: {subject}, Size: {count}')


def check():
    check_size()
    check_subjects()



def main():
    #MMLU
    craft_mmlu()
    check()

    #ARC-C and ARC-E
    craft_arc()
    check()

    #BIGBENCH
    craft_bigbench()
    check()

    #BOOLQ
    craft_boolq()
    check()

    #WINOGRANDE
    craft_winogrande()
    check()

    #OBQA
    craft_obqa()
    check()

    #TRUTHFULQA
    craft_truthfulqa()
    check()

    #AGIEVAL
    craft_agieval()
    check()


    #Make datasets
    make_unshuffled()
    make_shuffled()
    make_datasets()



    #upload to huggingface Hub with each dataset as a config
    
    

if __name__ == "__main__":
    main()




