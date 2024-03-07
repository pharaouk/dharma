import json
import random
import string
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
import argparse
import yaml
import os

class AttributeDict(dict):
    __slots__ = ()
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

def make_dir(filename):
    directory = os.path.dirname(filename)

    if not os.path.exists(directory):
        os.makedirs(directory)


def load_config(config_file):
    with open(config_file, "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
            config = Config(config_dict)
            return config
        except yaml.YAMLError as exc:
            print(exc)



non_english_chars = '的一是不了人我在有他这为之大来以个中上们到说国和地也子时道出而要于就下得可你年生自那后能对あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔㅁㄴㅇㄹㅎㅗㅓㅏㅣأبتثجحخدذرزسشصضطظعغفقكلمنهوىيकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहàèìòùáéíóúäëïöüÿâêîôûçñß'

class DataProcessor:
    def __init__(self, file_name = ""):
        self.file_name = file_name

    def load_json_data(self,filename):
        with open(filename, 'r') as f:
            return [json.loads(line) for line in f]

    def write_json_data(self, filename, data):
        with open(filename, 'w') as f:
            for item in data:
                json.dump(item, f)
                f.write('\n')

    def append_json_data(self, filename, data):
        with open(filename, 'a') as f:
            for item in data:
                json.dump(item, f)
                f.write('\n')

    def filter_english(self, data):
        non_english_data = [item for item in data if sum(char in non_english_chars for char in item['input']) > 1]
        return [item for item in data if item not in non_english_data]
    def filter_non_english(self, data):
        non_english_data = [item for item in data if sum(char in non_english_chars for char in item['input']) > 1]
        return non_english_data

    def shuffle_data(self, data):
        random.shuffle(data)
        return data






processor = DataProcessor()




def count_answer_options(datasetname):
    print("-"*20)
    data = processor.load_json_data(datasetname)
    answer_counts = {}
    for item in data:
        answer = item.get('output')
        if answer not in answer_counts:
            answer_counts[answer] = 0
        answer_counts[answer] += 1
    for answer, count in answer_counts.items():
        print(f'Answer: {answer}, Count: {count}')





def make_shuffled(output_path):
    path_final = f'{output_path}/final_eval.json'
    path_source = f'{output_path}/final/dharma_eval_unshuffled.json'
    path_dest = f'{output_path}/final/dharma_eval_shuffled.json'
    with open(path_dest, 'r') as f:
        data = [json.loads(line) for line in f]
    processor.shuffle_data(data)
    with open(path_dest, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def make_unshuffled(output_path):
    path_final = f'{output_path}/final_eval.json'
    path_dest = f'{output_path}/final/dharma_eval_unshuffled.json'
    make_dir(path_dest)

    with open(path_final, 'r') as f:
        data = [json.loads(line) for line in f]
    with open(path_dest, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

   
def make_datasets(output_path):
    path_source = f'{output_path}/final/dharma_eval_unshuffled.json'
    path_dest = f'{output_path}/final/'

    with open(path_source, 'r') as f:
        data = [json.loads(line) for line in f]
    subjects = {}
    for item in data:
        subject = item['subject']
        if subject not in subjects:
            subjects[subject] = 0
        subjects[subject] += 1

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
        with open(f'{path_dest}{dataset}.json', 'w') as f:
            for item in items:
                f.write(json.dumps(item) + '\n')


def check_size(output_path):
    path_final = f'{output_path}/final_eval.json'

    with open(path_final, 'r') as f:
        data = []
        for line in f:
            data.append(json.loads(line))

    total_samples = len(data)
    print(total_samples)

def check_subjects(output_path):
    path_final = f'{output_path}/final_eval.json'

    with open(path_final, 'r') as f:
        data = [json.loads(line) for line in f]

    subject_counts = {}
    for row in data:
        subject = row['subject']
        if subject not in subject_counts:
            subject_counts[subject] = 0
        subject_counts[subject] += 1

    for subject, count in subject_counts.items():
        print(f'Subject: {subject}, Size: {count}')


def check(output):
    print("=="*20)
    check_size(output)
    check_subjects(output)


def filter_data(output_path):
    path_final = f'{output_path}/final_eval.json'
    data = processor.load_json_data(path_final)
    non_english_data = processor.filter_non_english(data)
    print(f"Found {len(non_english_data)} non-English samples.")
    data = [item for item in data if item not in non_english_data]
    processor.write_json_data(path_final, data)


