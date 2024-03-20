import os
import pandas as pd
from datasets import load_dataset
import numpy as np

from tqdm import tqdm
tqdm.pandas()

input_folder = '/shared_drive/user-files/laion_dataset_200M/'

def get_loader_path(n_loader):
    return f'/mnt/disks/disk-big2/laion200m-od-labels-loader{n_loader}'

input_subset_loaders_map = {f'{input_folder}laion200m-train-12_2M-subset.parquet': 
                            [get_loader_path(1)],
                            f'{input_folder}laion200m-train-12_2M-20_2M-subset.parquet': 
                            [get_loader_path(3)],
                            f'{input_folder}laion200m-train-20_2M-33_2M-subset.parquet': 
                            [get_loader_path(3), get_loader_path(4)],
                            f'{input_folder}laion200m-train-33_2M-50M-subset.parquet':
                            [get_loader_path(4)],
                            f'{input_folder}laion200m-train-50M-100M-subset.parquet':
                            [get_loader_path(5), get_loader_path(6)],
                            f'{input_folder}laion200m-train-100M-150M-subset.parquet':
                            [get_loader_path(1), get_loader_path(2), get_loader_path(3)],
                            f'{input_folder}laion200m-train-150M-160_8M-subset.parquet':
                            [get_loader_path(4), get_loader_path(5)]}


for input_subset_path, loader_folder_pathes in tqdm(input_subset_loaders_map.items()):
    input_dataset = load_dataset("parquet", data_files={"train": input_subset_path}, split="train", 
                                 num_proc=12, cache_dir='/mnt/disks/disk-big2/huggingface/cache/')
    df_input_dataset = input_dataset.to_pandas()
    df_input_dataset = df_input_dataset.drop_duplicates(subset=['url'])

    # all_od_labels_files = []
    for i, label_folder in enumerate(loader_folder_pathes):
        od_labels_files = os.listdir(label_folder)
        od_labels_files = [os.path.join(label_folder, file) for file in od_labels_files]
        # all_od_labels_files.extend(od_labels_files)

    # processed_urls = load_dataset("parquet", data_files={"train": all_od_labels_files}, split="train")
        processed_urls = load_dataset("parquet", data_files={"train": od_labels_files}, split="train", 
                                      num_proc=12, cache_dir='/mnt/disks/disk-big2/huggingface/cache/')
        processed_urls_invalid = processed_urls.filter(lambda example: example['label'] == 'person')

        # UNIQUE URLs
        list_invalid_urls = set(processed_urls_invalid['url'])

        if i == 0:
            df_input_dataset_valid = df_input_dataset[~df_input_dataset['url'].isin(list_invalid_urls)]
            del list_invalid_urls
            del processed_urls_invalid
            del processed_urls
        else:
            df_input_dataset_valid = df_input_dataset_valid[~df_input_dataset_valid['url'].isin(list_invalid_urls)]
            del list_invalid_urls
            del processed_urls_invalid
            del processed_urls


    df_input_dataset_valid = df_input_dataset_valid.reset_index(drop=True)

    valid_dataset_len = int(len(df_input_dataset_valid))
    print(f'Length of the VALID input subset {input_subset_path}:  {valid_dataset_len}')
    df_input_dataset_valid.to_parquet(f'/mnt/disks/disk-big2/laion200m-people-filtered-subsets/{valid_dataset_len}.parquet')





