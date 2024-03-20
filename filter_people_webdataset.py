import webdataset as wds
from tqdm import tqdm
import numpy as np
from itertools import islice
import os
import pandas as pd
from datasets import load_dataset

folders = ['/mnt/disks/disk-big2/laion200m-od-labels-loader1'] 
        #    '/mnt/disks/disk-big2/laion200m-od-labels-loader2', 
        #    '/mnt/disks/disk-big2/laion200m-od-labels-loader3',
        #    '/mnt/disks/disk-big2/laion200m-od-labels-loader4', 
        #    '/mnt/disks/disk-big2/laion200m-od-labels-loader5', 
        #    '/mnt/disks/disk-big2/laion200m-od-labels-loader6']

full_list_invalid_urls = []

# 50 min for all 6 loaders
for folder in tqdm(folders):
    od_labels_files = os.listdir(folder)
    od_labels_files = [os.path.join(folder, file) for file in od_labels_files]
    processed_urls = load_dataset("parquet", data_files={"train": od_labels_files}, split="train")
    processed_urls_invalid = processed_urls.filter(lambda example: example['label'] == 'person')

    # UNIQUE URLs
    list_invalid_urls = list(processed_urls_invalid['url'])
    full_list_invalid_urls.extend(list_invalid_urls)
    full_list_invalid_urls = list(set(full_list_invalid_urls))

dataset = (
        wds.WebDataset(['/mnt/disks/disk-big2/laion200m-data-512/0-8_64M/00000.tar']) 
                        # '/mnt/disks/disk-big2/laion200m-data-512/0-8_64M/00001.tar'])
        .decode("rgb")
        .to_tuple("__key__", "jpg", "json")
    )

output = '/mnt/disks/disk-big2/temp.tar'

valid_images_count = 0
maxcount = 10000

with wds.TarWriter(output) as dst:
    for key, image, json_data in tqdm(islice(dataset, 0, maxcount)):
        if json_data['url'] not in full_list_invalid_urls:
            valid_images_count += 1
            sample = {
                "__key__": key,
                "jpg": image,
                "json": json_data
            }
            dst.write(sample)
        else:
            continue

print(f'Valid images count: {valid_images_count}')