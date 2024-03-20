import pandas as pd
from datasets import load_dataset
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from tqdm import tqdm

data_dir = '/shared_drive/user-files/laion_dataset_200M/laion200m-data'
device = torch.device('cuda')


def make_train_dataset(data_dir, n_loader, num_loaders):
    def get_folders_in_parent_directory(parent_dir):
        folders = []
        for entry in os.listdir(parent_dir):
            entry_path = os.path.join(parent_dir, entry)
            if os.path.isdir(entry_path) and entry != 'images_test':
                folders.append(entry_path)
        return folders

    data_subset_folders = get_folders_in_parent_directory(data_dir)

    webdataset_files = []
    for i, subset_folder in enumerate(data_subset_folders):
        tar_subset_files = os.listdir(subset_folder)
        tar_subset_files = [os.path.join(subset_folder, file) for file in tar_subset_files if file.endswith('.tar')] 
        webdataset_files.extend(tar_subset_files)

    portion_count = int(len(webdataset_files)/ num_loaders)
    webdataset_files_loader_subset = webdataset_files[portion_count*n_loader : portion_count*(n_loader+1)]

    webdataset_files_loader_subset_valid = []
    for file in webdataset_files_loader_subset:
        file_json = file.replace('.tar', '_stats.json')
        if os.path.exists(file_json):
            webdataset_files_loader_subset_valid.append(file)

    # webdataset_files_loader_subset_valid_FOLDERS = [file.split('/')[-2] for file in webdataset_files_loader_subset_valid]
    # webdataset_files_loader_subset_valid_FOLDERS = list(set(webdataset_files_loader_subset_valid_FOLDERS))
    # print(f'Loader {n_loader+1} contain images from folders: ')
    # print(webdataset_files_loader_subset_valid_FOLDERS)
    # print()

    iterable_dataset = load_dataset("webdataset", data_files={"train": webdataset_files_loader_subset_valid}, split="train", 
                       streaming=True)
    
    # shuffled_iterable_dataset = iterable_dataset.shuffle(seed=seed, buffer_size=buffer_size)
    # import pdb; pdb.set_trace()

    def get_detr_transformed_image_and_urls(example):
        new_example = {}
        url = example['json']['url']

        image = example['jpg']
        image = image.convert("RGB")

        # for detr model
        image_mean = [
            0.485,
            0.456,
            0.406
        ]

        image_std = [
            0.229,
            0.224,
            0.225
        ]

        image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float16),
                transforms.Normalize(image_mean, image_std),
            ]
        )

        new_example['pixel_values'] = image_transforms(image)
        new_example['url'] = url
        
        return new_example

    # def get_url(example):
    #     url = example['json']['url']
    #     example['url'] = url
    #     return example
    
    iterable_dataset = iterable_dataset.map(get_detr_transformed_image_and_urls)
    # iterable_dataset = iterable_dataset.map(get_url)

    return iterable_dataset


# train_dataset = make_train_dataset(data_dir=data_dir,
#                                    seed=42, buffer_size=100, resolution=256)


# def collate_fn(train_dataset):
#     image = train_dataset[0]
#     print(image)
    # import pdb; pdb.set_trace()
    # pixel_values = torch.stack([example["pixel_values"] for example in train_dataset])
    # pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    # # conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    # # conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    # # input_ids = torch.stack([example["input_ids"] for example in examples])

    # return {
    #     "pixel_values": pixel_values,
        # "conditioning_pixel_values": conditioning_pixel_values,
        # "input_ids": input_ids,
    # }

### Using with epochs
# for epoch in range(n_epochs):
#     my_iterable_dataset.set_epoch(epoch)
#     for example in my_iterable_dataset:  # fast + reshuffled at each epoch using `effective_seed = seed + epoch`
#         pass


# def collate_fn(train_dataset):
#     pixel_values = torch.stack([example["pixel_values"] for example in train_dataset])
#     urls = [example["url"] for example in train_dataset]
#     return pixel_values, urls


# IMAGE_RESOLUTION = 256
# train_dataset = make_train_dataset(data_dir=data_dir, n_loader=0, num_loaders=6)
# train_dataset = make_train_dataset(data_dir=data_dir, n_loader=1, num_loaders=6)
# train_dataset = make_train_dataset(data_dir=data_dir, n_loader=2, num_loaders=6)
# train_dataset = make_train_dataset(data_dir=data_dir, n_loader=3, num_loaders=6)
# train_dataset = make_train_dataset(data_dir=data_dir, n_loader=4, num_loaders=6)
# train_dataset = make_train_dataset(data_dir=data_dir, n_loader=5, num_loaders=6)

# dataloader = DataLoader(train_dataset, batch_size=128, 
#                         num_workers=1,
#                         collate_fn=collate_fn)

# for step, batch in tqdm(enumerate(dataloader)):
#     batch_tensor = batch[0]
    # batch_tensor = batch_tensor.to(device)
#     batch_urls = batch[1]
#     print(step)
#     import pdb; pdb.set_trace() 
