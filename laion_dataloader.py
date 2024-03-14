import pandas as pd
from datasets import load_dataset
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader
import torch
from torchvision import transforms


data_dir = '/shared_drive/user-files/laion_dataset_200M/laion200m-data'

def make_train_dataset(data_dir, seed, buffer_size, resolution):
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

    iterable_dataset = load_dataset("webdataset", data_files={"train": webdataset_files}, split="train", 
                       streaming=True)
    shuffled_iterable_dataset = iterable_dataset.shuffle(seed=seed, buffer_size=buffer_size)

    # train_dataset = shuffled_iterable_dataset.select_columns(['jpg', 'txt'])

    # def get_transformed_image(example):
    #     image = example['jpg']
    #     image = image.convert("RGB")

    #     image_transforms = transforms.Compose(
    #         [
    #             transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    #             transforms.CenterCrop(resolution),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.5], [0.5]),
    #         ]
    #     )

    #     example['pixel_values'] = image_transforms(image)
    #     return example

    def get_url(example):
        url = example['json']['url']
        example['url'] = url
        return example
    
    # train_dataset = train_dataset.map(get_transformed_image)
    shuffled_iterable_dataset = shuffled_iterable_dataset.map(get_url)

    return shuffled_iterable_dataset


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

# dataloader = DataLoader(train_dataset, batch_size=32, 
#                         num_workers=4,
#                         collate_fn=collate_fn)

# for step, batch in enumerate(dataloader):
#     print(step)
#     import pdb; pdb.set_trace() 
