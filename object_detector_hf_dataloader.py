from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import pandas as pd
import pdb

from laion_dataloader import make_train_dataset

data_dir = '/shared_drive/user-files/laion_dataset_200M/laion200m-data'
IMAGE_RESOLUTION = 256
device = torch.device('cuda')
batch_size = 128
num_workers = 6

train_dataset = make_train_dataset(data_dir=data_dir,
                                   seed=42, buffer_size=100, resolution=IMAGE_RESOLUTION)

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

model = model.to(device)

target_sizes = torch.tensor([[IMAGE_RESOLUTION, IMAGE_RESOLUTION]]*batch_size).to(device)

def collate_fn(train_dataset): 
    images = []
    urls = []
    
    for example in train_dataset:
        images.append(example['jpg'].convert("RGB"))
        urls.append(example['url'])
        
    return processor(images=images, return_tensors="pt", do_resize=False), urls


def save_results_to_parquet(results, urls, model, step):
    rows = []
    columns = ['url', 'label', 'score', 'top_left_x', 'top_left_y', 'bottom_right_x', 'bottom_right_y']
    for i, result_per_image in enumerate(results):
        for score, label, box in zip(result_per_image["scores"], result_per_image["labels"], result_per_image["boxes"]):
            url = urls[i]
            box = box.detach().cpu()
            top_left_x, top_left_y, bottom_right_x, bottom_right_y = box[0].item(), box[1].item(), box[2].item(), box[3].item()
            label_name = model.config.id2label[label.item()]
            score = np.round(score.detach().cpu().item(), 2)
    
            row = [url, label_name, score, top_left_x, top_left_y, bottom_right_x, bottom_right_y]
            rows.append(row)
        
    df = pd.DataFrame(rows, columns=columns)
    df.to_parquet(f'/shared_drive/user-files/laion_dataset_200M/od-test/{step}_batch.parquet')


dataloader = DataLoader(train_dataset, 
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=True,
                        collate_fn=collate_fn)


for step, batch in tqdm(enumerate(dataloader)):
    batch_urls = batch[1]
    batch_tensors = batch[0]

    batch_tensors['pixel_values'] = batch_tensors['pixel_values'].to(device)
    batch_tensors['pixel_mask'] = batch_tensors['pixel_mask'].to(device)

    outputs = model(**batch_tensors)

    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)

    save_results_to_parquet(results=results, 
                                 urls=batch_urls, 
                                 model=model, 
                                 step=step)

