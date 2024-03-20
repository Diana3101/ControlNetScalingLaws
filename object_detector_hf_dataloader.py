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
from transformers.image_transforms import center_to_corners_format
from typing import List, Union
import torch.nn as nn
from time import time
import pdb
import argparse

# import multiprocessing
# multiprocessing.set_start_method('spawn')
# print(multiprocessing.get_start_method())


import wandb
from wandb import AlertLevel
# wandb.login()

from laion_dataloader import make_train_dataset

DATA_DIR = '/shared_drive/user-files/laion_dataset_200M/laion200m-data'
IMAGE_RESOLUTION = 256
DEVICE = torch.device('cuda')
BATCH_SIZE = 128
NUM_WORKERS = 2

model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = model.to(DEVICE, dtype=torch.float16)


def save_results_to_parquet(results, urls, model, step, n_loader):
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
    df.to_parquet(f'/mnt/disks/disk-big2/laion200m-od-labels-loader{n_loader+1}/{step}_batch.parquet')

def post_process_object_detection(
        outputs, threshold: float = 0.5, target_sizes = None
    ):
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        if target_sizes is not None:
            if len(out_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        # t = time()
        prob = nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)
        # print(time() - t)

        # t = time()
        # Convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(out_bbox)
        # print(time() - t)

        mask = scores > threshold
    
        # mask = mask.cpu()
        # scores = scores.cpu()
        # labels = labels.cpu()
        # boxes = boxes.cpu()

        # t = time()
        # Convert from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]
        # print(time() - t)

        # t = time()
        results = []
    
        # print(len(scores))
        # mask_idx, class_idx = torch.where(mask)
        # class_idx[i, i*128]

        for i in range(len(scores)):
        # for s, l, b in zip(scores, labels, boxes):
            score = scores[i][mask[i]]
            label = labels[i][mask[i]]
            box = boxes[i][mask[i]]
            # print(s.shape)
            results.append({"scores": score, "labels": label, "boxes": box})
        # print(time() - t)
        # print()

        return results


def collate_fn(train_dataset):
    pixel_values = torch.stack([example["pixel_values"] for example in train_dataset])
    urls = [example["url"] for example in train_dataset]
    return pixel_values, urls


def run_od_model(args):
    train_dataset = make_train_dataset(data_dir=DATA_DIR, n_loader=args.n_loader, num_loaders=args.num_loaders)

    dataloader = DataLoader(train_dataset, 
                        batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS,
                        pin_memory=True,
                        collate_fn=collate_fn)
    
    t = time()
    for step, batch in tqdm(enumerate(dataloader)):
        # pdb.set_trace()
        try:
            batch_tensor = batch[0]
            batch_urls = batch[1]
            batch_tensor = batch_tensor.to(DEVICE)

            batch_dict = {'pixel_values': batch_tensor}
            
            outputs = model(**batch_dict)
            
            target_sizes = torch.tensor([[IMAGE_RESOLUTION, IMAGE_RESOLUTION]]*batch_tensor.shape[0]).to(DEVICE)
            results = post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)
            
            save_results_to_parquet(results=results, 
                                                urls=batch_urls, 
                                                model=model, 
                                                step=step,
                                                n_loader=args.n_loader)

            wandb.log({f"loader-{args.n_loader}_batch": step})

        except:
            wandb.alert(title=f"Batch Warning!",
                        text=f"Problem with batch {step}",
                        level=AlertLevel.WARN)
            pdb.set_trace()

    print(f"Time for {args.n_loader}: {time() - t} sec")
    wandb.alert(title=f"Run {args.n_loader} finished!",
            text = f"Objects successfully detected in all {step+1} batches in {np.round((time() - t)/3600, 2)} hours !!! :)",
                   level=AlertLevel.INFO)
   

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_loader", type=int)
    parser.add_argument("--num_loaders", type=int, default=6)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    run = wandb.init(
            # Set the project where this run will be logged
            project="object-detector",
            # Track hyperparameters and run metadata
            config={
            },
        )
    run_od_model(args)
    
    wandb.finish()
