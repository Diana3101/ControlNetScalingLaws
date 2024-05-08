#!/bin/bash
accelerate launch train_laion_controlnet_webdataset.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --dataset_name="scaling-laws-diff-exp/laion500k" \
 --dataset_length=500000 \
 --download_dataset_before_streaming \
 --cache_dir="/mnt/disk_60tb/huggingface/cache/datasets/" \
 --output_dir="/mnt/disk_60tb/checkpoints_laion/laion500k-canny" \
 --dataloader_num_workers=12 \
 --mixed_precision="bf16" \
 --resolution=512 \
 --condition_type='canny' \
 --validation_set_folder='validation_set_laion_canny' \
 --validation_steps=1000 \
 --num_validation_images=1 \
 --train_batch_size=8 \
 --gradient_accumulation_steps=8 \
 --gradient_checkpointing \
 --allow_tf32 \
 --use_8bit_adam \
 --learning_rate=1e-4 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=4000 \
 --checkpointing_steps=1000 \
 --seed=0 \
 --num_train_epochs=16 \
 --report_to=wandb \
 --tracker_project_name='laion_train_controlnet_MAIN'
#  --resume_from_checkpoint="/mnt/disk_60tb/checkpoints_laion/laion500k-canny/checkpoint-1000" \

