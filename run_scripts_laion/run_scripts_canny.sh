#!/bin/bash
accelerate launch train_laion_controlnet_webdataset.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --dataset_name="scaling-laws-diff-exp/laion1k" \
 --dataset_length=1000 \
 --download_dataset_before_streaming \
 --cache_dir="/shared_drive/user-files/huggingface/datasets/cache" \
 --output_dir="/mnt/disks/disk_1tb/checkpoints_laion/laion1k-canny-10ksteps" \
 --dataloader_num_workers=1 \
 --mixed_precision="bf16" \
 --resolution=512 \
 --condition_type='canny' \
 --validation_set_folder='validation_set_laion_canny' \
 --validation_steps=500 \
 --num_validation_images=1 \
 --train_batch_size=32 \
 --gradient_accumulation_steps=16 \
 --gradient_checkpointing \
 --allow_tf32 \
 --use_8bit_adam \
 --learning_rate=1e-4 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=4000 \
 --checkpointing_steps=500 \
 --seed=0 \
 --num_train_epochs=5000 \
 --report_to=wandb \
 --tracker_project_name='laion_train_controlnet_MAIN'

#  --multi_gpu
# --gradient_accumulation_steps=16/8 \
