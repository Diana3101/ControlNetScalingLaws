#!/bin/bash
python validate_laion_controlnet.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --controlnet_checkpoint_pathes "/mnt/disks/disk_1tb/checkpoints_laion/laion1k-depth/checkpoint-10000/controlnet" "/mnt/disks/disk_1tb/checkpoints_laion/laion10k-depth/checkpoint-10000/controlnet" \
 --seed=0 \
 --test_set_name='scaling-laws-diff-exp/test-set-1k-res-512' \
 --cache_dir="/shared_drive/user-files/huggingface/datasets" \
 --condition_type='depth' \
 --num_validation_images=1 \
 --mixed_precision="bf16" \
 --resolution=512 \
 --batch_size=60 \
 --num_workers=12 \
 --predicted_images_dir="/mnt/disks/disk_1tb/validation/depth_predicted" \
 --wandb_project_name="laion_validate_controlnet"

#  --controlnet_checkpoints_folders "/mnt/disks/disk_1tb/checkpoints_laion/laion10k-depth" \