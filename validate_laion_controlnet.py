import argparse
import logging
import os
import random
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

import numpy as np
import pandas as pd
import re
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data as data
import torchvision.transforms.functional as tvf
from huggingface_hub import create_repo, upload_folder
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import logging
from torchvision import transforms
from datasets import load_dataset

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)

from diffusers.utils import check_min_version, is_wandb_available

from metrics.get_metrics_batched import get_edge_metrics, get_depth_metrics

import warnings
warnings.simplefilter(action='ignore')

torch.backends.cuda.matmul.allow_tf32 = True


if is_wandb_available():
    import wandb
    from wandb import AlertLevel

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()  # StreamHandler sends log messages to the console
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

# flags for reproducibility
torch.backends.cudnn.deterministic = True
### setted by default
torch.backends.cudnn.benchmark = False

def init_zoe_model(device, dtype):
    # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
    repo = "isl-org/ZoeDepth"
    # Zoe_N
    model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)

    model_zoe_n=model_zoe_n.to(device)
    model_zoe_n.core= model_zoe_n.core.to(dtype)
    model_zoe_n.conv2 = model_zoe_n.conv2.to(dtype)
    model_zoe_n.seed_bin_regressor = model_zoe_n.seed_bin_regressor.to(dtype)
    model_zoe_n.seed_projector =model_zoe_n.seed_projector.to(dtype)
    model_zoe_n.projectors= model_zoe_n.projectors.to(dtype)
    model_zoe_n.attractors=model_zoe_n.attractors.to(dtype)
    model_zoe_n.conditional_log_binomial=model_zoe_n.conditional_log_binomial.to(dtype)

    return model_zoe_n

def sort_by_number(file_name):
    if '.jpg' in file_name:
        match = re.search(r'(\d+)(?=\.jpg)', file_name)
    elif '.png' in file_name:
        match = re.search(r'(\d+)(?=\.png)', file_name)
    else:
        match = re.search(r'\d+', file_name)  # Find the first number in the file name
    if match:
        return int(match.group())  # Convert the matched number to an integer
    else:
        return float('inf')  # If no number found, place the file at the end of the list


def get_controlnet_loss(pixel_values, conditioning_pixel_values, input_ids, vae, noise_scheduler,
                        text_encoder, controlnet, unet, weight_dtype):
    # Convert images to latent space
    latents = vae.encode(pixel_values).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    # Sample a random timestep for each image
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Get the text embedding for conditioning
    encoder_hidden_states = text_encoder(input_ids)[0]

    controlnet_image = conditioning_pixel_values

    down_block_res_samples, mid_block_res_sample = controlnet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        controlnet_cond=controlnet_image,
        return_dict=False,
    )

    # Predict the noise residual
    model_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        down_block_additional_residuals=[
            sample.to(dtype=weight_dtype) for sample in down_block_res_samples
        ],
        mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
    ).sample

    # Get the target for loss depending on the prediction type
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    return loss.detach().item()

class ValidationDatasetManual(torch.utils.data.Dataset):
    def __init__(self, validation_images_pathes, validation_prompts):
        self.validation_images_pathes = validation_images_pathes
        self.validation_prompts = validation_prompts

    def get_transformed_sample(self, validation_image, validation_target, prompt):
        image_transforms = transforms.Compose(
                [
                    transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(args.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
    
        condition_image_transforms = transforms.Compose(
                [
                    transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(args.resolution),
                    transforms.ToTensor(),
                ]
            )
        
        pixel_values = image_transforms(validation_target)
        conditioning_pixel_values = condition_image_transforms(validation_image)
        
        validation_target_tensor = transforms.ToTensor()(validation_target)
        validation_image_tensor = transforms.ToTensor()(validation_image)
        
        sample = {'pixel_values': pixel_values, 'conditioning_pixel_values': conditioning_pixel_values,
                    'prompt': prompt,
                    'validation_target_tensor': validation_target_tensor, 
                    'validation_image_tensor': validation_image_tensor}
        return sample

    def __len__(self):
        return len(self.validation_images_pathes)

    def __getitem__(self, idx):
        validation_image = Image.open(self.validation_images_pathes[idx]).convert("RGB")
        validation_target_path = self.validation_images_pathes[idx].replace(f'condition_{args.condition_type}/conditioning', 'target')
        validation_target = Image.open(validation_target_path).convert("RGB")
        prompt = self.validation_prompts[idx]

        return self.get_transformed_sample(validation_image=validation_image, 
                                           validation_target=validation_target, 
                                           prompt=prompt)


def make_valid_dataset(args):
    dataset = load_dataset(args.test_set_name, split='test',
                    cache_dir=args.cache_dir)
        
    def preprocess_test(examples, condition_type):
        target_images = [image.convert("RGB") for image in examples['target_image']]
        validation_images = [image.convert("RGB") for image in examples[f'conditioning_image_{condition_type}']]
        validation_prompts = examples['prompt']

        image_transforms = transforms.Compose(
                [
                    transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(args.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
    
        condition_image_transforms = transforms.Compose(
                [
                    transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(args.resolution),
                    transforms.ToTensor(),
                ]
            )

        pixel_values = [image_transforms(image) for image in target_images]
        conditioning_pixel_values = [condition_image_transforms(image) for image in validation_images]

        validation_target_tensors = [transforms.ToTensor()(image) for image in target_images]
        validation_image_tensors = [transforms.ToTensor()(image) for image in validation_images]

        sample = {'pixel_values': pixel_values, 'conditioning_pixel_values': conditioning_pixel_values,
                    'prompt': validation_prompts,
                    'validation_target_tensor': validation_target_tensors, 
                    'validation_image_tensor': validation_image_tensors}
        return sample
    
    validation_dataset = dataset.with_transform(lambda x: preprocess_test(x, args.condition_type))
    return validation_dataset
        

def log_validation(vae, text_encoder, tokenizer, unet, controlnet, args, weight_dtype, zoe_depth_model, device, checkpoint_step,
                   noise_scheduler, image_idxs_wandb, image_idxs_to_save, checkpoint_path, train_folder=None):
    logger.info("Running validation... ")

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    # pipeline.set_progress_bar_config(disable=False)
    pipeline.set_progress_bar_config(disable=True)

    pipeline.enable_vae_slicing()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    
    wandb_logs = []

    if args.condition_type == 'canny':
        val_avg_img_ods, val_avg_img_ap, val_avg_img_ods_blur, val_avg_img_ap_blur  = [], [], [], []
    elif args.condition_type == 'depth':
        val_avg_metrics_dict = dict(a1=[], a2=[], a3=[], abs_rel=[], 
                                    rmse=[], log_10=[], rmse_log=[], silog=[], sq_rel=[])

    if args.test_set_name:
        valid_dataset = make_valid_dataset(args)
    else:
        if len(args.validation_image) == len(args.validation_prompt):
            validation_images = args.validation_image
            validation_prompts = args.validation_prompt
        elif len(args.validation_image) == 1:
            validation_images = args.validation_image * len(args.validation_prompt)
            validation_prompts = args.validation_prompt
        elif len(args.validation_prompt) == 1:
            validation_images = args.validation_image
            validation_prompts = args.validation_prompt * len(args.validation_image)
        else:
            raise ValueError(
                "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
            )
        
        valid_dataset = ValidationDatasetManual(validation_images_pathes=validation_images,
                                        validation_prompts=validation_prompts)
    

    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, 
                                        batch_size=args.batch_size, 
                                        num_workers=args.num_workers,
                                        pin_memory=True)
    
    # logger.info(f"Dataloader created")
    
    batch_losses = []
    formatted_images = []

    def tokenize_captions(captions):
        inputs = tokenizer(
                captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    with torch.inference_mode():
        for n_batch, batch in tqdm(enumerate(valid_dataloader)):
            # ### TEST
            # if n_batch > 1:
            #     break
            # logger.info(f"Start batch calculations ...")
            image_idxs_to_save_batch_map = {}
            image_idxs_to_save_batch = []
            for x in image_idxs_to_save:
                if x < (args.batch_size*(n_batch+1)) and x > (args.batch_size*n_batch):
                    image_idxs_to_save_batch_map[x-(args.batch_size*n_batch)] = x
                    image_idxs_to_save_batch.append(x-(args.batch_size*n_batch))
        
            prompts = batch['prompt']
            pixel_values = batch['pixel_values'].to(device=device, dtype=weight_dtype, non_blocking=True)
            conditioning_pixel_values = batch['conditioning_pixel_values'].to(device=device, dtype=weight_dtype, non_blocking=True)
            with torch.autocast("cuda"):
                input_ids = tokenize_captions(prompts).to(device=device, non_blocking=True)

            with torch.autocast("cuda"):
                loss = get_controlnet_loss(pixel_values, conditioning_pixel_values, input_ids, vae, noise_scheduler,
                                text_encoder, controlnet, unet, weight_dtype)
                del pixel_values
                del conditioning_pixel_values
                del input_ids
                torch.cuda.empty_cache()
                
            # logger.info(f"Loss calculated")
            batch_losses.append(loss)

            validation_target_tensors = batch['validation_target_tensor'].to(device=device, dtype=weight_dtype, non_blocking=True)
            validation_image_tensors = batch['validation_image_tensor'].to(device=device, dtype=weight_dtype, non_blocking=True)

            with torch.autocast("cuda"):
                pred_batch_images = pipeline(
                    prompts, validation_image_tensors, num_inference_steps=20, generator=generator,
                    output_type='numpy'
                )[0]
                torch.cuda.empty_cache()

            # logger.info(f"Batch predicted")
            pred_batch_images = torch.from_numpy(pred_batch_images).to(device=device, dtype=weight_dtype, non_blocking=True)
            pred_batch_images = pred_batch_images.permute(0, 3, 1, 2)
            
            validation_image_tensors = validation_image_tensors[:,0, :, :].unsqueeze(1)


            if args.condition_type == 'canny':
                validation_image_tensors = validation_image_tensors.to(dtype=torch.float32)
                pred_batch_images = pred_batch_images.to(dtype=torch.float32)
                validation_target_tensors = validation_target_tensors.to(dtype=torch.float32)

                (avg_img_ods, avg_img_ap,
                avg_img_ods_blur, avg_img_ap_blur,
                pred_batch_images_edges, pred_batch_images_edges_blured) = get_edge_metrics(pred_batch_images=pred_batch_images, 
                                                                validation_image_tensors=validation_image_tensors)
                
                # logger.info(f"Metrics calculated")

                val_avg_img_ods.append(avg_img_ods)
                val_avg_img_ap.append(avg_img_ap)
                val_avg_img_ods_blur.append(avg_img_ods_blur)
                val_avg_img_ap_blur.append(avg_img_ap_blur)

                validation_images = validation_image_tensors[image_idxs_to_save_batch].clone()
                del validation_image_tensors

                pred_batch_images_small = pred_batch_images[image_idxs_to_save_batch].clone()
                del pred_batch_images                                                   
                prompts = [prompt for i, prompt in enumerate(prompts) if i in image_idxs_to_save_batch]

                pred_batch_images_edges_small = pred_batch_images_edges[image_idxs_to_save_batch].clone()
                del pred_batch_images_edges

                pred_batch_images_edges_blured_small = pred_batch_images_edges_blured[image_idxs_to_save_batch].clone()
                del pred_batch_images_edges_blured

                validation_targets = validation_target_tensors[image_idxs_to_save_batch].clone()
                del validation_target_tensors

                torch.cuda.empty_cache()

                for idx in range(len(image_idxs_to_save_batch)):
                    original_idx = image_idxs_to_save_batch_map[image_idxs_to_save_batch[idx]]
                    pred_edges_pil = transforms.functional.to_pil_image(pred_batch_images_edges_small[idx])
                    pred_edges_pil.save(f"{checkpoint_path}/predicted_edges_{original_idx}.png")

                    pred_edges_blur_pil = transforms.functional.to_pil_image(pred_batch_images_edges_blured_small[idx])
                    pred_edges_blur_pil.save(f"{checkpoint_path}/predicted_edges_blur_{original_idx}.png")

                    pred_img_pil = transforms.functional.to_pil_image(pred_batch_images_small[idx])
                    pred_img_pil.save(f"{checkpoint_path}/predicted_image_{original_idx}.png")


                    if original_idx in image_idxs_wandb:
                        val_img_pil = transforms.functional.to_pil_image(validation_images[idx].squeeze())
                        target_pil = transforms.functional.to_pil_image(validation_targets[idx])
                        formatted_images.append(wandb.Image(val_img_pil, caption="Controlnet conditioning"))
                        formatted_images.append(wandb.Image(pred_edges_pil, caption="Prediction image: Canny edge"))

                        formatted_images.append(wandb.Image(pred_edges_blur_pil, caption="Prediction image: Canny edge BLUR"))
                        formatted_images.append(wandb.Image(target_pil, caption="Target"))

                        formatted_images.append(wandb.Image(pred_img_pil, caption=prompts[idx]))

                del pred_batch_images_edges_small
                del pred_batch_images_edges_blured_small
                del pred_batch_images_small
                del validation_images
                del validation_targets
                torch.cuda.empty_cache()

            elif args.condition_type == 'depth':
                avg_metrics_dict, pred_depth_images = get_depth_metrics(pred_batch_images=pred_batch_images, 
                                                                    validation_target_tensors=validation_target_tensors,
                                                                    zoe_depth_model=zoe_depth_model)
                
                # logger.info(f"Metrics calculated")
                
                for key, value in avg_metrics_dict.items():
                    val_avg_metrics_dict[key].append(value)

                validation_image_tensors = validation_image_tensors.to(dtype=torch.float32)
                pred_batch_images = pred_batch_images.to(dtype=torch.float32)
                validation_target_tensors = validation_target_tensors.to(dtype=torch.float32)

                validation_images = validation_image_tensors[image_idxs_to_save_batch].clone()
                del validation_image_tensors

                pred_batch_images_small = pred_batch_images[image_idxs_to_save_batch].clone()
                del pred_batch_images

                prompts = [prompt for i, prompt in enumerate(prompts) if i in image_idxs_to_save_batch]

                validation_targets = validation_target_tensors[image_idxs_to_save_batch].clone()
                del validation_target_tensors

                pred_depth_images_small = pred_depth_images[image_idxs_to_save_batch].clone()
                del pred_depth_images

                torch.cuda.empty_cache()

                for idx in range(len(image_idxs_to_save_batch)):
                    original_idx = image_idxs_to_save_batch_map[image_idxs_to_save_batch[idx]]

                    pred_img_pil = transforms.functional.to_pil_image(pred_batch_images_small[idx])
                    pred_img_pil.save(f"{checkpoint_path}/predicted_image_{original_idx}.jpg")

                    pred_depth_pil = transforms.functional.to_pil_image(pred_depth_images_small[idx].squeeze())
                    pred_depth_pil.save(f"{checkpoint_path}/predicted_depth_{original_idx}.jpg")

                    if original_idx in image_idxs_wandb:
                        val_img_pil = transforms.functional.to_pil_image(validation_images[idx].squeeze())
                        target_pil = transforms.functional.to_pil_image(validation_targets[idx])

                        formatted_images.append(wandb.Image(val_img_pil, caption="Target depth map: ControlNet conditioning"))
                        formatted_images.append(wandb.Image(pred_depth_pil, caption="Predicted depth map"))
                        formatted_images.append(wandb.Image(target_pil, caption="Target"))

                        formatted_images.append(wandb.Image(pred_img_pil, caption=prompts[idx]))

                del pred_batch_images_small
                del pred_depth_images_small
                del validation_images
                del validation_targets
                torch.cuda.empty_cache()

    if args.condition_type == 'canny':
        if train_folder:
            wandb_logs.append({
                                f"{train_folder}_validation": formatted_images,
                                f"{train_folder}_loss": np.mean(batch_losses),
                                f"{train_folder}_edge_metric/ODS": np.mean(val_avg_img_ods),
                                f"{train_folder}_edge_metric/AP": np.mean(val_avg_img_ap),
                                f"{train_folder}_edge_metric_blur/ODS": np.mean(val_avg_img_ods_blur),
                                f"{train_folder}_edge_metric_blur/AP": np.mean(val_avg_img_ap_blur)})
        else:
            wandb_logs.append({
                                f"validation": formatted_images,
                                "loss": np.mean(batch_losses),
                                f"edge_metric/ODS": np.mean(val_avg_img_ods),
                                f"edge_metric/AP": np.mean(val_avg_img_ap),
                                f"edge_metric_blur/ODS": np.mean(val_avg_img_ods_blur),
                                f"edge_metric_blur/AP": np.mean(val_avg_img_ap_blur)})
        
    elif args.condition_type == 'depth':
        if train_folder:
            wandb_logs_dict_to_append = {f"{train_folder}_validation": formatted_images, 
                                         f"{train_folder}_loss": np.mean(batch_losses)}
            for key, metrics_list in val_avg_metrics_dict.items():
                wandb_logs_dict_to_append[f"{train_folder}_depth_metric/{key}"] = np.mean(metrics_list)
        else:
            wandb_logs_dict_to_append = {f"validation": formatted_images, "loss": np.mean(batch_losses)}
            for key, metrics_list in val_avg_metrics_dict.items():
                wandb_logs_dict_to_append[f"depth_metric/{key}"] = np.mean(metrics_list)

        wandb_logs.append(wandb_logs_dict_to_append)

    wandb_logs = {key: value for d in wandb_logs for key, value in d.items()}
    wandb.log(wandb_logs, step=checkpoint_step)

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--controlnet_checkpoints_folders",
        type=str,
        default=None,
        nargs="+",
        help="Path to the folders with controlnet checkpoints.",
    )

    parser.add_argument(
        "--controlnet_checkpoint_pathes",
        type=str,
        default=None,
        nargs="+",
        help="Pathes to the folders `controlnet`, where the one path - one controlnet checkpoint.",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--test_set_name",
        type=str,
        default=None,
        help = "Name of the test dataset on the HuggingFace"
    )

    parser.add_argument(
        "--condition_type",
        type=str,
        default=None,
        help=(
            "Type of condition images. Possible values: 'canny' or 'depth'."
        ),
    )

    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )

    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="",
    )

    parser.add_argument(
        "--predicted_images_dir",
        type=str,
        required=True,
        help="dir to save predicted images.",
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default='validate_controlnet',
        required=True
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    args = parser.parse_args()

    return args


def main(args):
    device = torch.device('cuda')
    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.condition_type == 'depth':
        zoe_depth_model = init_zoe_model(device=device, dtype=weight_dtype)
    else:
        zoe_depth_model = None

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    tracker_config = dict(vars(args))

    run = wandb.init(
        project=args.wandb_project_name,
        config=tracker_config
    )
    
    image_idxs_to_save = [1,3,4,5,11,12,17,22,23,25,28,30,31,36,37,40,43,46,47,69] + list(range(250, 300)) + list(range(500, 530))
    image_idxs_wandb = [17,23,25,28,30]

    idxs_assert = False
    for idx in image_idxs_wandb:
        if idx not in image_idxs_to_save:
            idxs_assert = True
            break
    if idxs_assert:
        logger.warn(f"`image_idxs_wandb` should be subset of `image_idxs_to_save`!")
    
    def get_contronet_pathes(controlnet_checkpoints_folder):
        controlnet_pathes = []
        folders_with_checkpoints = os.listdir(controlnet_checkpoints_folder)
        folders_with_checkpoints = [folder for folder in folders_with_checkpoints if 'checkpoint' in folder]
        folders_with_checkpoints = sorted(folders_with_checkpoints, key=sort_by_number)

        for folder in folders_with_checkpoints:
            controlnet_path = os.path.join(controlnet_checkpoints_folder, folder, 'controlnet')
            controlnet_pathes.append(controlnet_path)

            # ### CUSTOM
            # if '8.5epochs' in controlnet_checkpoints_folder:
            #     controlnet_path = os.path.join(controlnet_checkpoints_folder, folder, 'controlnet')
            #     checkpoint_step = int(controlnet_path.split('/')[-2].split('-')[-1])
            #     if checkpoint_step < 6045:
            #         controlnet_pathes.append(controlnet_path)
            # else:
            #     controlnet_path = os.path.join(controlnet_checkpoints_folder, folder, 'controlnet')
            #     controlnet_pathes.append(controlnet_path)

        return controlnet_pathes
    
    if args.controlnet_checkpoints_folders:
        all_controlnet_pathes = []

        for controlnet_checkpoints_folder in args.controlnet_checkpoints_folders:
            controlnet_pathes = get_contronet_pathes(controlnet_checkpoints_folder)
            all_controlnet_pathes.extend(controlnet_pathes)

        for controlnet_path in tqdm(all_controlnet_pathes):
            checkpoint_step = int(controlnet_path.split('/')[-2].split('-')[-1])

            if checkpoint_step % 1000 != 0:
                continue

            train_folder = controlnet_path.split('/')[-3]
            checkpoint_folder = controlnet_path.split('/')[-2]

            if not os.path.exists(os.path.join(args.predicted_images_dir, train_folder)):
                os.makedirs(os.path.join(args.predicted_images_dir, train_folder))
            if not os.path.exists(os.path.join(args.predicted_images_dir, train_folder, checkpoint_folder)):
                os.makedirs(os.path.join(args.predicted_images_dir, train_folder, checkpoint_folder))
                    
            checkpoint_path = os.path.join(args.predicted_images_dir, train_folder, checkpoint_folder)
            
            logger.info(f"Loading existing controlnet weights for checkpoint step {checkpoint_step}")
            logger.info(f"Full checkpoint path: {controlnet_path}")
            controlnet = ControlNetModel.from_pretrained(controlnet_path, 
                                                     torch_dtype=weight_dtype, 
                                                     use_safetensors=True)
            controlnet = controlnet.to(device)

            log_validation(
                        vae,
                        text_encoder,
                        tokenizer,
                        unet,
                        controlnet,
                        args,
                        weight_dtype,
                        zoe_depth_model=zoe_depth_model,
                        device=device,
                        checkpoint_step=checkpoint_step,
                        noise_scheduler=noise_scheduler,
                        image_idxs_wandb=image_idxs_wandb,
                        image_idxs_to_save=image_idxs_to_save,
                        checkpoint_path=checkpoint_path
                    )
            

    elif args.controlnet_checkpoint_pathes:
        for controlnet_checkpoint_path in args.controlnet_checkpoint_pathes:
            checkpoint_step = int(controlnet_checkpoint_path.split('/')[-2].split('-')[-1])
            train_folder = controlnet_checkpoint_path.split('/')[-3]
            checkpoint_folder = controlnet_checkpoint_path.split('/')[-2]

            if not os.path.exists(os.path.join(args.predicted_images_dir, train_folder)):
                os.makedirs(os.path.join(args.predicted_images_dir, train_folder))
            if not os.path.exists(os.path.join(args.predicted_images_dir, train_folder, checkpoint_folder)):
                os.makedirs(os.path.join(args.predicted_images_dir, train_folder, checkpoint_folder))
                    
            checkpoint_path = os.path.join(args.predicted_images_dir, train_folder, checkpoint_folder)
            logger.info(f"Loading existing controlnet weights for model {train_folder} at checkpoint step {checkpoint_step}")
            controlnet = ControlNetModel.from_pretrained(controlnet_checkpoint_path, 
                                                        torch_dtype=weight_dtype, 
                                                        use_safetensors=True)
            controlnet = controlnet.to(device)

            log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
                            args,
                            weight_dtype,
                            zoe_depth_model=zoe_depth_model,
                            device=device,
                            checkpoint_step=checkpoint_step,
                            noise_scheduler=noise_scheduler,
                            image_idxs_wandb=image_idxs_wandb,
                            image_idxs_to_save=image_idxs_to_save,
                            checkpoint_path=checkpoint_path,
                            train_folder=train_folder
                        )
    else:
        logger.info("You have to give `controlnet_checkpoints_folders` OR `controlnet_checkpoint_pathes`!")

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
            