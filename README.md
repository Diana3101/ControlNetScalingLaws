# ControlNetScalingLaws

## How to train ControlNet
- create environment:
```bash
conda env create -f controlnet-scalinglaws.yaml
conda activate controlnet-scalinglaws
```

- choose the `run_scripts_laion/run_scripts_depth.sh` or  `run_scripts_laion/run_scripts_canny.sh`, depending of what control signal you want to use
- change the next parameters:
    - `--dataset_name` choose one from here: https://huggingface.co/scaling-laws-diff-exp 
    - `--dataset_length` **assign manually length of the dataset**: \
        The exact length of the datasets: \
        Length of 1k dataset: 1000 \
        Length of 5k dataset: 5000\
        Length of 10k dataset: 10000\
        Length of 25k dataset: 25000\
        Length of 50k dataset: 50000\
        Length of 100k dataset: 100000\
        Length of 250k dataset: 250000\
        Length of 500k dataset: 500000\
        Length of 1M dataset: 1000320\

    - `--cache_dir` and `output_dir` to the pathes on your instance
    - `--train_batch_size` and `--gradient_accumulation_steps`\
        Recommended settings:\
            - For *8xNVIDIA L4x20Gb* train_batch_size=8, gradient_accumulation_steps=8 (total batch size will be 512). It uses 18GB of each GPU.\
            - For *1xA100x80Gb* train_batch_size=32, gradient_accumulation_steps=16 (total batch size will be 512).  It uses 51GB of GPU.\

    - `--num_train_epochs`. Use the approptiate value from the column *"N epochs"* from the [CNSL-laion-experiments table](https://docs.google.com/spreadsheets/d/12bNp6eeGgbk5-dIs7k14o6D3QN097ezFrFxOkJutdXQ/edit#gid=0) 

- run the following command from the root folder (ControlNetScalinglaws):
```bash
./run_scripts_laion/run_scripts_depth.sh
```
or 
```bash
./run_scripts_laion/run_scripts_canny.sh
```
- please, change status of the training run in the [CNSL-laion-experiments table](https://docs.google.com/spreadsheets/d/12bNp6eeGgbk5-dIs7k14o6D3QN097ezFrFxOkJutdXQ/edit#gid=0)


## How to validate ControlNet
- choose the `/run_scripts_laion/validation/run_scripts_depth.sh` or  `/run_scripts_laion/validation/run_scripts_canny.sh`, depending of what control signal you want to use
- change the next parameters:
    - `--controlnet_checkpoint_pathes` provide one or several pathes to the controlnet weights, in the same format as "/mnt/disks/disk_1tb/checkpoints_laion/laion1k-depth/checkpoint-10000/controlnet"
    - **OR** if you want to run validation on ALL checkpoint saved for the one training, replace the parameter `--controlnet_checkpoint_pathes` with `--controlnet_checkpoints_folders` and assign the value like this "/mnt/disks/disk_1tb/checkpoints_laion/laion1k-depth"
    - `--cache_dir` to the path on your instance
    - `--predicted_images_dir` to the path on your instance, where the predicted images will be saved.
    - `--batch_size` if you need:\
        CANNY validation need 20GB GPU with batch_size=60\
        DEPTH validation need 40GB with batch_size=60

- run the following command from the root folder (ControlNetScalinglaws):
```bash
./run_scripts_laion/validation/run_scripts_depth.sh
```
or 
```bash
./run_scripts_laion/validation/run_scripts_canny.sh
```
- please, change status of the validation in the [CNSL-laion-experiments table](https://docs.google.com/spreadsheets/d/12bNp6eeGgbk5-dIs7k14o6D3QN097ezFrFxOkJutdXQ/edit#gid=0)