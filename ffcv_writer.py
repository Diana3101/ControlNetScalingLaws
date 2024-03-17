import pdb

import webdataset as wds

from glob import glob
from os import path
import os

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, JSONField, BytesField

data_dir = '/shared_drive/user-files/laion_dataset_200M/laion200m-data'

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

    tar_subset_files_valid = []
    for file in tar_subset_files:
        if file.endswith('.tar'):
            file_json = file.replace('.tar', '_stats.json')
            if os.path.exists(os.path.join(subset_folder, file_json)):
                tar_subset_files_valid.append(os.path.join(subset_folder, file))
            else:
                 print(f"{os.path.join(subset_folder, file)} wasn't fully saved!")
        

    # tar_subset_files = [os.path.join(subset_folder, file) for file in tar_subset_files if file.endswith('.tar')] 
    webdataset_files.extend(tar_subset_files_valid)

print(f'Length of webdataset_files: {len(webdataset_files)}')

# my_shards = glob(path.join(FOLDER, '*'))
# tar_my_shards = [file for file in my_shards if file.endswith('.tar')][:1]

def pipeline(dataset):
    # pdb.set_trace()
    return dataset.decode('rgb8').to_tuple("jpg", "json")


writer = DatasetWriter('/mnt/disks/disk-big2/ffcv-laion200m.beton', {
    'image': RGBImageField(),
    'label': JSONField(),
}, num_workers=12)

writer.from_webdataset(webdataset_files, pipeline)