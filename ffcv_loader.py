from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

# Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
my_dataset = make_my_dataset()
write_path = '/output/path/for/converted/ds.beton'

# Pass a type for each data field
writer = DatasetWriter(write_path, {
    # Tune options to optimize dataset size, throughput at train-time
    'image': RGBImageField(max_resolution=256, jpeg_quality=jpeg_quality),
    'label': IntField()
})

# Write dataset
writer.from_indexed_dataset(my_dataset)