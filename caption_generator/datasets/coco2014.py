"""This contains the definition of how to create the COCO 2014 dataset."""

from pathlib import Path
import os
from ast import literal_eval
from PIL import Image
import toml
from .dataset import Dataset

DATASET_PATH = Path(__file__).parents[2]/'data'/'raw'/'coco2014'
METADATA_PATH = DATASET_PATH/'metadata.toml'
TRAIN_IMG_FILENAME = 'COCO_train2014_{}.jpg'
VAL_IMG_FILENAME = 'COCO_val2014_{}.jpg'
TEST_IMG_FILENAME = 'COCO_test2014_{}.jpg'

class COCO2014(Dataset):
    """A subclass defining the COCO 2014 dataset."""
    def __init__(self, is_validation_set=False, is_test_set=False):
        super(COCO2014, self).__init__(DATASET_PATH, METADATA_PATH, is_validation_set, is_test_set)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_test_set:
            img = self.create_img_filename(self.samples[idx][0], is_img_id=False)
        else:
            img = self.create_img_filename(str(self.samples[idx][0]))

        with Image.open(img) as img:
            img = self.transforms(img)
            return img, literal_eval(self.samples[idx][-1]) if not self.is_test_set else None

    def create_img_filename(self, img, is_img_id=True):
        metadata = toml.load(self.metadata_path)
        img_filename = TRAIN_IMG_FILENAME
        dir_file_path = os.path.join(self.dataset_path, metadata['train_ds_filename'])

        if self.is_validation_set:
            img_filename = VAL_IMG_FILENAME
            dir_file_path = os.path.join(self.dataset_path, metadata['val_ds_filename'])
        elif self.is_test_set:
            img_filename = TEST_IMG_FILENAME
            dir_file_path = os.path.join(self.dataset_path, metadata['test_ds_filename'])

        if is_img_id:
            img = img_filename.format('0' * (12 - len(img)) + img)

        return os.path.join(dir_file_path[:-4], img)
