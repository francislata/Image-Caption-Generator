"""This contains the base class that represents any dataset used."""

import os
from collections import defaultdict
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import toml
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO
from .util import download_from_url, extract_zip

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class Dataset(TorchDataset):
    """Base class of different datasets which handles dataset download, extraction and usage."""
    def __init__(self, dataset_path, metadata_path, is_validation_set=False, is_test_set=False, img_size=(226, 226)): #pylint: disable=line-too-long,too-many-arguments
        self.dataset_path = dataset_path
        self.metadata_path = metadata_path
        self.is_validation_set = is_validation_set
        self.is_test_set = is_test_set
        self.samples = []
        self.transforms = Compose([
            Resize(img_size),
            ToTensor(),
            Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

        self._load_dataset()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def _load_dataset(self):
        """Loads (and optionally downloads) the dataset."""
        metadata = toml.load(self.metadata_path)
        ds_url, anns_url = metadata['train_ds_url'], metadata['anns_url']
        ds_path = os.path.join(self.dataset_path, metadata['train_ds_filename'])
        ds_csv_filename = os.path.join(self.dataset_path, metadata['train_ds_csv_filename'])
        anns_path = os.path.join(self.dataset_path, metadata['anns_filename'])
        anns_file_path = os.path.join(anns_path[:-4], metadata['train_ds_anns_filename'])

        if self.is_validation_set:
            ds_url = metadata['val_ds_url']
            ds_path = os.path.join(self.dataset_path, metadata['val_ds_filename'])
            ds_csv_filename = os.path.join(self.dataset_path, metadata['val_ds_csv_filename'])
            anns_file_path = os.path.join(anns_path[:-4], metadata['val_ds_anns_filename'])
        elif self.is_test_set:
            ds_url = metadata['test_ds_url']
            ds_path = os.path.join(self.dataset_path, metadata['test_ds_filename'])
            ds_csv_filename = os.path.join(self.dataset_path, metadata['test_ds_csv_filename'])
            anns_url, anns_path, anns_file_path = None, None, None

        # Download dataset
        download_from_url(ds_url, ds_path)

        if not os.path.isdir(ds_path[:-4]):
            extract_zip(ds_path, self.dataset_path)

        # Download annotations
        if anns_url and anns_path:
            download_from_url(anns_url, anns_path)

            if not os.path.isdir(anns_path[:-4]):
                extract_zip(anns_path, self.dataset_path)

        if os.path.isfile(ds_csv_filename):
            self.samples = pd.read_csv(ds_csv_filename)
        else:
            self.samples = self._create_img_to_lbl_csv(ds_path[:-4], anns_file_path, ds_csv_filename) #pylint: disable=line-too-long

        self.samples = self.samples.values.tolist()

    def _create_img_to_lbl_csv(self, ds_path, anns_file_path, csv_filename): #pylint: disable=no-self-use
        """Creates a CSV file used for mapping images to its labels."""
        img_caps_mapping, image_ids, annotations = defaultdict(list), [], []

        if anns_file_path:
            coco_captions = COCO(anns_file_path)

            print('Mapping images to annotations...')

            for image_caption in coco_captions.loadAnns(coco_captions.getAnnIds()):
                img_caps_mapping[image_caption['image_id']].append(image_caption['caption'])

            for img_id, captions in tqdm(iterable=img_caps_mapping.items(), total=len(img_caps_mapping.items())): #pylint: disable=line-too-long
                image_ids.append(img_id)
                annotations.append(captions)

            print('Done!\n')
        else:
            print('Retrieving images...')

            image_ids = os.listdir(ds_path)

            print('Done!\n')

        print('Saving CSV to {}...'.format(csv_filename))

        df_mapping = {'images': image_ids}
        if anns_file_path:
            df_mapping['annotations'] = annotations

        ds_df = pd.DataFrame(df_mapping)
        ds_df.to_csv(csv_filename, index=False)

        print('Done!\n')

        return ds_df
