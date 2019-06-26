"""This contains the definition of how to create the COCO 2014 dataset."""

from collections import defaultdict
from pathlib import Path
import os
from typing import Union, Tuple, DefaultDict, Optional
from PIL import Image
import toml
import pandas as pd
from tqdm import tqdm
from torch import LongTensor, Tensor
from pycocotools.coco import COCO
from .dataset import Dataset
from .util import download_from_url, extract_zip
from .vocab import Vocab

DATASET_PATH = Path(__file__).parents[2]/'data'/'raw'/'coco2014'
METADATA_PATH = DATASET_PATH/'metadata.toml'
TRAIN_IMG_FILENAME = 'COCO_train2014_{}.jpg'
VAL_IMG_FILENAME = 'COCO_val2014_{}.jpg'
TEST_IMG_FILENAME = 'COCO_test2014_{}.jpg'

class COCO2014(Dataset):
    """A subclass defining the COCO 2014 dataset."""
    def __init__(self, is_validation_set: bool = False, is_test_set: bool = False) -> None:
        super(COCO2014, self).__init__(DATASET_PATH, METADATA_PATH, is_validation_set, is_test_set)

        self.vocab: Optional[Vocab] = None
        self.load_dataset()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, LongTensor], Tensor]:
        if self.is_test_set:
            img_filename = self._create_img_filename(self.samples[idx][0], is_img_id=False)
        else:
            img_filename = self._create_img_filename(str(self.samples[idx][0]))

        with Image.open(img_filename).convert('RGB') as img:
            img = self.transforms(img)

            if self.is_test_set:
                return img

            return img, LongTensor(self.vocab.preprocess_annotation(self.samples[idx][-1])) #type: ignore #pylint: disable=line-too-long

    def load_dataset(self) -> None:
        metadata = toml.load(self.metadata_path)
        ds_url, anns_url = metadata['train_ds_url'], metadata['anns_url']
        ds_path = os.path.join(self.dataset_path, metadata['train_ds_filename'])
        ds_csv_filename = os.path.join(self.dataset_path, metadata['train_ds_csv_filename'])
        anns_path: Optional[str] = os.path.join(self.dataset_path, metadata['anns_filename'])

        if anns_path:
            anns_file_path: Optional[str] = os.path.join(anns_path[:-4],
                                                         metadata['train_ds_anns_filename'])

        if self.is_validation_set:
            ds_url = metadata['val_ds_url']
            ds_path = os.path.join(self.dataset_path, metadata['val_ds_filename'])
            ds_csv_filename = os.path.join(self.dataset_path, metadata['val_ds_csv_filename'])

            if anns_path:
                anns_file_path = os.path.join(anns_path[:-4], metadata['val_ds_anns_filename'])
        elif self.is_test_set:
            ds_url = metadata['test_ds_url']
            ds_path = os.path.join(self.dataset_path, metadata['test_ds_filename'])
            ds_csv_filename = os.path.join(self.dataset_path, metadata['test_ds_csv_filename'])
            anns_url, anns_path, anns_file_path = None, None, None

        # Download dataset
        download_from_url(ds_url, ds_path)

        if not os.path.isdir(ds_path[:-4]):
            extract_zip(ds_path, str(self.dataset_path))

        # Download annotations
        if anns_url and anns_path:
            download_from_url(anns_url, anns_path)

            if not os.path.isdir(anns_path[:-4]):
                extract_zip(anns_path, str(self.dataset_path))

        if os.path.isfile(ds_csv_filename):
            samples = pd.read_csv(ds_csv_filename)
        else:
            samples = self._create_img_to_lbl_csv(ds_path[:-4],
                                                  anns_file_path,
                                                  ds_csv_filename)

        self.samples = samples.values.tolist()

    def _create_img_filename(self, img: str, is_img_id: bool = True) -> str:
        """Creates the complete image filename based on the image ID."""
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

    def _create_img_to_lbl_csv(self, #pylint: disable=no-self-use
                               ds_path: str,
                               anns_file_path: Optional[str],
                               csv_filename: str) -> pd.DataFrame:
        """Creates a CSV file used for mapping images to its labels."""
        img_caps_mapping: DefaultDict[str, list] = defaultdict(list)
        image_ids, annotations = [], []

        if anns_file_path:
            coco_captions = COCO(anns_file_path)

            print('Mapping images to annotations...')

            for image_caption in coco_captions.loadAnns(coco_captions.getAnnIds()):
                img_caps_mapping[image_caption['image_id']].append(image_caption['caption'])

            for img_id, captions in tqdm(iterable=img_caps_mapping.items(),
                                         total=len(img_caps_mapping.items())):
                image_ids.append(img_id)
                annotations.append(captions[0])

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
