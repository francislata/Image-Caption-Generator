"""This contains how to manage the vocabulary of image annotations."""

import os
from typing import Optional, List, Tuple, Any, Dict
import pandas as pd
import spacy
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch import stack, Tensor
from .util import save_to_json, load_json

STOI_FILENAME = 'stoi.json'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'

class Vocab: #pylint: disable=too-few-public-methods
    """This subclass manages the vocabulary of image annotations."""
    def __init__(self,
                 ds_path: str,
                 train_csv_file_path: str,
                 val_csv_file_path: Optional[str] = None,
                 lang_id: str = 'en') -> None:
        self.nlp = spacy.load(lang_id)
        self.stoi_path = os.path.join(ds_path, STOI_FILENAME)
        self.stoi = self._load_stoi()

        if not os.path.isfile(self.stoi_path):
            self._load_vocab(train_csv_file_path)

            if val_csv_file_path:
                self._load_vocab(val_csv_file_path)

            save_to_json(self.stoi, self.stoi_path)

    def preprocess_annotation(self, annotation: List[str]) -> List[int]:
        """Preprocesses the tokenized annotation."""
        preprocessed_annotation = [self.stoi[SOS_TOKEN]]

        for token in self.nlp(annotation):
            preprocessed_annotation.append(self.stoi[token.text.lower()])

        preprocessed_annotation.append(self.stoi[EOS_TOKEN])

        return preprocessed_annotation

    def pad_annotations(self,
                        samples: Any) -> Tuple[Tensor, Tensor, List[int]]:
        """Pads a batch of annotations when iterating through using dataloader."""
        samples = sorted(samples, key=lambda x: len(x[1]), reverse=True)
        imgs = stack([x for x, _ in samples], dim=0)
        lbls = pad_sequence([y for _, y in samples], padding_value=self.stoi[PAD_TOKEN])
        lbls_lengths = [len(y) for _, y in samples]

        return imgs, lbls, lbls_lengths

    def _load_stoi(self) -> Dict[str, int]:
        """Loads the JSON file for the strings to index mapping."""
        if os.path.isfile(self.stoi_path):
            return load_json(self.stoi_path)

        return {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2}

    def _load_vocab(self, csv_file_path: str) -> None:
        """Loads (and creates the vocabulary) from dataset."""
        ds_df = pd.read_csv(csv_file_path)

        print('Building vocabulary of {}...'.format(csv_file_path))

        samples = ds_df.values.tolist()

        for sample in tqdm(iterable=samples, total=len(samples)):
            for token in self.nlp(sample[-1]):
                if not token.text.lower() in self.stoi:
                    self.stoi[token.text.lower()] = len(self.stoi)

        print('Done!\n')
