"""This contains how to manage the vocabulary of image annotations."""

import os
import pandas as pd
import spacy
from tqdm import tqdm
from .util import save_to_json, load_json

STOI_FILENAME = 'stoi.json'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'

class Vocab: #pylint: disable=too-few-public-methods
    """This subclass manages the vocabulary of image annotations."""
    def __init__(self, ds_path, train_csv_file_path, val_csv_file_path=None, lang_id='en'):
        self.nlp = spacy.load(lang_id)
        self.stoi_path = os.path.join(ds_path, STOI_FILENAME)
        self.stoi = self._load_stoi()

        self._load_vocab(train_csv_file_path)

        if val_csv_file_path:
            self._load_vocab(val_csv_file_path)

        if not os.path.isfile(self.stoi_path):
            save_to_json(self.stoi, self.stoi_path)

    def _load_stoi(self):
        """Loads the JSON file for the strings to index mapping."""
        if os.path.isfile(self.stoi_path):
            return load_json(self.stoi_path)

        return {SOS_TOKEN: 0, EOS_TOKEN: 1}

    def _load_vocab(self, csv_file_path):
        """Loads (and creates the vocabulary) from dataset."""
        ds_df = pd.read_csv(csv_file_path)

        if not 'tokenized_annotations' in ds_df.columns:
            print('Tokenizing and building vocabulary of {}...'.format(csv_file_path))

            tokenized_annotations = []
            samples = ds_df.values.tolist()

            for sample in tqdm(iterable=samples, total=len(samples)):
                tokenized_annotation = []

                for token in self.nlp(sample[-1]):
                    tokenized_annotation.append(token)

                    if not token.text.lower() in self.stoi:
                        self.stoi[token.text.lower()] = len(self.stoi)

                tokenized_annotations.append(tokenized_annotation)

            ds_df['tokenized_annotations'] = tokenized_annotations
            ds_df.to_csv(csv_file_path, index=False)

            print('Done!\n')
