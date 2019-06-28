"""This contains a script on how to train a model by specifying arguments."""

from argparse import ArgumentParser, Namespace
from importlib import import_module
from pathlib import Path
import sys
import os
import json
import toml

RAW_DATA_PATH = str(Path(__file__).resolve().parents[1]/'data'/'raw')
DS_MODULE = 'image_caption_generator.datasets'
MODELS_MODULE = 'image_caption_generator.models'
NETWORKS_MODULE = 'image_caption_generator.networks'

def main(args: Namespace) -> None:
    """The main entrypoint of the program."""

    # Create datasets
    ds_module = import_module(DS_MODULE)
    train_ds = getattr(ds_module, args.dataset)()
    val_ds = getattr(ds_module, args.dataset)(is_validation_set=True)
    test_ds = getattr(ds_module, args.dataset)(is_test_set=True)

    # Create vocabulary
    ds_path = os.path.join(RAW_DATA_PATH, args.dataset.lower())
    metadata = toml.load(os.path.join(ds_path, 'metadata.toml'))
    vocab = getattr(ds_module, 'Vocab')(ds_path,
                                        os.path.join(ds_path, metadata['train_ds_csv_filename']),
                                        val_csv_file_path=os.path.join(ds_path, metadata['val_ds_csv_filename'])) #pylint: disable=line-too-long

    # Set vocabulary of all datasets
    train_ds.vocab = vocab
    val_ds.vocab = vocab
    test_ds.vocab = vocab

    # Create model
    network_cls = getattr(import_module(NETWORKS_MODULE), args.network)
    models_module = import_module(MODELS_MODULE)
    network_kwargs = json.loads(args.network_kwargs) if args.network_kwargs else {}
    network_kwargs = {'vocab': train_ds.vocab, **network_kwargs}
    model = getattr(models_module, args.model)(train_ds,
                                               network_cls=network_cls,
                                               network_kwargs=network_kwargs,
                                               val_ds=val_ds,
                                               test_ds=test_ds)

    # Train model
    train_dl_kwargs = json.loads(args.train_dl_kwargs)
    train_dl_kwargs = {'collate_fn': train_ds.vocab.pad_annotations, **train_dl_kwargs}
    model.train(args.num_epochs, train_dl_kwargs)

def _parse_args() -> Namespace:
    """Parses and returns parsed command-line arguments."""
    arg_parser = ArgumentParser(description='Trains a given model using a specific dataset.')

    # Required arguments
    arg_parser.add_argument('model', type=str, help='the model to use')
    arg_parser.add_argument('network', type=str, help='the network for the model to use')
    arg_parser.add_argument('dataset', type=str, help='the dataset to use')
    arg_parser.add_argument('train_dl_kwargs',
                            type=str,
                            help='the keyword arguments for the training dataloader')

    # Optional arguments
    arg_parser.add_argument('--num-epochs',
                            type=int,
                            default=10,
                            help='the number of epochs to train the model for')
    arg_parser.add_argument('--val-dl-kwargs',
                            type=str,
                            help='the keyword arguments for the validation dataloader')
    arg_parser.add_argument('--test-dl-kwargs',
                            type=str,
                            help='the keyword arguments for the test dataloader')
    arg_parser.add_argument('--model-kwargs', type=str, help='the keyword arguments for the model')
    arg_parser.add_argument('--network-kwargs',
                            type=str,
                            help='the keyword arguments for the network')

    return arg_parser.parse_args()


if __name__ == '__main__':
    # Add project for paths to search for modules
    sys.path.append(str(Path(__file__).resolve().parents[1]))

    main(_parse_args())
