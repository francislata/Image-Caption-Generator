"""This contains the model for the ResNet101 and LSTM network."""

from typing import Optional, Any
from .model import Model
from ..networks.resnet101_lstm import ResNet101LSTM
from ..datasets.dataset import Dataset

class ResNet101LSTMModel(Model):
    """This subclass represents the ResNet101 and LSTM network's model."""
    def __init__(self,
                 network_kwargs: Any,
                 train_ds: Dataset,
                 val_ds: Optional[Dataset] = None,
                 test_ds: Optional[Dataset] = None) -> None:
        if train_ds.vocab:
            network = ResNet101LSTM(train_ds.vocab, **network_kwargs)

            super(ResNet101LSTMModel, self).__init__(network, train_ds, val_ds, test_ds=test_ds)
