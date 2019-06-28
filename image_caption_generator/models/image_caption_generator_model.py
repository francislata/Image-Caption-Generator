"""This contains the image caption generator model."""

from typing import Optional, Mapping
from .model import Model
from ..networks import ResNet101LSTM
from ..datasets import Dataset

class ImageCaptionGeneratorModel(Model):
    """This contains the image caption generator model."""
    def __init__(self,
                 train_ds: Dataset,
                 network_cls: type = ResNet101LSTM,
                 network_kwargs: Optional[Mapping] = None,
                 val_ds: Optional[Dataset] = None,
                 test_ds: Optional[Dataset] = None):
        super(ImageCaptionGeneratorModel, self).__init__(network_cls,
                                                         train_ds,
                                                         val_ds=val_ds,
                                                         test_ds=test_ds,
                                                         network_kwargs=network_kwargs)

    def learning_rate(self):
        return 1e-2
