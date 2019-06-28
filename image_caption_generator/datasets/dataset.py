"""This contains the base class that represents any dataset used."""

from typing import Tuple, List, Union, Any, Optional
from pathlib import Path
from torch import LongTensor, Tensor
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from .vocab import Vocab

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class Dataset(TorchDataset):
    """Base class of different datasets which handles dataset download, extraction and usage."""
    def __init__(self,
                 dataset_path: Path,
                 metadata_path: Path,
                 is_validation_set: bool = False,
                 is_test_set: bool = False,
                 img_size: Tuple[int, int] = (226, 226)) -> None:
        self.dataset_path = dataset_path
        self.metadata_path = metadata_path
        self.is_validation_set = is_validation_set
        self.is_test_set = is_test_set
        self.samples: List[Tuple[str, str]] = []
        self.transforms = Compose([
            Resize(img_size),
            ToTensor(),
            Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
        self.vocab: Optional[Vocab] = None

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, LongTensor], Tensor]:
        raise NotImplementedError

    def load_dataset(self) -> None:
        """Loads (and optionally downloads) the dataset."""
        raise NotImplementedError

    def create_dataloader(self, **kwargs: Any) -> DataLoader:
        """Returns a dataloader that represents the dataset."""
        return DataLoader(self, **kwargs)
