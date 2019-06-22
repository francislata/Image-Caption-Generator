"""This contains the base class that represents any dataset used."""

from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class Dataset(TorchDataset):
    """Base class of different datasets which handles dataset download, extraction and usage."""
    def __init__(self,
                 dataset_path,
                 metadata_path,
                 is_validation_set=False,
                 is_test_set=False,
                 img_size=(226, 226)):
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

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def load_dataset(self):
        """Loads (and optionally downloads) the dataset."""
        raise NotImplementedError

    def create_dataloader(self, **kwargs):
        """Returns a dataloader that represents the dataset."""
        return DataLoader(self, **kwargs)
