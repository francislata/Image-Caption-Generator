from torch.utils.data import Dataset
from ast import literal_eval
import pandas as pd

class COCODataset(Dataset):
    """This subclass represents the COCO dataset."""

    def __init__(self, image_file_path, csv_file_path):
        self.image_file_path = image_file_path
        self.samples = pd.read_csv(csv_file_path).values.tolist()

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        return self.samples[index][0], literal_eval(self.samples[index][-1])
