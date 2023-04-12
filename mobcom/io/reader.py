import pandas as pd
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

DATA_FILE_PATH = "../../data/train.csv"

class Reader:
    def default_data(self):
        data = Dataset(DATA_FILE_PATH) 
        

    def download_data(self):
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )

        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )

        return training_data, test_data