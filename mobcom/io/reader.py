import pandas as pd
from torchvision import datasets
from torch.utils.data import Dataset
from torch import Tensor
from torchvision.transforms import ToTensor

DATA_FILE_PATH = "data/train.csv"

class Reader:
    @staticmethod
    def default_data():
        data = pd.read_csv(DATA_FILE_PATH) 
        dataset = Tensor(data)

        return dataset
        
    @staticmethod
    def download_data():
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