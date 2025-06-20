from torch.utils.data import Dataset
from torchvision import transforms
from src.utils.config import instanciate_module
from onnxruntime.quantization import CalibrationDataReader
import numpy as np


class MedMNISTDataset(Dataset):
    """
    PyTorch Dataset wrapper for MedMNIST datasets using dynamic instantiation.
    """

    def __init__(
        self,
        data_dir: str,
        dataset_name: str = 'BreastMNIST',
        image_size: int = 128,
        split: str = 'train',
        transform=None
    ):
        self.transform = transform or transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])

        self.dataset = instanciate_module(
            module_name='medmnist',
            class_name=dataset_name,
            params={
                "root": data_dir,
                "download": True,
                "transform": self.transform,
                "size": image_size,
                "split": split,
            }
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        return x, y.squeeze(0)
