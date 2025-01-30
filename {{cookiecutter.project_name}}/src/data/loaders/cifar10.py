import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from . import AbstractDataloader

class CIFAR10Loader(AbstractDataloader):
    def __init__(self, data_dir: str = './data', input_size: tuple = (28, 28), batch_size: int = 8, num_workers: int = 2):
        super(CIFAR10Loader, self).__init__()
        self.data_dir = data_dir
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers 

        self.transform_no_aug = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
        ])

    def train(self):
        trainset = tv.datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=self.transform_no_aug)
        dataloader = DataLoader(trainset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True)
        return dataloader

    def val(self):
        trainset = tv.datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=self.transform_no_aug)
        dataloader = DataLoader(trainset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False,
                                drop_last=True,
                                pin_memory=True)
        return dataloader

    def test(self):
        return self.val()