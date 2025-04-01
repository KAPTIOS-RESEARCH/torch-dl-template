from torch.utils.data import Subset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

class DefaultDataloader(object):
    def __init__(self, 
                 train_data_dir: str, 
                 val_data_dir: str,
                 batch_size: int = 4, 
                 num_workers: int = 4,
                 debug: bool = True):
        
        super(DefaultDataloader, self).__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.debug = debug
        self.batch_size = batch_size
        self.num_workers = num_workers 
        
        self.transform = ToTensor()

    def train(self):
        train_dataset = MNIST(
            root=self.train_data_dir,
            train=True,
            transform=self.transform,
            download=True
        )
        
        if self.debug:
            train_dataset = Subset(train_dataset, range(self.batch_size * 2))
        
        dataloader = DataLoader(train_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=True,
                                pin_memory=True)
        return dataloader

    def val(self):
        val_dataset = MNIST(
            root=self.train_data_dir,
            train=False,
            transform=self.transform,
            download=True
        )
        
        if self.debug:
            val_dataset = Subset(val_dataset, range(self.batch_size * 2))
            
        dataloader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=True,
                                pin_memory=True)
        return dataloader

    def test(self):
        return self.val()