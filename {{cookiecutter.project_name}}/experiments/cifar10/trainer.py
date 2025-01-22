import torch                    
from torch import nn
from tqdm import tqdm
from src.core.trainer import BaseTrainer

class CIFAR10Trainer(BaseTrainer):

    def __init__(self, model: nn.Module, parameters: dict, device: str):
        super(CIFAR10Trainer, self).__init__(model, parameters, device)

    def train(self, train_loader):
        self.model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        with tqdm(train_loader, leave=False, desc="Running training phase") as pbar:
            for data, targets in train_loader:
                inputs, labels = data.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_preds.append(outputs.cpu())
                train_labels.append(labels.cpu())
                pbar.update(1)

        train_loss /= len(train_loader)
        return train_loss
    

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0.0
        test_preds = []
        test_labels = []

        with torch.no_grad():
            with tqdm(test_loader, leave=False, desc="Running testing phase") as pbar:
                for data, targets in test_loader:
                    inputs, labels = data.to(
                        self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    test_loss += loss.item()

                    test_preds.append(outputs.cpu())
                    test_labels.append(labels.cpu())
                    pbar.update(1)

        test_loss /= len(test_loader)

        return test_loss