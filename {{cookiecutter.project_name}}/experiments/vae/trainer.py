import torch                    
from torch import nn
from tqdm import tqdm
from src.core.trainer import BaseTrainer
from src.optimisation.losses import VAELoss

class VAETrainer(BaseTrainer):

    def __init__(self, model: nn.Module, parameters: dict, device: str):
        super(VAETrainer, self).__init__(model, parameters, device)
        if not self.criterion:
            self.criterion = VAELoss()

    def train(self, train_loader):
        self.model.train()
        train_loss = 0.0

        with tqdm(train_loader, leave=False, desc="Running training phase") as pbar:
            for data, _ in train_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_x, x, mu, logvar = self.model(data)
                loss = self.criterion(recon_x, x, mu, logvar)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                pbar.update(1)

        train_loss /= len(train_loader)
        return train_loss
    

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0.0

        with tqdm(test_loader, leave=False, desc="Running testing phase") as pbar:
            for data, _ in test_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_x, mu, logvar = self.model(data)
                loss = self.criterion(recon_x, data, mu, logvar)
                loss.backward()
                self.optimizer.step()
                test_loss += loss.item()
                pbar.update(1)

        test_loss /= len(test_loader)
        return test_loss