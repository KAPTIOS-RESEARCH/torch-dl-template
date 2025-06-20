import torch
from torch import nn
from tqdm import tqdm
from src.core.trainer import BaseTrainer


class MedMNISTrainer(BaseTrainer):

    def __init__(self, model: nn.Module, parameters: dict, device: str):
        super(MedMNISTrainer, self).__init__(model, parameters, device)
        if not self.criterion:
            self.criterion = nn.NLLLoss(reduction='sum')

    def train(self, train_loader, evaluator):
        self.model.train()
        train_loss = 0.0
        all_preds = []
        all_targets = []
        with tqdm(train_loader, leave=False, desc="Running training phase") as pbar:
            for sample in train_loader:
                data, targets = sample
                data, targets = data.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())
                pbar.update(1)

        all_targets = torch.cat(all_targets)
        all_preds = torch.cat(all_preds)

        metrics = evaluator.compute_metrics(all_preds, all_targets)
        train_loss /= len(train_loader)
        return train_loss, metrics

    def test(self, val_loader, evaluator):
        self.model.eval()
        test_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            with tqdm(val_loader, leave=False, desc="Running testing phase") as pbar:
                for idx, sample in enumerate(val_loader):
                    data, targets = sample
                    data, targets = data.to(
                        self.device), targets.to(self.device)
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                    test_loss += loss.item()
                    all_preds.append(outputs.cpu())
                    all_targets.append(targets.cpu())
                    pbar.update(1)

        all_targets = torch.cat(all_targets)
        all_preds = torch.cat(all_preds)
        metrics = evaluator.compute_metrics(all_preds, all_targets)
        test_loss /= len(val_loader)
        return test_loss, metrics
