import wandb
import logging
import time
import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from src.utils.config import instanciate_module
from src.optimisation.early_stopping import EarlyStopping


class BaseTrainer(object):

    def __init__(self, model: nn.Module, parameters: dict, device: str):
        self.model = model
        self.parameters = parameters
        self.device = device
        self.early_stop = EarlyStopping(
            patience=parameters['early_stopping_patience'], enable_wandb=parameters['track']) if parameters['early_stopping_patience'] else None

        # OPTIMIZER
        self.optimizer = Adam(
            self.model.parameters(),
            lr=parameters['lr'],
            weight_decay=parameters['weight_decay']
        )

        # LR SCHEDULER
        self.lr_scheduler = None
        lr_scheduler_type = parameters['lr_scheduler'] if 'lr_scheduler' in parameters.keys(
        ) else 'none'

        if lr_scheduler_type == 'plateau':
            self.lr_scheduler = ReduceLROnPlateau(
                optimizer=self.optimizer, mode='min', factor=0.1, patience=3)
        elif lr_scheduler_type == 'exponential':
            self.lr_scheduler = ExponentialLR(
                optimizer=self.optimizer, gamma=0.97)

        # LOSS FUNCTION
        self.criterion = instanciate_module(parameters['loss']['module_name'],
                                            parameters['loss']['class_name'],
                                            parameters['loss']['parameters'])

    def train(self, dl: DataLoader, evaluator):
        raise NotImplementedError

    def test(self, dl: DataLoader, evaluator):
        raise NotImplementedError

    def fit(self, train_dl, test_dl, log_dir: str, evaluator):
        start_time = time.time()
        num_epochs = self.parameters['num_epochs']
        best_loss = None
        for epoch in range(num_epochs):
            train_loss, train_metrics = self.train(train_dl, evaluator)
            test_loss, test_metrics = self.test(test_dl, evaluator)

            if self.parameters['track']:
                log_data = {
                    f"Train/{self.parameters['loss']['class_name']}": train_loss,
                    f"Test/{self.parameters['loss']['class_name']}": test_loss,
                    "_step_": epoch
                }
                if train_metrics:
                    for metric_name, value in train_metrics.items():
                        log_data[f"Train/{metric_name}"] = value
                if test_metrics:
                    for metric_name, value in test_metrics.items():
                        log_data[f"Test/{metric_name}"] = value

                wandb.log(log_data)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(test_loss)

            metric_log_str = " | ".join([
                f"{name}: {train_metrics[name]:.4f} / {test_metrics[name]:.4f}"
                for name in train_metrics
            ]) if train_metrics else None

            logging.info(
                f"Epoch {epoch + 1} / {num_epochs} - "
                f"Loss: {train_loss:.4f} / {test_loss:.4f} - "
                f"{metric_log_str}"
            )

            if self.early_stop is not None:
                self.early_stop(self.model, test_loss, log_dir, epoch)
                if self.early_stop.stop:
                    logging.info(
                        f"Val loss did not improve for {self.early_stop.patience} epochs.")
                    logging.info(
                        'Training stopped by early stopping mechanism.')
                    break
            else:
                if best_loss is None or test_loss < best_loss:
                    best_loss = test_loss
                    model_object = {
                        'weights': self.model.state_dict(),
                        'min_loss': best_loss,
                        'last_epoch': epoch
                    }
                    torch.save(model_object, os.path.join(
                        log_dir, 'best_model.pth'))

        end_time = time.time()
        logging.info(
            f"Training completed in {end_time - start_time:.2f} seconds.")

        if self.parameters['track']:
            wandb.finish()
