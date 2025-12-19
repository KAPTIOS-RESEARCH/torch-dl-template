from comet_ml import Experiment
import logging
import time
import torch
import os
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from src.utils.config import instanciate_module
from src.optimisation.early_stopping import EarlyStopping
from src.utils.summary import get_acronym

class BaseTrainer:
    def __init__(self, model: nn.Module, parameters: dict, device: str, comet_exp: Experiment):
        self.model = model
        self.parameters = parameters
        self.device = device
        self.comet_exp = comet_exp

        self.early_stop = EarlyStopping(
            patience=parameters.get('early_stopping_patience')
        ) if parameters.get('early_stopping_patience') else None

        self.optimizer = self._init_optimizer(parameters)
        self.lr_scheduler = self._init_scheduler(parameters)
        self.criterion = instanciate_module(
            parameters['loss']['module_name'],
            parameters['loss']['class_name'],
            parameters['loss']['parameters']
        )

    # ----------------------------
    # Helper to init optimizer
    # ----------------------------
    def _init_optimizer(self, parameters):
        if parameters.get('optimizer') is not None:
            return instanciate_module(
                parameters['optimizer']['module_name'],
                parameters['optimizer']['class_name'],
                {**parameters['optimizer']['parameters'], "params": self.model.parameters()}
            )
        else:
            return Adam(
                self.model.parameters(),
                lr=parameters['lr'],
                weight_decay=parameters['weight_decay']
            )

    # ----------------------------
    # Helper to init scheduler
    # ----------------------------
    def _init_scheduler(self, parameters):
        if parameters.get('lr_scheduler') is not None:
            return instanciate_module(
                parameters['lr_scheduler']['module_name'],
                parameters['lr_scheduler']['class_name'],
                {**parameters['lr_scheduler']['parameters'], 'optimizer': self.optimizer}
            )
        else:
            return None

    # ----------------------------
    # Core epoch runner
    # ----------------------------
    def _run_epoch(self, loader, epoch=None, num_epochs=None, evaluator=None, phase: str = 'training'):
        is_train = phase == 'training'
        self.model.train(mode=is_train)
        total_loss = 0.0
        current_lr = self.optimizer.param_groups[0]['lr']

        desc = (
            f"Epoch [{epoch + 1}/{num_epochs}] - {phase} - LR {current_lr:.6f}"
            if epoch is not None
            else f"Running {phase} phase"
        )

        has_metrics = evaluator is not None
        metric_sums = {k: 0.0 for k in evaluator.metrics} if has_metrics else {}

        with tqdm(loader, desc=desc, ncols=150) as pbar:
            for i, (data, targets) in enumerate(loader, start=1):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

                if has_metrics:
                    with torch.no_grad():
                        batch_metrics = evaluator.compute_metrics(outputs, targets)
                        for k, v in batch_metrics.items():
                            metric_sums[k] += v.item() if isinstance(v, torch.Tensor) else v

                postfix = {'Loss': f'{total_loss / i:.6f}'}

                if has_metrics:
                    avg_metrics = {k: metric_sums[k] / i for k in metric_sums}
                    for k, avg_val in avg_metrics.items():
                        postfix[get_acronym(k)] = f"{avg_val:.4f}"

                pbar.set_postfix(postfix)
                pbar.update(1)

        avg_loss = total_loss / len(loader)
        avg_metrics = (
            {k: v / len(loader) for k, v in metric_sums.items()}
            if has_metrics
            else {}
        )

        return avg_loss, avg_metrics


    # ----------------------------
    # Centralized logging helper
    # ----------------------------
    def _log_metrics(self, epoch, loss=None, metrics=None):
        metrics_to_log = {"loss": loss, **metrics}
        self.comet_exp.log_metrics(metrics_to_log, step=epoch, epoch=epoch, include_context=True)

    # ----------------------------
    # Phase wrappers
    # ----------------------------
    def train(self, train_loader, epoch=None, num_epochs=None, evaluator=None):
        with self.comet_exp.train():
            loss, metrics = self._run_epoch(train_loader, epoch, num_epochs, evaluator, phase='training')
            self._log_metrics(epoch, loss, metrics)
        return loss, metrics

    def validation(self, val_loader, epoch=None, num_epochs=None, evaluator=None):
        with self.comet_exp.validate():
            loss, metrics = self._run_epoch(val_loader, epoch, num_epochs, evaluator, phase='validation')
            self._log_metrics(epoch, loss, metrics)
        return loss, metrics

    def test(self, test_loader, epoch=None, num_epochs=None, evaluator=None):
        with self.comet_exp.test():
            _, metrics = self._run_epoch(test_loader, epoch, num_epochs, evaluator, phase='test')
            m = {}
            for k, v in metrics.items():
                m[f"test_{k}"] = v 
            self.comet_exp.log_others(m)
        return None, metrics

   
    def fit(self, train_dl, val_dl, test_dl, log_dir: str, evaluator):
        start_time = time.time()
        num_epochs = self.parameters['num_epochs']
        best_loss = None
        final_epoch = 0

        for epoch in range(num_epochs):
            final_epoch = epoch
            train_loss, _ = self.train(train_dl, epoch, num_epochs, evaluator)
            val_loss, _ = self.validation(val_dl, epoch, num_epochs, evaluator)

            if self.lr_scheduler:
                self.lr_scheduler.step(train_loss)

            if self.early_stop:
                self.early_stop(self.model, val_loss, log_dir, epoch)
                if self.early_stop.stop:
                    logging.info(f"Val loss did not improve for {self.early_stop.patience} epochs.")
                    logging.info("Training stopped by early stopping mechanism.")
                    break
            else:
                if best_loss is None or val_loss < best_loss:
                    best_loss = val_loss
                    model_object = {
                        'model_state_dict': self.model.state_dict(),
                        'min_loss': best_loss,
                        'last_epoch': epoch
                    }
                    torch.save(model_object, os.path.join(log_dir, 'best_model.pth'))

        _, test_metrics = self.test(test_dl, final_epoch, num_epochs, evaluator)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time:.2f} seconds.")
        return test_metrics