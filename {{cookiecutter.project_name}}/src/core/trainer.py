import wandb, torch, os, logging
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from src.utils.config import instanciate_module

class EarlyStopping:

    def __init__(self, patience=5, delta=0, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.stop = False

    def __call__(self, model: nn.Module, val_loss: float, log_dir: str):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            checkpoint = {
                'net': model.state_dict(),
                'min_test_loss': self.best_loss,
            }
            torch.save(checkpoint, os.path.join(log_dir, 'best_model.pth'))
        else:
            self.counter += 1
            if self.verbose:
                logging.info(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.stop = True

            
class BaseTrainer(object):

    def __init__(self, model: nn.Module, parameters: dict, device: str):
        self.model = model
        self.parameters = parameters
        self.device = device
        self.early_stop = EarlyStopping(patience=parameters['early_stopping_patience'])
        
        # OPTIMIZER
        self.optimizer = Adam(
            self.model.parameters(), 
            lr=parameters['lr'],
            weight_decay=parameters['weight_decay']
        )
        
        # LR SCHEDULER
        self.lr_scheduler = None
        lr_scheduler_type = parameters['lr_scheduler'] if 'lr_scheduler' in parameters.keys() else 'none'

        if lr_scheduler_type == 'cosine':
            self.lr_scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=100)
        elif lr_scheduler_type == 'plateau':
            self.lr_scheduler = ReduceLROnPlateau(optimizer=self.optimizer, mode='min', factor=0.1)
        elif lr_scheduler_type == 'exponential':
            self.lr_scheduler = ExponentialLR(optimizer=self.optimizer, gamma=0.97)

        # LOSS FUNCTION
        self.criterion = instanciate_module(parameters['loss']['module_name'],
                                   parameters['loss']['class_name'], 
                                   parameters['loss']['parameters'])

    def train(self, dl: DataLoader):
        raise NotImplementedError
    
    def test(self, dl: DataLoader):
        raise NotImplementedError
    
    def fit(self, train_dl, test_dl, log_dir: str):
        num_epochs = self.parameters['num_epochs']
        for epoch in range(num_epochs):
            train_loss = self.train(train_dl)
            test_loss = self.test(test_dl)
            wandb.log({"Train/Loss_": train_loss, '_step_': epoch})
            wandb.log({"Test/Loss_": test_loss, '_step_': epoch})
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(test_loss)

            self.early_stop(self.model, test_loss, log_dir)

            logging.info(
                f"Epoch {epoch+1} / {num_epochs} -  Train/Test Loss: {train_loss:.4f} | {test_loss:4f}")

            if self.early_stop.stop:
                logging.info(
                    f"Val loss did not improve for {self.early_stop.patience} epochs.")
                logging.info('Training stopped by early stopping mecanism.')
                break
