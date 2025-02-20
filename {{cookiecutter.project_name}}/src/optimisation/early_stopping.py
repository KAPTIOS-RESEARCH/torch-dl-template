import torch, os, logging

class EarlyStopping:

    def __init__(self, patience=5, delta=0, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.stop = False

    def __call__(self, model: torch.nn.Module, val_loss: float, log_dir: str):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
        else:
            self.counter += 1
            if self.verbose:
                logging.info(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.stop = True