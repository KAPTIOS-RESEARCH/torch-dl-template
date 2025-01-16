import logging
from abc import ABC, abstractmethod
from uuid import uuid4
from src.utils.config import set_seed, instanciate_module
from src.loggers.wandb import WnBLogger
from src.utils.device import get_available_device


class AbstractExperiment(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load_model(self):
        raise NotImplementedError
    
    @abstractmethod
    def load_dataloader(self):
        raise NotImplementedError
    
    @abstractmethod
    def load_trainer(self):
        raise NotImplementedError

    @abstractmethod
    def run_training(self):
        raise NotImplementedError
    
    @abstractmethod
    def run_eval(self):
        raise NotImplementedError


class BaseExperiment(AbstractExperiment):

    def __init__(self, project_name: str, config: dict):
        super().__init__()

        experiment_id = str(uuid4())[:4]
        experiment_name = f"{config['name']}_{experiment_id}"
        set_seed(config['seed'])

        logging.info('Initialization of the experiment - {}'.format(experiment_name))
        self.device = get_available_device()

        # CORE INIT
        self.model = self.load_model(config['model'])
        self.dataloader = self.load_dataloader(config['dataloader'])
        self.trainer = self.load_trainer(config['trainer'])

        # W&B
        #logger = WnBLogger(project_name, experiment_name, config = config)
        #logger.watch(self.model)


    def load_model(self, model_config):
        md_name = model_config['module_name']
        cls_name = model_config['class_name']
        params = model_config['parameters']
        model = instanciate_module(md_name, cls_name, params)
        model.to(self.device)
        return model
    
    def load_dataloader(self, dataloader_config):
        md_name = dataloader_config['module_name']
        cls_name = dataloader_config['class_name']
        params = dataloader_config['parameters']
        return instanciate_module(md_name, cls_name, params)
    
    def load_trainer(self, trainer_config):
        md_name = trainer_config['module_name']
        cls_name = trainer_config['class_name']
        params = trainer_config['parameters']
        return instanciate_module(
            md_name, 
            cls_name, 
            {
                "device": self.device, 
                "model": self.model, 
                "parameters": params
            }
        ) 

    def run_training(self):
        train_dl = self.dataloader.train()
        val_dl = self.dataloader.val()
        self.trainer.train(train_dl)

    def run_eval(self):
        test_dl = self.dataloader.test()
        pass
