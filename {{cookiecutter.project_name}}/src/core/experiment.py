import logging
import os
from src.utils.config import set_seed, instanciate_module
from src.utils.device import get_available_device
from src.core.trainer import BaseTrainer
from torch import nn
from src.utils.summary import print_model_size
from comet_ml import CometExperiment 
from codecarbon import EmissionsTracker

class BaseExperiment:

    def __init__(self, project_name: str, id: str, name: str, comet_exp: CometExperiment, config: dict):
        
        self.project_name = project_name
        self.id = id
        self.name = name
        self.comet_exp = comet_exp
     
        # LOGGER INIT        
        self.log_dir = os.path.join('./logs', self.name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.codeCarbonTracker = EmissionsTracker(
            experiment_id=self.id,
            experiment_name=self.name,
            output_dir=self.log_dir,
            output_file='emissions.csv',
            log_level='error',
            measure_power_secs=10,
            save_to_file=True,
        )
        
        set_seed(config['seed'])

        logging.info(
            'Initialization of the experiment - {}'.format(self.name))
        self.device = get_available_device()
        logging.info(f'Experiments running on device : {self.device}')
        # CORE INIT
        self.model = self.load_model(config['model'])
        self.dataloader = self.load_dataloader(config['dataloader'])
        self.trainer = self.load_trainer(config['trainer'])
        self.evaluator = self.load_evaluator(config['evaluator']) if 'evaluator' in config else None

    def load_model(self, model_config) -> nn.Module:
        md_name = model_config['module_name']
        cls_name = model_config['class_name']
        params = model_config['parameters']
        model = instanciate_module(md_name, cls_name, params)
        model.to(self.device)
        print_model_size(model)
        return model

    def load_dataloader(self, dataloader_config):
        md_name = dataloader_config['module_name']
        cls_name = dataloader_config['class_name']
        params = dataloader_config['parameters']
        return instanciate_module(md_name, cls_name, params)

    def load_trainer(self, trainer_config) -> BaseTrainer:
        md_name = trainer_config['module_name']
        cls_name = trainer_config['class_name']
        params = trainer_config['parameters']
        return instanciate_module(
            md_name,
            cls_name,
            {
                "device": self.device,
                "model": self.model,
                "parameters": {**params},
                "comet_exp": self.comet_exp
            }
        )

    def load_evaluator(self, evaluator_config):
        md_name = evaluator_config['module_name']
        cls_name = evaluator_config['class_name']
        params = evaluator_config['parameters']
        return instanciate_module(md_name, cls_name, params)

    
    def run(self):
        self.codeCarbonTracker.start_task('data')
        train_dl = self.dataloader.train()
        val_dl = self.dataloader.val()
        test_dl = self.dataloader.test()
        data_emissions = self.codeCarbonTracker.stop_task()
        self.codeCarbonTracker.start_task('training')
        self.trainer.fit(train_dl, val_dl, test_dl, self.log_dir, self.evaluator)
        training_emissions = self.codeCarbonTracker.stop_task()
        self.codeCarbonTracker.stop()
        self.comet_exp.log_metric("training_energy", training_emissions.energy_consumed)
        self.comet_exp.log_metric("training_emissions", training_emissions.emissions)
        self.comet_exp.log_metric("data_energy", data_emissions.energy_consumed)
        self.comet_exp.log_metric("data_emissions", data_emissions.emissions)
        return data_emissions, training_emissions