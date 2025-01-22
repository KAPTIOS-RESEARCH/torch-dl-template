import logging, wandb
from argparse import ArgumentParser
from src.utils.config import load_config_file, instanciate_module
from src.core.experiment import AbstractExperiment
from datetime import datetime

if __name__ == "__main__":

    project_name = "{{cookiecutter.project_name}}"

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {project_name} - %(levelname)s - %(message)s'
    )

    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    config = load_config_file(args.config_path)

    # LOGGER INIT

    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
    wandb.init(project=project_name, name=project_name, config=config, id=date_time)

    # EXPERIMENT INIT

    experiment_cls = config['experiment']['class_name']
    experiment_md = config['experiment']['module_name']
    experiment: AbstractExperiment = instanciate_module(
        experiment_md, 
        experiment_cls, 
        {"config": config}
    )

    experiment.run_training()
    
    