import warnings, comet_ml
warnings.filterwarnings("ignore")
import logging, os, torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from uuid import uuid4
from datetime import datetime
from argparse import ArgumentParser
from src.utils.config import load_config_file, instanciate_module
from src.core.experiment import BaseExperiment

if __name__ == "__main__":

    project_name = "{{cookiecutter.project_name}}"

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {project_name} Model Training - %(levelname)s - %(message)s'
    )

    logging.getLogger("comet_ml").setLevel(logging.WARNING)

    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    config = load_config_file(args.config_path)
    
    comet_exp = None
    comet_config = config.get("comet", {})
    api_key = comet_config.get("api_key", None)

    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
    id = str(uuid4())[:4]
    experiment_name = f"{config['name']}_{id}"
    
    comet_experiment = comet_ml.start(
            api_key=api_key,
            project_name=project_name,
            workspace=comet_config.get("workspace"),
            online=api_key is not None,
    )
    comet_experiment.set_name(experiment_name)
        
    experiment_cls = config['experiment']['class_name']
    experiment_md = config['experiment']['module_name']
    experiment: BaseExperiment = instanciate_module(
        experiment_md,
        experiment_cls,
        {
            "project_name": project_name, 
            "name": experiment_name, 
            "comet_exp": comet_experiment, 
            "config": config
        }
    )

    data_emissions, training_emissions = experiment.run()
    model_file_path = os.path.join(experiment.log_dir, 'best_model.pth')
    logging.info(f'⚡ Saving energy consumption metrics to {model_file_path}')
    model_object = torch.load(model_file_path, map_location=torch.device('cpu'))
    torch.save(model_object, model_file_path)
    logging.info(f"✅ Updated model saved with energy_consumption at {model_file_path}")
