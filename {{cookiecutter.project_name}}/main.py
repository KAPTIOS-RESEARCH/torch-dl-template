import os, logging, warnings, comet_ml
warnings.filterwarnings("ignore")
import torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from uuid import uuid4
from datetime import datetime
from argparse import ArgumentParser
from src.utils.config import load_config_file, instanciate_module
from src.core.experiment import BaseExperiment
from dotenv import load_dotenv

if __name__ == "__main__":

    project_name = "{{cookiecutter.project_name}}"

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {project_name} Model Training - %(levelname)s - %(message)s'
    )
    
    logging.getLogger("codecarbon").setLevel(logging.ERROR)
    logging.getLogger("comet_ml").setLevel(logging.ERROR)
    
    if os.path.exists(".env"):
        load_dotenv(".env")
        logging.info("Loaded .env")
    elif os.path.exists(".env.example"):
        load_dotenv(".env.example")
        logging.info("Loaded .env.example")
    else:
        logging.error("No .env or .env.example file found")
    
    comet_api_key = os.getenv("COMET_API_KEY")
    comet_workspace = os.getenv("COMET_WORKSPACE")

    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    config = load_config_file(args.config_path)
    
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
    experiment_id = str(uuid4())[:4]
    experiment_name = f"{config['name']}_{experiment_id}"
    
    comet_experiment = comet_ml.start(
            api_key=comet_api_key,
            project_name=project_name,
            workspace=comet_workspace,
            online=comet_api_key is not None,
    )
    comet_experiment.set_name(experiment_name)
        
    experiment_cls = config['experiment']['class_name']
    experiment_md = config['experiment']['module_name']
    experiment: BaseExperiment = instanciate_module(
        experiment_md,
        experiment_cls,
        {
            "project_name": project_name, 
            "id": experiment_id,
            "name": experiment_name, 
            "comet_exp": comet_experiment, 
            "config": config
        }
    )
    data_emissions, training_emissions = experiment.run()
    model_file_path = os.path.join(experiment.log_dir, 'best_model.pth')
    logging.info(f"✅ Training done. Model was saved in {model_file_path}")
