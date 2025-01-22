import wandb
from datetime import datetime

class WnBLogger(object):
    def __init__(self, project_name, experiment_name, config):
        now = datetime.now()
        date_time = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
        wandb.init(project=project_name, name=experiment_name, config=config, id=date_time)
    
    def watch(self, model):
        wandb.watch(model)