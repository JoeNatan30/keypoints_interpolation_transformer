import argparse
import os
import parseMain

import numpy as np
import pandas as pd
import wandb

CONFIG_FILENAME = "config.json"
PROJECT_WANDB = "fill_missings_transformer"
ENTITY = "joenatan30" #joenatan30
TAG = ["embedding"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[parseMain.get_default_args()], add_help=False)
    args = parser.parse_args()

    run = wandb.init(project=PROJECT_WANDB,
                     entity=ENTITY,
                     config=args,
                     name=args.experiment_name,
                     #mode="offline",
                     job_type="model-training",
                     tags=TAG,
                     save_code=True)

    run.notes = args.notes 
    config = wandb.config
    
    #files_list = ["3_train.py", "dataloader.py", "model.py", "parseMain.py"]
    #for file in files_list:
    #    wandb.run.log_code(f"./{file}")
    wandb.run.log_code(".")
    
    
    train(args)