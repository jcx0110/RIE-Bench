import random
import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.verbose = True
import hydra
from hydra.utils import instantiate
import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')
sys.path.insert(0, 'src')
sys.path.insert(0, './alfred')

from reibench.alfred_evaluator_human import AlfredEvaluator
from dotenv import load_dotenv
load_dotenv() 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
load_dotenv(override=False)
load_dotenv()
gpu = os.getenv("CUDA_VISIBLE_DEVICES")
print(f"Using GPU: {gpu}")


@hydra.main(version_base=None, config_path="../conf", config_name="config_alfred")
def main(cfg):
    print(cfg)

    random.seed(cfg.planner.random_seed)
    torch.manual_seed(cfg.planner.random_seed)
    np.random.seed(cfg.planner.random_seed)

    data_types = cfg.data_types
    for data_type in data_types:
        print(f"Evaluating data_type: {data_type}")
        evaluator = None
        cfg.data_type = data_type
        if cfg.name == 'alfred':
            evaluator = AlfredEvaluator(cfg)
        else:
            raise ValueError("Unknown configuration name. Must be 'alfred' or 'wah'.")
        evaluator.evaluate()
            

    
    
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available! GPU is being used.")
    
    main()  
