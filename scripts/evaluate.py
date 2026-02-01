import random
import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.verbose = True
import hydra
from hydra.utils import instantiate
import os
import sys

# Suppress IPython HTML/display output to avoid "<IPython.core.display.HTML object>" spam (e.g. from ai2thor)
try:
    import IPython.display as _disp
    _disp.display = lambda *args, **kwargs: None
except Exception:
    pass

# Ensure project root is on path (for alfred.* imports when run via conda run / from scripts/)
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

sys.path.insert(0, '.')
sys.path.insert(0, '..')
sys.path.insert(0, 'src')
sys.path.insert(0, './alfred')

from reibench.alfred_evaluator import AlfredEvaluator
from dotenv import load_dotenv
load_dotenv() 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
load_dotenv(override=False)
load_dotenv()
gpu = os.getenv("CUDA_VISIBLE_DEVICES")
print(f"Using GPU: {gpu}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    from reibench.utils.config_mapper import get_data_types
    
    random.seed(cfg.planner.random_seed)
    torch.manual_seed(cfg.planner.random_seed)
    np.random.seed(cfg.planner.random_seed)

    # Use config mapper to get data_types from new or old format
    data_types = get_data_types(cfg)
    if cfg.name == 'alfred':
        evaluator = AlfredEvaluator(cfg)
    else:
        raise ValueError("Unknown configuration name. Must be 'alfred' or 'wah'.")
    for data_type in data_types:
        print(f"Evaluating data_type: {data_type}")
        cfg.data_type = data_type
        evaluator.evaluate()
            

    
    
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available! GPU is being used.")
    
    main()  
