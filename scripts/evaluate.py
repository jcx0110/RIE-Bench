import random
import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.verbose = True
import hydra
from hydra.utils import instantiate
import os
import sys

# Ensure project root is on path (for alfred.* imports when run via conda run / from scripts/)
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

# #region agent log
def _dbg_log(msg, data=None, hid="H1"):
    try:
        import json
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        p = os.path.join(_root, ".cursor", "debug.log")
        with open(p, "a") as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"import-debug","hypothesisId":hid,"location":"scripts/evaluate.py","message":msg,"data":data or {},"timestamp":__import__("time").time()}) + "\n")
    except Exception:
        pass
# #endregion

sys.path.insert(0, '.')
sys.path.insert(0, '..')
sys.path.insert(0, 'src')
sys.path.insert(0, './alfred')

# #region agent log
cwd = os.getcwd()
_dbg_log("cwd and path before import", {"cwd": cwd, "sys_path_first10": list(sys.path)[:10]}, "H1")
script_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else cwd
p_alfred_data_preprocess = os.path.join(proj_root, "alfred", "data", "preprocess.py")
p_data_raw_alfred_preprocess = os.path.join(proj_root, "data", "raw", "alfred", "preprocess.py")
_dbg_log("path check", {"script_dir": script_dir, "proj_root": proj_root, "exists_alfred_data_preprocess": os.path.exists(p_alfred_data_preprocess), "exists_data_raw_alfred_preprocess": os.path.exists(p_data_raw_alfred_preprocess), "path_alfred_data": p_alfred_data_preprocess, "path_data_raw": p_data_raw_alfred_preprocess}, "H2")
_dbg_log("__file__", {"__file__": __file__, "abspath_file": os.path.abspath(__file__)}, "H3")
# #endregion

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
    print(cfg)

    from reibench.utils.config_mapper import get_data_types
    
    random.seed(cfg.planner.random_seed)
    torch.manual_seed(cfg.planner.random_seed)
    np.random.seed(cfg.planner.random_seed)

    # Use config mapper to get data_types from new or old format
    data_types = get_data_types(cfg)
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
