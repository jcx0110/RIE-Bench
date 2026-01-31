"""
Configuration mapper for converting between new and old configuration formats.
Maintains backward compatibility while supporting the new modular config structure.
"""

from omegaconf import DictConfig


def get_planner_framework(cfg: DictConfig) -> str:
    """
    Get planner framework from new or old config format.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Framework name (saycan, LLM+P, dag_plan, hpe_plan)
    """
    # New format: framework.planner_framework
    if hasattr(cfg, 'framework') and hasattr(cfg.framework, 'planner_framework'):
        return cfg.framework.planner_framework
    
    # Old format: planner.planner_framework
    if hasattr(cfg, 'planner') and hasattr(cfg.planner, 'planner_framework'):
        return cfg.planner.planner_framework
    
    # Default
    return "saycan"


def get_model_name(cfg: DictConfig) -> str:
    """
    Get model name from new or old config format.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Model name string
    """
    # New format: model.model_name
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'model_name'):
        return cfg.model.model_name
    
    # Old format: planner.model_name
    if hasattr(cfg, 'planner') and hasattr(cfg.planner, 'model_name'):
        return cfg.planner.model_name
    
    # Default
    return "meta-llama/Llama-3.1-8B-Instruct"


def get_prompting_method(cfg: DictConfig) -> dict:
    """
    Get prompting method configuration from new or old config format.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Dictionary with aware_hint, COT, TOCC, ICL flags
    """
    result = {
        'aware_hint': False,
        'COT': False,
        'TOCC': False,
        'ICL': False
    }
    
    # New format: method.*
    if hasattr(cfg, 'method'):
        if hasattr(cfg.method, 'aware_hint'):
            result['aware_hint'] = cfg.method.aware_hint
        if hasattr(cfg.method, 'COT'):
            result['COT'] = cfg.method.COT
        if hasattr(cfg.method, 'TOCC'):
            result['TOCC'] = cfg.method.TOCC
        if hasattr(cfg.method, 'ICL'):
            result['ICL'] = cfg.method.ICL
    
    # Old format: prompting_method.*
    if hasattr(cfg, 'prompting_method'):
        if hasattr(cfg.prompting_method, 'aware_hint'):
            result['aware_hint'] = cfg.prompting_method.aware_hint
        if hasattr(cfg.prompting_method, 'COT'):
            result['COT'] = cfg.prompting_method.COT
        if hasattr(cfg.prompting_method, 'TOCC'):
            result['TOCC'] = cfg.prompting_method.TOCC
    
    return result


def get_data_type(cfg: DictConfig) -> str:
    """
    Get data type from new or old config format.
    Converts re_level + context_type to 'X-Y' format.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Data type string (e.g., '1-1', '2-3')
    """
    # New format: task_difficulty.*
    if hasattr(cfg, 'task_difficulty'):
        if hasattr(cfg.task_difficulty, 'data_type'):
            return cfg.task_difficulty.data_type
        
        # Convert re_level + context_type to data_type
        if hasattr(cfg.task_difficulty, 're_level') and hasattr(cfg.task_difficulty, 'context_type'):
            re_level_map = {'explicit': '1', 'mixed': '2', 'implicit': '3'}
            context_type_map = {'standard': '1', 'noised': '2', 'short': '3'}
            
            re_level = cfg.task_difficulty.re_level
            context_type = cfg.task_difficulty.context_type
            
            if re_level in re_level_map and context_type in context_type_map:
                return f"{re_level_map[re_level]}-{context_type_map[context_type]}"
    
    # Old format: data_type
    if hasattr(cfg, 'data_type') and cfg.data_type is not None:
        return str(cfg.data_type)
    
    # Default
    return "1-1"


def get_data_types(cfg: DictConfig) -> list:
    """
    Get list of data types from config.
    
    Args:
        cfg: Configuration object
        
    Returns:
        List of data type strings
    """
    # Old format: data_types list
    if hasattr(cfg, 'data_types') and cfg.data_types is not None:
        return cfg.data_types
    
    # If only single data_type, convert to list
    data_type = get_data_type(cfg)
    return [data_type]


def convert_re_level_to_number(re_level: str) -> int:
    """Convert re_level string to number (explicit=1, mixed=2, implicit=3)."""
    mapping = {'explicit': 1, 'mixed': 2, 'implicit': 3}
    return mapping.get(re_level, 1)


def convert_context_type_to_number(context_type: str) -> int:
    """Convert context_type string to number (standard=1, noised=2, short=3)."""
    mapping = {'standard': 1, 'noised': 2, 'short': 3}
    return mapping.get(context_type, 1)


def get_model_config(cfg: DictConfig) -> dict:
    """
    Get model configuration from new or old format.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Dictionary with model configuration
    """
    result = {}
    
    # New format: model.*
    if hasattr(cfg, 'model'):
        if hasattr(cfg.model, 'model_name'):
            result['model_name'] = cfg.model.model_name
        if hasattr(cfg.model, 'use_accelerate_device_map'):
            result['use_accelerate_device_map'] = cfg.model.use_accelerate_device_map
        if hasattr(cfg.model, 'load_in_8bit'):
            result['load_in_8bit'] = cfg.model.load_in_8bit
        if hasattr(cfg.model, 'device'):
            result['device'] = cfg.model.device
        if hasattr(cfg.model, 'hf_auth_token'):
            result['hf_auth_token'] = cfg.model.hf_auth_token
        if hasattr(cfg.model, 'openai_api_key'):
            result['openai_api_key'] = cfg.model.openai_api_key
    
    # Old format: planner.*
    if hasattr(cfg, 'planner'):
        if 'model_name' not in result and hasattr(cfg.planner, 'model_name'):
            result['model_name'] = cfg.planner.model_name
        if 'use_accelerate_device_map' not in result and hasattr(cfg.planner, 'use_accelerate_device_map'):
            result['use_accelerate_device_map'] = cfg.planner.use_accelerate_device_map
        if 'device' not in result and hasattr(cfg.planner, 'device'):
            result['device'] = cfg.planner.device
        if 'hf_auth_token' not in result and hasattr(cfg.planner, 'hf_auth_token'):
            result['hf_auth_token'] = cfg.planner.hf_auth_token
        if 'openai_api_key' not in result and hasattr(cfg.planner, 'openai_api_key'):
            result['openai_api_key'] = cfg.planner.openai_api_key
    
    return result

