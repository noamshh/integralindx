import yaml
from pathlib import Path

# get project root (assuming this file is at src/utils/paths.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent

def _convert_to_paths(obj):
    if isinstance(obj, dict):
        return {key: _convert_to_paths(value) for key, value in obj.items()}
    elif isinstance(obj, str):
        return PROJECT_ROOT / obj
    else:
        return obj

def get_paths():
    config_file = PROJECT_ROOT / 'config' / 'paths.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return _convert_to_paths(config)