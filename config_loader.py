# config_loader.py
import os
import glob
import yaml
from typing import List, Tuple
from config import Config

def load_configs_from_dir(config_dir: str) -> List[Tuple[str, Config]]:
    paths = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))
    if not paths:
        raise FileNotFoundError(f"No .yaml configs found in: {config_dir}")

    configs = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        cf = Config.from_dict(d)
        configs.append((p, cf))
    return configs
