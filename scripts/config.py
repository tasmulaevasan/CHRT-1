from omegaconf import OmegaConf
from pathlib import Path
import os

def load():
    BASE = Path(os.environ.get("CHRT-1", Path.cwd()))
    colab_path = Path("/content/drive/MyDrive/CHRT-1")
    if colab_path.exists() and not BASE.exists():
        BASE = colab_path

    cfg = OmegaConf.load("config.yaml")

    for key, rel_path in cfg.paths.items():
        cfg.paths[key] = Path(BASE) / str(rel_path)
    return cfg

def update_version(mode):
    cfg.ver[mode] += 1
    paths_as_str = {k: str(v) for k, v in cfg.paths.items()}
    cfg_for_save = cfg.copy()
    cfg_for_save.paths = paths_as_str
    OmegaConf.save(cfg_for_save, "config.yaml")

cfg = load()