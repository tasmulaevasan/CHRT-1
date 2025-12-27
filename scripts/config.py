import os
from pathlib import Path
from omegaconf import OmegaConf

def load():
    cfg = OmegaConf.load("config.yaml")

    is_kaggle = Path("/kaggle/input").exists()
    is_colab = Path("/content/drive").exists()

    if is_kaggle:
        BASE = Path("/kaggle/working/chrt")
    elif is_colab:
        BASE = Path("/content/drive/MyDrive/CHRT-1")
    else:
        BASE = Path(os.environ.get("CHRT-1", str(Path.cwd())))

    for key, rel_path in cfg.paths.items():
        cfg.paths[key] = BASE / Path(rel_path)
    return cfg

def update_version(mode):
    global cfg
    cfg.ver[mode] += 1
    paths_as_str = {k: str(v) for k, v in cfg.paths.items()}
    cfg_for_save = cfg.copy()
    cfg_for_save.paths = paths_as_str
    config_path = Path(__file__).parent.parent / "config.yaml"
    OmegaConf.save(cfg_for_save, config_path)

cfg = load()