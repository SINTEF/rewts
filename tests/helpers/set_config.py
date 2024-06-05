from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict


def cfg_set_paths_and_hydra_args(cfg, output_path):
    with open_dict(cfg):
        cfg.paths.output_dir = str(output_path)
        cfg.paths.log_dir = str(output_path) + "/"
        cfg.hydra.job.num = 0
        cfg.hydra.job.id = 0

    HydraConfig().set_config(cfg)

    return cfg
