import copy

import darts.models.forecasting.torch_forecasting_model
import pyrootutils
import pytorch_lightning.loggers.mlflow

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
from typing import List, Optional, Tuple
from omegaconf import DictConfig, OmegaConf, open_dict
import math
import os
import shutil

from src import utils
import src.utils.plotting
import src.utils.model_io
from src.datamodules.components import ChunkedTimeSeriesDataModule

import src.eval
import src.train

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    datamodule = hydra.utils.instantiate(cfg.datamodule, _convert_="partial")
    if cfg.model_type == "ensemble":
        assert isinstance(datamodule, ChunkedTimeSeriesDataModule)
        datamodule.hparams.dataset_length = int(datamodule.hparams.dataset_length * cfg.HOPT_DATA_FRACTION)
        if datamodule.num_chunks is None:
            log.error("Datamodule property num_chunks is not defined. Ensure that the arguments dataset_length and chunk_length are set.")
            raise ValueError

        n_ensemble_models = datamodule.num_chunks

        metric_dict = {"ensemble_models": []}
        run_base_dir = cfg.paths.output_dir

        for model_i in range(n_ensemble_models):
            model_cfg = copy.deepcopy(cfg)
            with open_dict(model_cfg):
                if cfg.get("ensemble_disable_logger"):
                    model_cfg.logger = None
                model_cfg.datamodule.chunk_idx = model_i

                model_cfg.paths.output_dir = os.path.join(run_base_dir, str(model_i))
                os.makedirs(model_cfg.paths.output_dir, exist_ok=True)
                shutil.copytree(os.path.join(run_base_dir, ".hydra"), os.path.join(model_cfg.paths.output_dir, ".hydra"))
                #OmegaConf.save(cfg, os.path.join(cfg.paths.output_dir, ".hydra", "config.yaml"), resolve=False)
            m_dict, _ = src.train.train(model_cfg)
            metric_dict["ensemble_models"].append(m_dict)

        # update config.model_dir to point to the newly trained ensemble models
        eval_model_dir = [os.path.join(run_base_dir, str(model_i)) for model_i in range(n_ensemble_models)]
    elif cfg.model_type == "global":
        metric_dict, _ = src.train.train(cfg)
        eval_model_dir = cfg.paths.output_dir
    else:
        raise ValueError(f"Unsupported model_type {cfg.model_type}")

    with open_dict(cfg):
        cfg.datamodule = OmegaConf.merge(cfg.datamodule, cfg.eval_datamodule)
        cfg.model_dir = eval_model_dir

    eval_cfg = src.utils.load_saved_config(cfg.model_dir, cfg)
    eval_metric_dict, object_dict = src.eval.evaluate(eval_cfg)
    metric_dict.update(eval_metric_dict)

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    utils.enable_eval_resolver()
    main()
