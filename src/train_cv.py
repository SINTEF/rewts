import darts.models.forecasting.torch_forecasting_model
import pandas as pd
import pyrootutils
import pytorch_lightning.loggers.mlflow

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import copy
import os
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict

import src.datamodules.components.timeseries_datamodule
import src.eval
import src.models.utils
import src.predict
import src.train
import src.utils.plotting
from src import utils
from src.utils.utils import call_function_with_data

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train_cv(cfg: DictConfig) -> Tuple[dict, dict]:
    """Train and evaluate a number of models over a configured set of cross validation folds.
    Metrics will be returned for each fold in addition to metrics averaged over the folds. The
    returned objects are from the last fold, i.e. objects from all folds are not saved in order to
    reduce memory requirements. This is subject to change in the future.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    assert OmegaConf.select(cfg, "cross_validation") is not None

    run_base_dir = cfg.paths.output_dir

    fold_metric_dicts = []

    cv_folds = hydra.utils.instantiate(cfg.cross_validation.folds)

    log.info(f"Training models on {len(cv_folds)} cross validation folds!")
    for fold_i, fold in enumerate(cv_folds):
        fold_cfg = copy.deepcopy(cfg)

        for split_name in fold:
            if isinstance(fold[split_name][0][0], pd.Timestamp):
                for fold_ds_i in range(len(fold[split_name])):
                    fold[split_name][fold_ds_i] = [
                        str(fold_ds) for fold_ds in fold[split_name][fold_ds_i]
                    ]
            with open_dict(fold_cfg):
                fold_cfg.datamodule.train_val_test_split[split_name] = fold[split_name]

        with open_dict(fold_cfg):
            fold_cfg.paths.output_dir = os.path.join(run_base_dir, str(fold_i))
            if OmegaConf.select(fold_cfg, "cross_validation.disable_fold_logger", default=True):
                fold_cfg.logger = None
            if OmegaConf.select(fold_cfg, "logger.mlflow") is not None:
                fold_cfg.logger.mlflow.run_name = f"{fold_cfg.logger.mlflow.run_name}_fold{fold_i}"
        os.makedirs(fold_cfg.paths.output_dir, exist_ok=True)
        shutil.copytree(
            os.path.join(run_base_dir, ".hydra"),
            os.path.join(fold_cfg.paths.output_dir, ".hydra"),
        )

        log.info(f"Training model on fold {fold_i}!")
        fold_metrics, fold_objects = src.train.train(fold_cfg)
        fold_metric_dicts.append(fold_metrics)

    metric_dict = {}
    all_metric_names = set()
    for fold_metric_dict in fold_metric_dicts:
        all_metric_names = all_metric_names.union(fold_metric_dict.keys())
    for metric_name in all_metric_names:
        if not re.search(r"series\d+$", metric_name):
            metric_dict[metric_name] = np.mean(
                [fold_metric_dicts[f_i][metric_name] for f_i in range(len(fold_metric_dicts))]
            )
        for fold_i, fold_metric_dict in enumerate(fold_metric_dicts):
            if metric_name in fold_metric_dict:
                metric_dict[f"{metric_name}_fold{fold_i}"] = fold_metric_dict[metric_name]

    logger = src.utils.instantiate_loggers(cfg.get("logger", None))
    if logger:
        for lg in logger:
            if lg:
                log.info("Logging hyperparameters!")
                lg.log_metrics({k: np.mean(v) for k, v in metric_dict.items()})
                lg.finalize("success")

    return metric_dict, fold_objects


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # train the model
    metric_dict, _ = train_cv(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    utils.enable_eval_resolver()
    utils.enable_powerset_resolver()
    main()
