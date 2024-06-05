import os
import time
from typing import List

import darts.dataprocessing
import hydra
import mlflow.exceptions
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import Logger

import src.datamodules
import src.models.utils
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("Logger config is empty.")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            if lg_conf._target_ == "pytorch_lightning.loggers.mlflow.MLFlowLogger" and not OmegaConf.select(lg_conf, "tracking_uri", default="file:").startswith("file:"):
                db_path = ":".join(lg_conf["tracking_uri"].split(":")[1:])
                if not os.path.exists(os.path.dirname(db_path)):
                    os.makedirs(os.path.dirname(db_path), exist_ok=True)
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))
    return logger


def instantiate_objects(cfg: DictConfig):
    """Helper function to initialize objects from config.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Dictionary of initialized objects.
    """
    if cfg.get("scale_model_parameters"):
        try:
            cfg = src.models.utils.scale_model_parameters(cfg)
        except ValueError as e:
            log.error(
                "Parameter scaling is not supported for this model, please set the scale_model_parameters argument to False"
            )
            raise e

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: src.datamodules.components.TimeSeriesDataModule = hydra.utils.instantiate(
        cfg.datamodule, _convert_="partial"
    )

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model, _convert_="partial")

    object_dict = {"cfg": cfg, "datamodule": datamodule, "model": model}

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    object_dict["logger"] = logger

    if cfg.get("trainer", None) is not None and src.models.utils.is_torch_model(cfg):
        log.info("Instantiating callbacks...")
        callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    else:
        trainer = None
        callbacks = None

    object_dict["trainer"] = trainer
    object_dict["callbacks"] = callbacks

    return object_dict


def instantiate_saved_objects(cfg: DictConfig):
    """Instantiates all objects needed for evaluation / prediction from the provided config using
    hydra. This includes the datamodule, model, trainer and loggers. The instantiated objects are
    returned in a dictionary.

    :param cfg: The config to use for instantiation.
    :return: A dictionary containing the instantiated objects.
    """
    if OmegaConf.select(cfg, "model_dir") is None:
        # TODO: assert not ensemble model?
        if not src.models.utils.is_local_model(OmegaConf.select(cfg, "model")):
            raise ValueError(
                "Either cfg.model_dir must be set and pointing to a log folder resulting from train.py, or a LocalForecastingModel must be set at cfg.model"
            )
        object_dict = instantiate_objects(cfg)
        object_dict["datamodule"].setup("fit")
    else:
        log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
        datamodule: src.datamodules.TimeSeriesDataModule = hydra.utils.instantiate(
            cfg.datamodule, _convert_="partial"
        )

        if cfg.get("ensemble_models"):
            assert (
                "ensemble" in cfg
            ), "When the cfg.ensemble_models argument is set, you must specify an ensemble model that these models operate within."
            assert len(cfg.model_dir) == len(cfg.ensemble_models)

            # TODO: perhaps change architecture of datamodule to transform on data request? i.e. in get_data

            # data provided to ensemble model must be non-transformed, since each model has its own pipeline.
            # thus, have to make sure that the load function does not overwrite the pipeline with the loaded one.
            # If datamodule originally has a MissingValuesFiller, we keep that to avoid nans in ensemble weight fitting.
            ensemble_pipeline = None
            if datamodule.hparams.processing_pipeline is not None:
                original_pipeline = src.datamodules.utils.ensure_pipeline_per_component(
                    datamodule.hparams.processing_pipeline, datamodule.hparams.data_variables
                )
                transformers = {
                    component: getattr(pipeline, "_transformers", [])
                    for component, pipeline in original_pipeline.items()
                }
                ensemble_pipeline = {component: None for component in original_pipeline}

                for component, component_transformers in transformers.items():
                    for transformer in component_transformers:
                        if transformer.name == "MissingValuesFiller":
                            ensemble_pipeline[component] = darts.dataprocessing.Pipeline(
                                [transformer], copy=True
                            )
                            ensemble_pipeline[component]._fit_called = True
                            break
                datamodule.hparams.processing_pipeline = ensemble_pipeline

            datamodule.setup("fit")

            log.info(f"Instantiating ensemble model: {cfg.ensemble._target_}")
            models = []
            data_pipelines = []
            for model_i, model_cfg in enumerate(cfg.ensemble_models):
                models.append(
                    src.models.utils.load_model(
                        model_cfg=model_cfg, model_dir=cfg.model_dir[model_i], ckpt=cfg.ckpt
                    )
                )
                datamodule.load_state(os.path.join(cfg.model_dir[model_i], "datamodule"))
                data_pipelines.append(datamodule.hparams.processing_pipeline)
            model = hydra.utils.instantiate(
                cfg.ensemble, models, data_pipelines=data_pipelines, datamodule=datamodule
            )

            # finally, overwrite the loaded ones with the pipeline we possibly made above
            datamodule.hparams.processing_pipeline = ensemble_pipeline
        else:
            datamodule.setup("fit", load_dir=os.path.join(cfg.model_dir, "datamodule"))

            log.info(f"Instantiating model <{cfg.model._target_}>")
            model = src.models.utils.load_model(
                model_cfg=cfg.model, model_dir=cfg.model_dir, ckpt=cfg.ckpt
            )

        log.info("Instantiating loggers...")
        logger: List[Logger] = instantiate_loggers(cfg.get("logger"))  # , model_dir=cfg.model_dir)

        object_dict = {"datamodule": datamodule, "model": model, "logger": logger}

        if OmegaConf.select(cfg, "trainer._target_", default=None) is not None:
            log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
            trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
            object_dict["trainer"] = trainer

    return object_dict
