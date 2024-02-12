import darts.models.forecasting.torch_forecasting_model
import pyrootutils
import pytorch_lightning.loggers.mlflow

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import List, Optional, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

from src import utils
import src.utils.plotting
import src.utils.model_io
import src.datamodules.components.timeseries_datamodule
from src.utils.utils import call_function_with_data

import os
from pathlib import Path

import src.eval

log = utils.get_pylogger(__name__)


def initialize_objects(cfg: DictConfig):
    if cfg.get("scale_model_parameters"):
        try:
            cfg = utils.scale_model_parameters(cfg)
        except ValueError as e:
            log.error(
                "Parameter scaling is not supported for this model, please set the scale_model_parameters argument to False")
            raise e

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: src.datamodules.components.TimeSeriesDataModule = hydra.utils.instantiate(cfg.datamodule,
                                                                                          _convert_="partial")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model, _convert_="partial")

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model
    }

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    object_dict["logger"] = logger

    if cfg.get("trainer", None) is not None and utils.is_torch_model(cfg):
        log.info("Instantiating callbacks...")
        callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    else:
        trainer = None
        callbacks = None

    object_dict["trainer"] = trainer
    object_dict["callbacks"] = callbacks

    return object_dict


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
    try:
        OmegaConf.resolve(cfg)
        resolved_cfg_path = Path(cfg.paths.output_dir) / ".hydra" / "resolved_config.yaml"
        if not resolved_cfg_path.parent.exists():
            os.makedirs(resolved_cfg_path.parent, exist_ok=True)
        OmegaConf.save(cfg, resolved_cfg_path)
        log.info(f"Saved resolved config to: {resolved_cfg_path}")
    except:
        log.warning("Could not save resolved config.")

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed") is not None:  # TODO: separate seed for non-torch models?
        pl.seed_everything(cfg.seed, workers=True)

    time_metrics = {}
    metric_dict = {}
    object_dict = initialize_objects(cfg)
    model = object_dict["model"]
    datamodule = object_dict["datamodule"]
    trainer = object_dict["trainer"]
    callbacks = object_dict["callbacks"]
    logger = object_dict["logger"]

    fit_kwargs = dict(cfg.get("fit", {}))

    if trainer is not None:
        fit_kwargs["trainer"] = trainer

    if cfg.get("train"):
        # TODO: support for fit_from_dataset
        # TODO: dynamically decide what arguments to pass such as trainer
        datamodule.setup("fit")
        datamodule.save_state(os.path.join(cfg.paths.output_dir, "datamodule"))

        # Ensure model is properly saved when using customer trainer object
        if trainer is not None:  # maybe also only if is torchmodel?
            assert isinstance(model, darts.models.forecasting.torch_forecasting_model.TorchForecastingModel), \
                "Pytorch lightning trainer should only be used with TorchForecastingModels"

            if any(isinstance(cb, pl.callbacks.model_checkpoint.ModelCheckpoint) for cb in trainer.callbacks):
                src.utils.model_io.ensure_torch_model_saving(model, cfg.paths.output_dir)
            if any(isinstance(cb, pl.callbacks.progress.RichProgressBar) for cb in trainer.callbacks):
                fit_kwargs["verbose"] = False
            if cfg.get("ckpt_path", None) is not None:
                assert os.path.exists(cfg.ckpt_path), "Provided checkpoint file could not be found"
                model.load_ckpt_path = cfg.ckpt_path

        if cfg.get("plot_datasets"):
            presenters = logger if len(logger) > 0 else ["savefig"]
            for presenter in presenters:
                if src.utils.plotting.is_supported_presenter(presenter):
                    if isinstance(presenter, pytorch_lightning.loggers.TensorBoardLogger):
                        dataset_plot_kwargs = dict(global_step=0)
                    elif isinstance(presenter, pytorch_lightning.loggers.MLFlowLogger):
                        dataset_plot_kwargs = dict(fname="datasets")
                    else:
                        dataset_plot_kwargs = dict(fname=os.path.join(cfg.paths.output_dir, "plots", "datasets"))
                    datamodule.plot_data(presenter=presenter,
                                         separate_components=cfg.plot_datasets.get("separate_components", False),
                                         **dataset_plot_kwargs)


        log.info("Starting training!")

        # execution time is measured through callbacks for pytorch code
        with src.utils.time_block(enabled=cfg.get("measure_execution_time") and trainer is None,
                                  metric_dict=time_metrics,
                                  log_file=os.path.join(cfg.paths.output_dir, "train_exec_time.log") if logger is None else None
                                  ):
            call_function_with_data(model.fit, datamodule, main_split="train", model=model, **fit_kwargs)
        metric_dict.update({f"train_{k}": v for k, v in time_metrics.items()})

        src.utils.model_io.save_model(model, cfg.paths.output_dir)

        if trainer is not None:
            metric_dict.update({k: v.numpy().item() for k, v in trainer.callback_metrics.items()})
        if cfg.get("validate", False) and not (trainer is not None and cfg.eval.get("runner", "trainer") == "trainer"):
            if datamodule.has_split_data("val"):
                with open_dict(cfg):
                    cfg.eval.split = "val"
                val_metric_dict = src.eval.run(cfg, datamodule, model, logger=logger, trainer=trainer)
                metric_dict.update(val_metric_dict)
            else:
                log.info("Validate argument was true but datamodule has no validation data. Skipping validation!")

    if cfg.get("test", False):
        if datamodule.has_split_data("test"):
            log.info("Starting testing!")
            ckpt_path = None
            if trainer is not None:
                ckpt_path = trainer.checkpoint_callback.best_model_path
                if ckpt_path == "":
                    log.warning("Best ckpt not found! Using current weights for testing...")
                    ckpt_path = None
                else:
                    log.info(f"Best ckpt path: {ckpt_path}")
            with open_dict(cfg):
                cfg.eval.split = "test"
            test_metrics = src.eval.run(cfg, datamodule, model, logger=logger, trainer=trainer, ckpt_path=ckpt_path)
            metric_dict.update(test_metrics)
        else:
            log.info("Test argument was true but datamodule has no test data. Skipping testing!")

    # TODO: possibly also do this in task_wrapper if the script crashes somehow?
    if logger:  # Have to do this after call to model.fit, as model is initialized during fit call.
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)
        if trainer is None:
            for lg in logger:
                lg.log_metrics(metric_dict)
                lg.finalize("success")

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
