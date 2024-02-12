import pyrootutils

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import time
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Optional, Union, Sequence, Type

import omegaconf
from omegaconf import open_dict
import hydra
import xarray
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
import mlflow.exceptions
import time
from contextlib import contextmanager

import darts.timeseries
import darts.models.forecasting

import math
import numpy as np
import os
import inspect
import logging
import collections
import glob
import re
import copy
from hydra.core.hydra_config import HydraConfig

import src.utils.model_io
import src.datamodules

from src.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished
    - Logging the exception if occurs
    - Logging the task total execution time
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):
        OmegaConf.register_new_resolver("mlflow-exp-name", lambda x: x.replace(".yaml", "").replace(".yml", ""), replace=True)

        # apply extra utilities
        extras(cfg)

        # execute the task
        try:
            start_time = time.time()
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")  # save exception to `.log` file
            raise ex
        finally:
            path = Path(cfg.paths.output_dir, "exec_time.log")
            content = f"'{cfg.task_name}' execution time: {time.time() - start_time} (s)"
            save_file(path, content)  # save task execution time (even if exception occurs)
            close_loggers()  # close loggers (even if exception occurs so multirun won't fail)

        log.info(f"Output dir: {cfg.paths.output_dir}")

        return metric_dict, object_dict

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    if cfg.extras.get("torch_num_cpu_threads"):
        import torch

        torch.set_num_threads(int(cfg.extras.torch_num_cpu_threads))
        # NB!: Can only be called once, on the second call python will crash.
        # Dont think this setting has much impact anyway, so disabling for now
        #if torch.get_num_interop_threads != int(cfg.extras.torch_num_cpu_threads):
        #    torch.set_num_interop_threads(int(cfg.extras.torch_num_cpu_threads))

    if cfg.extras.get("select_gpu"):
        import os
        import torch

        gpu_ids = cfg.extras.select_gpu.get("gpu_ids")
        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))  # this call will initialize cuda on all gpus for some reason
        else:
            assert is_sequence(gpu_ids)
        if len(gpu_ids) > 0:
            if cfg.extras.select_gpu.get("strategy", "job_id") == "job_id":
                hydra_config = HydraConfig().get()
                job_id = int(hydra_config.job.get("num", 0))  # will be None for mode == RUN therefore returns 0
                gpu_idx = int(job_id % len(gpu_ids))
                gpu_id = str(gpu_ids[gpu_idx])
            elif cfg.extras.select_gpu.get("strategy", "job_id") == "random":
                gpu_id = str(np.random.choice(gpu_ids))
            else:
                raise ValueError(f"Unsupported strategy for gpu_select {cfg.extras.select_gpu.get('strategy')}")
            log.info(f"GPU select strategy = {cfg.extras.select_gpu.get('strategy')} chose id {gpu_id}")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        else:
            log.warning("A GPU-selection strategy was set, but no GPU is available.")

    if cfg.extras.get("matplotlib_savefig_format"):
        import matplotlib
        matplotlib.rcParams["savefig.format"] = cfg.extras.get("matplotlib_savefig_format")

    if cfg.extras.get("matplotlib_backend"):
        import matplotlib
        matplotlib.use(cfg.extras.get("matplotlib_backend"))

    if cfg.extras.get("disable_pytorch_lightning_output"):  # TODO: rename "output"
        import logging
        import warnings
        import os
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        warnings.filterwarnings("ignore", ".*does not have many workers.*")
        warnings.filterwarnings("ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*")
        warnings.filterwarnings("ignore", ".*The number of training batches.*")
        warnings.filterwarnings("ignore", ".*The `srun` command is available on your system but is not used.*")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        try:
            resolve = cfg.extras.print_config.get("resolve", True)
        except:
            resolve = True
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=resolve, save_to_file=True)


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


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
            try:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))
            except mlflow.exceptions.MlflowException:
                # There is a race condition to create the experiment between parallel jobs
                time.sleep(np.random.uniform(0.25, 1))
                logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict.get("trainer", None)
    logger = object_dict["logger"]

    hparams["model"] = cfg["model"]

    # save number of model parameters

    hparams["datamodule"] = cfg["datamodule"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")
    hparams["eval"] = cfg.get("eval")
    hparams["ensemble"] = cfg.get("ensemble", {})
    experiment_kwargs = ["scale_model_parameters", "model_type", "chunk_idx_end", "chunk_idx_start", "chunk_idx"]
    hparams["experiment"] = {k: cfg.get(k) for k in experiment_kwargs if k in cfg}

    if hasattr(getattr(model, "model", None), "parameters"):
        hparams["model/params/total"] = sum(p.numel() for p in model.model.parameters())
        hparams["model/params/trainable"] = sum(
            p.numel() for p in model.model.parameters() if p.requires_grad
        )
        hparams["model/params/non_trainable"] = sum(
            p.numel() for p in model.model.parameters() if not p.requires_grad
        )

    for lg in logger:
        lg.log_hyperparams(hparams)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    try:
        metric_value = metric_dict[metric_name].item()
    except AttributeError:  # metric value is a float, not a numpy array
        metric_value = metric_dict[metric_name]
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


def linear_scale(values: Union[np.ndarray, float], new_max: float, new_min: float, old_max: float, old_min: float) -> Union[np.ndarray, float]:
    """
    Linearly scale values from one range (old_min, old_max) to another range (new_min, new_max).
    :param values:
    :param new_max:
    :param new_min:
    :param old_max:
    :param old_min:
    :return:
    """
    return (new_max - new_min) * (values - old_min) / (old_max - old_min) + new_min


def get_absolute_project_path(project_path: Union[str, Path]):
    """
    Returns the absolute path to the provided path in the project. If the provided path is already an absolute
    path, it is returned as is. If it is a relative path, it is joined with the root directory.

    :param project_path: The provided project_path, possibly relative to project root.
    :return: The absolute path to the provided project_path.
    """
    if os.path.isabs(project_path):
        return project_path
    else:
        return root / project_path


def initialize_hydra(config_path: str, overrides_dot: List[str] = [], overrides_dict: Optional[Dict[str, Any]] = None, return_hydra_config=False, print_config: bool = False, job_name=None) -> DictConfig:
    """
    Initialize hydra and compose config. Optionally, override config values with overrides_dict and overrides_dot.
    Overrides_dot is a list of strings in dot notation, e.g. ["model.transformer.encoder.layers=6"], and is useful
    for overriding whole sections of the config, e.g. ["model=xgboost"]. Overrides_dict is a dictionary of overrides
    in the form {"model": {"transformer": {"encoder": {"layers": 6}}}}, and is useful when doing many overrides of
    nested config values.

    :param config_path: Path to config file.
    :param overrides_dot: List of overrides in dot notation.
    :param overrides_dict: Dictionary of overrides.
    :param return_hydra_config: Whether to return hydra config.
    :param print_config: Whether to print config.

    :return: Config.
    """
    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    hydra.initialize(version_base="1.3", config_path=os.path.dirname(config_path), job_name=job_name)
    cfg = hydra.compose(config_name=os.path.basename(config_path), return_hydra_config=return_hydra_config, overrides=overrides_dot)

    if overrides_dict is not None:
        cfg_overrides = OmegaConf.create(overrides_dict)
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, cfg_overrides)

    if return_hydra_config:
        # Generate and set run output directory
        with open_dict(cfg):
            cfg.paths.output_dir = cfg.hydra.run.dir
        os.makedirs(os.path.join(cfg.paths.output_dir, ".hydra"), exist_ok=True)

        # Save the config without Hydra config entry
        cfg_save = copy.deepcopy(cfg)
        cfg_save_hydra = cfg_save.pop('hydra', None)
        OmegaConf.save(cfg_save_hydra, os.path.join(cfg.paths.output_dir, ".hydra", 'hydra.yaml'))
    else:
        cfg_save = cfg

    if cfg.get("paths", {}).get("output_dir") is not None:
        OmegaConf.save(cfg_save, os.path.join(cfg.paths.output_dir, ".hydra", 'config.yaml'))

    if "extras" in cfg:
        original_extras = cfg.get("extras")
        with open_dict(cfg):
            if cfg.get("extras", {}).get("print_config") is not None:
                cfg.extras.print_config = False
        extras(cfg)
        with open_dict(cfg):
            cfg.extras = original_extras

    if cfg.get("model_dir"):
        cfg = load_saved_config(cfg.model_dir, cfg, print_config=False)

    if "hydra" in cfg:
        HydraConfig().set_config(cfg)
    if print_config:
        rich_utils.print_config_tree(cfg)

    return cfg


def generate_dotpath_value_pairs(cfg: DictConfig, parent_key=None):
    """
    Recursively generate all dotpath-value pairs from config.
    :param cfg: Hydra config.
    :param parent_key: Used to recursively generate dotpaths.
    :return: Collection of two-tuples of dotpaths and values.
    """
    for key, value in cfg.items():
        if isinstance(value, collections.Mapping):
            yield from generate_dotpath_value_pairs(value, parent_key=f"{parent_key}.{key}" if parent_key is not None else key)
        else:
            if parent_key is None:
                yield key, value
            else:
                yield f"{parent_key}.{key}", value


def is_ensemble_model(log_dir):
    if is_sequence(log_dir):
        return True
    else:
        if os.path.exists(os.path.join(get_absolute_project_path(log_dir), "multirun.yaml")):
            return True
        elif len(glob.glob(str(get_absolute_project_path(log_dir)))) > 1:
            return True
        else:
            return False


def _load_saved_config(log_dir: str, cfg_overrides: Optional[DictConfig] = None, print_config: bool = False):
    """
    Load saved config from log_dir with optional overrides from cfg_overrides. Will first try to load resolved config,
    if that fails, will load non-resolved config. Note that loading non-resolved config can yield non-reproducible
    configuration, e.g. for paths that resolve the current time.

    :param log_dir: Path to log directory.
    :param cfg_overrides: Config overrides.
    :param print_config: Whether to print config.

    :return: Config.
    """
    resolved_config_path = os.path.join(log_dir, ".hydra", "resolved_config.yaml")

    config_path = os.path.join(log_dir, ".hydra", "config.yaml")
    cfg = OmegaConf.load(config_path)
    if not os.path.exists(resolved_config_path):  # backwards compatability
        log.warning("Could not find resolved config. Loading non-resolved config instead. Note that this can yield non-reproducible configuration, e.g. for paths that resolve the current time.")
    else:
        resolved_cfg = OmegaConf.load(resolved_config_path)

        # replace paths referencing project-root in resolved config (which is an absolute path) so that models can be transferred between projects
        dotpath_values = list(generate_dotpath_value_pairs(resolved_cfg))
        original_root_dir = resolved_cfg.paths.root_dir
        for dotpath, value in dotpath_values:
            if not is_sequence(value):
                value = [value]
            for v in value:
                if isinstance(v, str) and original_root_dir in v:
                    OmegaConf.update(resolved_cfg, dotpath, v.replace(original_root_dir, cfg.paths.root_dir))
        cfg = resolved_cfg

    if cfg_overrides is not None:
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, cfg_overrides)

    with open_dict(cfg):
        if src.utils.utils.is_sequence(cfg.model_dir):
            cfg.model_dir = [src.utils.get_absolute_project_path(project_path=md) for md in cfg.model_dir]
        else:
            cfg.model_dir = src.utils.get_absolute_project_path(project_path=cfg.model_dir)
            if cfg.get("ckpt") == "best" and is_torch_model(cfg):
                cfg.ckpt = src.utils.model_io.get_best_checkpoint(checkpoint_dir=os.path.join(cfg.model_dir, "checkpoints"))

    if print_config:
        rich_utils.print_config_tree(cfg)

    return cfg


def load_saved_config(log_dir: Union[str, Sequence[str]], cfg_overrides: Optional[DictConfig] = None, print_config: bool = False):  # TODO: shouldnt callbacks also be per model? To ensure checkpoints are correct and stuff
    if is_sequence(log_dir) and len(log_dir) == 1:
        log_dir = log_dir[0]
    if is_ensemble_model(log_dir):  # TODO: if is sequence and not any wildcards, pass through as is?
        model_dir = log_dir
        if not is_sequence(log_dir):
            model_dir = [log_dir]
        model_dirs = []
        for md in model_dir:
            if os.path.exists(os.path.join(get_absolute_project_path(md), "multirun.yaml")):
                md_candidates = [os.path.join(md, fname) for fname in os.listdir(get_absolute_project_path(md))]
            else:
                md_candidates = glob.glob(str(get_absolute_project_path(project_path=md)))
                if len(md_candidates) == 0:
                    log.warning(f"Model directory not found: {md}")
            for md_candidate in md_candidates:
                model_name = os.path.basename(md_candidate)
                if "multiruns" in md_candidate:
                    if re.match("[0-9]+", model_name):
                        model_dirs.append(md_candidate)
                else:
                    model_dirs.append(md_candidate)
        if not len(model_dirs) > 1:
            log.error("When creating an ensemble model, at least two model_dirs must be specified. See configs/eval.yaml for documentation.")
            raise ValueError("Invalid ensemble configuration: model_dir")

        with open_dict(cfg_overrides):
            cfg_overrides.model_dir = model_dirs
        model_cfgs = []
        for model_dir in cfg_overrides.model_dir:
            model_cfgs.append(_load_saved_config(get_absolute_project_path(project_path=model_dir), cfg_overrides, print_config=print_config))
        cfg = model_cfgs[0]
        with open_dict(cfg):
            cfg.ensemble_models = [m_cfg.model for m_cfg in model_cfgs]
    else:
        cfg = _load_saved_config(get_absolute_project_path(project_path=log_dir), cfg_overrides, print_config=print_config)

    return cfg


def initialize_saved_objects(cfg: DictConfig):
    """
    Instantiates all objects needed for evaluation from the provided config using hydra. This includes the datamodule,
    model, trainer and loggers. The instantiated objects are returned in a dictionary.

    :param cfg: The config to use for instantiation.

    :return: A dictionary containing the instantiated objects.
    """
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: src.datamodules.TimeSeriesDataModule = hydra.utils.instantiate(cfg.datamodule, _convert_="partial")

    if cfg.get("ensemble_models"):
        assert "ensemble" in cfg, "When the cfg.ensemble_models argument is set, you must specify an ensemble model that these models operate within."
        assert len(cfg.model_dir) == len(cfg.ensemble_models)

        # TODO: perhaps change architecture of datamodule to transform on data request? i.e. in get_data

        # data provided to ensemble model must be non-transformed, since each model has its own pipeline.
        # thus, have to make sure that the load function does not overwrite the pipeline with the loaded one.
        # If datamodule originally has a MissingValuesFiller, we keep that to avoid nans in ensemble weight fitting.
        processing_pipeline = None
        if datamodule.hparams.processing_pipeline is not None:
            for t_i, transformer in enumerate(getattr(datamodule.hparams.processing_pipeline, "_transformers", [])):
                if transformer.name == "MissingValuesFiller":
                    processing_pipeline = darts.dataprocessing.Pipeline([transformer], copy=True)
                    processing_pipeline._fit_called = True
                    break
        datamodule.hparams.processing_pipeline = processing_pipeline
        datamodule.setup("fit")#, load_dir=os.path.join(cfg.model_dir, "0", "datamodule"))

        log.info(f"Instantiating ensemble model: {cfg.ensemble._target_}")
        models = []
        data_pipelines = []
        for model_i, model_cfg in enumerate(cfg.ensemble_models):
            models.append(src.utils.model_io.load_model(model_cfg=model_cfg, model_dir=cfg.model_dir[model_i], ckpt=cfg.ckpt))
            datamodule.load_state(os.path.join(cfg.model_dir[model_i], "datamodule"))
            data_pipelines.append(datamodule.hparams.processing_pipeline)
        model = hydra.utils.instantiate(cfg.ensemble, models, data_pipelines=data_pipelines, datamodule=datamodule)
        datamodule.hparams.processing_pipeline = processing_pipeline
    else:
        datamodule.setup("fit", load_dir=os.path.join(cfg.model_dir, "datamodule"))

        log.info(f"Instantiating model <{cfg.model._target_}>")
        model = src.utils.model_io.load_model(model_cfg=cfg.model, model_dir=cfg.model_dir, ckpt=cfg.ckpt)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))#, model_dir=cfg.model_dir)

    object_dict = {
        "datamodule": datamodule,
        "model": model,
        "logger": logger
    }

    if cfg.get("trainer") is not None and cfg.trainer.get("_target_") is not None and cfg.get("eval", {}).get("runner", None) != "backtest":
        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
        object_dict["trainer"] = trainer

    return object_dict


class DisableLogger():
    def __enter__(self):
       logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
       logging.disable(logging.NOTSET)


def check_model_covariate_support(model: darts.models.forecasting.forecasting_model.ForecastingModel) -> List[str]:
    """
    Takes a darts forecasting model and returns the subset of [past_covariates, future_covariates, static_covariates],
    the model is configured to use. The resulting output is therefore both a function of what the model inherently
    supports and which of its supported covariate types are enabled.

    :param model: Darts forecasting model.
    :return: List of supported covariate types.
    """
    supported_covariates = set()

    if isinstance(model, darts.models.forecasting.ensemble_model.EnsembleModel):
        models = model.forecasting_models
    else:
        models = [model]

    for model in models:
        if isinstance(model, darts.models.forecasting.torch_forecasting_model.TorchForecastingModel):
            with DisableLogger():
                try:
                    model._verify_past_future_covariates(past_covariates=1, future_covariates=None)
                    supported_covariates.add("past_covariates")
                except ValueError:
                    pass
                try:
                    model._verify_past_future_covariates(past_covariates=None, future_covariates=1)
                    supported_covariates.add("future_covariates")
                except ValueError:
                    pass
        elif isinstance(model, darts.models.forecasting.regression_model.RegressionModel):
            for cov in ["past", "future"]:
                if cov in model.lags:
                    supported_covariates.add(f"{cov}_covariates")
        elif isinstance(model, darts.models.forecasting.forecasting_model.LocalForecastingModel):
            for cov in ["past", "future"]:
                if getattr(model, f"supports_{cov}_covariates", False):
                    supported_covariates.add(f"{cov}_covariates")
        else:
            raise ValueError(f"Unexpected model type: {type(model)}")

    return supported_covariates


def get_model_supported_data(datamodule, model, data_types=None, main_split="train", warn_unsupported_covariates=True):
    """
    Get data from datamodule that is supported by model. This is a function of what the model inherently supports,
    which of its supported data types are enabled for the model, and which of the enabled data types are available in
    the datamodule. The data is returned in the form of a dictionary, with keys corresponding to the data types.

    :param datamodule: Datamodule to get data from.
    :param model: Darts forecasting model.
    :param data_types: List of data types to get. If None, will get all data types supported by the model.
    :param main_split: Main split to get data for.
    :param warn_unsupported_covariates: Whether to warn about covariates present in the datamodule that the model does
    not support/is not configured to use.

    :return: Dictionary of data.
    """
    if data_types is None:
        data_types = ["series", "past_covariates", "future_covariates", "static_covariates"]

    model_supported_covariates = check_model_covariate_support(model)
    for cov in ["past", "future", "static"]:  # TODO: define all covariates somewhere?
        cov = f"{cov}_covariates"
        if cov in data_types and not cov in model_supported_covariates:
            data_types = [param for param in data_types if not param.endswith(cov)]
        if warn_unsupported_covariates and cov not in model_supported_covariates and datamodule.has_split_covariate_type(main_split, cov):
            log.warning(f"Datamodule has {cov} but model does not support or is not configured to use them.")

    return datamodule.get_data(data_types, main_split=main_split)


def call_function_with_data(function, datamodule, main_split="train", model=None, raise_exception_on_missing_argument=True, **function_kwargs):
    """
    Call function with data from datamodule as keyword arguments. The required data types to get from the datamodule are
    inferred from the function signature. A model can optionally be provided, in which case the data will be filtered
    to only include data types supported by the model. The data is passed as keyword arguments to the function,
    with the names of the keyword arguments corresponding to the data types.

    :param function: Function to call.
    :param datamodule: Datamodule to get data from.
    :param main_split: Main split to get data for.
    :param model: Darts forecasting model.
    :param raise_exception_on_missing_argument: Whether to raise an exception if a required argument is missing.
    :param function_kwargs: Additional keyword arguments to pass to the function.

    :return: Output of function.
    """
    function_parameters = inspect.signature(function).parameters

    if model is not None:
        data_kwargs = get_model_supported_data(datamodule, model, data_types=list(function_parameters), main_split=main_split)
    else:
        data_kwargs = datamodule.get_data(list(function_parameters), main_split=main_split)

    all_kwargs = {}
    for kwarg_name in list(function_parameters):
        if kwarg_name in data_kwargs:
            all_kwargs[kwarg_name] = data_kwargs[kwarg_name]
        elif kwarg_name in function_kwargs:
            all_kwargs[kwarg_name] = function_kwargs[kwarg_name]
        else:
            if function_parameters[kwarg_name].kind == inspect.Parameter.VAR_KEYWORD:  # this is the **kwargs argument
                continue
            elif function_parameters[kwarg_name].default == inspect.Parameter.empty:
                if raise_exception_on_missing_argument:
                    raise ValueError(f"The required argument {kwarg_name} to the function {function.__name__} was not supplied, and raise_exception_on_missing_argument is True")
                log.info(f"The required argument {kwarg_name} to the function {function.__name__} was not supplied, setting to default value None")
                all_kwargs[kwarg_name] = None
            else:
                all_kwargs[kwarg_name] = function_parameters[kwarg_name].default
    return function(**all_kwargs)


def data_is_binary(data: Union[darts.timeseries.TimeSeries, np.ndarray, xarray.DataArray]) -> bool:
    """
    Check if data is binary.

    :param data: Data to check.

    :return: Whether data is binary.
    """
    if isinstance(data, darts.timeseries.TimeSeries):
        data = data.values()
    return np.count_nonzero((data != 0) & (data != 1)) == 0


def hist_bin_num_freedman_diaconis(x: Union[darts.timeseries.TimeSeries, np.ndarray, xarray.DataArray]) -> Union[int, List[float]]:
    """
    Calculate number of bins for histogram using Freedman-Diaconis rule. If data is binary, returns two bins centered on
    0 and 1.

    :param x: Data to calculate number of bins for.

    :return: Number of bins.
    """
    if isinstance(x, darts.timeseries.TimeSeries):
        x = x.values()
    q25, q75 = np.nanpercentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
    if bin_width == 0.0:
        if data_is_binary(x):
            return [-0.4, 0.4, 0.6, 1.4]
        else:
            raise ValueError(f"Cannot calculate number of bins given data x (q25: {q25:.2f}, q75: {q75:.2f}, bin_width: {bin_width:.2f}, any_nans: {np.any(np.isnan(x))})")
    else:
        bins = round((np.nanmax(x) - np.nanmin(x)) / bin_width)
        return bins


def timeseries_from_dataloader_and_model_outputs(dataloader, model_outputs):
    """
    Takes in a dataloader and a list of model outputs, and returns a timeseries with the predictions from the model outputs.

    :param dataloader: Dataloader to get the timeseries index from.
    :param model_outputs: List of model outputs.
    """
    return darts.timeseries.TimeSeries.from_times_and_values(dataloader.dataset.index.index, np.concatenate([mo.prediction for mo in model_outputs], axis=0))


def interpolate_nan_values(values: "np.ndarray[Any, Any]") -> Tuple["np.ndarray[Any, Any]", "np.ndarray[Any, Any]"]:
    """
    Takes in a 1d array with possible nan values, and replaces these nan values by linear interpolation. Returns array with
    replaced values and boolean mask of where True indicates a replaced value.
    :param values: 1d array of values where some values can be NaN
    :return: 1d array where NaN values have been linearly interpolated, boolean mask where True signifies a replaced value
    """
    missing = np.isnan(values)
    nan_indices = lambda x: np.array(x).nonzero()[0]

    values[missing] = np.interp(nan_indices(missing), nan_indices(~missing), values[~missing])

    return values, missing


def string_format_datetime(time_object, string_format):
    """
    Helper function to string format datetime-objects. Time object can either be an iterable, for which case the
    function returns a list of the string-formatted elements from time_object, or a single np.datetime64 object, for
    which the function returns that object string-formatted.
    :param time_object: np.datetime64 object or iterable of np.datetime64 objects.
    :param string_format: string-format specifier.
    :return: string formatted np.datetime64 object or list.
    """
    import pandas as pd

    try:
        iter(time_object)
        is_iterable = True
        test_type_object = time_object[0]
    except TypeError:
        is_iterable = False
        test_type_object = time_object

    if is_iterable:
        res = []
        if isinstance(test_type_object, np.datetime64):
            for t in time_object:
                res.append(t.astype('datetime64[s]').item().strftime(string_format))
        elif isinstance(test_type_object, pd.Timestamp):
            for t in time_object:
                res.append(t.strftime(string_format))
        else:
            raise ValueError
    else:
        if isinstance(test_type_object, np.datetime64):
            res = time_object.astype('datetime64[s]').item().strftime(string_format)
        elif isinstance(test_type_object, pd.Timestamp):
            res = time_object.strftime(string_format)
        else:
            raise ValueError

    return res


def get_lstm_hidden_dim_from_params(n_params: int, n_layers: int, n_models: int, inputs: Dict[str, List[str]]):
    n_inputs = sum(len(data_type) for data_type in inputs.values() if data_type is not None)
    #return round((np.sqrt(n_models * (n_inputs + 4) ** 2 * n_models + 3 * n_params) - (n_inputs + 4) * n_models) / (6 * n_models))
    n_stacked_layers = n_layers - 1
    return round((np.sqrt(n_models * (n_models * (n_inputs + 2 * n_stacked_layers + 2) ** 2 + 2 * n_params * n_stacked_layers + n_params)) - n_models * (n_inputs + 2 * n_stacked_layers + 2)) / (2 * (2 * n_models * n_stacked_layers + n_models)))


def get_lstm_n_params_from_hidden_dim(hidden_dim: int, n_layers: int, n_models: int, inputs: Dict[str, List[str]]):
    n_inputs = sum(len(data_type) for data_type in inputs.values() if data_type is not None)
    n_stacked_layers = n_layers - 1
    return round(n_models * (4 * (hidden_dim * n_inputs + hidden_dim ** 2 + hidden_dim) + n_stacked_layers * 4 * (2 * hidden_dim ** 2 + hidden_dim)))


def get_global_hidden_dim_from_ensemble(ensemble_hidden_dim: int, n_layers: int, n_ensemble_models: int, inputs: Dict[str, List[str]]):
    n_ensemble_params = get_lstm_n_params_from_hidden_dim(ensemble_hidden_dim, n_layers, n_ensemble_models, inputs)
    return get_lstm_hidden_dim_from_params(n_ensemble_params, n_layers, 1, inputs)


def calculate_TCN_matching_num_filters(input_chunk_length: int, output_chunk_length: int, kernel_size: int,
                                       num_layers: Optional[int], dilation_base: int, target_size: int,
                                       ensemble_num_filters: int, inputs: Dict[str, List[str]], n_ensemble_models: int) -> int:
    """
    Calculates the number of filters needed for a single TCN model to have a total number of parameters
    roughly equal to that of an ensemble of N identical TCN models.

    Parameters
    ----------
    input_chunk_length : int
        Number of past time steps that are fed to the forecasting module.
    output_chunk_length : int
        Number of time steps the torch module will predict into the future at once.
    kernel_size : int
        The size of every kernel in a convolutional layer.
    num_layers : Optional[int]
        The number of convolutional layers. If None, it will be calculated based on other parameters.
    dilation_base : int
        The base of the exponent that will determine the dilation on every level.
    target_size : int
        The dimensionality of the output time series.
    N : int
        Number of models in the ensemble.

    Returns
    -------
    int
        The calculated number of filters for the single TCN model to match the parameter count of the ensemble.
    """
    input_size = sum(len(data_type) for data_type in inputs.values() if data_type is not None)

    # Calculate the number of layers if not provided
    if num_layers is None:
        if dilation_base > 1:
            num_layers = math.ceil(
                math.log(
                    (input_chunk_length - 1)
                    * (dilation_base - 1)
                    / (kernel_size - 1)
                    / 2
                    + 1,
                    dilation_base,
                )
            )
        else:
            num_layers = np.ceil(
                (input_chunk_length - 1) / (kernel_size - 1) / 2
            )

    # Calculate parameters for a single model with a given number of filters
    def calculate_parameters(num_filters):
        total_params = 0
        for layer in range(num_layers):
            input_dim = input_size if layer == 0 else num_filters
            output_dim = target_size if layer == num_layers - 1 else num_filters
            total_params += (kernel_size * input_dim * num_filters) + num_filters  # conv1
            total_params += (kernel_size * num_filters * output_dim) + output_dim  # conv2
            if input_dim != output_dim:
                total_params += (input_dim * output_dim) + output_dim  # conv3 (if needed)
        return total_params

    # Calculate the total parameters for one model in the ensemble
    single_model_params = calculate_parameters(ensemble_num_filters)

    ensemble_total_params = single_model_params * n_ensemble_models

    # Adjust num_filters for the single model to match the ensemble's total parameters
    # Start with an initial guess and adjust until global model has at least as many parameters as ensemble
    matching_num_filters = ensemble_num_filters
    while calculate_parameters(matching_num_filters) <= ensemble_total_params:
        matching_num_filters += 1

    return matching_num_filters


def is_sequence(obj: Any) -> bool:
    return isinstance(obj, (list, tuple, omegaconf.ListConfig))


def enable_eval_resolver():
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)


def is_torch_model(model_or_cfg: Union[darts.models.forecasting.forecasting_model.ForecastingModel, omegaconf.DictConfig]):
    if isinstance(model_or_cfg, omegaconf.DictConfig):
        try:
            if hasattr(model_or_cfg.get("model", {}), "_target_"):
                model_class = hydra.utils.get_class(model_or_cfg.model._target_)
            else:
                model_class = hydra.utils.get_class(model_or_cfg._target_)
        except omegaconf.errors.ConfigAttributeError:
            log.warning("A config object was passed but it does not contain the required keys model._target_")
            return None
    else:
        assert isinstance(model_or_cfg, darts.models.forecasting.forecasting_model.ForecastingModel), f"Invalid type for function {type(model_or_cfg)}, expected a darts.models.forecasting.forecasting_model.ForecastingModel object"
        model_class = type(model_or_cfg)
    return issubclass(model_class, darts.models.forecasting.torch_forecasting_model.TorchForecastingModel)


def scale_model_parameters(cfg: DictConfig) -> DictConfig:
    SUPPORTED_MODELS = ["darts.models.forecasting.rnn_model.RNNModel",
                        "darts.models.forecasting.tcn_model.TCNModel",
                        "darts.models.forecasting.xgboost.XGBModel"]
    assert "model" in cfg, "config object must have a model entry"

    if cfg.model._target_ not in SUPPORTED_MODELS:
        log.warning(f"Model {cfg.model._target_} is not a supported model {SUPPORTED_MODELS}. No parameter scaling was performed.")

        return cfg
    elif "chunk_idx" not in cfg:
        log.warning("No chunk_idx is set in config, cannot infer number of models in ensemble. No parameter scaling was performed.")
        return cfg

    if cfg.model._target_ == "darts.models.forecasting.rnn_model.RNNModel":
        hidden_dim_before = cfg.model.hidden_dim
        with open_dict(cfg):
            cfg.model.hidden_dim = get_global_hidden_dim_from_ensemble(
                ensemble_hidden_dim=cfg.model.hidden_dim,
                n_layers=cfg.model.n_rnn_layers,
                n_ensemble_models=1 + cfg.chunk_idx - cfg.chunk_idx_start,
                inputs=cfg.datamodule.data_variables,
            )
        log.info(f"model.hidden_dim was scaled from {hidden_dim_before} to {cfg.model.hidden_dim}")
    elif cfg.model._target_ == "darts.models.forecasting.tcn_model.TCNModel":
        num_filters_before = cfg.model.num_filters
        tcn_args = inspect.signature(hydra.utils.get_class(cfg.model._target_).__init__).parameters
        matching_args = {k: tcn_args[k].default if k in tcn_args and tcn_args[k].default is not inspect.Parameter.empty else None for k in inspect.signature(calculate_TCN_matching_num_filters).parameters}
        matching_args.update({k: v for k, v in cfg.model.items() if k in matching_args})
        matching_args.update(dict(
            target_size=len(cfg.datamodule.data_variables.target) * cfg.model.output_chunk_length,
            ensemble_num_filters=cfg.model.num_filters,
            inputs=cfg.datamodule.data_variables,
            n_ensemble_models=1 + cfg.chunk_idx - cfg.chunk_idx_start
        ))
        with open_dict(cfg):
            cfg.model.num_filters = calculate_TCN_matching_num_filters(
                **matching_args
            )
        log.info(f"model.num_filters was scaled from {num_filters_before} to {cfg.model.num_filters}")
    elif cfg.model._target_ == "darts.models.forecasting.xgboost.XGBModel":
        n_ensemble_models = 1 + cfg.chunk_idx - cfg.chunk_idx_start
        if n_ensemble_models > 1:
            n_estimators_before = cfg.model.get("n_estimators", 100)
            n_estimators_new = n_estimators_before * n_ensemble_models
            with open_dict(cfg):
                cfg.model.n_estimators = n_estimators_new
            log.info(f"model.n_estimators was scaled from {n_estimators_before} to {n_estimators_new}")

    return cfg


@contextmanager
def time_block(enabled, metric_dict=None, log_file=None):
    start_time = time.perf_counter() if enabled else None
    try:
        yield
    finally:
        if enabled:
            end_time = time.perf_counter()
            execution_time = end_time - start_time

            log.info(f"Execution time in block: {execution_time} seconds\n")
            # Optionally log to a file
            if log_file:
                with open(log_file, "a") as file:
                    file.write(f"Execution time: {execution_time} seconds\n")

            # Optionally add to a metrics dictionary
            if metric_dict is not None and isinstance(metric_dict, dict):
                metric_dict['execution_time'] = execution_time
