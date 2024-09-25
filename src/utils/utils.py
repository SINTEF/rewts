import copy
import inspect
import logging
import os
import time
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import darts.models.forecasting
import darts.timeseries
import notifiers
import numpy as np
import omegaconf
import pytorch_lightning.loggers
import xarray
from hydra.core.hydra_config import HydraConfig
from notifiers.exceptions import NotificationError
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.utilities import rank_zero_only

import src.utils.plotting
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
        def send_notification(event: str, additional_message: Optional = None) -> None:
            if OmegaConf.select(cfg, f"extras.notifications.{event}"):
                for endpoint, details in cfg.extras.notifications[event].items():
                    try:
                        notifier = notifiers.get_notifier(endpoint)

                        if "message" not in details:
                            details["message"] = ""
                        if additional_message is not None:
                            details["message"] += f". {additional_message}"

                        notifier.notify(raise_on_errors=True, **details)
                        log.info(f"Notification sent for {event} event to {endpoint}")
                    except NotificationError as e:
                        log.error(
                            f"Failed to send notification for {event} event to {endpoint}: {e}"
                        )

        OmegaConf.register_new_resolver(
            "mlflow-exp-name", lambda x: x.replace(".yaml", "").replace(".yml", ""), replace=True
        )

        # Apply extra utilities
        extras(cfg)

        try:
            send_notification("begin")

            start_time = time.time()
            metric_dict, object_dict = task_func(cfg=cfg)
            execution_time = time.time() - start_time

            # Task finished successfully
            send_notification("end", f"Execution time: {execution_time:.2f} seconds.")

        except Exception as ex:
            execution_time = time.time() - start_time

            # Task failed
            send_notification(
                "fail",
                f"Task '{cfg.task_name}' failed with exception: {ex}. Execution time: {execution_time:.2f} seconds.",
            )
            log.exception("")  # Save exception to `.log` file
            raise ex

        finally:
            path = Path(cfg.paths.output_dir, "exec_time.log")
            content = f"'{cfg.task_name}' execution time: {execution_time} (s)"
            save_file(path, content)  # Save task execution time (even if exception occurs)
            close_loggers()  # Close loggers (even if exception occurs so multirun won't fail)

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

    if cfg.extras.get(
        "enable_eval_resolver"
    ):  # TODO: check if this can be removed since we have a function for this now
        from omegaconf import OmegaConf

        OmegaConf.register_new_resolver("eval", eval, replace=True)

    if cfg.extras.get("torch_num_cpu_threads"):
        import torch

        torch.set_num_threads(int(cfg.extras.torch_num_cpu_threads))
        # NB!: Can only be called once, on the second call python will crash.
        # Dont think this setting has much impact anyway, so disabling for now
        # if torch.get_num_interop_threads != int(cfg.extras.torch_num_cpu_threads):
        #    torch.set_num_interop_threads(int(cfg.extras.torch_num_cpu_threads))

    if cfg.extras.get("select_gpu"):
        import os

        import torch

        gpu_ids = cfg.extras.select_gpu.get("gpu_ids")
        if gpu_ids is None:
            gpu_ids = list(
                range(torch.cuda.device_count())
            )  # this call will initialize cuda on all gpus for some reason
        else:
            assert is_sequence(gpu_ids)
        if len(gpu_ids) > 0:
            if cfg.extras.select_gpu.get("strategy", "job_id") == "job_id":
                hydra_config = HydraConfig().get()
                job_id = int(
                    hydra_config.job.get("num", 0)
                )  # will be None for mode == RUN therefore returns 0
                gpu_idx = int(job_id % len(gpu_ids))
                gpu_id = str(gpu_ids[gpu_idx])
            elif cfg.extras.select_gpu.get("strategy", "job_id") == "random":
                gpu_id = str(np.random.choice(gpu_ids))
            else:
                raise ValueError(
                    f"Unsupported strategy for gpu_select {cfg.extras.select_gpu.get('strategy')}"
                )
            log.info(
                f"GPU select strategy = {cfg.extras.select_gpu.get('strategy')} chose id {gpu_id}"
            )
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
        import os
        import warnings

        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        warnings.filterwarnings("ignore", ".*does not have many workers.*")
        warnings.filterwarnings(
            "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
        )
        warnings.filterwarnings("ignore", ".*The number of training batches.*")
        warnings.filterwarnings(
            "ignore", ".*The `srun` command is available on your system but is not used.*"
        )
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

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
        except Exception as e:
            resolve = True
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=resolve, save_to_file=True)


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


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
    experiment_kwargs = [
        "scale_model_parameters",
        "model_type",
        "chunk_idx_end",
        "chunk_idx_start",
        "chunk_idx",
    ]
    hparams["chunk_experiment"] = {k: cfg.get(k) for k in experiment_kwargs if k in cfg}
    hparams["run_dir"] = OmegaConf.select(cfg, "paths.output_dir")

    if hasattr(getattr(model, "model", None), "parameters"):
        hparams["model/params/total"] = sum(p.numel() for p in model.model.parameters())
        hparams["model/params/trainable"] = sum(
            p.numel() for p in model.model.parameters() if p.requires_grad
        )
        hparams["model/params/non_trainable"] = sum(
            p.numel() for p in model.model.parameters() if not p.requires_grad
        )

    # log number of trees used by xgboost
    if getattr(getattr(model, "model", None), "best_iteration", None) is not None:
        hparams["model/num_trees"] = model.model.best_iteration

    if OmegaConf.select(cfg, "log_hyperparameters_custom") is not None:
        for dot_path in OmegaConf.select(cfg, "log_hyperparameters_custom"):
            hparams[dot_path] = OmegaConf.select(cfg, dot_path)

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


def linear_scale(
    values: Union[np.ndarray, float],
    new_max: float,
    new_min: float,
    old_max: float,
    old_min: float,
) -> Union[np.ndarray, float]:
    """Linearly scale values from one range (old_min, old_max) to another range (new_min, new_max).

    :param values:
    :param new_max:
    :param new_min:
    :param old_max:
    :param old_min:
    :return:
    """
    return (new_max - new_min) * (values - old_min) / (old_max - old_min) + new_min


class DisableLogger:
    """Utility context manager to disable logging."""

    def __enter__(self):
        """Entry point of logging disable :return:"""
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        """Exit point of logging disable :return:"""
        logging.disable(logging.NOTSET)


def check_model_covariate_support(
    model: darts.models.forecasting.forecasting_model.ForecastingModel,
) -> List[str]:
    """Takes a darts forecasting model and returns the subset of [past_covariates,
    future_covariates, static_covariates], the model is configured to use. The resulting output is
    therefore both a function of what the model inherently supports and which of its supported
    covariate types are enabled.

    :param model: Darts forecasting model.
    :return: List of supported covariate types.
    """
    supported_covariates = set()

    if isinstance(model, darts.models.forecasting.ensemble_model.EnsembleModel):
        models = model.forecasting_models
    else:
        models = [model]

    for model in models:
        if isinstance(
            model, darts.models.forecasting.torch_forecasting_model.TorchForecastingModel
        ):
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


def get_model_supported_data(
    datamodule, model, data_types=None, main_split="train", warn_unsupported_covariates=True
):
    """Get data from datamodule that is supported by model. This is a function of what the model
    inherently supports, which of its supported data types are enabled for the model, and which of
    the enabled data types are available in the datamodule. The data is returned in the form of a
    dictionary, with keys corresponding to the data types.

    :param datamodule: Datamodule to get data from.
    :param model: Darts forecasting model.
    :param data_types: List of data types to get. If None, will get all data types supported by the
        model.
    :param main_split: Main split to get data for.
    :param warn_unsupported_covariates: Whether to warn about covariates present in the datamodule
        that the model does not support/is not configured to use.
    :return: Dictionary of data.
    """
    if data_types is None:
        data_types = ["series", "past_covariates", "future_covariates", "static_covariates"]

    model_supported_covariates = check_model_covariate_support(model)
    for cov in ["past", "future", "static"]:  # TODO: define all covariates somewhere?
        cov = f"{cov}_covariates"
        if cov in data_types and cov not in model_supported_covariates:
            data_types = [param for param in data_types if not param.endswith(cov)]
        if (
            warn_unsupported_covariates
            and cov not in model_supported_covariates
            and datamodule.has_split_covariate_type(main_split, cov)
        ):
            log.warning(
                f"Datamodule has {cov} but model does not support or is not configured to use them."
            )

    return datamodule.get_data(data_types, main_split=main_split)


def call_function_with_data(
    function,
    datamodule,
    main_split="train",
    model=None,
    raise_exception_on_missing_argument=True,
    **function_kwargs,
):
    """Call function with data from datamodule as keyword arguments. The required data types to get
    from the datamodule are inferred from the function signature. A model can optionally be
    provided, in which case the data will be filtered to only include data types supported by the
    model. The data is passed as keyword arguments to the function, with the names of the keyword
    arguments corresponding to the data types.

    :param function: Function to call.
    :param datamodule: Datamodule to get data from.
    :param main_split: Main split to get data for.
    :param model: Darts forecasting model.
    :param raise_exception_on_missing_argument: Whether to raise an exception if a required
        argument is missing.
    :param function_kwargs: Additional keyword arguments to pass to the function.
    :return: Output of function.
    """
    function_parameters = inspect.signature(function).parameters

    if model is not None:
        data_kwargs = get_model_supported_data(
            datamodule, model, data_types=list(function_parameters), main_split=main_split
        )
    else:
        data_kwargs = datamodule.get_data(list(function_parameters), main_split=main_split)

    return call_function_with_resolved_arguments(
        function,
        raise_exception_on_missing_argument=raise_exception_on_missing_argument,
        **function_kwargs,
        **data_kwargs,
    )


def call_function_with_resolved_arguments(
    function: Callable, raise_exception_on_missing_argument: bool = True, **kwargs
):
    """Helper function to call a function with only its supported arguments from a dictionary of
    arguments. The function is inspected for the arguments it accepts, and only these are selected.

    :param function: Function to resolve arguments for.
    :param raise_exception_on_missing_argument: Whether to raise an exception if a required
        argument is missing.
    :param kwargs: Additional keyword arguments to pass to the function.
    :return: Function results with resolved arguments
    """
    resolved_args = {}
    function_parameters = inspect.signature(function).parameters

    for param in inspect.signature(function).parameters:
        if param in kwargs:
            resolved_args[param] = kwargs[param]
        else:
            if (
                function_parameters[param].kind == inspect.Parameter.VAR_KEYWORD
            ):  # this is the **kwargs argument
                continue
            elif function_parameters[param].default == inspect.Parameter.empty:
                if raise_exception_on_missing_argument:
                    raise ValueError(
                        f"The required argument {param} to the function {function.__name__} was not supplied, and raise_exception_on_missing_argument is True"
                    )
                log.info(
                    f"The required argument {param} to the function {function.__name__} was not supplied, setting to default value None"
                )
                resolved_args[param] = None
            else:
                resolved_args[param] = function_parameters[param].default
    return function(**resolved_args)


def data_is_binary(data: Union[darts.timeseries.TimeSeries, np.ndarray, xarray.DataArray]) -> bool:
    """Check if data is binary.

    :param data: Data to check.
    :return: Whether data is binary.
    """
    if isinstance(data, darts.timeseries.TimeSeries):
        data = data.values()
    return np.count_nonzero((data != 0) & (data != 1)) == 0


def hist_bin_num_freedman_diaconis(
    x: Union[darts.timeseries.TimeSeries, np.ndarray, xarray.DataArray]
) -> Union[int, List[float]]:
    """Calculate number of bins for histogram using Freedman-Diaconis rule. If data is binary,
    returns two bins centered on 0 and 1.

    :param x: Data to calculate number of bins for.
    :return: Number of bins.
    """
    if isinstance(x, darts.timeseries.TimeSeries):
        x = x.values()
    q25, q75 = np.nanpercentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    if bin_width == 0.0:
        if data_is_binary(x):
            return [-0.4, 0.4, 0.6, 1.4]
        else:
            raise ValueError(
                f"Cannot calculate number of bins given data x (q25: {q25:.2f}, q75: {q75:.2f}, bin_width: {bin_width:.2f}, any_nans: {np.any(np.isnan(x))})"
            )
    else:
        bins = round((np.nanmax(x) - np.nanmin(x)) / bin_width)
        return bins


def timeseries_from_dataloader_and_model_outputs(dataloader, model_outputs):
    """Takes in a dataloader and a list of model outputs, and returns a timeseries with the
    predictions from the model outputs.

    :param dataloader: Dataloader to get the timeseries index from.
    :param model_outputs: List of model outputs.
    """
    return darts.timeseries.TimeSeries.from_times_and_values(
        dataloader.dataset.index.index,
        np.concatenate([mo.prediction for mo in model_outputs], axis=0),
    )


def interpolate_nan_values(
    values: "np.ndarray[Any, Any]",
) -> Tuple["np.ndarray[Any, Any]", "np.ndarray[Any, Any]"]:
    """Takes in a 1d array with possible nan values, and replaces these nan values by linear
    interpolation. Returns array with replaced values and boolean mask of where True indicates a
    replaced value.

    :param values: 1d array of values where some values can be NaN
    :return: 1d array where NaN values have been linearly interpolated, boolean mask where True
        signifies a replaced value
    """
    missing = np.isnan(values)

    def nan_indices(x):
        return np.array(x).nonzero()[0]

    values[missing] = np.interp(nan_indices(missing), nan_indices(~missing), values[~missing])

    return values, missing


def string_format_datetime(time_object, string_format):
    """Helper function to string format datetime-objects. Time object can either be an iterable,
    for which case the function returns a list of the string-formatted elements from time_object,
    or a single np.datetime64 object, for which the function returns that object string-formatted.

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
                res.append(t.astype("datetime64[s]").item().strftime(string_format))
        elif isinstance(test_type_object, pd.Timestamp):
            for t in time_object:
                res.append(t.strftime(string_format))
        else:
            raise ValueError
    else:
        if isinstance(test_type_object, np.datetime64):
            res = time_object.astype("datetime64[s]").item().strftime(string_format)
        elif isinstance(test_type_object, pd.Timestamp):
            res = time_object.strftime(string_format)
        else:
            raise ValueError

    return res


def is_sequence(obj: Any) -> bool:
    """Check if object is a sequence."""
    return isinstance(obj, (list, tuple, omegaconf.ListConfig))


def enable_eval_resolver():
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)


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
                metric_dict["execution_time"] = execution_time


def get_presenters_and_kwargs(presenters, fig_dir, fig_name, logger=None, trainer=None):
    """Utility function to process figure presenters and their arguments according to an output
    directory, a figure name, and logger and trainer objects.

    :param presenters: Sequence of possible presenters. Defaults to savefig if None.
    :param fig_dir: Output directory for figures
    :param fig_name: Name of figure
    :param logger: Sequence of logger objects
    :param trainer: Trainer object
    :return: processed sequence of presenters and corresponding presenter arguments.
    """
    if presenters is None:
        presenters = ["savefig"]
    else:
        if is_sequence(presenters):
            presenters = list(presenters)
        else:
            presenters = [presenters]
    if logger is not None and len(logger) > 0:
        presenters.extend(logger)

    presenter_kwargs = []
    for presenter in presenters:
        if src.utils.plotting.is_supported_presenter(presenter):
            if isinstance(presenter, pytorch_lightning.loggers.TensorBoardLogger):
                p_kwargs = dict(
                    global_step=trainer.global_step if trainer is not None else 0, tag=fig_name
                )
            elif presenter == "savefig" or isinstance(
                presenter, pytorch_lightning.loggers.mlflow.MLFlowLogger
            ):
                p_kwargs = dict(fname=os.path.join(fig_dir, fig_name))
            else:
                p_kwargs = dict(fname=fig_name)
            presenter_kwargs.append(p_kwargs)

    return presenters, presenter_kwargs


# TODO: perhaps it is better to return callable that does nothing rather than None?
def get_inverse_transform_data_func(
    inverse_transform_data_cfg: Union[None, Dict[str, Any]], datamodule, split: str
) -> Union[Callable, None]:
    """
    Helper function to test if inverse transformation is possible and if so returns a callable that inverse transforms
    data according to the provided configuration.
    Args:
        inverse_transform_data_cfg: Configuration options for inverse transformation
        datamodule: The datamodule used to inverse transform data.
        split: Name of split for which inverse transformation should be tested. One of ["train", "val", "test"].

    Returns: Callable that inverse transforms data or None.
    """
    if not inverse_transform_data_cfg:
        return None

    try:
        partial_ok = inverse_transform_data_cfg.get("partial_ok", False)
    except AttributeError:
        partial_ok = False

    # First try to inverse transform some data to ensure it is possible.
    try:
        # TODO: this can be a lot of data? Perhaps take some subset? (but hard to know exactly how much cause it could e.g. have a 24-hour diff.)
        datamodule.inverse_transform_data(
            datamodule.get_data(["target"], main_split=split)["target"], partial=partial_ok
        )
    except Exception as e:
        log.exception("Inverse transform of datamodule pipeline failed with the following error")
        return None

    return lambda ts: datamodule.inverse_transform_data(
        darts.utils.ts_utils.seq2series(ts), partial=partial_ok
    )


def inverse_transform_data(
    inverse_transform_data_func: Union[None, Callable],
    data: Union[
        darts.TimeSeries,
        Sequence[darts.TimeSeries],
        Dict[str, Union[darts.TimeSeries, Sequence[darts.TimeSeries]]],
    ],
) -> Union[
    darts.TimeSeries,
    Sequence[darts.TimeSeries],
    Dict[str, Union[darts.TimeSeries, Sequence[darts.TimeSeries]]],
]:
    """Helper function to inverse transform a data structure.

    Data can either be a single TimeSeries object, a sequence of TimeSeries objects or a dictionary
    of a single or sequence of TimeSeries. Result is returned in the same data structure.
    :param inverse_transform_data_func: Function for inverse transforming data
    :param data: Data to be transformed
    :return: Inverse transformed data, or input data if inverse_transform_data_func is None or
        error occurred during inverse transformation.
    """
    if inverse_transform_data_func is None:
        return data

    assert callable(
        inverse_transform_data_func
    ), "inverse_transform_data_func must be a callable function"

    try:
        if isinstance(data, darts.TimeSeries):
            return inverse_transform_data_func(data)
        elif isinstance(data, list) and isinstance(data[0], darts.TimeSeries):
            return [inverse_transform_data_func(ts) for ts in data]
        elif isinstance(data, dict):
            res = {}
            for (
                data_type
            ) in data:  # TODO: can we transform them all at the same time? Is that more efficient?
                if data[data_type] is not None:
                    res[data_type] = inverse_transform_data_func(data[data_type])
                else:
                    res[data_type] = None
            return res
        else:
            raise ValueError(
                f"data has unexpected structure. Expect either a TimeSeries, a Sequence of TimeSeries or a dict. You have: {type(data)}"
            )
    except Exception as e:
        log.exception(
            "Inverse transform of datamodule pipeline failed with the following error. Returning data"
        )
        return data
