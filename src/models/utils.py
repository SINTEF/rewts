import glob
import inspect
import logging
import math
import os
from typing import Dict, List, Optional, Union

import darts.models.forecasting
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict

import src.models.ensemble_model
from src.utils import get_pylogger

log = get_pylogger(__name__)


def _get_model_class(
    model_or_cfg: Union[darts.models.forecasting.forecasting_model.ForecastingModel, DictConfig]
):
    if isinstance(model_or_cfg, DictConfig):
        model_target = OmegaConf.select(model_or_cfg, "model._target_")
        if model_target is None or model_target == "hydra.utils.get_class":
            model_target = OmegaConf.select(model_or_cfg, "_target_")
            if model_target is None:
                raise ValueError(
                    "Could not find _target_ attribute in config. Expected either model._target_ or _target_ to be present."
                )

        model_class = hydra.utils.get_class(model_target)
    else:
        assert isinstance(
            model_or_cfg, darts.models.forecasting.forecasting_model.ForecastingModel
        ), f"Invalid type for function {type(model_or_cfg)}, expected a darts.models.forecasting.forecasting_model.ForecastingModel object"
        model_class = type(model_or_cfg)

    return model_class


def is_torch_model(
    model_or_cfg: Union[darts.models.forecasting.forecasting_model.ForecastingModel, DictConfig]
) -> bool:
    """Check if a model (model instance, or model config) is a torch model.

    :param model_or_cfg: Model instance or config to be instantiated.
    :return: Whether model is a torch model.
    """
    model_class = _get_model_class(model_or_cfg)
    return issubclass(
        model_class, darts.models.forecasting.torch_forecasting_model.TorchForecastingModel
    )


def is_local_model(
    model_or_cfg: Union[darts.models.forecasting.forecasting_model.ForecastingModel, DictConfig]
) -> bool:
    """Check if a model (model instance, or model config) is a LocalForecastingModel.

    :param model_or_cfg: Model instance or config to be instantiated.
    :return: Whether model is a local model.
    """
    model_class = _get_model_class(model_or_cfg)
    return issubclass(
        model_class, darts.models.forecasting.forecasting_model.LocalForecastingModel
    )


def is_transferable_model(
    model_or_cfg: Union[darts.models.forecasting.forecasting_model.ForecastingModel, DictConfig]
) -> bool:
    """Check if a model (model instance, or model config) is a transferable model, i.e. it can
    predict on data it has not trained on.

    :param model_or_cfg: Model instance or config to be instantiated.
    :return: Whether model is a transferable model.
    """
    model_class = _get_model_class(model_or_cfg)

    is_local = issubclass(
        model_class, darts.models.forecasting.forecasting_model.LocalForecastingModel
    )

    return not is_local or issubclass(
        model_class,
        darts.models.forecasting.forecasting_model.TransferableFutureCovariatesLocalForecastingModel,
    )


def is_regression_model(
    model_or_cfg: Union[darts.models.forecasting.forecasting_model.ForecastingModel, DictConfig]
) -> bool:
    """Check if a model (model instance, or model config) is a RegressionModel.

    :param model_or_cfg: Model instance or config to be instantiated.
    :return: Whether model is a torch model.
    """
    model_class = _get_model_class(model_or_cfg)
    return issubclass(model_class, darts.models.forecasting.regression_model.RegressionModel)


def is_rewts_model(
    model_or_cfg: Union[darts.models.forecasting.forecasting_model.ForecastingModel, DictConfig]
) -> bool:
    """Check if a model (model instance, or model config) is a ReWTS ensemble model.

    :param model_or_cfg: Model instance or config to be instantiated.
    :return: Whether model is a ReWTS Ensemble model.
    """
    model_class = _get_model_class(model_or_cfg)
    return issubclass(model_class, src.models.ensemble_model.ReWTSEnsembleModel)


class SuppressMissingCheckpointWarning(logging.Filter):
    """Suppress warning about missing checkpoints."""

    def filter(self, record):
        """Filter records to remove warnings about missing checkpoints.

        :param record:
        :return:
        """
        if (
            "Model was loaded without weights since no PyTorch LightningModule checkpoint ('.ckpt') could be found at"
            in record.getMessage()
        ):
            return False
        return True


logging.getLogger("darts.models.forecasting.torch_forecasting_model").addFilter(
    SuppressMissingCheckpointWarning()
)


def get_best_checkpoint(checkpoint_dir) -> Union[str, None]:
    """Returns the path to the best checkpoint file in the given directory. The best checkpoint is
    identified by matching the names of the files in the directory against the pattern default
    pattern for best checkpoints, i.e. "epoch_*.ckpt". If no files match this pattern, the last
    checkpoint is returned. If no checkpoint files are found, None is returned.

    :param checkpoint_dir: Path to the directory containing the checkpoint files.
    :return: Path to the best checkpoint file if it exists, otherwise None.
    """
    if not os.path.exists(checkpoint_dir):
        log.warning(f"No checkpoint directory exists at {os.path.split(checkpoint_dir)[0]}")
        return None

    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "epoch_*.ckpt"))
    if len(ckpt_files) == 0:
        ckpt = "last.ckpt"
        assert os.path.exists(os.path.join(checkpoint_dir, ckpt)), "No checkpoints found."
        log.info(
            "Best checkpoint was requested but no checkpoints matching default pattern were found, using last.ckpt."
        )
    elif len(ckpt_files) == 1:
        ckpt = os.path.basename(ckpt_files[0])
        log.info(f"Found best checkpoint file: {ckpt}")
    else:
        ckpt = os.path.basename(ckpt_files[-1])
        log.info(
            f"Multiple checkpoints matching best pattern were found, selected the following checkpoint: {ckpt}"
        )

    return ckpt


def load_model(model_cfg, model_dir, ckpt=None):
    """Loads and returns a model from the given directory. If the model is a TorchForecastingModel,
    the checkpoint must be provided, otherwise an AssertionError is raised. If the model is not a
    TorchForecastingModel, the checkpoint is ignored.

    :param model_cfg: The model configuration object.
    :param model_dir: The directory containing the model file
    :param ckpt: The name of the checkpoint file to load weights and state from.
    :return: The loaded model.
    """
    model_class = hydra.utils.get_class(model_cfg._target_)
    if is_torch_model(model_cfg):
        assert (
            ckpt is not None
        ), "For TorchForecastingModels the model parameters are saved in the checkpoint object. The name of the checkpoint to load must therefore be provided"
        if ckpt == "best":  # TODO: see if it is now redundant to check this condition here
            ckpt = get_best_checkpoint(checkpoint_dir=os.path.join(model_dir, "checkpoints"))
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(model_dir, "checkpoints", ckpt)

        try:
            model = model_class.load(
                os.path.join(
                    model_dir,
                    getattr(darts.models.forecasting.torch_forecasting_model, "INIT_MODEL_NAME"),
                )
            )
            model.model = model._load_from_checkpoint(
                ckpt, **model.pl_module_params
            )  # TODO: probably should not be necessary to provide pl_module params.
        except RuntimeError:
            log.info("Model could not be loaded, attempting to map model to and load on CPU.")
            model = model_class.load(
                os.path.join(
                    model_dir,
                    getattr(darts.models.forecasting.torch_forecasting_model, "INIT_MODEL_NAME"),
                ),
                map_location="cpu",
            )
            model.model = model._load_from_checkpoint(
                ckpt, map_location="cpu", **model.pl_module_params
            )
        model.load_cpkt_path = ckpt
        model._fit_called = True  # TODO: gets set to False in the load method, can we check somehow if the model was actually fit?
    else:
        model = model_class.load(os.path.join(model_dir, "model.pkl"))

    return model


def ensure_torch_model_saving(model, model_work_dir) -> None:
    """Ensures that the given model is configured to save checkpoints in the given directory. If
    the model is not a TorchForecastingModel, a warning is logged and no operation is performed.

    :param model: The model to ensure saving for.
    :param model_work_dir: The directory to save checkpoints in.
    :return: None
    """
    if is_torch_model(model):
        model.save_checkpoints = True
        model.work_dir = model_work_dir
        model.model_name = ""
    else:
        log.info(
            "function was called with non torch model as argument, no operation was performed."
        )


def save_model(model, save_dir, save_name="model") -> None:
    """Saves the given model to the given directory. If the model is a TorchForecastingModel, the
    method performs no operation apart from logging an info message, as the model is saved through
    pytorch lightning callbacks.

    :param model: The model to save.
    :param save_dir: The directory to save the model in.
    :return: None
    """
    if is_torch_model(model):
        log.info(
            "Torch model saving is configured through pytorch lightning callbacks, no operation was done."
        )
        # TODO: maybe should save anyway? e.g. if something has changed in the object.
    else:
        save_path = os.path.join(save_dir, save_name + ".pkl")
        model.save(save_path)
        log.info(f"Model was saved to {save_path}")


def get_lstm_hidden_dim_from_params(
    n_params: int, n_layers: int, n_models: int, inputs: Dict[str, List[str]]
):
    n_inputs = sum(len(data_type) for data_type in inputs.values() if data_type is not None)
    # return round((np.sqrt(n_models * (n_inputs + 4) ** 2 * n_models + 3 * n_params) - (n_inputs + 4) * n_models) / (6 * n_models))
    n_stacked_layers = n_layers - 1
    return round(
        (
            np.sqrt(
                n_models
                * (
                    n_models * (n_inputs + 2 * n_stacked_layers + 2) ** 2
                    + 2 * n_params * n_stacked_layers
                    + n_params
                )
            )
            - n_models * (n_inputs + 2 * n_stacked_layers + 2)
        )
        / (2 * (2 * n_models * n_stacked_layers + n_models))
    )


def get_lstm_n_params_from_hidden_dim(
    hidden_dim: int, n_layers: int, n_models: int, inputs: Dict[str, List[str]]
):
    n_inputs = sum(len(data_type) for data_type in inputs.values() if data_type is not None)
    n_stacked_layers = n_layers - 1
    return round(
        n_models
        * (
            4 * (hidden_dim * n_inputs + hidden_dim**2 + hidden_dim)
            + n_stacked_layers * 4 * (2 * hidden_dim**2 + hidden_dim)
        )
    )


def get_global_hidden_dim_from_ensemble(
    ensemble_hidden_dim: int, n_layers: int, n_ensemble_models: int, inputs: Dict[str, List[str]]
):
    n_ensemble_params = get_lstm_n_params_from_hidden_dim(
        ensemble_hidden_dim, n_layers, n_ensemble_models, inputs
    )
    return get_lstm_hidden_dim_from_params(n_ensemble_params, n_layers, 1, inputs)


def calculate_TCN_matching_num_filters(
    input_chunk_length: int,
    output_chunk_length: int,
    kernel_size: int,
    num_layers: Optional[int],
    dilation_base: int,
    target_size: int,
    ensemble_num_filters: int,
    inputs: Dict[str, List[str]],
    n_ensemble_models: int,
) -> int:
    """Calculates the number of filters needed for a single TCN model to have a total number of
    parameters roughly equal to that of an ensemble of N identical TCN models.

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
                    (input_chunk_length - 1) * (dilation_base - 1) / (kernel_size - 1) / 2 + 1,
                    dilation_base,
                )
            )
        else:
            num_layers = np.ceil((input_chunk_length - 1) / (kernel_size - 1) / 2)

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


def scale_model_parameters(cfg: DictConfig) -> DictConfig:
    SUPPORTED_MODELS = [
        "darts.models.forecasting.rnn_model.RNNModel",
        "darts.models.forecasting.tcn_model.TCNModel",
        "darts.models.forecasting.xgboost.XGBModel",
    ]
    assert "model" in cfg, "config object must have a model entry"

    if cfg.model._target_ not in SUPPORTED_MODELS:
        log.warning(
            f"Model {cfg.model._target_} is not a supported model {SUPPORTED_MODELS}. No parameter scaling was performed."
        )

        return cfg
    elif "chunk_idx" not in cfg:
        log.warning(
            "No chunk_idx is set in config, cannot infer number of models in ensemble. No parameter scaling was performed."
        )
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
        matching_args = {
            k: (
                tcn_args[k].default
                if k in tcn_args and tcn_args[k].default is not inspect.Parameter.empty
                else None
            )
            for k in inspect.signature(calculate_TCN_matching_num_filters).parameters
        }
        matching_args.update({k: v for k, v in cfg.model.items() if k in matching_args})
        matching_args.update(
            dict(
                target_size=len(cfg.datamodule.data_variables.target)
                * cfg.model.output_chunk_length,
                ensemble_num_filters=cfg.model.num_filters,
                inputs=cfg.datamodule.data_variables,
                n_ensemble_models=1 + cfg.chunk_idx - cfg.chunk_idx_start,
            )
        )
        with open_dict(cfg):
            cfg.model.num_filters = calculate_TCN_matching_num_filters(**matching_args)
        log.info(
            f"model.num_filters was scaled from {num_filters_before} to {cfg.model.num_filters}"
        )
    elif cfg.model._target_ == "darts.models.forecasting.xgboost.XGBModel":
        n_ensemble_models = 1 + cfg.chunk_idx - cfg.chunk_idx_start
        if n_ensemble_models > 1:
            n_estimators_before = cfg.model.get("n_estimators", 100)
            n_estimators_new = n_estimators_before * n_ensemble_models
            with open_dict(cfg):
                cfg.model.n_estimators = n_estimators_new
            log.info(
                f"model.n_estimators was scaled from {n_estimators_before} to {n_estimators_new}"
            )

    return cfg
