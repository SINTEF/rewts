import pyrootutils
import pytorch_lightning

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import List, Tuple, Optional, Dict, Any, Union
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from src import utils
import src.utils.model_io
import src.utils.plotting
from src.utils.utils import call_function_with_data
import torch.utils.data
import darts.models.forecasting.torch_forecasting_model
import src.datamodules
import inspect
import functools
import pandas as pd
import src.eval
import matplotlib.pyplot as plt

import os

log = utils.get_pylogger(__name__)


def _assert_predict_cfg(cfg: DictConfig):
    assert cfg.model_dir
    assert cfg.predict.indices
    assert cfg.predict.kwargs, "Must supply arguments for the model.predict function"
    assert cfg.predict.kwargs.n, "Must supply the argument n which sets the forecasting horizon."


def process_predict_index(predict_index: Union[str, pd.Timestamp, int, float], predict_n, model, data, fit_ensemble_weights) -> Union[pd.Timestamp, int]:
    # maybe have to scale predict_index to account for needing lags and future covariates at each end?
    if isinstance(predict_index, float):
        assert 0 <= predict_index <= 1, "Prediction indices with float-values must be between 0 (first predictable sample) and 1 (last predictable sample)."
        new_max = int((len(data["series"]) - 1))
        new_min = max(model.extreme_lags[0] * -1, 1)
        if data.get("future_covariates") is not None:
            if model.extreme_lags[-1] is not None:
                new_max = min(new_max,
                              (len(data["future_covariates"]) - 1) - (predict_n - 1) - max(
                                  model.extreme_lags[-1], 0))
            if model.extreme_lags[-2] is not None:
                new_min = max(model.extreme_lags[-2] * -1, new_min)
        if data.get("past_covariates") is not None and model.extreme_lags[2] is not None:
            new_max = min(new_max, len(data["past_covariates"]) - (predict_n - 1))
            new_min = max(model.extreme_lags[2] * -1, new_min)
        if fit_ensemble_weights and isinstance(model, src.models.ensemble_model.TSEnsembleModel):
            new_min += model.lookback_data_length
            if model.lookback_data_length >= len(data["series"]) + model.extreme_lags[0]:
                log.error("fit_ensemble_weights is True, but the predict dataset is shorter than what is required by the fit function. Cannot predict on this dataset.")
        predict_index = round(src.utils.linear_scale(predict_index, new_max, new_min, 1.0, 0.0))
        if isinstance(data["series"].time_index, pd.RangeIndex):
            predict_index += data["series"].time_index.start
    else:
        if isinstance(predict_index, str):
            predict_index = pd.Timestamp(predict_index)
        # TODO: assert index is predictable

    return predict_index


def run(cfg: DictConfig, datamodule: src.datamodules.TimeSeriesDataModule, model) -> Tuple[Dict[str, List[float]], List[darts.TimeSeries], List[plt.Figure]]:
    _assert_predict_cfg(cfg)

    if isinstance(model, darts.models.forecasting.forecasting_model.LocalForecastingModel):
        raise NotImplementedError("LocalForecastingModels are not supported yet.")

    log.info("Starting prediction!")

    metric_funcs = cfg.predict.get("metric", None)

    if metric_funcs is not None:
        metric_funcs = hydra.utils.instantiate(metric_funcs)
        if callable(metric_funcs):
            metric_funcs = [metric_funcs]

        metric_names = [src.eval._get_metric_name(cfg.predict.split, metric_func) for metric_func in metric_funcs]
        metric_dict = {metric_name: [] for metric_name in metric_names}
    else:
        metric_dict = {}

    # TODO: add support for naive models... (they dont take in series in model.predict, so have to fit before...)

    predict_split_data = src.utils.get_model_supported_data(datamodule, model, main_split=cfg.predict.split)
    if datamodule.num_series_for_split(cfg.predict.split) > 1:
        predict_split_data = {k: v[cfg.predict.get("series_index", 0)] for k, v in predict_split_data.items()}

    figs = []
    predictions = []

    if cfg.predict.get("inverse_transform_data"):
        inverse_transform_data_func = lambda ts: datamodule.inverse_transform_data(ts, partial=cfg.predict.inverse_transform_data.get("partial_ok", False))
    else:
        inverse_transform_data_func = None

    for p_i, predict_index in enumerate(cfg.predict.get("indices")):
        predict_index = process_predict_index(predict_index, cfg.predict.kwargs.n, model, predict_split_data, cfg.predict.get("fit_ensemble_weights", False))
        if predict_index == len(predict_split_data["series"]) - 1:
            predict_series = predict_split_data["series"]
        else:
            predict_series = predict_split_data["series"].drop_after(predict_index)

        if isinstance(model, src.models.ensemble_model.TSEnsembleModel):
            if cfg.predict.get("fit_ensemble_weights", False):
                model.reset()
                model._fit_weights = True
                model._weights_last_update = -model.fit_weights_every
            else:
                model._fit_weights = False
        prediction = model.predict(series=predict_series,
                                   **{data_type: data for data_type, data in predict_split_data.items() if data_type != "series"},
                                   **cfg.predict.kwargs)

        if metric_funcs is not None:
            for m_i, m in enumerate(metric_funcs):
                metric_dict[metric_names[m_i]].append(m(predict_split_data["series"], prediction, intersect=True))

        fig_title = f"Prediction for {cfg.predict.split} dataset at time {prediction.start_time()}"
        if cfg.predict.get("presenter") is not None:
            presenter_kwargs = dict(fname=os.path.join(cfg.paths.output_dir, "plots", "predictions", fig_title.lower().replace(" ", "_").replace(":", "-")))
        else:
            presenter_kwargs = None
        for metric_name, metric_values in metric_dict.items():
            fig_title += f" {'_'.join(metric_name.split('_')[1:])}={metric_values[-1]:.2E}"

        figs.append(src.utils.plotting.plot_prediction(prediction,
                                                       predict_split_data,
                                                       model,
                                                       cfg.predict.presenter,
                                                       predict_n=cfg.predict.kwargs.n,
                                                       plot_covariates=cfg.predict.plot.get("covariates", False),
                                                       plot_encodings=cfg.predict.plot.get("encodings", False),
                                                       plot_prediction_point=cfg.predict.plot.get("prediction_point", False),
                                                       plot_weights=cfg.predict.plot.get("weights", False),
                                                       inverse_transform_data_func=inverse_transform_data_func,
                                                       fig_title=fig_title,
                                                       presenter_kwargs=[presenter_kwargs]))
        predictions.append(prediction)

        if isinstance(model, src.models.ensemble_model.TSEnsembleModel):
            model.save_weights(os.path.join(cfg.paths.output_dir, f"predict_{p_i}_weights.npy"))

    return metric_dict, predictions, figs


@utils.task_wrapper
def predict(cfg: DictConfig) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    _assert_predict_cfg(cfg)

    object_dict = src.utils.initialize_saved_objects(cfg)
    model = object_dict["model"]
    datamodule = object_dict["datamodule"]
    trainer = object_dict.get("trainer", None)
    logger = object_dict.get("logger", None)

    object_dict["cfg"] = cfg

    metric_dict, predictions, figs = run(cfg, datamodule, model)

    object_dict["predictions"] = predictions
    object_dict["figs"] = figs

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    assert cfg.model_dir
    cfg = src.utils.load_saved_config(cfg.model_dir, cfg)
    predict(cfg)


if __name__ == "__main__":
    src.utils.utils.enable_eval_resolver()
    main()
