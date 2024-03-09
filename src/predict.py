import omegaconf.errors
import pyrootutils

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import darts.models.forecasting.torch_forecasting_model
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts.models.forecasting.forecasting_model import ForecastingModel
from omegaconf import DictConfig, OmegaConf

import src.datamodules
import src.eval
import src.metrics.utils
import src.models.utils
import src.utils.plotting
from src import utils
from src.utils import call_function_with_resolved_arguments

log = utils.get_pylogger(__name__)


def _assert_predict_cfg(cfg: DictConfig):
    """Check that required arguments are present in the config.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: None
    """
    assert cfg.predict.indices
    assert cfg.predict.kwargs, "Must supply arguments for the model.predict function"
    assert cfg.predict.kwargs.n, "Must supply the argument n which sets the forecasting horizon."


def process_predict_index(
    predict_index: Union[str, pd.Timestamp, int, float],
    predict_n: int,
    model: ForecastingModel,
    data: Dict[str, Union[darts.TimeSeries, Sequence[darts.TimeSeries]]],
    retrain: bool = False,
    fit_ensemble_weights: bool = False,
) -> Union[pd.Timestamp, int]:
    """
    Helper function to take a dataset index to be predicted on and process it: 1) Float indices are treated as the
    percentile datapoint and converted to an integer index, 2) int indices are returned as is, 3) str indices are
    converted to pd.Timestamps and timezone information is removed.
    :param predict_index: dataset index to predict
    :param predict_n: Number of steps to predict into the future
    :param model: model used for prediction. Used to infer necessary amount of lags.
    :param data: Data to predict on, used to assert that necessary data is present.
    :param retrain: Whether the model is to be retrained on prediction data before predicting.
    :param fit_ensemble_weights: Whether ensemble weights are to be fitted before predicting, requires at least ensemble.lookback_data_length datapoints before prediction index
    :return: processed predict_index
    """
    extreme_lags_indices = dict(
        min_target_lag=0,
        max_target_lag=1,
        min_past_cov_lag=2,
        max_past_cov_lag=3,
        min_future_cov_lag=4,
        max_future_cov_lag=5,
        output_chunk_shift=6,
    )

    def get_extreme_lag(lag_name):
        """Utility function to get the extreme lag (i.e. furthest time index required) for a named
        data type."""
        return model.extreme_lags[extreme_lags_indices[lag_name]]

    if get_extreme_lag("output_chunk_shift") > 0:
        raise NotImplementedError

    # maybe have to scale predict_index to account for needing lags and future covariates at each end?
    if isinstance(predict_index, float):
        assert (
            0 <= predict_index <= 1
        ), "Prediction indices with float-values must be between 0 (first predictable sample) and 1 (last predictable sample)."
        new_max = int(len(data["series"]) - 1)
        new_min = max(get_extreme_lag("min_target_lag") * -1, 1)
        if retrain:
            new_min = max(new_min, model.min_train_series_length)
        if data.get("future_covariates") is not None:
            if get_extreme_lag("max_future_cov_lag") is not None:
                new_max = min(
                    new_max,
                    (len(data["future_covariates"]) - 1)
                    - (predict_n - 1)
                    - max(get_extreme_lag("max_future_cov_lag"), 0),
                )
            if get_extreme_lag("min_future_cov_lag") is not None:
                new_min = max(get_extreme_lag("min_future_cov_lag") * -1, new_min)
        if (
            data.get("past_covariates") is not None
            and get_extreme_lag("min_past_cov_lag") is not None
        ):
            new_max = min(new_max, len(data["past_covariates"]) - (predict_n - 1))
        if fit_ensemble_weights and src.models.utils.is_rewts_model(model):
            new_min += model.lookback_data_length
            if model.lookback_data_length >= len(data["series"]) + get_extreme_lag(
                "min_target_lag"
            ):
                log.error(
                    "fit_ensemble_weights is True, but the predict dataset is shorter than what is required by the fit function. Cannot predict on this dataset."
                )
        predict_index = round(src.utils.linear_scale(predict_index, new_max, new_min, 1.0, 0.0))
        if isinstance(data["series"].time_index, pd.RangeIndex):
            predict_index += data["series"].time_index.start
    else:
        if isinstance(predict_index, str):
            predict_index = pd.Timestamp(predict_index)
            if predict_index.tz is not None:
                predict_index = predict_index.tz_localize(None)
        # TODO: assert index is predictable

    return predict_index


def run(
    cfg: DictConfig,
    datamodule: src.datamodules.TimeSeriesDataModule,
    model,
    logger=None,
    trainer=None,
) -> Tuple[
    Dict[str, List[float]], Union[None, List[darts.TimeSeries]], Union[None, List[plt.Figure]]
]:
    """Run prediction on specific predict indices, producing predictions, figures, and metrics.

    :param cfg: Configuration composed by Hydra.
    :param datamodule: Datamodule containing data to predict on.
    :param model: Model to predict with.
    :param logger: Logger objects to log figures and metrics to.
    :param trainer: Trainer object for pytorch models when predict.retrain is True.
    :return: metrics, predictions, figures
    """
    _assert_predict_cfg(cfg)

    if not datamodule.has_split_data(cfg.predict.split):
        log.error(
            f"No data is available for the predict split ({cfg.predict.split}). Skipping prediction."
        )
        return {}, None, None

    inverse_transform_data_func = src.utils.get_inverse_transform_data_func(
        cfg.predict.get("inverse_transform_data"), datamodule, cfg.predict.split
    )

    metric_funcs = cfg.predict.get("metric", None)
    if metric_funcs is not None:
        metric_funcs = hydra.utils.instantiate(metric_funcs)
        metric_funcs, metric_names = src.metrics.utils.process_metric_funcs(
            metric_funcs, "predict", inverse_transform_data_func
        )

        metric_dict = {metric_name: [] for metric_name in metric_names}
    else:
        metric_dict = {}

    # TODO: add support for naive models... (they dont take in series in model.predict, so have to fit before...)

    predict_split_data = src.utils.get_model_supported_data(
        datamodule, model, main_split=cfg.predict.split
    )
    if datamodule.num_series_for_split(cfg.predict.split) > 1:
        predict_split_data = {
            k: v[cfg.predict.get("series_index", 0)] for k, v in predict_split_data.items()
        }

    figs = []
    predictions = []

    log.info("Starting prediction!")

    retrain = OmegaConf.select(cfg.predict, "retrain", default=False)
    for p_i, predict_index in enumerate(cfg.predict.get("indices")):
        predict_index = process_predict_index(
            predict_index,
            cfg.predict.kwargs.n,
            model,
            predict_split_data,
            retrain=retrain,
            fit_ensemble_weights=OmegaConf.select(
                cfg, "predict.ensemble_weights.fit", default=False
            ),
        )
        if predict_index == len(predict_split_data["series"]) - 1:
            predict_series = predict_split_data["series"]
        # TODO: is this creating a copy? Can we optimize by giving a view or something?
        else:
            predict_series = predict_split_data["series"].drop_after(predict_index)

        if src.models.utils.is_rewts_model(model):
            if OmegaConf.select(cfg, "predict.ensemble_weights.fit", default=False):
                model.reset()
                model._fit_weights = True
                model._weights_last_update = -model.fit_weights_every
            else:
                model._fit_weights = False
        log.info(f"Predicting index {predict_index}")
        if retrain or not src.models.utils.is_transferable_model(model):
            log.info("Model is retrained on data preceding prediction point!")

            fit_kwargs = OmegaConf.select(cfg, "fit", default={})
            if isinstance(fit_kwargs, DictConfig):
                fit_kwargs = OmegaConf.to_container(fit_kwargs)
            fit_kwargs["trainer"] = trainer

            model = call_function_with_resolved_arguments(
                model.fit,
                series=predict_series,
                **{
                    data_type: data
                    for data_type, data in predict_split_data.items()
                    if data_type != "series"
                },
                **fit_kwargs,
            )

            prediction = model.predict(
                **{
                    data_type: data
                    for data_type, data in predict_split_data.items()
                    if data_type != "series"
                },
                **cfg.predict.kwargs,
            )
        else:
            prediction = model.predict(
                series=predict_series,
                **{
                    data_type: data
                    for data_type, data in predict_split_data.items()
                    if data_type != "series"
                },
                **cfg.predict.kwargs,
            )

        if metric_funcs is not None:
            target_prediction_intersection = predict_split_data["series"].slice_intersect(
                prediction
            )
            for m_i, m in enumerate(metric_funcs):
                metric_dict[metric_names[m_i]].append(
                    m(target_prediction_intersection, prediction)
                )

        fig_title = f"Prediction for {cfg.predict.split} dataset at time {prediction.start_time()}"

        presenters, presenter_kwargs = src.utils.get_presenters_and_kwargs(
            cfg.predict.plot.get("presenter"),
            os.path.join(cfg.paths.output_dir, "plots"),
            os.path.join("predictions", fig_title.lower().replace(" ", "_").replace(":", "-")),
            logger=logger,
            trainer=None,
        )

        for metric_name, metric_values in metric_dict.items():
            fig_title += f" {'_'.join(metric_name.split('_')[1:])}={metric_values[-1]:.2E}"

        figs.append(
            src.utils.plotting.plot_prediction(
                prediction,
                predict_split_data,
                model,
                presenters,
                inverse_transform_data_func=inverse_transform_data_func,
                fig_title=fig_title,
                presenter_kwargs=presenter_kwargs,
                **cfg.predict.plot.get("kwargs", {}),
            )
        )
        predictions.append(prediction)

        if src.models.utils.is_rewts_model(model) and OmegaConf.select(
            cfg, "predict.ensemble_weights.save", default=False
        ):
            model.save_weights(
                os.path.join(cfg.paths.output_dir, "ensemble_weights", f"predict_{p_i}_weights")
            )

    # TODO: decide if data format for prediction metrics (dict with key index? list? use reduction?)
    if logger is not None:
        for lg in logger:
            lg.log_metrics({k: np.mean(v) for k, v in metric_dict.items()})

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

    object_dict = src.utils.instantiate_saved_objects(cfg)
    model = object_dict["model"]
    datamodule = object_dict["datamodule"]
    trainer = object_dict.get("trainer", None)
    logger = object_dict.get("logger", None)

    object_dict["cfg"] = cfg

    metric_dict, predictions, figs = run(cfg, datamodule, model, logger, trainer)

    object_dict["predictions"] = predictions
    object_dict["figs"] = figs

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for prediction.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    cfg = src.utils.verify_and_load_config(cfg)
    predict(cfg)


if __name__ == "__main__":
    src.utils.utils.enable_eval_resolver()
    main()
