import pyrootutils
import pytorch_lightning

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import inspect
import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import darts.models.forecasting.torch_forecasting_model
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from darts import TimeSeries
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import Logger

import src.datamodules
import src.metrics.utils
import src.models.ensemble_model
import src.models.utils
import src.train
import src.utils.plotting
from src import utils
from src.utils.utils import call_function_with_data

log = utils.get_pylogger(__name__)


# TODO: assert we have enough data
def run(
    cfg: DictConfig,
    datamodule: src.datamodules.TimeSeriesDataModule,
    model,
    trainer: Optional[pytorch_lightning.Trainer] = None,
    logger: Optional[List[Logger]] = None,
    ckpt_path: Optional[str] = None,
) -> Tuple[
    Dict[str, Union[float, List[float]]],
    Dict[str, Union[TimeSeries, Sequence[TimeSeries], plt.Figure, List[plt.Figure]]],
]:
    """Runs the evaluation of the model. The results of the evaluation are logged to the logger and
    saved to the model directory, and returned as a dictionary with keys corresponding to the
    metric names and values corresponding to the metric values.

    :param cfg: The config object.
    :param datamodule: The datamodule.
    :param model: The model.
    :param trainer: The PyTorch Lightning Trainer.
    :param logger: The logger(s).
    :param ckpt_path: The path to the checkpoint to load.
    :return: The metrics.
    """
    metric_dict, object_dict = {}, {}

    if not datamodule.has_split_data(cfg.eval.split):
        log.error(
            f"No data is available for the eval split ({cfg.eval.split}). Skipping evaluation."
        )
        return metric_dict, object_dict

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed") is not None:
        pl.seed_everything(cfg.seed, workers=True)

    if isinstance(model, darts.models.forecasting.torch_forecasting_model.TorchForecastingModel):
        if hasattr(model.model, "set_mc_dropout"):
            model.model.set_mc_dropout(cfg.eval.get("mc_dropout", False))

    time_metrics = {}

    if "kwargs" in cfg.eval:
        eval_kwargs = dict(cfg.eval.kwargs)
    else:
        eval_kwargs = {}

    metric_funcs = eval_kwargs.pop("metric", None)

    if metric_funcs is None:
        metric_funcs = inspect.signature(model.backtest).parameters["metric"].default
    else:
        metric_funcs = hydra.utils.instantiate(metric_funcs, _convert_="partial")

    inverse_transform_data_func = src.utils.get_inverse_transform_data_func(
        cfg.eval.get("inverse_transform_data"), datamodule, cfg.eval.split
    )
    # TODO: also apply the inverse transformation optimization when not plotting somehow?
    metric_funcs, metric_names = src.metrics.utils.process_metric_funcs(
        metric_funcs,
        cfg.eval.split,
        inverse_transform_data_func=(
            None if cfg.eval.get("plot") is not None else inverse_transform_data_func
        ),
    )

    is_multiple_series = datamodule.num_series_for_split(cfg.eval.split) > 1

    log.info("Starting evaluation!")
    if cfg.eval.get("plot") is not None or cfg.eval.get("predictions", None):
        with src.utils.time_block(
            enabled=cfg.eval.get("measure_execution_time"),
            metric_dict=time_metrics,
            log_file=(
                os.path.join(cfg.paths.output_dir, "eval_exec_time.log")
                if logger is None
                else None
            ),
        ):
            predictions_seq = call_function_with_data(
                model.historical_forecasts,
                datamodule,
                main_split=cfg.eval.split,
                model=model,
                last_points_only=False,
                **eval_kwargs,
            )

        if len(predictions_seq) == 0:
            log.error(
                "No predictions were obtained with current config. Ensure cfg.eval.split has enough data to produce predictions given eval.kwargs.forecast_horizon, and model.input_chunk_length, etc. "
            )
            return metric_dict, object_dict

        if datamodule.num_series_for_split(cfg.eval.split) > 1:
            is_multiple_series = True
        else:
            is_multiple_series = False
            predictions_seq = [predictions_seq]

        for pred_i in range(len(predictions_seq)):
            if inverse_transform_data_func is not None:
                predictions_seq[pred_i] = src.utils.inverse_transform_data(
                    inverse_transform_data_func, predictions_seq[pred_i]
                )
            if eval_kwargs.get("forecast_horizon", 1) == eval_kwargs.get("stride", 1):
                predictions_seq[pred_i] = darts.timeseries.concatenate(predictions_seq[pred_i])

        prediction_data_seq = src.utils.get_model_supported_data(
            datamodule, model, main_split=cfg.eval.split
        )

        if inverse_transform_data_func is not None:
            prediction_data_seq = src.utils.inverse_transform_data(
                inverse_transform_data_func, prediction_data_seq
            )

        # TODO: a substantial speedup can be achieved if possible with TimeSeries instead of Sequence of TimeSeries
        metrics = model.backtest(
            prediction_data_seq["series"],
            historical_forecasts=predictions_seq if is_multiple_series else predictions_seq[0],
            metric=metric_funcs,
        )

        metrics = np.array(np.atleast_1d(metrics)).reshape(len(predictions_seq), len(metric_funcs))

        # SineDataModule feature to scale metrics by the amplitude of each chunk
        if (
            "data_args" in datamodule.hparams
            and cfg.eval.get("metrics_scale_amplitude")
            and (is_multiple_series or len(datamodule.hparams["data_args"]) == 1)
        ):
            metrics /= np.array(
                [series_args["amplitude"] for series_args in datamodule.hparams["data_args"]]
            ).reshape(-1, 1)

        if OmegaConf.select(cfg.eval, "predictions.save"):
            # TODO: argument to control if data/predictions should be saved transformed or non-transformed
            if is_multiple_series:
                predictions_to_save = predictions_seq
            else:
                predictions_to_save = predictions_seq[0]

            pred_folder = os.path.join(cfg.paths.output_dir, "predictions")
            os.makedirs(pred_folder, exist_ok=True)

            # TODO: use built in timeseries to pickle?
            predictions_save_path = os.path.join(pred_folder, "predictions.pkl")
            log.info(f"Saving predictions to {predictions_save_path}")
            with open(predictions_save_path, "wb") as f:
                pickle.dump(predictions_to_save, f)

            if OmegaConf.select(cfg.eval, "predictions.save.data"):
                predictions_data_save_path = os.path.join(pred_folder, "data.pkl")
                log.info(f"Also saving the data predicted on to {predictions_data_save_path}")
                with open(predictions_data_save_path, "wb") as f:
                    pickle.dump(prediction_data_seq, f)

        if OmegaConf.select(cfg.eval, "predictions.return"):
            if is_multiple_series:
                object_dict["predictions"] = predictions_seq
            else:
                object_dict["predictions"] = predictions_seq[0]

            if OmegaConf.select(cfg.eval, "predictions.return.data"):
                object_dict["predictions_data"] = prediction_data_seq

        if cfg.eval.get("plot"):
            log.info("Plotting predictions!")
            object_dict["figs"] = []
            for pred_i, predictions in enumerate(predictions_seq):
                if OmegaConf.select(
                    cfg.eval, "plot.every_n_prediction", default=1
                ) > 1 and eval_kwargs.get("forecast_horizon", 1) > eval_kwargs.get("stride", 1):
                    assert isinstance(
                        predictions, list
                    ), "Predictions were expected to be a list, please report."
                    predictions = predictions[:: cfg.eval.plot.every_n_prediction]

                plot_title = None
                if OmegaConf.select(cfg.eval, "plot.title"):
                    plot_title = OmegaConf.select(cfg.eval, "plot.title")
                if OmegaConf.select(cfg.eval, "plot.title_add_metrics"):
                    if plot_title is None:
                        plot_title = ""
                    plot_title += " ".join(
                        f"{'_'.join(metric_name.split('_')[1:])}={metrics[pred_i, m_i]:.2E}"
                        for m_i, metric_name in enumerate(metric_names)
                    )

                if is_multiple_series:
                    fig_name = f"{cfg.eval.split}-{pred_i}-predictions"
                else:
                    fig_name = f"{cfg.eval.split}-predictions"
                fig_name = os.path.join(
                    "predictions", fig_name
                )  # TODO: put predictions as parent folder in fig name so that it gets organized in MLFlow logger

                presenters, presenter_kwargs = src.utils.get_presenters_and_kwargs(
                    OmegaConf.select(cfg.eval, "plot.presenter", default=["savefig", None]),
                    os.path.join(cfg.paths.output_dir, "plots"),
                    fig_name,
                    logger=logger,
                    trainer=trainer,
                )

                figs = src.utils.plotting.plot_prediction(
                    predictions,
                    {
                        k: v[pred_i] if is_multiple_series else v
                        for k, v in prediction_data_seq.items()
                    },
                    model,
                    presenters,
                    fig_title=plot_title,
                    presenter_kwargs=presenter_kwargs,
                    **OmegaConf.select(cfg.eval, "plot.kwargs", default={}),
                )
                for fig in figs:
                    if fig is not None:
                        object_dict["figs"].append(fig)
    else:
        eval_kwargs["metric"] = metric_funcs
        with src.utils.time_block(
            enabled=cfg.eval.get("measure_execution_time"),
            metric_dict=time_metrics,
            log_file=(
                os.path.join(cfg.paths.output_dir, "eval_exec_time.log")
                if logger is None
                else None
            ),
        ):
            metrics = call_function_with_data(
                model.backtest,
                datamodule,
                main_split=cfg.eval.split,
                model=model,
                **eval_kwargs,
            )
        num_eval_series = datamodule.num_series_for_split(cfg.eval.split)
        metrics = np.array(np.atleast_1d(metrics)).reshape(num_eval_series, len(metric_funcs))
        # SineDataModule feature to scale metrics by the amplitude of each chunk
        if (
            "data_args" in datamodule.hparams
            and cfg.eval.get("metrics_scale_amplitude")
            and (is_multiple_series or len(datamodule.hparams["data_args"]) == 1)
        ):
            metrics /= np.array(
                [series_args["amplitude"] for series_args in datamodule.hparams["data_args"]]
            ).reshape(-1, 1)

    if cfg.eval.metrics_per_series and metrics.shape[0] > 1:
        metric_dict = {
            f"{metric_name}_{s_i}": metrics[s_i, m_i]
            for m_i, metric_name in enumerate(metric_names)
            for s_i in range(metrics.shape[0])
        }

    metrics = np.mean(metrics, axis=0)  # get mean metrics over series

    metric_dict.update({metric_name: metrics[m_i] for m_i, metric_name in enumerate(metric_names)})
    metric_dict.update({f"eval_{k}": v for k, v in time_metrics.items()})

    if src.models.utils.is_rewts_model(model) and OmegaConf.select(
        cfg, "eval.ensemble_weights.save", default=False
    ):
        model.save_weights(
            os.path.join(
                cfg.paths.output_dir, "ensemble_weights", f"eval_{cfg.eval.split}_weights"
            )
        )

    if cfg.eval.get("log_metrics"):  # TODO: translate split_loss into split_rmse?
        split_range = datamodule.get_split_range(cfg.eval.split)
        if isinstance(split_range[0], pd.Timestamp):
            split_range = tuple(index.strftime("%Y-%m-%d") for index in split_range)
        results_to_serialize = dict(
            dataset_range=split_range,
            split=cfg.eval.split,
            model_name=str(model),  # make prettier?
            forecast_horizon=cfg.eval.kwargs.get("forecast_horizon", 1),
            stride=cfg.eval.kwargs.get("stride", 1),
            start=cfg.eval.kwargs.get("start", None),
            metrics={k: float(v) for k, v in metric_dict.items()},
        )
        if isinstance(
            model,
            darts.models.forecasting.torch_forecasting_model.TorchForecastingModel,
        ):
            results_to_serialize["model_name"] = model.__class__.__name__
            results_to_serialize["ckpt"] = (
                os.path.basename(ckpt_path) if ckpt_path is not None else ckpt_path
            )
        OmegaConf.save(
            OmegaConf.create(results_to_serialize),
            os.path.join(cfg.paths.output_dir, f"eval_{cfg.eval.split}_results.yaml"),
        )

        if logger is not None:
            for lg in logger:
                lg.log_metrics(metric_dict)

    return metric_dict, object_dict


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    object_dict = src.utils.instantiate_saved_objects(cfg)

    model = object_dict["model"]
    datamodule = object_dict["datamodule"]
    trainer = object_dict.get("trainer", None)
    logger = object_dict.get("logger", [])

    object_dict["cfg"] = cfg

    metric_dict, run_object_dict = run(cfg, datamodule, model, logger=logger, trainer=trainer)
    object_dict.update(run_object_dict)

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)
        for lg in logger:
            lg.finalize("success")

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    cfg = src.utils.verify_and_load_config(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    src.utils.utils.enable_eval_resolver()
    main()
