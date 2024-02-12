import pyrootutils
import pytorch_lightning

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import List, Tuple, Optional
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf, ListConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import pytorch_lightning as pl
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
import numpy as np
import os
import pickle

import src.models.ensemble_model


log = utils.get_pylogger(__name__)

_EVAL_RUNNERS = ["trainer", "backtest"]


def _get_metric_name(split, metric_func):
    """
    Returns the name of the metric function to use for logging on the form <split>_<metric_func.__name__>. Gets the name
    of the metric from the function name.

    :param split: The split to use for the metric name.
    :param metric_func: The metric function to use.

    :return: The name of the metric.
    """
    if isinstance(metric_func, functools.partial):
        metric_name = metric_func.func.__name__
    else:
        metric_name = metric_func.__name__

    return f"{split}_{metric_name}"


def _run_darts(cfg: DictConfig, datamodule: src.datamodules.TimeSeriesDataModule, model, logger: Optional[List[Logger]] = None, trainer: Optional[pytorch_lightning.Trainer] = None):
    """
    Runs the evaluation of a darts model using the darts functions backtest/historical_forecasts performed on the split
    specified in the config. The results are returned as a dictionary containing the metric names as keys and metric
    values as values. The results are also logged using the provided loggers. If the plot_predictions flag is set to
    True and the forecast_horizon is 1 the predictions are plotted and saved to the model directory and the loggers.

    :param cfg: The config to use for evaluation.
    :param datamodule: The datamodule to use for evaluation.
    :param model: The model to evaluate.
    :param logger: The loggers to use for logging.
    :param trainer: If a tensorboard logger is used, the trainer is used to get the current step.

    :return: The results of the evaluation.
    """
    if isinstance(model, darts.models.forecasting.torch_forecasting_model.TorchForecastingModel):
        log.info("Your model is a TorchForecastingModel but you have chosen to evaluate with the backtest method. "
                 "This method generates and evaluates successive datasets of a single sample for each datapoint in the evaluation dataset. "
                 "A substantial speedup can be achieved by using the runner=trainer instead.")

    time_metrics = {}

    if "kwargs" in cfg.eval:
        eval_kwargs = dict(cfg.eval.kwargs)
    else:
        eval_kwargs = {}

    metric_funcs = eval_kwargs.pop("metric", None)

    if metric_funcs is None:
        metric_funcs = inspect.signature(model.backtest).parameters['metric'].default
    else:
        metric_funcs = hydra.utils.instantiate(metric_funcs, _convert_="partial")

    if callable(metric_funcs):
        metric_funcs = [metric_funcs]

    metric_names = [_get_metric_name(cfg.eval.split, metric_func) for metric_func in metric_funcs]

    if cfg.eval.get("inverse_transform_data", False):
        partial_ok = cfg.eval.inverse_transform_data.get("partial_ok", False)
        inverse_transform_data_func = lambda ts: datamodule.inverse_transform_data(ts, partial=partial_ok)
        try:
            datamodule.inverse_transform_data(getattr(datamodule, f"data_{cfg.eval.split}")["target"][:1], partial=partial_ok)
            for i, metric_func in enumerate(metric_funcs):
                metric_funcs[i] = lambda actuals, preds, *args, m_func=metric_func, **kwargs: \
                    m_func(
                        datamodule.inverse_transform_data(darts.utils.utils.seq2series(actuals), partial=partial_ok),
                        datamodule.inverse_transform_data(darts.utils.utils.seq2series(preds), partial=partial_ok),
                        *args,
                        **kwargs
                    )
        except Exception as e:
            log.exception("Inverse transform of datamodule pipeline failed with the following error")
    else:
        inverse_transform_data_func = None

    if cfg.eval.get("plot") is not None or cfg.eval.get("save_predictions", False):
        with src.utils.time_block(enabled=cfg.eval.get("measure_execution_time"),
                                  metric_dict=time_metrics,
                                  log_file=os.path.join(cfg.paths.output_dir, "eval_exec_time.log") if logger is None else None
                                  ):
            predictions_seq = call_function_with_data(model.historical_forecasts, datamodule, main_split=cfg.eval.split, model=model, last_points_only=False, **eval_kwargs)

        if len(predictions_seq) == 0:
            log.error("No predictions were obtained with current config. Ensure cfg.eval.split has enough data to produce predictions given eval.kwargs.forecast_horizon, and model.input_chunk_length, etc. ")
            return {}

        prediction_data_seq = src.utils.get_model_supported_data(datamodule, model, main_split=cfg.eval.split)
        metrics = model.backtest(**prediction_data_seq, historical_forecasts=predictions_seq, metric=metric_funcs)

        if datamodule.num_series_for_split(cfg.eval.split) > 1:
            is_multiple_series = True
            metrics = np.array(metrics).reshape(len(predictions_seq), len(metric_funcs))
        else:
            is_multiple_series = False
            predictions_seq = [predictions_seq]
            metrics = np.array(np.atleast_1d(metrics)).reshape(1, len(metric_funcs))

        # SineDataModule feature to scale metrics by the amplitude of each chunk
        if "data_args" in datamodule.hparams and cfg.eval.get("metrics_scale_amplitude") and (is_multiple_series or len(datamodule.hparams["data_args"]) == 1):
            metrics /= np.array([series_args["amplitude"] for series_args in datamodule.hparams["data_args"]]).reshape(-1, 1)

        if cfg.eval.get("save_predictions", False):
            if inverse_transform_data_func is not None:
                predictions_to_save = [inverse_transform_data_func(seq_p) if isinstance(seq_p, darts.TimeSeries) else [inverse_transform_data_func(p) for p in seq_p] for seq_p in predictions_seq]
                pred_data_to_save = {k: inverse_transform_data_func(v) if v is not None else v for k, v in prediction_data_seq.items()}
            else:
                predictions_to_save = predictions_seq
                pred_data_to_save = prediction_data_seq
            if len(predictions_to_save) == 1:
                predictions_to_save = predictions_to_save[0]

            pred_folder = os.path.join(cfg.paths.output_dir, "predictions")
            os.makedirs(pred_folder, exist_ok=True)
            with open(os.path.join(pred_folder, "predictions.pkl"), 'wb') as f:
                pickle.dump(predictions_to_save, f)
            with open(os.path.join(pred_folder, "data.pkl"), 'wb') as f:
                pickle.dump(pred_data_to_save, f)

        if cfg.eval.get("plot") is not None:
            for pred_i, predictions in enumerate(predictions_seq):
                if eval_kwargs.get("forecast_horizon", 1) == 1:
                    predictions = darts.timeseries.concatenate(predictions)
                elif cfg.eval.plot.get("every_n_prediction", 1) > 1:
                    predictions = predictions[::cfg.eval.plot.every_n_prediction]

                plot_title = None
                if cfg.eval.plot.get("title"):
                    plot_title = cfg.eval.plot.get("title")
                if cfg.eval.plot.get("title_add_metrics"):
                    if plot_title is None:
                        plot_title = ""
                    plot_title += " ".join(f"{'_'.join(metric_name.split('_')[1:])}={metrics[pred_i, m_i]:.2E}" for m_i, metric_name in enumerate(metric_names))

                if not "presenter" in cfg.eval.plot:
                    presenters = ["savefig"]
                else:
                    presenters = cfg.eval.plot.presenter
                    if src.utils.utils.is_sequence(presenters):
                        presenters = list(presenters)
                    else:
                        presenters = [presenters]
                if logger is not None and len(logger) > 0:
                    presenters.extend(logger)
                if is_multiple_series:
                    fig_name = f"{cfg.eval.split}-{pred_i}-predictions"
                else:
                    fig_name = f"{cfg.eval.split}-predictions"
                fig_name = os.path.join("predictions", fig_name)

                presenter_kwargs = []
                for presenter in presenters:
                    if src.utils.plotting.is_supported_presenter(presenter):
                        if isinstance(presenter, pytorch_lightning.loggers.TensorBoardLogger):
                            p_kwargs = dict(global_step=trainer.global_step if trainer is not None else 0, tag=fig_name)
                        elif presenter == "savefig":
                            p_kwargs = dict(fname=os.path.join(cfg.paths.output_dir, "plots", fig_name))
                        else:
                            p_kwargs = dict(fname=fig_name)
                        presenter_kwargs.append(p_kwargs)
                src.utils.plotting.plot_prediction(predictions,
                                                   {k: v[pred_i] if (is_multiple_series and v is not None) else v for k, v in prediction_data_seq.items()},
                                                   model,
                                                   presenters,
                                                   predict_n=1,
                                                   plot_covariates=cfg.eval.plot.get("covariates", True),
                                                   plot_encodings=cfg.eval.plot.get("encodings", False),
                                                   inverse_transform_data_func=inverse_transform_data_func,
                                                   fig_title=plot_title,
                                                   plot_weights=cfg.eval.plot.get("weights", False),
                                                   presenter_kwargs=presenter_kwargs)
    else:
        eval_kwargs["metric"] = metric_funcs
        with src.utils.time_block(enabled=cfg.eval.get("measure_execution_time"),
                                  metric_dict=time_metrics,
                                  log_file=os.path.join(cfg.paths.output_dir, "eval_exec_time.log") if logger is None else None
                                  ):
            metrics = call_function_with_data(model.backtest, datamodule, main_split=cfg.eval.split, model=model, **eval_kwargs)
        num_eval_series = datamodule.num_series_for_split(cfg.eval.split)
        metrics = np.array(np.atleast_1d(metrics)).reshape(num_eval_series, len(metric_funcs))

    if cfg.eval.metrics_per_series and metrics.shape[0] > 1:
        metric_dict = {f"{metric_name}_{s_i}": metrics[s_i, m_i] for m_i, metric_name in enumerate(metric_names) for s_i in range(metrics.shape[0])}
    else:
        metric_dict = {}

    metrics = np.mean(metrics, axis=0)  # get mean metrics over series

    metric_dict.update({metric_name: metrics[m_i] for m_i, metric_name in enumerate(metric_names)})
    metric_dict.update({f"eval_{k}": v for k, v in time_metrics.items()})

    return metric_dict


def _run_trainer(cfg: DictConfig, datamodule: src.datamodules.TimeSeriesDataModule, model,  trainer: pytorch_lightning.Trainer, ckpt_path: Optional[str] = None):
    """
    Runs the evaluation of the model using the PyTorch Lightning Trainer validation loop. This is the default method
    for evaluating models that inherit from TorchForecastingModel, as it provides a substantial speedup over the
    darts backtest method. However, it is not possible to plot predictions when using this method. The results of the
    evaluation are logged to the logger and saved to the model directory, and returned as a dictionary with keys
    corresponding to the metric names and values corresponding to the metric values.

    :param cfg: The config object.
    :param datamodule: The datamodule.
    :param model: The model.
    :param trainer: The PyTorch Lightning Trainer.
    :param ckpt_path: The path to the checkpoint to load.

    :return: The metrics.
    """
    assert isinstance(model, darts.models.forecasting.torch_forecasting_model.TorchForecastingModel), "The trainer evaluator is only available for models that inherit from TorchForecastingModel. Use runner=backtest instead."

    if "kwargs" in cfg.eval:
        eval_kwargs = dict(cfg.eval.kwargs)
    else:
        eval_kwargs = {}

    if isinstance(model, darts.models.TCNModel):
        log.info("You are using a TCNModel with runner=trainer. Note that for the TCNModel, "
                 "metrics are evaluated on the entire output (i.e. both past and future) as opposed to when using the "
                 "backtest method which only evaluates metrics on future time steps.")

    if cfg.eval.get("plot"):
        log.info("Plotting of predictions is currently not supported when using runner=trainer.")

    eval_dataset = call_function_with_data(model._build_train_dataset, datamodule, main_split=cfg.eval.split,
                                           # TODO: improve handling of this.
                                           model=model, raise_exception_on_missing_argument=False,
                                           max_samples_per_ts=eval_kwargs.get("max_samples_per_ts", None))
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=model.batch_size,
        shuffle=False,
        num_workers=2,  # TODO: make configurable
        pin_memory=True,
        drop_last=False,
        collate_fn=model._batch_collate_fn,
    )

    log.info("Starting evaluation!")
    metric_dict = trainer.validate(model=model.model, dataloaders=eval_dataloader, ckpt_path=ckpt_path)
    metric_dict = metric_dict[0]  # a list with entries for every dataloader, we use only one dataloader
    if cfg.eval.split != "val":
        metric_dict = {f"{cfg.eval.split}_{'_'.join(k.split('_')[1:])}": v for k, v in metric_dict.items()}

    return metric_dict


def run(cfg: DictConfig, datamodule: src.datamodules.TimeSeriesDataModule, model, trainer: Optional[pytorch_lightning.Trainer] = None, logger: Optional[List[Logger]] = None, ckpt_path: Optional[str] = None):
    """
    Runs the evaluation of the model. The results of the evaluation are logged to the logger and saved to the model
    directory, and returned as a dictionary with keys corresponding to the metric names and values corresponding to the
    metric values. The evaluation is performed using the method specified in the config object. The default method is
    to use backtest method of the darts model, but this can be changed to the PyTorch Lightning Trainer validation loop
    by setting runner=trainer in the config object (this is only available for models that inherit from
    TorchForecastingModel).

    :param cfg: The config object.
    :param datamodule: The datamodule.
    :param model: The model.
    :param trainer: The PyTorch Lightning Trainer.
    :param logger: The logger(s).
    :param ckpt_path: The path to the checkpoint to load.

    :return: The metrics.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed") is not None:
        pl.seed_everything(cfg.seed, workers=True)
    runner = cfg.eval.get("runner")
    if runner is None:
        if utils.is_torch_model(model):
            log.info("Since runner was not set, runner=trainer was automatically chosen because the model is a TorchForecastingModel")
            runner = "trainer"
        else:
            log.info("Since runner was not set, runner=backtest was automatically chosen because the model is not a TorchForecastingModel")
            runner = "backtest"

    if isinstance(model, darts.models.forecasting.torch_forecasting_model.TorchForecastingModel):
        if hasattr(model.model, "set_mc_dropout"):
            model.model.set_mc_dropout(cfg.eval.get("mc_dropout", False))

    log.info("Starting evaluation!")
    if runner == "trainer":
        try:
            if cfg.eval.get("inverse_transform_data", None) is not None:
                log.warning("The eval.inverse_transform_data argument was set, but inverse transformation is not possible with runner=trainer. Ignoring argument.")
            metric_dict = _run_trainer(cfg=cfg, datamodule=datamodule, model=model, trainer=trainer, ckpt_path=ckpt_path)
        except Exception as e:
            log.exception("Tried evaluating model with runner=trainer, but got the following error:")
            log.info("Retrying with runner=backtest")
            runner = "backtest"
            with open_dict(cfg):
                cfg.eval.kwargs.retrain = False  # we do not retrain when using runner=trainer
            metric_dict = _run_darts(cfg=cfg, datamodule=datamodule, logger=logger, model=model)
    elif runner == "backtest":
        metric_dict = _run_darts(cfg=cfg, datamodule=datamodule, model=model, logger=logger, trainer=trainer)
    else:
        raise ValueError(f"Config argument runner must be one of the following {_EVAL_RUNNERS}, you have {cfg.eval.runner}")

    with open_dict(cfg):
        cfg.eval.runner = runner

    if isinstance(model, src.models.ensemble_model.TSEnsembleModel):
        model.save_weights(os.path.join(cfg.paths.output_dir, f"eval_{cfg.eval.split}_weights.npy"))

    if cfg.eval.get("log_metrics"):  # TODO: translate split_loss into split_rmse?
        split_range = datamodule.get_split_range(cfg.eval.split)
        if isinstance(split_range[0], pd.Timestamp):
            split_range = tuple(index.strftime("%Y-%m-%d") for index in split_range)
        results_to_serialize = dict(dataset_range=split_range,
                                    split=cfg.eval.split,
                                    model_name=str(model),  # make prettier?
                                    forecast_horizon=cfg.eval.kwargs.get("forecast_horizon", 1),
                                    stride=cfg.eval.kwargs.get("stride", 1),
                                    start=cfg.eval.kwargs.get("start", None),
                                    metrics={k: float(v) for k, v in metric_dict.items()})
        if isinstance(model, darts.models.forecasting.torch_forecasting_model.TorchForecastingModel):
            results_to_serialize["model_name"] = model.__class__.__name__
            results_to_serialize["ckpt"] = os.path.basename(ckpt_path) if ckpt_path is not None else ckpt_path
        OmegaConf.save(OmegaConf.create(results_to_serialize), os.path.join(cfg.paths.output_dir, f"eval_{cfg.eval.split}_results.yaml"))

        if logger is not None:  # TODO: should reflect the forecast horizon etc. used now. Maybe make new run if settings are not the same?
            for lg in logger:
                lg.log_metrics(metric_dict)

    return metric_dict


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
    assert cfg.model_dir

    object_dict = src.utils.initialize_saved_objects(cfg)
    model = object_dict["model"]
    datamodule = object_dict["datamodule"]
    trainer = object_dict.get("trainer", None)
    logger = object_dict.get("logger", [])

    object_dict["cfg"] = cfg

    metric_dict = run(cfg, datamodule, model, logger=logger, trainer=trainer)

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)
        for lg in logger:
            lg.finalize("success")

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    cfg = src.utils.load_saved_config(cfg.model_dir, cfg)
    evaluate(cfg)


if __name__ == "__main__":
    src.utils.utils.enable_eval_resolver()
    main()
