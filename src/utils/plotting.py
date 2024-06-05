import datetime
import os
import pathlib
import platform
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import darts.utils.utils
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker
import mlflow.exceptions
import numpy as np
import pandas as pd
import pytorch_lightning.loggers

import src.models.utils
import src.utils
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


PRESENTERS = [
    pytorch_lightning.loggers.tensorboard.TensorBoardLogger,
    pytorch_lightning.loggers.mlflow.MLFlowLogger,
    pytorch_lightning.loggers.logger.DummyLogger,
    "savefig",
    "show",
    None,
]


def is_supported_presenter(presenter) -> bool:
    """Checks if presenter is supported by present_figure.

    :param presenter: Presenter to check.
    :return: True if presenter is supported, False otherwise.
    """
    for supported_presenter in PRESENTERS:
        try:
            is_supported = isinstance(presenter, supported_presenter)
        except TypeError:
            is_supported = presenter == supported_presenter
        if is_supported:
            return True

    return False


def has_valid_extension(fig: plt.Figure, fname: str):
    """Helper function to check if the given filename has a valid extension/file format supported
    by the current active matplotlib backend.

    :param fig: Figure from which the supported filetypes are extracted.
    :param fname: Filename to check
    :return: True if fname has a valid extension supported by the active matplotlib backend, False
        otherwise
    """
    _, ext = os.path.splitext(fname)
    if not ext:
        return False
    return ext[1:].lower() in list(fig.canvas.get_supported_filetypes())


def plot_darts_timeseries(
    series_to_be_plotted: Union[darts.TimeSeries, Sequence[darts.TimeSeries]],
    title: str = "",
    presenter: Any = None,
    **presenter_kwargs,
) -> Optional[List[plt.Figure]]:
    """Plots a darts TimeSeries object. If the series is a sequence of TimeSeries objects, plots
    each series in a separate figure.

    :param series_to_be_plotted: TimeSeries object to be plotted.
    :param title: Title of the plot.
    :param presenter: Presenter to use for plotting. Can be a pytorch_lightning logger, a string
        ["savefig", "show"] or None.
    :param presenter_kwargs: Keyword arguments to pass to the presenter.
    :return: List of figures plotted if presenter is None, None otherwise.
    """
    figs = []
    series_list = darts.utils.ts_utils.series2seq(series_to_be_plotted)

    for series_i, series in enumerate(series_list):
        series.plot(new_plot=True)
        fig = plt.gcf()
        if not isinstance(series_to_be_plotted, darts.TimeSeries):
            title += f" {series_i}"
        fig.suptitle(title)
        stage_plot_kwargs = {}
        stage_plot_kwargs.update(presenter_kwargs)
        if isinstance(presenter, pytorch_lightning.loggers.tensorboard.TensorBoardLogger):
            stage_plot_kwargs.update(dict(tag=title))
        elif "fname" in stage_plot_kwargs:
            stage_plot_kwargs["fname"] = os.path.join(stage_plot_kwargs["fname"], f"{title}")
        figs.append(present_figure(fig, presenter, **stage_plot_kwargs))

    return figs


def plot_prediction(
    prediction: Union[Sequence[darts.TimeSeries], darts.TimeSeries],
    prediction_data: Dict[str, darts.TimeSeries],
    model,
    presenters,
    separate_target=False,
    plot_covariates=True,
    plot_weights=False,
    plot_encodings=True,
    plot_past=True,
    plot_prediction_point=False,
    inverse_transform_data_func=None,
    fig_title=None,
    presenter_kwargs=None,
):
    """Plots prediction(s) of a model versus the actual target series as well as any covariates
    consumed by the model.

    :param prediction: Prediction to plot.
    :param prediction_data: Dictionary containing the actual target series and any covariates
        consumed by the model.
    :param model: Model that produced the prediction.
    :param presenters: Presenter(s) to use for plotting. Can be a single presenter or a list of
        presenters where each presenter is a pytorch_lightning logger, a string ["savefig", "show"]
        or None.
    :param separate_target: If True and the target is multivariate, the target variables are
        plotted in separate subplots.
    :param plot_covariates: Whether to plot the covariates of the model.
    :param plot_encodings: Whether to plot the encodings of the model. Note that plot_covariates
        must also be True.
    :param plot_past: Whether to plot past values before the prediction point. Disables
        plot_prediction_point if True.
    :param plot_prediction_point: Whether to plot the prediction point as a vertical line.
    :param inverse_transform_data_func: Function to inverse transform the data before plotting.
    :param fig_title: Title of the plot.
    :param presenter_kwargs: Keyword arguments to pass to the presenter.
    :return: List of figures plotted if presenter is None, None otherwise.
    """
    if not isinstance(presenters, list):
        presenters = [presenters]
    if presenter_kwargs is not None:
        assert len(presenters) == len(presenter_kwargs) and type(presenters) is type(
            presenter_kwargs
        )

    lag_data_type_translator = {
        "series": 0,
        "past_covariates": 2,
        "future_covariates": 3,
    }

    plot_prediction_point = plot_prediction_point and plot_past
    separate_target = separate_target and not prediction_data["series"].is_univariate

    if isinstance(prediction, darts.TimeSeries):
        prediction = [prediction]
    else:
        prediction = sorted(prediction, key=lambda x: x.start_time())
        plot_prediction_point = False  # there are multiple prediction points

    # Get start and stop indices for actual target series and covariates consumed by model
    # (which might be longer than prediction because of lags etc.)
    plot_data = {}
    for data_type in ["series", "past_covariates", "future_covariates"]:
        if prediction_data.get(data_type) is None or (
            "covariates" in data_type and not plot_covariates
        ):
            plot_data[data_type] = None
        else:
            if prediction[-1].start_time() > prediction_data[data_type].end_time():
                assert (
                    prediction[-1].start_time()
                    == prediction_data[data_type].end_time() + prediction_data[data_type].freq
                )
                # TODO: is this assert necessary? We require that prediction starts maximum one step after the end of the data
                start_idx = len(prediction_data[data_type]) - 1
            else:
                start_idx = prediction_data[data_type].get_index_at_point(
                    prediction[0].start_time()
                )
            if plot_past and model.extreme_lags[lag_data_type_translator[data_type]] is not None:
                start_idx -= abs(model.extreme_lags[lag_data_type_translator[data_type]])
                start_idx = max(0, start_idx)

            if prediction[-1].end_time() > prediction_data[data_type].end_time():
                end_idx = len(prediction_data[data_type]) - 1
            else:
                end_idx = prediction_data[data_type].get_index_at_point(prediction[-1].end_time())
                if (
                    data_type == "future_covariates"
                    and model.extreme_lags[lag_data_type_translator[data_type] + 1] is not None
                ):
                    end_idx += abs(model.extreme_lags[lag_data_type_translator[data_type] + 1] + 1)
                    end_idx = min(end_idx, len(prediction_data[data_type]) - 1)
            if isinstance(prediction_data[data_type].time_index, pd.DatetimeIndex):
                plot_data[data_type] = prediction_data[data_type].slice(
                    prediction_data[data_type].get_timestamp_at_point(start_idx),
                    prediction_data[data_type].get_timestamp_at_point(end_idx),
                )
            elif isinstance(prediction_data[data_type].time_index, pd.RangeIndex):
                start_idx += prediction_data[data_type].time_index.start
                end_idx += prediction_data[data_type].time_index.start
                plot_data[data_type] = prediction_data[data_type].slice(start_idx, end_idx)
            else:
                raise ValueError(
                    f"Not supported time_index type {type(prediction_data[data_type].time_index)}"
                )

    if inverse_transform_data_func is not None:
        prediction = src.utils.inverse_transform_data(inverse_transform_data_func, prediction)
        plot_data = src.utils.inverse_transform_data(inverse_transform_data_func, plot_data)

    # extract model encodings
    if plot_encodings and getattr(model.encoders, "encoding_available", False):
        predict_n = prediction[0].n_timesteps
        enc_past_covariates, enc_future_covariates = model.generate_predict_encodings(
            n=predict_n,
            series=plot_data["series"],
            past_covariates=plot_data["past_covariates"],
            future_covariates=plot_data["future_covariates"],
        )
        if enc_past_covariates is not None:
            plot_data["past_covariates"] = enc_past_covariates
        if enc_future_covariates is not None:
            plot_data["future_covariates"] = enc_future_covariates

    nplots = sum(pred_data is not None for pred_data in plot_data.values())
    if separate_target:
        nplots += plot_data["series"].n_components - 1
    if plot_weights:
        if not src.models.utils.is_rewts_model(model):
            log.warning(
                "plot_weights can only be used with the ReWTSEnsembleModel. Setting argument to false"
            )
            plot_weights = False
        elif len(model._weights_history) == 0:
            log.warning(
                "plot_weights was True but ReWTSEnsembleModel weight history is empty. Ensure weights have been fitted and model has not been reset."
            )
            plot_weights = False
        else:
            nplots += 1
    fig, axs = create_figure(nplots, 1, sharex=True, figsize=(6.4, 4.8 * nplots))
    plot_i = 0
    for data_type in plot_data:
        if plot_data[data_type] is not None:
            if data_type == "series":
                for target_plot_i in range(
                    plot_data["series"].n_components if separate_target else 1
                ):
                    plt.sca(axs[plot_i])
                    if separate_target:
                        series_to_plot = plot_data["series"].univariate_component(target_plot_i)
                    else:
                        series_to_plot = plot_data["series"]
                    series_to_plot.plot(label="actual")
                    axs[plot_i].set_title("target")
                    if plot_prediction_point:
                        if prediction[0].start_time() > series_to_plot.end_time():
                            assert (
                                prediction[0].start_time()
                                == series_to_plot.end_time() + series_to_plot.freq
                            )
                            # TODO: is this assert necessary? We require that prediction starts maximum one step after the end of the data
                            prediction_point_index = len(series_to_plot) - 2
                        else:
                            prediction_point_index = (
                                series_to_plot.get_index_at_point(prediction[0].start_time()) - 1
                            )
                        axs[plot_i].axvline(
                            series_to_plot.time_index[prediction_point_index],
                        )
                    for pred in prediction:
                        if separate_target:
                            pred = pred.univariate_component(target_plot_i)
                        if len(prediction) == 1:
                            pred.plot(label="prediction")
                        else:
                            pred.plot(label="_nolegend_")
                    plot_i += 1
            else:
                plt.sca(axs[plot_i])
                plot_data[data_type].plot()
                axs[plot_i].set_title(data_type)
                plot_i += 1
    if plot_weights:
        weights, time_indices = zip(*model._weights_history)
        plot_ensemble_weights(weights=weights, time_indices=time_indices, ax=axs[-1])

    if fig_title is not None:
        fig.suptitle(fig_title)

    figs = []
    for p_i, presenter in enumerate(presenters):
        if presenter_kwargs is not None and presenter_kwargs[p_i] is not None:
            p_i_kwargs = presenter_kwargs[p_i]
        else:
            p_i_kwargs = {}
        figs.append(present_figure(fig, presenter, **p_i_kwargs))

    return figs


def create_figure(
    nrows: int, ncols: int = 1, constrained_layout: bool = True, sharex=False, **kwargs
) -> Tuple[plt.Figure, Union[plt.axis, List[plt.axis]]]:
    """Utility function to create a figure with a given number of subplots.

    :param nrows: Number of rows in the figure.
    :param ncols: Number of columns in the figure.
    :param constrained_layout: Whether to use constrained layout.
    :param sharex: Whether to share x axis.
    :param kwargs: Additional keyword arguments to pass to plt.figure.
    :return: Tuple of figure and axes. axes is always a np.array of axes.
    """
    if nrows * ncols > 1:
        fig, axs = plt.subplots(
            nrows, ncols, constrained_layout=constrained_layout, sharex=sharex, **kwargs
        )
        axs = axs.ravel()
    else:
        fig = plt.figure(constrained_layout=constrained_layout, **kwargs)
        axs = np.array([plt.gca()])

    return fig, axs


def present_figure(
    fig: plt.Figure, presenter: Any, **presenter_kwargs: Optional[Dict[str, Any]]
) -> Union[plt.Figure, None]:
    """Utility function to present a figure using a given presenter. If presenter is None, the
    figure is returned. If the presenter is a TensorBoardLogger, the figure is added to the
    TensorBoardLogger. If the presenter is a MLFlowLogger, the figure is saved to the MLFlowLogger.
    If the presenter is show, the figure is shown. If the presenter is savefig, the figure is saved
    to the path given in the fname keyword argument. If presenter is not None, the figure is
    closed.

    :param fig: Figure to present.
    :param presenter: Presenter to use.
    :param presenter_kwargs: Keyword arguments to pass to the presenter.
    :return: fig if presenter is None, None otherwise.
    """
    if presenter is None:
        return fig

    if isinstance(presenter, pytorch_lightning.loggers.tensorboard.TensorBoardLogger):
        assert "tag" in presenter_kwargs and "global_step" in presenter_kwargs
        tag_split = presenter_kwargs.pop("tag").split(" ")
        if len(tag_split) == 0:
            tag = tag_split
        elif "/" not in tag_split[0]:
            tag = f"{tag_split[0]}/{'_'.join(tag_split[1:])}"
        else:
            tag = "_".join(tag_split)
        presenter.experiment.add_figure(tag, fig, **presenter_kwargs)
    elif isinstance(presenter, pytorch_lightning.loggers.mlflow.MLFlowLogger):
        if "fname" in presenter_kwargs:
            fname = presenter_kwargs.pop("fname")
            if os.path.isabs(fname):
                # try to get the relative path to the model directory if it follows standard naming scheme
                fname_path = pathlib.Path(fname)
                found_base_dir = False
                for part_i, part in enumerate(fname_path.parts):
                    try:
                        datetime.datetime.strptime(part, "%Y-%m-%d_%H-%M-%S")
                        found_base_dir = True
                        if fname_path.parts[part_i + 1] == "plots":
                            part_i += 1
                        fname = os.path.join(*fname_path.parts[part_i + 1 :])
                        break
                    except ValueError:
                        pass
                if not found_base_dir:
                    fname = os.path.basename(fname)
        else:
            try:
                fname = fig._suptitle.get_text()
            except AttributeError as e:
                log.error(
                    "Either a fname kwarg has to be provided or a title must be set for the figure."
                )
                raise e
            assert (
                fname is not None and fname != ""
            ), "Either a fname kwarg has to be provided or a title must be set for the figure."
            fname = fname.replace(".", "d")
        if platform.system() == "Windows":
            fname = os.path.basename(
                fname
            )  # there is a bug on windows where it does not create subdirectories...
        if not has_valid_extension(fig, fname):
            fname += f".{presenter_kwargs.get('format', 'png')}"
        try:
            presenter.experiment.log_figure(presenter.run_id, fig, fname)
        except mlflow.exceptions.MlflowException:
            # There is a race condition to create the experiment between parallel jobs
            time.sleep(np.random.uniform(0.25, 1))
            presenter.experiment.log_figure(presenter.run_id, fig, fname)
    elif isinstance(presenter, pytorch_lightning.loggers.logger.DummyLogger):
        pass
    elif presenter == "savefig":
        assert "fname" in presenter_kwargs
        if "bbox_inches" not in presenter_kwargs:
            presenter_kwargs["bbox_inches"] = "tight"
        os.makedirs(
            os.path.dirname(presenter_kwargs["fname"]), exist_ok=True
        )  # TODO: rename fname to output_dir, fname or something? is a bit misleading providing fname to a function and then have it merged with something else e.g. in plot_dataset
        if not has_valid_extension(fig, presenter_kwargs["fname"]):
            presenter_kwargs["fname"] += f".{presenter_kwargs.get('format', 'png')}"
        fig.savefig(**presenter_kwargs)
    elif presenter == "show":
        plt.show(block=presenter_kwargs.get("block", False))
    else:
        raise ValueError(f"Unrecognized presenter {presenter}, must be one of {PRESENTERS}")
    plt.close(fig)


def plot_ensemble_weights(
    weights, ax=None, time_indices=None, presenter: Any = None, **presenter_kwargs
):
    if isinstance(weights, (list, tuple)):
        weights = np.array(weights)
    weights = weights.squeeze()
    assert len(weights.shape) <= 2

    if ax is None:
        fig, ax = create_figure(1, 1)
        ax = ax[0]
    else:
        fig = ax.figure

    model_indices = range(1, weights.shape[-1] + 1)
    if len(weights.shape) == 1:  # bar plot
        ax.bar(model_indices, weights.ravel())
        ax.set_xlim(model_indices[0], model_indices[-1])
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator())
        ax.set_xlabel("Model #")
        ax.set_ylabel("Model Weights")
        ax.set_xticks(model_indices)
        ax.set_xticklabels(model_indices, rotation=45)
    elif len(weights.shape) == 2:  # stacked area plot
        # Choose a colormap
        cmap = plt.cm.viridis

        # Create a Normalize object for scaling data values to the [0, 1] range for the colormap
        norm = mcolors.Normalize(vmin=min(model_indices), vmax=max(model_indices))

        # Generate a list of colors from the colormap
        colors = [cmap(norm(value)) for value in model_indices]

        # Create a ScalarMappable and initialize with the colormap and the Normalize object
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # You have to set_array with some values for the ScalarMappable

        # Add a color bar to the figure based on the ScalarMappable
        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.05)
        cbar.set_label("Model Indices")

        if time_indices is None:
            time_indices = np.arange(1, weights.shape[0] + 1)
        ax.stackplot(time_indices, weights.T, colors=colors)

        # Set the range of the color bar with key points
        if len(model_indices) > 20:
            key_indices = [min(model_indices), max(model_indices) // 2, max(model_indices)]
            cbar.set_ticks(key_indices)

            # Set the labels of the color bar with key points
            key_labels = [f"M{i}" for i in key_indices]
            cbar.set_ticklabels(key_labels)
        else:
            # Set the range of the color bar
            cbar.set_ticks(
                np.linspace(min(model_indices), max(model_indices), num=len(model_indices))
            )

            # Set the labels of the color bar
            cbar.set_ticklabels(["M" + str(i) for i in model_indices])

        # Customization
        ax.set_xlabel("Fit #")
        ax.set_ylabel("Model Weights")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(True)
    else:
        raise ValueError

    return present_figure(fig, presenter, **presenter_kwargs)


# TODO: use is supported presenter?
def multiple_present_figure(
    fig: plt.Figure, presenter: List[Any], presenter_kwargs: Optional[List[Dict[str, Any]]] = None
) -> Union[plt.Figure, None]:  # TODO: do we have to wait with closing figure until last one?
    """Utility function to present a figure using a given presenter. If presenter is None, the
    figure is returned. If the presenter is a TensorBoardLogger, the figure is added to the
    TensorBoardLogger. If the presenter is a MLFlowLogger, the figure is saved to the MLFlowLogger.
    If the presenter is show, the figure is shown. If the presenter is savefig, the figure is saved
    to the path given in the fname keyword argument. If presenter is not None, the figure is
    closed.

    :param fig: Figure to present.
    :param presenter: List of presenters to use.
    :param presenter_kwargs: List of keyword arguments to pass to the presenter.
    :return: fig if one of the presenters is None, None otherwise.
    """
    if presenter is None:
        return fig

    if presenter_kwargs is None:
        presenter_kwargs = [{}] * len(presenter)
    else:
        assert type(presenter) is type(presenter_kwargs) and len(presenter) == len(
            presenter_kwargs
        ), "A set of kwargs for each presenter must be provided"

    res = None
    for p_i in range(len(presenter)):
        p_kwargs = presenter_kwargs[p_i]
        if isinstance(p_kwargs, dict):
            p_res = present_figure(fig, presenter[p_i], **presenter_kwargs[p_i])
        else:
            p_res = present_figure(fig, presenter[p_i])
        if p_res is not None:
            res = p_res

    return res
