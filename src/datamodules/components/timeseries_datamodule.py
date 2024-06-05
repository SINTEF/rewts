import copy
import itertools
import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import darts.dataprocessing.pipeline
import darts.timeseries
import darts.utils.model_selection
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule

import src.datamodules.components.dataloaders as dataloaders
import src.datamodules.utils
import src.utils.plotting
from src.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


_ALLOWED_SPLIT_TYPES = (tuple, list, int, float, str)
_SUPPORTED_SAVE_BACKENDS = ("pickle",)
_SUPPORTED_DATA_TYPES = (pd.DataFrame, darts.timeseries.TimeSeries)
_SUPPORTED_FREQ_TYPES = (str, type(None), pd.Timedelta)
_VALID_RESAMPLE_KEYS = ("method", "freq")
_VALID_SPLIT_NAMES = ("train", "val", "test", "predict")
_VALID_INDEX_TYPES = (pd.RangeIndex, pd.DatetimeIndex)
_PLOT_SEPARATE_MAX = 3
_PLOT_SPLIT_COLOR = {"train": "blue", "val": "red", "test": "green"}
_PLOT_COMPONENT_COLORMAP = plt.cm.tab10.colors


def _assert_compatible_with_index(
    value: Union[int, pd.Timestamp, float], index: Union[pd.RangeIndex, pd.DatetimeIndex]
):
    if isinstance(value, float):
        return
    elif isinstance(value, int):
        assert isinstance(
            index, (pd.RangeIndex, pd.DatetimeIndex)
        ), "An int-object is only compatible with pd.RangeIndex. You can use the timeseries.get_index_at_point function to convert."
    elif isinstance(value, (str, pd.Timestamp)):
        assert isinstance(
            index, pd.DatetimeIndex
        ), "A Timestamp-object is only compatible with pd.DatetimeIndex. You can use the timeseries.get_timestamp_at_point to convert."
    else:
        raise ValueError


def _assert_valid_type_and_index(data):
    assert isinstance(
        data, _SUPPORTED_DATA_TYPES
    ), f"The supported types for self.data are {_SUPPORTED_DATA_TYPES}, you have {type(data)}"

    if isinstance(data, darts.timeseries.TimeSeries):
        index = data.time_index
    elif isinstance(data, pd.DataFrame):
        index = data.index
    else:
        raise ValueError("Should not be possible")

    assert isinstance(
        index, _VALID_INDEX_TYPES
    ), f"The supported types of data indexes are {_VALID_INDEX_TYPES}, you have {type(index)}"


class TimeSeriesDataModule(LightningDataModule):
    """Example of LightningDataModule for a generic TimeSeriesDataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(  # TODO: add support for choosing which factory for timeseries?
        self,
        data_variables: Dict[str, Tuple[str]],
        data_source: Optional[Dict[str, Any]] = None,
        processing_pipeline: Optional[darts.dataprocessing.pipeline.Pipeline] = None,
        data_dir: str = "data/",
        train_val_test_split: Optional[Dict[str, Union[float, Tuple[str, str]]]] = None,
        freq: Optional[str] = None,
        resample: Optional[Dict[str, str]] = None,
        precision: int = 32,
        check_for_nan: bool = True,
        predict_split: str = "val",  # TODO: remove
        labels_component_wise: bool = False,
        crop_data_range: Optional[List[Union[str, pd.Timestamp, int]]] = None,
    ):
        assert predict_split in ["train", "val", "test"]
        assert precision in [8, 16, 32, 64]
        assert "target" in data_variables
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data: Optional[Union[pd.DataFrame, darts.timeseries.TimeSeries]] = None
        self.data_train: Optional[Dict[str, darts.timeseries.TimeSeries]] = None
        self.data_val: Optional[Dict[str, darts.timeseries.TimeSeries]] = None
        self.data_test: Optional[Dict[str, darts.timeseries.TimeSeries]] = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None, load_dir: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!

        The setup function processes the data and splits it into train, val and test sets as
        configured by the arguments to the __init__ method. It also applies the processing pipeline
        to the data.

        This function can be called directly if a data source has been configured. If not, you need
        to make a subclass in which you override the setup function.

        :param stage: The pytorch lightning stage to prepare the dataset for
        :param load_dir: The folder to which the state of the datamodule is saved for later
            reproduction (e.g. fitted scalers etc.).
        :return: None
        """
        assert (
            self.hparams.data_source is not None
        ), "This function can only be called directly if a data source has been configured. Either configure a data source to be loaded, or make a subclass in which you override the setup function."
        if not self.data_train and not self.data_val and not self.data_test:
            self.data = dataloaders.DataLoader(**self.hparams.data_source).load()

            self._finalize_setup(load_dir=load_dir)

    def _finalize_setup(self, load_dir: Optional[str] = None):
        """This function must be called by the setup function of any subclass of
        TimeSeriesDataModule. It performs the final steps of the setup function, including fitting
        the processing pipeline, splitting the data into train, val and test sets and transforming
        the sets, and resampling the data if needed. It also checks that the data is valid and that
        the hparams are valid.

        :param load_dir: The folder to which the state of the datamodule is saved for later
            reproduction.
        :return: None
        """
        assert self.data is not None, "You need to set your dataset to the self.data attribute"
        assert isinstance(
            self.hparams.freq, _SUPPORTED_FREQ_TYPES
        ), f"The supported types for freq are {_SUPPORTED_FREQ_TYPES}, you have {type(self.hparams.freq)}"

        if load_dir is not None:
            self.load_state(load_dir)

        all_data_variables = []
        for dv_name, dvs in self.hparams.data_variables.items():
            if dvs is not None:
                if len(dvs) == 0:
                    self.hparams.data_variables[dv_name] = None
                    continue
                if isinstance(dvs, str):
                    dvs = [dvs]
                all_data_variables.extend(dvs)

        if not len(all_data_variables) == len(set(all_data_variables)):
            seen = set()
            dupes = []

            for dv in all_data_variables:
                if dv in seen:
                    dupes.append(dv)
                else:
                    seen.add(dv)

            raise ValueError(
                f"data_variables must have unique entries. The variable(s) {dupes} appears in multiple entries."
            )

        assert (
            self.hparams.data_variables.get("target", []) is not None
            and len(self.hparams.data_variables.get("target", [])) > 0
        ), "You need to provide a target variable"
        if not isinstance(self.data, dict):
            self.data = {0: self.data}

        for dataset_name in self.data:
            _assert_valid_type_and_index(self.data[dataset_name])

            if self.hparams.crop_data_range is not None:
                self.data[dataset_name] = self.crop_dataset_range(
                    self.data[dataset_name], self.hparams.crop_data_range
                )

            if self.hparams.resample is not None:
                # TODO: should this logic be somewhere else?
                required_keys = list(_VALID_RESAMPLE_KEYS)
                for key, value in self.hparams.resample.items():
                    if key not in _VALID_RESAMPLE_KEYS:
                        log.info(
                            f"Unrecognized argument {key} given for resample. The argument is ignored."
                        )
                    elif key in required_keys:
                        required_keys.remove(key)
                assert (
                    len(required_keys) == 0
                ), f"The following required keys are missing from resample {required_keys}"

                self.data[dataset_name] = self.resample_dataset(
                    self.data[dataset_name], **self.hparams.resample
                )

            if isinstance(self.data[dataset_name], pd.DataFrame):
                self.data[dataset_name].columns.name = (
                    None  # fix potential bug with Timeseries.from_dataset
                )
                self.data[dataset_name] = darts.timeseries.TimeSeries.from_dataframe(
                    self.data[dataset_name],
                    value_cols=all_data_variables,
                    fill_missing_dates=True,
                    freq=self.hparams.freq,
                )  # TODO: add support for other arguments

            self.data[dataset_name] = self.set_dataset_precision(
                self.data[dataset_name], precision=self.hparams.precision
            )

        self.hparams.train_val_test_split = self.process_train_val_test_split(
            self.data, self.hparams.train_val_test_split
        )

        for split_name in ["train", "val", "test"]:
            if not self.has_split_data(split_name):
                continue

            if self.hparams.processing_pipeline is not None:
                self.hparams.processing_pipeline = (
                    src.datamodules.utils.ensure_pipeline_per_component(
                        self.hparams.processing_pipeline, self.hparams.data_variables
                    )
                )
                if split_name == "train":
                    self.fit_processing_pipeline(self._get_split_data_raw("train"))
                else:
                    if not src.datamodules.utils.pipeline_is_fitted(
                        self.hparams.processing_pipeline
                    ):
                        if not self.has_split_data("train"):
                            raise RuntimeError(
                                "A pipeline has been configured, but no training set has been provided on which it can be fitted. Either pass load_dir containing the state of a pipeline or configure a training set."
                            )
                        else:
                            raise RuntimeError(
                                "processing_pipeline.transform was called before it was fitted. Ensure datamodule configuration is correct. Contact developer as this is an unexpected error."
                            )

            if self.hparams.check_for_nan:
                split_data = self.get_split_data(
                    split_name, transform=self.hparams.processing_pipeline is not None
                )
                for covariate_type in split_data:
                    if split_data[covariate_type] is None:
                        continue
                    for series in darts.utils.ts_utils.series2seq(split_data[covariate_type]):
                        if np.any(np.isnan(series.all_values())):
                            raise ValueError(
                                f"The {split_name} dataset contains nan-values in {covariate_type} and the check_for_nan attribute is set to True. Please check if data-processing pipeline is configured correctly."
                            )

    def get_data(
        self,
        data_kwargs: List[str],
        main_split: str = "train",
        transform: bool = True,
    ) -> Dict[str, darts.timeseries.TimeSeries]:
        """Get dictionary of training and validation data (if defined) formatted with the names
        expected by the darts model methods such as fit, predict, historical_forecast, backtest,
        etc. I.e. intended use is for instance: model.fit(**datamodule.get_data(), other=value1,
        kwargs=value2).

        :param data_kwargs: List of strings specifying which data to return. The strings should be
            the same as the arguments of the model.fit method. The strings can be prefixed with
            "val_" to specify that the data should be returned for the validation split. If no
            prefix is given, the data is returned for the main split.
        :param main_split: String specifying which split should be used as the main split. The main
            split is used for all data that is not prefixed with "val_".
        :param transform: Whether the data should be transformed using processing_pipeline or not.
        :return: Dictionary of data formatted with the names expected by the model.fit method.
        """
        assert self.has_split_data(main_split), f"No data has been set for split {main_split}"
        data_translator = {
            "target": "target",
            "series": "target",
            "past_covariates": "past_covariates",
            "future_covariates": "future_covariates",  # TODO: static covariates
        }
        main_split_data = self.get_split_data(main_split, transform=transform)
        if any(kwarg.startswith("val_") for kwarg in data_kwargs):
            val_split_data = self.get_split_data("val", transform=transform)

        res = {}
        for kwarg in data_kwargs:
            if kwarg.startswith("val_"):  # TODO: can there be others, e.g. test?
                if not self.has_split_data("val"):
                    continue
                kwarg_name_split = kwarg.split("_")
                kwarg_split = kwarg_name_split[0]
                kwarg = "_".join(kwarg_name_split[1:])
                kwarg_split_data = val_split_data
            else:
                kwarg_split = main_split
                kwarg_split_data = main_split_data
            if kwarg in data_translator:
                kwarg_value = kwarg_split_data.get(data_translator[kwarg])
                if kwarg_split == main_split:
                    res[kwarg] = kwarg_value
                else:
                    res[f"{kwarg_split}_{kwarg}"] = kwarg_value
            elif hasattr(
                self.hparams, kwarg
            ):  # TODO: is this safe? Meant for stuff like num_loader_workers, max_samples_per_ts
                res[kwarg] = getattr(self.hparams, kwarg)

        return res

    def save_state(self, save_dir: str, backend: str = "pickle") -> None:
        """Save the state of the datamodule to a directory. This includes the state of the fitted
        processing pipeline such that data will be transformed in the same way as during training,
        even if a different data split is used.

        :param save_dir: The directory to which to save the state.
        :param backend: The backend to use for saving. Currently only "pickle" is supported.
        :return: None
        """
        assert (
            backend in _SUPPORTED_SAVE_BACKENDS
        ), f"The supported saving/loading backends are {_SUPPORTED_SAVE_BACKENDS}, you have {backend}"
        os.makedirs(save_dir, exist_ok=True)
        if backend == "pickle":
            with open(os.path.join(save_dir, "pipeline.pkl"), "wb") as pipeline_pkl:
                pickle.dump(self.hparams.processing_pipeline, pipeline_pkl)
        else:
            raise NotImplementedError

    def load_state(self, load_dir: str, backend: str = "pickle") -> None:
        """Load the state of the datamodule from a directory. This includes the state of the fitted
        processing pipeline such that data will be transformed in the same way as during training,
        even if a different data split is used.

        :param load_dir: The directory from which to load the state.
        :param backend: The backend to use for loading. Currently only "pickle" is supported.
        :return: None
        """
        assert (
            backend in _SUPPORTED_SAVE_BACKENDS
        ), f"The supported saving/loading backends are {_SUPPORTED_SAVE_BACKENDS}, you have {backend}"
        if backend == "pickle":
            if os.path.exists(os.path.join(load_dir, "pipeline.pkl")):
                with open(os.path.join(load_dir, "pipeline.pkl"), "rb") as pipeline_pkl:
                    self.hparams.processing_pipeline = pickle.load(pipeline_pkl)
                log.info("Loaded processing_pipeline from file.")
            else:
                log.warning("processing_pipeline file was not found.")
        else:
            raise NotImplementedError

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):  # TODO: could also put state of pipeline here?
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    def plot_data(
        self,
        split: Optional[str] = None,
        datasets: Optional[Union[str, Sequence[str]]] = None,
        slice: Optional[
            Tuple[Union[float, int, pd.Timestamp], Union[float, int, pd.Timestamp]]
        ] = None,  # TODO: convert for user?
        separate_splits: bool = False,
        separate_components: Optional[bool] = None,
        transformed: bool = False,
        presenter: Any = None,
        **presenter_kwargs,
    ) -> Union[List[plt.Figure], None]:  # TODO: consider rewriting to using TSD to save space
        """Plots the dataset splits. If no split is specified, all splits are plotted.

        :param split: The split to plot. If None, all splits are plotted.
        :param datasets: Name of dataset(s) to be plotted, i.e. entries in the self.data
            dictionary. If none, all datasets will be plotted.
        :param slice: A tuple of (start, end) to slice the data before plotting.
        :param separate_splits: Whether to plot splits in separate figures, or in the same figure
            with colored background to indicate different splits.
        :param separate_components: Whether to plot the components of the split separately. If
            None, components will be plotted separately if there are _PLOT_SEPARATE_MAX or fewer
            components, or together otherwise.
        :param transformed: Whether to plot the transformed data or non-transformed data.
        :param presenter: A presenter class to use for plotting. If None, the default presenter is
            used.
        :param presenter_kwargs: Keyword arguments to pass to the presenter.
        :return: A list of figures if presenter is None, otherwise None
        """
        if slice is not None:
            raise NotImplementedError  # get some blank plots
            _assert_compatible_with_index(slice[0], next(iter(self.data.values())).time_index)

        # TODO: should also allow for splitting across series?

        if isinstance(datasets, str):
            datasets = [datasets]

        if split is None:
            plot_splits = ["train", "val", "test"]
        else:
            plot_splits = [split]

        train_patch = matplotlib.patches.Patch(
            color=_PLOT_SPLIT_COLOR["train"], label="Train", alpha=0.25
        )
        val_patch = matplotlib.patches.Patch(
            color=_PLOT_SPLIT_COLOR["val"], label="Val", alpha=0.25
        )
        test_patch = matplotlib.patches.Patch(
            color=_PLOT_SPLIT_COLOR["test"], label="Test", alpha=0.25
        )

        figs = []
        for dataset_name in map(str, self.data):
            if datasets is not None and dataset_name not in datasets:
                continue

            component_colors = {}
            split_data = {
                s: self.get_split_data(s, transform=transformed, datasets=dataset_name)
                for s in plot_splits
            }

            color_cycles = {}

            for data_type in ["target", "past_covariates", "future_covariates"]:
                data_figures = {}
                labeled_components = set()

                for plot_split, split_series in split_data.items():
                    if split_series is None or split_series.get(data_type) is None:
                        continue

                    if separate_splits:
                        labeled_components = set()

                    if len(self.data) == 1 and dataset_name == "0":
                        title_prefix = ""
                    else:
                        title_prefix = f"Dataset {dataset_name} "

                    if separate_splits:
                        title_prefix = f"{title_prefix}{plot_split} "

                    series_seq = darts.utils.ts_utils.series2seq(split_series[data_type])

                    separate_series_components = separate_components or (
                        separate_components is None
                        and series_seq[0].n_components <= _PLOT_SEPARATE_MAX
                    )

                    if len(data_figures) == 0 or separate_splits:
                        # Initialize a figure for each component if separate_components is True
                        if separate_series_components:
                            for component in series_seq[0].components:
                                fig, ax = src.utils.plotting.create_figure(1, 1, figsize=(10, 5))
                                ax = ax[0]
                                data_figures[component] = (fig, ax)
                        else:
                            fig, ax = src.utils.plotting.create_figure(1, 1, figsize=(10, 5))
                            ax = ax[0]
                            data_figures["combined"] = (fig, ax)

                    # Plot each series within the split
                    for series in series_seq:
                        if slice:
                            if all(slice[i] not in series.time_index for i in range(2)):
                                continue
                            else:
                                series = series.slice(*slice)

                        for component_i, component in enumerate(series.components):
                            # Assign consistent colors
                            if component in data_figures:
                                ax = data_figures[component][1]
                            else:
                                ax = data_figures["combined"][1]

                            if ax not in color_cycles:
                                color_cycles[ax] = iter(_PLOT_COMPONENT_COLORMAP)

                            if component not in component_colors:
                                component_colors[component] = next(color_cycles[ax])
                            color = component_colors[component]

                            # Plot with label only if it's the first time for this component in this axis
                            label = f"{component}" if component not in labeled_components else None
                            series.univariate_component(component).plot(
                                ax=ax, label=label, color=color
                            )
                            ax.set_title(f"{title_prefix}{data_type}")
                            labeled_components.add(component)

                            # Background color for splits, only
                            if not separate_splits and (
                                separate_series_components or component_i == 0
                            ):
                                ax.axvspan(
                                    series.time_index[0],
                                    series.time_index[-1],
                                    color=_PLOT_SPLIT_COLOR[plot_split],
                                    alpha=0.1,
                                )

                    for component, (fig, ax) in data_figures.items():
                        ax.legend()
                        if not separate_splits:
                            handles, labels = ax.get_legend_handles_labels()
                            handles.extend([train_patch, val_patch, test_patch])
                            ax.legend(handles=handles, labels=labels + ["Train", "Val", "Test"])
                        figs.append(fig)

        for fig_i, fig in enumerate(figs):
            figs[fig_i] = src.utils.plotting.present_figure(
                fig, presenter=presenter, **presenter_kwargs
            )

        return figs

    def call_function_on_univariate_series(
        self,
        function: Callable,
        series_to_call: Dict[str, Any],
        presenter: Any = False,
        **presenter_kwargs,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """This function provides access to univariate components of the dataset splits. It can be
        used to call a function on the univariate series. If the function returns a figure, it can
        be presented using the presenter.

        :param function: The function to call on the univariate series.
        :param series_to_call: A dictionary specifying which series to call the function on. The dictionary should have the
            following structure: {"train": {"series_type": ["component_name"]}}}. The "train" key can be replaced by
            "val" or "test". The "series_type" key can be replaced by any of the series types in the dataset. The value of
            the series_type dictionary can be "all" to call the function on all series of the specified type, or a list of
            components names in the dataset. The value of the dictionary can be "all" to call the function on all series.
        :param presenter: A presenter class to use for plotting. If None, the default presenter is used.
        :param presenter_kwargs: Keyword arguments to pass to the presenter.

        :return A dictionary of function results, with the same structure as the input dictionary.
        """
        res = {}
        for split_name in ["train", "val", "test"]:
            if (series_to_call == "all" or split_name in series_to_call) and self.has_split_data(
                split_name
            ):
                split_series_to_call = (
                    "all" if series_to_call == "all" else series_to_call[split_name]
                )
                res[split_name] = {}
                for series_type, serieses in self.get_split_data(split_name).items():
                    if serieses is not None and (
                        split_series_to_call == "all" or series_type in split_series_to_call
                    ):
                        for series_i, series in enumerate(serieses):
                            component_series_to_call = (
                                "all"
                                if split_series_to_call == "all"
                                else split_series_to_call[series_type]
                            )
                            res[split_name][series_type] = {}
                            for component_name in series.components.values:
                                if (
                                    component_series_to_call == "all"
                                    or component_name in component_series_to_call
                                ):
                                    res[split_name][series_type][component_name] = function(
                                        series.univariate_component(component_name)
                                    )
                                    if presenter is not False:
                                        fig = plt.gcf()
                                        if len(serieses) > 1:
                                            title = f"{split_name}/{series_type}/{series_i}/{component_name}"
                                        else:
                                            title = f"{split_name}/{series_type}/{component_name}"
                                        plt.title(title)
                                        res[split_name][series_type][component_name] = (
                                            src.utils.plotting.present_figure(
                                                fig, presenter, **presenter_kwargs
                                            )
                                        )  # TODO: should we overwrite or return a separate argument? i.e. return res, figs

        return res

    def call_function_on_pairs_of_univariate_series(
        self,
        function: Callable,
        series_to_call: Dict[str, Any],
        presenter: Any = False,
        **presenter_kwargs,
    ) -> Dict[str, Dict[str, Dict[frozenset, Any]]]:
        """This function extends the capability to call a function on pairs of univariate series
        across different series types within the same split.

        :return: A dictionary of function results, structured by split, series type pair, and
            component pair.
        """
        included_series_types = ["target", "past_covariates", "future_covariates"]

        res = {}
        for split_name in ["train", "val", "test"]:
            if (series_to_call == "all" or split_name in series_to_call) and self.has_split_data(
                split_name
            ):
                split_series_to_call = (
                    "all" if series_to_call == "all" else series_to_call[split_name]
                )
                res[split_name] = {}

                # Get all series types for the current split
                split_data = self.get_split_data(split_name)
                series_types = [st for st in split_data.keys() if st in included_series_types]

                # Generate all unique pairs of series types
                for series_type_1, series_type_2 in itertools.combinations(series_types, 2):
                    series_type_pair_key = f"{series_type_1}_{series_type_2}"
                    res[split_name][series_type_pair_key] = {}
                    processed_pairs = set()  # Track processed component pairs to avoid redundancy

                    # Get series for each type
                    serieses_1 = (
                        split_data.get(series_type_1, None)
                        if series_type_1 in split_series_to_call or split_series_to_call == "all"
                        else None
                    )
                    serieses_2 = (
                        split_data.get(series_type_2, None)
                        if series_type_2 in split_series_to_call or split_series_to_call == "all"
                        else None
                    )

                    if serieses_1 is None or serieses_2 is None:
                        continue

                    # Iterate over all combinations of series from the two types
                    for series_1, series_2 in itertools.product(serieses_1, serieses_2):
                        for component_name_1, component_name_2 in itertools.product(
                            series_1.components.values, series_2.components.values
                        ):
                            component_pair = frozenset([component_name_1, component_name_2])
                            if component_pair not in processed_pairs:
                                processed_pairs.add(component_pair)
                                series_component_1 = series_1.univariate_component(
                                    component_name_1
                                )
                                series_component_2 = series_2.univariate_component(
                                    component_name_2
                                )
                                res[split_name][series_type_pair_key][component_pair] = function(
                                    series_component_1, series_component_2
                                )

                                if presenter is not False:
                                    fig = plt.gcf()
                                    title = f"{split_name}/{series_type_pair_key}/{component_name_1}_{component_name_2}"
                                    plt.title(title)
                                    res[split_name][series_type_pair_key][component_pair] = (
                                        src.utils.plotting.present_figure(
                                            fig, presenter, **presenter_kwargs
                                        )
                                    )

        return res

    def fit_processing_pipeline(
        self,
        dataset: Union[darts.TimeSeries, Sequence[darts.TimeSeries]],
        pipeline: Optional[Dict[str, darts.dataprocessing.Pipeline]] = None,
    ) -> None:
        """Fit a processing (data transformation) pipeline on the given dataset(s).

        If no pipeline is given, the pipeline set to hparams.processing_pipeline will be used.
        Dataset can either be a single darts.TimeSeries or a sequence of darts.TimeSeries.
        :param dataset: Dataset(s) to fit the pipeline on.
        :param pipeline: Pipeline to fit.
        :return:
        """
        if pipeline is None:
            if self.hparams.processing_pipeline is None:
                return ValueError(
                    "You either have to specify a processing pipeline as argument or have set a pipeline to self.hparams.processing_pipeline."
                )

            pipeline = self.hparams.processing_pipeline

        all_data_variables = []
        for dv_name, dvs in self.hparams.data_variables.items():
            if dvs is not None:
                if isinstance(dvs, str):
                    dvs = [dvs]
                all_data_variables.extend(dvs)

        if not isinstance(pipeline, dict) or not pipeline.keys() == set(all_data_variables):
            raise ValueError(
                "Processing_pipeline must be a dictionary with key data_variables and one pipeline per variable."
            )

        dataset = darts.utils.ts_utils.series2seq(dataset)
        for component in pipeline:
            if pipeline[component] is None:
                continue
            component_ds = [ds.univariate_component(component) for ds in dataset]
            pipeline[component].fit(component_ds)
            pipeline[component]._fit_called = True

    def transform_data(
        self,
        dataset: Union[darts.TimeSeries, Sequence[darts.TimeSeries]],
        pipeline: Optional[Dict[str, darts.dataprocessing.Pipeline]] = None,
        fit_pipeline: bool = False,
    ) -> Union[darts.TimeSeries, Sequence[darts.TimeSeries]]:
        """A function to transform a dataset using a darts Pipeline object containing a set of
        transformers. If fit_pipeline is True, the pipeline is fitted to the dataset before
        transforming it. If fit_pipeline is False, the pipeline is only transformed if it has been
        fitted before. If the pipeline has not been fitted, the dataset is returned without
        transformation.

        :param dataset: The dataset to transform.
        :param pipeline: The pipeline to use for transformation. If None, the pipeline specified in
            the __init__ method is used.
        :param fit_pipeline: Whether to fit the pipeline before transforming the dataset.
        :return: The transformed dataset.
        """
        if pipeline is None:
            if self.hparams.processing_pipeline is None:
                log.warning(
                    "Tried transforming dataset but no pipeline has been configured. Returning dataset."
                )
                return dataset

            pipeline = self.hparams.processing_pipeline

        if fit_pipeline:
            self.fit_processing_pipeline(dataset=dataset, pipeline=pipeline)

        # Have considered using the component_mask feature of darts transformers as an alternative to fitting pipeline
        # per component, however, it has two major issues:
        # 1. component_masks applies to the input series rather than the transformer components, such that the input
        # must be a superset of the fit data, not subset as we require
        # 2. Pipelines do not support component masks, such that we would need to operate on the transformers themselves

        datasets = darts.utils.ts_utils.series2seq(dataset)
        transformed = None
        for component in datasets[0].components:
            if component not in pipeline:
                log.exception(f"The component {component} is not in the transformation pipeline")
                raise ValueError(
                    f"The component {component} is not in the transformation pipeline"
                )
            elif pipeline[component] is None:
                transformed_component = [ds.univariate_component(component) for ds in datasets]
            else:
                transformed_component = pipeline[component].transform(
                    [ds.univariate_component(component) for ds in datasets]
                )
            if transformed is None:
                transformed = transformed_component
            else:
                transformed = [
                    transformed[ds_i].stack(transformed_component[ds_i])
                    for ds_i in range(len(datasets))
                ]

        return darts.utils.ts_utils.seq2series(transformed)

    def inverse_transform_data(
        self,
        dataset: Union[darts.TimeSeries, Sequence[darts.TimeSeries]],
        pipeline: Optional[Dict[str, darts.dataprocessing.Pipeline]] = None,
        partial: bool = False,
        verify_diff_transformers: bool = True,
    ) -> Union[darts.TimeSeries, Sequence[darts.TimeSeries]]:
        """A function to inverse transform a dataset using a darts Pipeline object containing a set
        of transformers. If all transformers in the pipeline are not invertible, the partial
        argument can be set to True to return the inverse transformation of the invertible
        transformers. If the pipeline has not been fitted, the dataset is returned without inverse
        transformation. If the pipeline is not invertible and partial is False, an error is raised.

        :param dataset: The dataset to inverse transform.
        :param pipeline: The pipeline to use for inverse transformation. If None, the pipeline
            specified in the __init__ method is used.
        :param partial: Whether to return the inverse transformation of the invertible transformers
            if the pipeline is not invertible.
        :return: The inverse transformed dataset.
        """
        if pipeline is None:
            if self.hparams.processing_pipeline is None:
                log.warning(
                    "Tried inverse transforming dataset but no pipeline has been configured. Returning dataset."
                )
                return dataset
            pipeline = self.hparams.processing_pipeline

            if not src.datamodules.utils.pipeline_is_fitted(pipeline):
                log.warning(
                    "Tried inverse transforming dataset but pipeline has not been fitted. Returning dataset."
                )
                return dataset

        if not partial and not all(p.invertible() for p in pipeline.values()):
            raise ValueError(
                "Pipeline is not invertible, and the partial argument was not set True."
            )

        datasets = darts.utils.ts_utils.series2seq(dataset)
        transformed = None
        for component in datasets[0].components:
            if component not in pipeline:
                log.exception(f"The component {component} is not in the transformation pipeline")
                raise ValueError(
                    f"The component {component} is not in the transformation pipeline"
                )
            if verify_diff_transformers:
                diff_transformer_indexes = [
                    t_i
                    for t_i, transformer in enumerate(pipeline[component]._transformers)
                    if isinstance(transformer, darts.dataprocessing.transformers.Diff)
                ]
                if len(diff_transformer_indexes) > 0:
                    sum_lags = sum(pipeline[component]._transformers[0]._lags)
                    assert (
                        pipeline[component]
                        ._transformers[diff_transformer_indexes[0]]
                        ._fitted_params[0][2]
                        + sum_lags
                        * pipeline[component]
                        ._transformers[diff_transformer_indexes[0]]
                        ._fitted_params[0][3]
                        == dataset.start_time()
                    ), "A pipeline with a darts.Diff transformer is only invertible if it is the first transformer in the pipeline, or used with data starting at the same time as the data it was fitted with."
            transformed_component = pipeline[component].inverse_transform(
                [ds.univariate_component(component) for ds in datasets], partial=partial
            )
            if transformed is None:
                transformed = transformed_component
            else:
                transformed = [
                    transformed[ds_i].stack(transformed_component[ds_i])
                    for ds_i in range(len(datasets))
                ]

        return darts.utils.ts_utils.seq2series(transformed)

    def num_series_for_split(self, split: str) -> Union[int, None]:
        """Convenience function to get the number of series for a given split."""
        split_data = self.get_split_data(split)
        if split_data is None:
            return None
        else:
            return len(darts.utils.ts_utils.series2seq(split_data["target"]))

    @staticmethod
    def subset_dataframe_on_index(df, t_s, t_e, variable=None):
        """Subset a pandas dataframe on its index between start time t_s and end time t_e, and
        possibly only on a specific variable.

        :param df (pd.DataFrame): DataFrame to subset. :param t_s (int, pd.Timestamp): Start time
            of subset :param t_e (int, pd.Timestamp): End time of subset
        :param variable (str): Variable to select subset for.
        :return (pd.DataFrame): DataFrame subset
        """
        if isinstance(t_s, int):
            assert isinstance(t_e, int)
            res = df.iloc[t_s:t_e]
            if variable is not None:
                res = res[variable]
        else:
            if variable is None:
                res = df.loc[(df.index >= t_s) & (df.index <= t_e)]
            else:
                res = df.loc[(df.index >= t_s) & (df.index <= t_e), variable]
        return res

    @staticmethod
    def crop_dataset_range(
        dataset: Union[pd.DataFrame, darts.timeseries.TimeSeries],
        crop_data_range: List[Union[str, pd.Timestamp, int]],
    ) -> Union[pd.DataFrame, darts.timeseries.TimeSeries]:
        """Crops a dataset along the index to a given range.

        :param dataset: The dataset to crop.
        :param crop_data_range: A list of two elements, the first being the start of the range, the
            second being the end. The elements can be strings, pd.Timestamps or ints. If strings,
            they are parsed as pd.Timestamps. If ints, they are interpreted as the number of points
            to crop from the start and end of the dataset.
        :return: The cropped dataset.
        """
        crop_start, crop_stop = crop_data_range
        if isinstance(crop_start, str):
            crop_start = pd.Timestamp(crop_start)
        if isinstance(crop_stop, str):
            crop_stop = pd.Timestamp(crop_stop)

        if isinstance(dataset, pd.DataFrame):
            dataset = TimeSeriesDataModule.subset_dataframe_on_index(
                dataset, crop_start, crop_stop
            )
        elif isinstance(dataset, darts.timeseries.TimeSeries):
            if dataset.has_datetime_index and isinstance(crop_start, int):
                crop_start = dataset.get_timestamp_at_point(crop_start)
            if dataset.has_datetime_index and isinstance(crop_stop, int):
                crop_stop = dataset.get_timestamp_at_point(crop_stop)
            dataset = dataset.slice(crop_start, crop_stop)
        else:
            raise ValueError

        return dataset

    @staticmethod
    def resample_dataset(
        dataset: Union[pd.DataFrame, darts.timeseries.TimeSeries], method: str, freq: str
    ) -> Union[pd.DataFrame, darts.timeseries.TimeSeries]:
        """Resamples a dataset to a given frequency. The dataset must have a pd.DatetimeIndex.

        :param dataset: The dataset to resample.
        :param method: The method to use for resampling. Can be "interpolate" for which the dataset
            is interpolated in time to the new frequency, or any of the methods available in
            pd.DataFrame.resample.
        :param freq: The frequency to resample to.
        :return: The resampled dataset.
        """
        # TODO: sequences of series?
        if isinstance(dataset, pd.DataFrame):
            assert isinstance(
                dataset.index, pd.DatetimeIndex
            ), f"Can only resample data with pd.DatetimeIndex, you have {type(dataset.index)}"

            if method == "interpolate":
                resample_index = dataset.resample(freq).binner
                old_index = dataset.index
                dataset = (
                    dataset.reindex(old_index.union(resample_index))
                    .interpolate(method="time")
                    .reindex(resample_index)
                )
                if dataset.iloc[0].isnull().values.any():
                    dataset = dataset.iloc[1:]
                if dataset.iloc[-1].isnull().values.any():
                    dataset = dataset.iloc[:-1]
                if dataset.isnull().values.any():
                    log.warning("There are nans in the data after resampling with interpolation.")
            else:
                agg_func = getattr(dataset.resample(rule=freq), method, None)
                if agg_func is None:
                    raise ValueError(
                        f"The provided aggregation function is not a valid pandas dispatching function: {method}"
                    )
                dataset = agg_func()
        elif isinstance(dataset, darts.timeseries.TimeSeries):
            dataset = dataset.resample(freq=freq)  # TODO: kwargs?
        else:
            raise NotImplementedError

        return dataset

    @staticmethod
    def set_dataset_precision(
        dataset: darts.TimeSeries, precision: Union[str, int]
    ) -> darts.TimeSeries:
        """Sets the precision of a dataset to a given precision.

        :param dataset: The dataset to set the precision of.
        :param precision: The precision to set the dataset to in bits.
        :return: The dataset with the new precision.
        """
        return dataset.astype(np.dtype(getattr(np, f"float{precision}")))

    @staticmethod
    def process_train_val_test_split(
        dataset: Dict[Any, darts.TimeSeries],
        train_val_test_split: Union[Dict[str, Union[float, Tuple[str, str]]], None],
    ) -> Union[Dict[str, Tuple[Union[float, int, str], Union[float, int, str]]], None]:
        """Takes in the train_val_test_split dictionary, asserts it has a valid structure, and
        processes it to a dictionary with the correct types. If applicable converts single value-
        style splits into list-style splits, i.e. if the train_val_test_split is {"train": 0.8,
        "val": 0.2} it will be converted to.

        {"train": [0, 0.8], "val": [0.8, 1.0]}.

        :param dataset: The dataset that the train_val_test_split is to be applied to.
        :param train_val_test_split: The train_val_test_split dictionary to process.
        :return: The processed train_val_test_split dictionary.
        """

        def get_split_indices(_split_values, _dataset):
            if isinstance(_split_values[0], str):
                if _split_values[0] == "start":
                    _split_values[0] = _dataset.start_time()
                else:
                    _split_values[0] = pd.Timestamp(_split_values[0])
                if _split_values[1] == "end":
                    _split_values[1] = _dataset.end_time()
                else:
                    _split_values[1] = pd.Timestamp(_split_values[1])

            if isinstance(_split_values[0], (float, pd.Timestamp)):
                _split_values = [_dataset.get_index_at_point(sv) for sv in _split_values]
                if _dataset.has_range_index:
                    _split_values = [sv + _dataset.start_time() for sv in _split_values]
            # avoid overlap by making end non-inclusive if next split starts at same point
            if (
                split_i < len(split_order) - 1
                and dataset_splits[split_name][-1] == dataset_splits[split_order[split_i + 1]][0]
            ):
                _split_values[-1] -= 1
            if _dataset.has_datetime_index:
                _split_values = [_dataset.get_timestamp_at_point(sv) for sv in _split_values]

            return _split_values

        if train_val_test_split is None:
            return None
        assert isinstance(train_val_test_split, dict)

        # test if splits are specified per dataset
        # if not, copy splits into dictionary with dataset keys as keys
        # then go through each dataset with the keys and run below code
        # TODO: how to handle float splits not per dataset, apply to each dataset or on total datapoints?

        if not train_val_test_split.keys() == dataset.keys():
            # Consider if we should help the user allocate over datapoints rather than over dataset
            # E.g. if train: 0.5 and val: 0.5 and we have two equally sized datasets, allocate one for training and
            # one for val rather than dividing each dataset into 50% for each.

            train_val_test_split = {
                dataset_name: copy.deepcopy(train_val_test_split)
                for dataset_name in dataset.keys()
            }

        for dataset_name in dataset:
            dataset_splits = train_val_test_split[dataset_name]
            if dataset_splits is None:
                continue
            dataset_splits = {k: v for k, v in dataset_splits.items() if v is not None}
            split_order = list(dataset_splits)
            assert all(
                [so in ["train", "test", "val"] for so in split_order]
            ), "Only the entries [train, val, test] are allowed in dataset_splits"
            split_values = list(dataset_splits.values())

            # TODO: add support for splits per dataset
            # check that all split values are the same type
            assert (
                type(split_values[0]) in _ALLOWED_SPLIT_TYPES
            ), f"Split values must have one of the following types: {_ALLOWED_SPLIT_TYPES}"
            # TODO: works if multiple list splits are mixed with single list-splits?
            assert all(
                [isinstance(sv, type(split_values[0])) for sv in split_values]
            ), f"All split values must be same type, you have {[type(sv) for sv in split_values]}"

            # If only provided one value per split, convert to a list of [start_index, stop_index]
            if not isinstance(split_values[0], (list, tuple)):
                _assert_compatible_with_index(
                    split_values[0], dataset[dataset_name].time_index
                )  # TODO: convert for user?
                if isinstance(split_values[0], float):
                    assert (
                        sum(split_values) <= 1.0
                    ), f"You have provided split_values as floats, but they add up to more than 1 ({sum(split_values)})"
                    prev_split = 0.0
                elif isinstance(split_values[0], int):
                    assert sum(split_values) <= len(
                        dataset[dataset_name]
                    ), f"You have provided split_values as ints, but they add up to more than the number of elements in the dataset ({len(dataset[dataset_name])}"
                    prev_split = 0
                elif isinstance(split_values[0], str) or isinstance(split_values[0], pd.Timestamp):
                    prev_split = dataset[dataset_name].start_time()
                    # TODO: warn if they are outside the dataset
                else:
                    raise ValueError

                for split_name in split_order:
                    if split_name in dataset_splits:
                        if isinstance(split_values[0], (float, int)):
                            this_split = prev_split + dataset_splits[split_name]
                            list_split_values = [prev_split, this_split]
                        elif isinstance(split_values[0], (str, pd.Timestamp)):
                            this_split = dataset_splits[split_name]
                            if isinstance(this_split, str):
                                assert this_split not in [
                                    "start",
                                    "end",
                                ], "Can not use the special strings [start, end] when not supplying explicit [start_point, end_point] values for each split."
                                this_split = pd.Timestamp(this_split)
                            list_split_values = [prev_split, this_split]
                        else:
                            raise ValueError
                        prev_split = this_split
                        dataset_splits[split_name] = list_split_values
            else:
                if any(isinstance(split_values[i][0], list) for i in range(len(split_values))):

                    def flatten_list_of_lists(list_of_lists):
                        result = []
                        for item in list_of_lists:
                            if isinstance(item[0], list):
                                result.extend(item)
                            else:
                                result.append(item)
                        return result

                    split_values_flat = flatten_list_of_lists(split_values)
                    for split_value in split_values_flat:
                        _assert_compatible_with_index(
                            split_value[0], dataset[dataset_name].time_index
                        )
                        # TODO: check that they are disjoint?
                        # TODO: yet to assert that all split_values have the same type (i.e. int/float/pd.Timestamp)
                else:
                    _assert_compatible_with_index(
                        split_values[0][0], dataset[dataset_name].time_index
                    )

            for split_i, split_name in enumerate(split_order):
                split_values = dataset_splits[split_name]
                if isinstance(split_values[0], list):
                    split_values = [
                        get_split_indices(sv, dataset[dataset_name]) for sv in split_values
                    ]
                else:
                    split_values = get_split_indices(split_values, dataset[dataset_name])
                train_val_test_split[dataset_name][split_name] = split_values

        return train_val_test_split

        # TODO: assert that the provided split has same type as the index (i.e. int for RangeIndex, Timestamp for DateTimeIndex)

    def get_split_data(
        self,
        split: str,
        transform: bool = True,
        datasets: Optional[Union[str, Sequence[str]]] = None,
    ) -> Union[None, Dict[str, Union[Sequence[darts.TimeSeries], darts.TimeSeries]]]:
        """Get data for a given split as a Dictionary with keys being the data_variable types
        (target, future_covariates, past_covariates), and values being the series associated with
        these based on the datasets bound to self.data and the splits configured in
        train_val_test_split.

        Optionally transform the data.
        :param split: Name of split, one of ['train', 'val', 'test']
        :param transform: Whether to transform the data with the processing_pipeline.
        :param datasets: Name of dataset(s) to get data for, i.e. entries in the self.data
            dictionary. If none, all datasets will be used.
        :return: Dictionary of split data or None if split has no data.
        """
        if not self.has_split_data(split):
            return None

        res = {
            k: [] if self.hparams.data_variables.get(k, None) else None
            for k in ["target", "past_covariates", "future_covariates"]
        }

        split_datasets = self._get_split_data_raw(split=split, datasets=datasets)

        for dataset in split_datasets:
            if transform:
                dataset = self.transform_data(dataset)
            for series_name in ["target", "past_covariates", "future_covariates"]:
                if self.hparams.data_variables.get(series_name, None) is not None:
                    res[series_name].append(
                        dataset.drop_columns(
                            [
                                dv
                                for dv in dataset.columns
                                if dv not in self.hparams.data_variables[series_name]
                            ]
                        )
                    )

        return {k: darts.utils.ts_utils.seq2series(v) for k, v in res.items()}

    def _get_split_data_raw(
        self, split: str, datasets: Optional[Union[str, Sequence[str]]] = None
    ) -> Union[Sequence[darts.TimeSeries], None]:
        """Helper function to get the data for a given split before organizing into the covariates
        as a dictionary.

        :param split: Name of split, one of ['train', 'val', 'test']
        :param datasets: Name of dataset(s) to get data for, i.e. entries in the self.data
            dictionary. If none, all datasets will be used.
        :return: darts.TimeSeries sequence for split.
        """
        if not self.has_split_data(split):
            return None
        if isinstance(datasets, str):
            datasets = [datasets]

        split_datasets = []

        for dataset_name, dataset in self.data.items():
            if datasets is not None and str(dataset_name) not in datasets:
                continue
            if (
                self.hparams.train_val_test_split is None
                or self.hparams.train_val_test_split[dataset_name] is None
            ):
                if split == "train":
                    split_datasets.append(dataset)
                continue
            elif self.hparams.train_val_test_split[dataset_name].get(split, None) is None:
                continue
            split_values = self.hparams.train_val_test_split[dataset_name].get(split)
            if isinstance(split_values[0], list):
                split_datasets.extend([dataset.slice(*split_v) for split_v in split_values])
            else:
                split_datasets.append(dataset.slice(*split_values))

        if len(split_datasets) == 0:
            return None

        return split_datasets

    def has_split_data(self, split: str) -> bool:
        """Returns True if the datamodule has data for the specified split, False otherwise.

        :param split: The split to check for data.
        :return: True if the datamodule has data for the specified split, False otherwise.
        """
        if self.hparams.train_val_test_split is None:
            return split == "train"

        return any(
            (dataset_split is None and split == "train")
            or (dataset_split is not None and dataset_split.get(split) is not None)
            for dataset_split in self.hparams.train_val_test_split.values()
        )

    def has_split_covariate_type(self, split: str, covariate_type: str) -> bool:
        """Returns True if the datamodule has the given covariate_type (past/future/static) for the
        specified split, False otherwise.

        :param split: The split to check for data.
        :param covariate_type: Type of covariate (past/future/static)
        :return: True if the datamodule has the covariate type for the specified split, False
            otherwise.
        """
        # TODO: perhaps better to just check if covariate in data_variables
        split_data = self.get_split_data(split, transform=False)
        return split_data is not None and split_data.get(covariate_type) is not None
