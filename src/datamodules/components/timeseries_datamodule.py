import itertools
import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import darts.dataprocessing.pipeline
import darts.timeseries
import darts.utils.model_selection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule

import src.datamodules.components.dataloaders as dataloaders
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
        _assert_valid_type_and_index(self.data)
        assert isinstance(
            self.hparams.freq, _SUPPORTED_FREQ_TYPES
        ), f"The supported types for freq are {_SUPPORTED_FREQ_TYPES}, you have {type(self.hparams.freq)}"

        if load_dir is not None:
            self.load_state(load_dir)
        else:
            if self.hparams.processing_pipeline is not None:
                assert (
                    self.hparams.train_val_test_split is None
                    or "train" in self.hparams.train_val_test_split
                ), "A training set is not configured, and no state for the data processing_pipeline was provided."

        all_data_variables = []
        for dv_name, dvs in self.hparams.data_variables.items():
            if dv_name != "actual_anomalies" and dvs is not None:
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

        if self.hparams.crop_data_range is not None:
            self.data = self.crop_dataset_range(self.data, self.hparams.crop_data_range)

        if self.hparams.data_variables.get("actual_anomalies", None) is not None:
            if isinstance(self.data, pd.DataFrame):
                self.data.columns.name = None  # fix potential bug with Timeseries.from_dataset
                self.labels = darts.timeseries.TimeSeries.from_dataframe(
                    self.data,
                    value_cols=self.hparams.data_variables["actual_anomalies"],
                    freq=self.hparams.freq,
                )
            elif isinstance(self.data, darts.timeseries.TimeSeries):
                raise NotImplementedError
                # self.labels = self.data["actual_anomalies"]
            else:
                raise ValueError
            self.labels = self.labels.astype(
                np.dtype(getattr(np, f"float{self.hparams.precision}"))
            )
            self.labels = darts.utils.missing_values.fill_missing_values(
                self.labels, fill=0.0
            )  # make configurable fill_value?
        # TODO: component_wise
        else:
            self.labels = None  # TODO: more integrated handling of labels ( big problem is that it should not go through pipeline, at least not all stages).

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

            self.data = self.resample_dataset(self.data, **self.hparams.resample)

        if isinstance(self.data, pd.DataFrame):
            self.data.columns.name = None  # fix potential bug with Timeseries.from_dataset
            self.data = darts.timeseries.TimeSeries.from_dataframe(
                self.data,
                value_cols=all_data_variables,
                fill_missing_dates=True,
                freq=self.hparams.freq,
            )  # TODO: add support for other arguments

        self.data = self.set_dataset_precision(self.data, precision=self.hparams.precision)

        self.hparams.train_val_test_split = self.process_train_val_test_split(
            self.data, self.hparams.train_val_test_split
        )

        for split_name, split_data in self.split_dataset(
            self.data, self.hparams.train_val_test_split
        ).items():
            setattr(self, f"data_{split_name}", split_data)

        for split_name in ["train", "val", "test"]:
            split_data = getattr(self, f"data_{split_name}")
            if split_data is None:
                continue

            if self.hparams.processing_pipeline is not None:
                if split_name == "train" and not getattr(
                    self.hparams.processing_pipeline, "_fit_called", False
                ):
                    split_data = self.transform_data(split_data, fit_pipeline=True)
                else:
                    if not getattr(self.hparams.processing_pipeline, "_fit_called", False):
                        if self.data_train is None:
                            raise RuntimeError(
                                "A pipeline has been configured, but no training set has been provided on which it can be fitted. Either pass load_dir containing the state of a pipeline or configure a training set."
                            )
                        else:
                            raise RuntimeError(
                                "processing_pipeline.transform was called before it was fitted. Ensure datamodule configuration is correct. Contact developer as this is an unexpected error."
                            )
                    split_data = self.transform_data(split_data)

            split_data_seq = darts.utils.utils.series2seq(split_data)
            split_series = {
                k: [] if self.hparams.data_variables.get(k, None) else None
                for k in ["target", "past_covariates", "future_covariates", "actual_anomalies"]
            }

            for split_data in split_data_seq:
                if self.labels is not None:
                    split_labels = self.labels.slice_intersect(split_data)
                else:
                    split_labels = None

                if self.hparams.check_for_nan:
                    if np.any(np.isnan(split_data.all_values())):
                        raise ValueError(
                            f"The {split_name} dataset contains nan-values and the check_for_nan attribute is set to True. Please check if data-processing pipeline is configured correctly."
                        )
                    if split_labels is not None and np.any(np.isnan(split_labels.all_values())):
                        raise ValueError(
                            f"The labels for {split_name} dataset contains nan-values and the check_for_nan attribute is set to True. Please check if data-processing pipeline is configured correctly."
                        )

                for series_name in ["target", "past_covariates", "future_covariates"]:
                    if self.hparams.data_variables.get(series_name, None) is not None:
                        split_series[series_name].append(
                            split_data.drop_columns(
                                [
                                    dv
                                    for dv in all_data_variables
                                    if dv not in self.hparams.data_variables[series_name]
                                ]
                            )
                        )

                if split_labels is not None:
                    split_series["actual_anomalies"].append(split_labels)

            setattr(self, f"data_{split_name}", split_series)

    def get_data(
        self, data_kwargs: List[str], main_split: str = "train"
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
        :return: Dictionary of data formatted with the names expected by the model.fit method.
        """
        assert self.has_split_data(main_split), f"No data has been set for split {main_split}"
        data_translator = {
            "target": "target",
            "series": "target",
            "past_covariates": "past_covariates",
            "future_covariates": "future_covariates",  # TODO: static covariates
            "actual_anomalies": "actual_anomalies",
        }
        res = {}
        for kwarg in data_kwargs:
            if kwarg.startswith("val_"):  # TODO: can there be others, e.g. test?
                kwarg_name_split = kwarg.split("_")
                kwarg_split = kwarg_name_split[0]
                kwarg = "_".join(kwarg_name_split[1:])
            else:
                kwarg_split = main_split
            if kwarg in data_translator:
                if data_translator[kwarg] in getattr(self, f"data_{kwarg_split}"):
                    kwarg_value = darts.utils.utils.seq2series(
                        getattr(self, f"data_{kwarg_split}")[data_translator[kwarg]]
                    )
                else:
                    kwarg_value = None
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

    def train_dataloader(self):  # TODO: remove these
        return self.data_train.to_dataloader(
            train=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return self.data_val.to_dataloader(
            train=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return self.data_test.to_dataloader(
            train=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def predict_dataloader(self):
        return getattr(self, f"data_{self.hparams.predict_split}").to_dataloader(
            train=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

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
        slice: Optional[
            Tuple[Union[float, int, pd.Timestamp], Union[float, int, pd.Timestamp]]
        ] = None,  # TODO: convert for user?
        predictions: Optional[darts.timeseries.TimeSeries] = None,
        separate_components: Optional[bool] = None,
        presenter: Any = None,
        **presenter_kwargs,
    ) -> Union[List[plt.Figure], None]:  # TODO: consider rewriting to using TSD to save space
        """Plots the dataset splits. If no split is specified, all splits are plotted.

        :param split: The split to plot. If None, all splits are plotted.
        :param slice: A tuple of (start, end) to slice the data before plotting.
        :param predictions: A TimeSeries containing predictions to plot.
        :param separate_components: Whether to plot the components of the split separately. If
            None, components will be plotted separately if there are _PLOT_SEPARATE_MAX or fewer
            components, or together otherwise.
        :param presenter: A presenter class to use for plotting. If None, the default presenter is
            used.
        :param presenter_kwargs: Keyword arguments to pass to the presenter.
        :return: A list of figures if presenter is None, otherwise None
        """
        if predictions is not None:
            raise NotImplementedError
        if slice is not None:
            _assert_compatible_with_index(slice[0], self.data.time_index)
        if split is not None:
            return self._plot_data_split(
                split=split,
                slice=slice,
                predictions=predictions,
                separate_components=separate_components,
                presenter=presenter,
                **presenter_kwargs,
            )
        else:
            figs = []
            for split in ["train", "val", "test"]:
                if self.has_split_data(split):
                    figs.extend(
                        self._plot_data_split(
                            split=split,
                            slice=slice,
                            predictions=predictions,
                            separate_components=separate_components,
                            presenter=presenter,
                            **presenter_kwargs,
                        )
                    )  # TODO: what about predictions now? Dictionary?

            return figs

    def _plot_data_split(
        self,
        split: str,
        slice: Optional[
            Tuple[Union[float, int, pd.Timestamp], Union[float, int, pd.Timestamp]]
        ] = None,
        predictions: Optional[darts.timeseries.TimeSeries] = None,
        separate_components: Optional[bool] = None,
        presenter: Any = "savefig",
        **presenter_kwargs,
    ) -> Union[List[plt.Figure], None]:
        """Plot a single split of the dataset.

        :param split: The split to plot.
        :param slice: A tuple of (start, end) to slice the data before plotting.
        :param predictions: A TimeSeries containing predictions to plot.
        :param separate_components: Whether to plot the components of the split separately. If
            None, components will be plotted separately if there are _PLOT_SEPARATE_MAX or fewer
            components, or together otherwise.
        :param presenter: A presenter class to use for plotting. If None, the default presenter is
            used.
        :param presenter_kwargs: Keyword arguments to pass to the presenter.
        :return: A list of figures if presenter is None, otherwise None
        """
        split_series = getattr(self, f"data_{split}")
        if split_series is None:
            log.info(f"No data to plot for split {split}")
            return None
        figs = []
        for series_name, serieses in split_series.items():  # TODO: sequences of series?
            if serieses is not None:
                for series_i, series in enumerate(serieses):
                    title_prefix = f"dataset {split} {series_name}"
                    if len(serieses) > 1:
                        title_prefix += f" {series_i}"
                    if slice is not None:
                        if all(slice[i] not in series.time_index for i in range(2)):
                            continue
                        else:
                            series = series.slice(*slice)
                    if separate_components is None:
                        separate_series_components = len(series.components) <= _PLOT_SEPARATE_MAX
                    else:
                        separate_series_components = separate_components
                    if separate_series_components:
                        for component_name in series.components.values:
                            figs.append(
                                src.utils.plotting.plot_darts_timeseries(
                                    series[component_name],
                                    title=title_prefix + f" {component_name}",
                                    presenter=presenter,
                                    **presenter_kwargs,
                                )
                            )
                    else:
                        figs.append(
                            src.utils.plotting.plot_darts_timeseries(
                                series, title=title_prefix, presenter=presenter, **presenter_kwargs
                            )
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
                for series_type, serieses in getattr(self, f"data_{split_name}").items():
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
                split_data = getattr(self, f"data_{split_name}")
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

    def get_split_range(
        self, split: str
    ) -> Union[Tuple[Union[int, pd.Timestamp], Union[int, pd.Timestamp]], None]:  # TODO: fix seq
        """Returns the range of a data split, i.e. a tuple with (start_index, stop_index). If data
        has DatetimeIndex this function returns pd.Timestamp, if data has RangeIndex this function
        returns int-indexes, if the split has no data, this function returns None.

        :param split: which data split to get range for. One of ["train", "val", "test", "predict"]
        :return: range of data split.
        """
        assert (
            split in _VALID_SPLIT_NAMES
        ), f"split must be one of {_VALID_SPLIT_NAMES}, you have split."
        if split == "predict":
            split = self.hparams.predict_split

        split_data = getattr(self, f"data_{split}")
        if split_data is None:
            return None
        else:
            split_data = split_data["target"][0]  # TODO: what to do with multiple sets?
            return split_data.start_time(), split_data.end_time()

    def transform_data(
        self,
        dataset: Union[darts.TimeSeries, Sequence[darts.TimeSeries]],
        pipeline: Optional[darts.dataprocessing.Pipeline] = None,
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
            pipeline.fit(dataset)
            pipeline._fit_called = True
        else:
            if not getattr(pipeline, "_fit_called", False):
                log.warning(
                    "Tried transforming dataset but pipeline has not been fitted. Returning dataset."
                )
                return dataset

        datasets = darts.utils.utils.series2seq(dataset)

        transformed = []
        for dataset in datasets:
            # TODO: might be wrong if someone passed a pipeline?
            if not dataset.width == self.data.width or any(
                dataset.components != self.data.components
            ):
                assert set(dataset.components).issubset(
                    self.data.components
                ), "transform_data was called with a dataset that includes components that the processing pipeline was not fitted for."
                dummy_dataset = self.data.slice_intersect(dataset)
                component_indexes = [
                    self.data.components.get_loc(comp) for comp in dataset.components
                ]
                dummy_dataset_values = dummy_dataset.all_values()
                dummy_dataset_values[:, component_indexes, :] = dataset.all_values()

                dummy_transformed = pipeline.transform(
                    dummy_dataset.with_values(dummy_dataset_values)
                )

                transformed.append(
                    dataset.with_values(dummy_transformed.all_values()[:, component_indexes, :])
                )
            else:
                transformed.append(pipeline.transform(dataset))

        return darts.utils.utils.seq2series(transformed)

    def inverse_transform_data(
        self,
        dataset: Union[darts.TimeSeries, Sequence[darts.TimeSeries]],
        pipeline: Optional[darts.dataprocessing.Pipeline] = None,
        partial: bool = False,
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
            elif not getattr(self.hparams.processing_pipeline, "_fit_called", False):
                log.warning(
                    "Tried inverse transforming dataset but pipeline has not been fitted. Returning dataset."
                )
                return dataset

            pipeline = self.hparams.processing_pipeline

        if not partial and not pipeline.invertible():
            raise ValueError(
                "Pipeline is not invertible, and the partial argument was not set True."
            )

        diff_transformer_indexes = [
            t_i
            for t_i, transformer in enumerate(pipeline._transformers)
            if isinstance(transformer, darts.dataprocessing.transformers.Diff)
        ]
        if len(diff_transformer_indexes) > 0:
            sum_lags = sum(pipeline._transformers[0]._lags)
            if len(diff_transformer_indexes) > 1 or diff_transformer_indexes[0] != 0:
                assert (
                    pipeline._transformers[diff_transformer_indexes[0]]._fitted_params[0][2]
                    + sum_lags
                    * pipeline._transformers[diff_transformer_indexes[0]]._fitted_params[0][3]
                    == dataset.start_time()
                ), "A pipeline with a darts.Diff transformer is only invertible if it is the first transformer in the pipeline, or used with data starting at the same time as the data it was fitted with."
            else:
                original_dataset = self.data.slice_n_points_before(
                    dataset.start_time(), sum_lags + 1
                )
                pipeline._transformers[0].fit(original_dataset)

        datasets = darts.utils.utils.series2seq(dataset)

        transformed = []
        for dataset in datasets:
            # TODO: might be wrong if someone passed a pipeline?
            if not dataset.width == self.data.width or (
                list(dataset.components.values) != [str(i) for i in range(dataset.width)]
                and any(dataset.components != self.data.components)
            ):
                assert set(dataset.components).issubset(
                    self.data.components
                ), "transform_data was called with a dataset that includes components that the processing pipeline was not fitted for."
                dummy_dataset = self.data.slice_intersect(dataset)
                component_indexes = [
                    self.data.components.get_loc(comp) for comp in dataset.components
                ]
                dummy_dataset_values = dummy_dataset.all_values()
                dummy_dataset_values[:, component_indexes, :] = dataset.all_values()

                dummy_transformed = pipeline.inverse_transform(
                    dummy_dataset.with_values(dummy_dataset_values), partial=partial
                )

                transformed.append(
                    dataset.with_values(dummy_transformed.all_values()[:, component_indexes, :])
                )
            else:
                transformed.append(pipeline.inverse_transform(dataset, partial=partial))

        return darts.utils.utils.seq2series(transformed)

    def num_series_for_split(self, split: str) -> Union[int, None]:
        """Convenience function to get the number of series for a given split."""
        split_data = getattr(self, f"data_{split}")
        if split_data is None:
            return None
        else:
            return len(split_data["target"])

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
        dataset: darts.TimeSeries,
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
        if train_val_test_split is None:
            return None
        assert isinstance(train_val_test_split, dict)
        train_val_test_split = {k: v for k, v in train_val_test_split.items() if v is not None}
        split_order = list(train_val_test_split)
        assert all(
            [so in ["train", "test", "val"] for so in split_order]
        ), "Only the entries [train, val, test] are allowed in train_val_test_split"
        split_values = list(train_val_test_split.values())

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
                split_values[0], dataset.time_index
            )  # TODO: convert for user?
            if isinstance(split_values[0], float):
                assert (
                    sum(split_values) <= 1.0
                ), f"You have provided split_values as floats, but they add up to more than 1 ({sum(split_values)})"
                prev_split = 0.0
            elif isinstance(split_values[0], int):
                assert sum(split_values) <= len(
                    dataset
                ), f"You have provided split_values as ints, but they add up to more than the number of elements in the dataset ({len(dataset)}"
                prev_split = 0
            elif isinstance(split_values[0], str) or isinstance(split_values[0], pd.Timestamp):
                prev_split = dataset.start_time()
                pass  # TODO: warn if they are outside the dataset
            else:
                raise ValueError

            for split_name in split_order:
                if split_name in train_val_test_split:
                    if isinstance(split_values[0], (float, int)):
                        this_split = prev_split + train_val_test_split[split_name]
                        list_split_values = [prev_split, this_split]
                    elif isinstance(split_values[0], (str, pd.Timestamp)):
                        this_split = train_val_test_split[split_name]
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
                    train_val_test_split[split_name] = list_split_values
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
                    _assert_compatible_with_index(split_value[0], dataset.time_index)
                    # TODO: check that they are disjoint?
                    # TODO: yet to assert that all split_values have the same type (i.e. int/float/pd.Timestamp)
            else:
                _assert_compatible_with_index(split_values[0][0], dataset.time_index)

        return train_val_test_split

        # TODO: assert that the provided split has same type as the index (i.e. int for RangeIndex, Timestamp for DateTimeIndex)

    @staticmethod
    def split_dataset(
        dataset: darts.TimeSeries,
        train_val_test_split: Union[Dict[str, Union[float, Tuple[str, str]]], None],
    ) -> Dict[str, darts.TimeSeries]:
        """Takes a dataset and a processed train_val_test_split dictionary and splits the dataset,
        return a dictionary with the split as keys and the split data as values. Ensure that the
        train_val_test_split dictionary has been processed by calling process_train_val_test_split
        before calling this function.

        :param dataset: darts.Timeseries object to be split
        :param train_val_test_split: Dictionary with keys split_name and values list of
            [start_index, stop_index] for that split.
        :return: Dictionary with keys split_name and values split_data.
        """

        def process_split_values(_split_values):
            assert isinstance(_split_values, (list, tuple)) and len(_split_values) == 2, (
                "The train_val_test_split argument does not have the correct structure. Have you called"
                "process_train_val_test_split first?"
            )
            if isinstance(_split_values[0], str):
                if _split_values[0] == "start":
                    _split_values[0] = dataset.start_time()
                else:
                    _split_values[0] = pd.Timestamp(_split_values[0])
                if _split_values[1] == "end":
                    _split_values[1] = dataset.end_time()
                else:
                    _split_values[1] = pd.Timestamp(_split_values[1])
                train_val_test_split[split_name] = _split_values

            if isinstance(_split_values[0], (float, pd.Timestamp)):
                _split_values = [dataset.get_index_at_point(sv) for sv in _split_values]
                if dataset.has_range_index:
                    _split_values = [sv + dataset.time_index[0] for sv in _split_values]
            # avoid overlap by making end non-inclusive if next split starts at same point
            if (
                split_i < len(split_order) - 1
                and train_val_test_split[split_name][-1]
                == train_val_test_split[split_order[split_i + 1]][0]
            ):
                _split_values[-1] -= 1
            if isinstance(dataset.time_index, pd.DatetimeIndex):
                _split_values = [dataset.get_timestamp_at_point(sv) for sv in _split_values]

            return _split_values

        splits = {}
        if train_val_test_split is None:
            splits["train"] = dataset

            return splits

        split_order = list(train_val_test_split)
        for split_i, split_name in enumerate(
            train_val_test_split
        ):  # TODO: use ordered_dict to ensure order?
            if split_name in train_val_test_split:
                split_values = train_val_test_split[split_name]
                if isinstance(split_values[0], list):
                    splits[split_name] = [
                        dataset.slice(*process_split_values(split_v)) for split_v in split_values
                    ]
                else:
                    splits[split_name] = dataset.slice(*process_split_values(split_values))
            else:
                raise ValueError

        return splits

    def has_split_data(self, split: str) -> bool:
        """Returns True if the datamodule has data for the specified split, False otherwise.

        :param split: The split to check for data.
        :return: True if the datamodule has data for the specified split, False otherwise.
        """
        return getattr(self, f"data_{split}") is not None

    def has_split_covariate_type(self, split: str, covariate_type: str) -> bool:
        """Returns True if the datamodule has the given covariate_type (past/future/static) for the
        specified split, False otherwise.

        :param split: The split to check for data.
        :param covariate_type: Type of covariate (past/future/static)
        :return: True if the datamodule has the covariate type for the specified split, False
            otherwise.
        """
        split_data = getattr(self, f"data_{split}")
        return split_data is not None and split_data.get(covariate_type) is not None
