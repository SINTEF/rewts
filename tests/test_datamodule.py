import copy
import itertools
import os
from pathlib import Path

import darts.dataprocessing
import darts.datasets
import hydra.utils
import numpy as np
import pandas as pd
import pytest
import sklearn.preprocessing
import torch

import src.datamodules.components
import src.datamodules.utils
import src.utils

# TODO: test illegal splits
# TODO: hvorfor ikke bare en sekvens av datamoduler?
# * hvordan velger man da processing pipeline? Kanskje fitte en pipeline ved å ta alle datasettene?
# * hvordan blir det med plotting av dataset?
# * hvordan skal det rent praktisk implementeres? Wrapper rundt liste med dm's?


_SCALER_PIPELINE_CONFIG = dict(
    _target_="darts.dataprocessing.Pipeline",
    transformers=[
        dict(
            _target_="darts.dataprocessing.transformers.Scaler",
            global_fit=True,
            scaler=dict(_target_="sklearn.preprocessing.StandardScaler"),
        )
    ],
)


@pytest.mark.parametrize("lags", [[4], 3])
def test_pipeline_inverse_diff(get_darts_example_dm, lags):
    """Test inverse transformation using pipeline with Diff transformer."""
    dm = get_darts_example_dm
    dm.hparams.train_val_test_split = {"train": 0.5, "val": 0.25, "test": 0.25}
    if isinstance(lags, list):
        sum_lags = sum(lags)
    else:
        sum_lags = lags

    dm.hparams.processing_pipeline = darts.dataprocessing.Pipeline(
        [
            darts.dataprocessing.transformers.Diff(lags=lags),
            darts.dataprocessing.transformers.Scaler(
                scaler=sklearn.preprocessing.StandardScaler()
            ),
        ]
    )
    dm.setup("fit")

    assert np.allclose(
        dm.data[0].values(), dm.inverse_transform_data(dm.transform_data(dm.data[0])).values()
    )
    for split in ["train", "val", "test"]:
        split_data_transformed = dm.get_data(["target"], main_split=split)["target"]
        original_split_data = dm.data[0].slice_n_points_before(
            split_data_transformed.end_time(), len(split_data_transformed) + sum_lags
        )
        # Only works with diff transformer when data to be transformed is aligned with start of dataset
        if split == "train":
            split_data_inversed = dm.inverse_transform_data(split_data_transformed, partial=True)
            assert np.allclose(original_split_data.values(), split_data_inversed.values())
        else:
            with pytest.raises(AssertionError):
                split_data_inversed = dm.inverse_transform_data(
                    split_data_transformed, partial=True
                )


@pytest.mark.parametrize("dataset_name", ["example_ettm1"])
@pytest.mark.parametrize("pipeline_config", [_SCALER_PIPELINE_CONFIG])
def test_pipeline_subset(get_darts_example_dm, pipeline_config):
    """Test that inverse transforming with a subset of the original data (e.g. only the target
    variable) works and returns correct result."""
    dm = get_darts_example_dm
    dm.hparams.processing_pipeline = hydra.utils.instantiate(pipeline_config, _convert_="partial")
    dm.setup("fit")

    data_train = dm.get_data(["target"], main_split="train")["target"]
    assert src.datamodules.utils.pipeline_is_fitted(dm.hparams.processing_pipeline)
    original_train_data = dm.data[0].slice_intersect(data_train)
    pipeline = hydra.utils.instantiate(pipeline_config, _convert_="partial")
    pipeline = src.datamodules.utils.ensure_pipeline_per_component(
        pipeline, dm.hparams.data_variables
    )
    transformed_train_data = dm.transform_data(
        original_train_data, fit_pipeline=True, pipeline=pipeline
    )
    transformed_train_data_subset = transformed_train_data.univariate_component(2)

    assert np.allclose(
        original_train_data.univariate_component(2).values(),
        dm.inverse_transform_data(transformed_train_data_subset).values(),
    )


@pytest.mark.parametrize("dataset_name", ["example_ettm1", "example_aus_beer"])
@pytest.mark.parametrize("pipeline_config", [_SCALER_PIPELINE_CONFIG, None])
def test_pipeline(get_darts_example_dm, pipeline_config):
    """Test setting up datamodule with and without pipeline."""
    dm = get_darts_example_dm
    dm.hparams.processing_pipeline = hydra.utils.instantiate(pipeline_config, _convert_="partial")
    dm.setup("fit")

    # ensure pipelines are different objects, not copies of each other
    if dm.data[0].n_components > 1 and dm.hparams.processing_pipeline is not None:
        pipeline_component = list(dm.hparams.processing_pipeline.keys())[0]
        assert all(
            id(dm.hparams.processing_pipeline[pipeline_component]) != id(pipeline)
            for component, pipeline in dm.hparams.processing_pipeline.items()
            if component != pipeline_component
        )

    data_train = dm.get_data(["target"], main_split="train")["target"]
    data_val = dm.get_data(["target"], main_split="val")["target"]
    if pipeline_config is None:
        assert (
            dm.data[0].slice_intersect(data_train).univariate_component(data_train.components[0])
            == data_train
        )
    else:
        assert src.datamodules.utils.pipeline_is_fitted(dm.hparams.processing_pipeline)
        original_train_data = dm.data[0].slice_intersect(data_train)
        original_val_data = dm.data[0].slice_intersect(data_val)
        assert not original_train_data == data_train

        pipeline = hydra.utils.instantiate(pipeline_config, _convert_="partial")
        pipeline = src.datamodules.utils.ensure_pipeline_per_component(
            pipeline, dm.hparams.data_variables
        )
        transformed_train_data = dm.transform_data(
            original_train_data, fit_pipeline=True, pipeline=pipeline
        )
        assert src.datamodules.utils.pipeline_is_fitted(pipeline)
        assert transformed_train_data.univariate_component(data_train.components[0]) == data_train
        assert (
            dm.transform_data(original_val_data, pipeline=pipeline).univariate_component(
                data_val.components[0]
            )
            == data_val
        ), "Pipeline was not fitted on just training data"

        assert np.allclose(
            original_train_data.all_values(),
            dm.inverse_transform_data(transformed_train_data).all_values(),
        )


@pytest.mark.parametrize("dataset_name", ["example_ettm1"])
@pytest.mark.parametrize("pipeline_config", [_SCALER_PIPELINE_CONFIG, None])
def test_inverse_transform_data_func(get_darts_example_dm, pipeline_config):
    """Integration tests of helper functions for creating inverse transformation function from
    datamodule and inverting different data structures."""
    dm = get_darts_example_dm
    dm.hparams.processing_pipeline = hydra.utils.instantiate(pipeline_config, _convert_="partial")
    dm.setup("fit")

    assert src.utils.get_inverse_transform_data_func(None, dm, "train") is None

    # TODO: test invalid inverse_transform function that raises exception

    inverse_transform_data_func = src.utils.get_inverse_transform_data_func(
        {"partial_ok": True}, dm, "train"
    )
    assert inverse_transform_data_func is not None
    assert callable(inverse_transform_data_func)

    all_transformed = dm.get_data(
        ["target", "past_covariates", "future_covariates"], main_split="train"
    )

    target_inverse_transformed = {}
    for data_type, data in all_transformed.items():
        if data is None:
            target_inverse_transformed[data_type] = None
        else:
            target_inverse_transformed[data_type] = dm.inverse_transform_data(data)

    for structure in ["timeseries", "sequence", "dict"]:
        transformed = copy.deepcopy(all_transformed)
        if structure == "timeseries":
            transformed = transformed["target"]
        elif structure == "sequence":
            transformed = [transformed["target"], transformed["target"]]
        elif structure == "dict":
            pass
        else:
            raise ValueError

        inverse_transformed = src.utils.inverse_transform_data(
            inverse_transform_data_func, transformed
        )
        if pipeline_config is None:
            if structure == "timeseries":
                original_data = dm.data[0].slice_intersect(inverse_transformed)
                original_data = original_data.drop_columns(
                    [
                        c
                        for c in original_data.components
                        if c not in inverse_transformed.components
                    ]
                )
                assert original_data == inverse_transformed
            else:
                continue
        else:
            if structure == "timeseries":
                assert inverse_transformed == target_inverse_transformed["target"]
            elif structure == "sequence":
                assert all(
                    inv_trans_ts == target_inverse_transformed["target"]
                    for inv_trans_ts in inverse_transformed
                )
            elif structure == "dict":
                for data_type, data in target_inverse_transformed.items():
                    assert inverse_transformed[data_type] == data


def test_pipeline_per_component_util(get_darts_example_dm):
    dm = get_darts_example_dm
    pipeline = src.datamodules.utils.ensure_pipeline_per_component(
        dm.hparams.processing_pipeline, dm.hparams.data_variables
    )
    assert isinstance(pipeline, dict)
    for cov_type, cov_components in dm.hparams.data_variables.items():
        if cov_components is not None:
            for component in cov_components:
                assert component in pipeline


@pytest.mark.parametrize("dataset_name", ["example_ettm1"])
def test_unique_component_pipelines(get_darts_example_dm):
    dm = get_darts_example_dm
    pipeline = dict()
    all_components = list(src.datamodules.utils.get_all_data_components(dm.hparams.data_variables))

    pipeline[all_components[0]] = dm.hparams.processing_pipeline
    pipeline[all_components[1]] = hydra.utils.instantiate(_SCALER_PIPELINE_CONFIG)
    pipeline[all_components[2]] = None

    dm.hparams.processing_pipeline = pipeline
    dm.setup("fit")

    dm.data[0] = dm.data[0].with_values(
        np.repeat(dm.data[0].univariate_component(0).all_values(), dm.data[0].n_components, axis=1)
    )
    transformed_data = dm.transform_data(dm.data[0])

    # assert that component with explicitly set None pipeline is same as component with no provided pipeline (default None)

    assert np.array_equal(
        transformed_data.univariate_component(all_components[2]).all_values(),
        transformed_data.univariate_component(all_components[3]).all_values(),
    )

    # assert that component with None pipeline is not transformed
    assert transformed_data.univariate_component(all_components[2]) == dm.data[
        0
    ].univariate_component(all_components[2])

    # assert that component with Scaler is different from non transformed component
    assert not np.array_equal(
        transformed_data.univariate_component(all_components[1]).all_values(),
        dm.data[0].univariate_component(all_components[2]).all_values(),
    )

    inverse_transformed = dm.transform_data(transformed_data)

    assert transformed_data.univariate_component(
        all_components[2]
    ) == inverse_transformed.univariate_component(all_components[2])


@pytest.mark.parametrize(
    "crop_data_range", [["1980-01-01", "2000-01-01"], ["1980-01-01", "2050-01-01"]]
)
def test_crop_data_range(get_darts_example_dm, crop_data_range):
    """Test the crop time range feature of datamodule."""
    dm = get_darts_example_dm
    dm.hparams.crop_data_range = crop_data_range
    dm.setup("fit")

    assert dm.data[0].start_time() >= pd.Timestamp(crop_data_range[0]) and dm.data[
        0
    ].end_time() <= pd.Timestamp(crop_data_range[1])


@pytest.mark.parametrize("precision", [64, 32, 16, "invalid"])
def test_precision(get_darts_example_dm, precision):
    """Test changing dataset precision."""
    dm = get_darts_example_dm
    dm.hparams.precision = precision

    if precision == "invalid":
        with pytest.raises(AttributeError):
            dm.setup("fit")
    else:
        dm.setup("fit")

        new_precision = 64 if precision != 64 else 32

        assert dm.data[0].dtype == np.dtype(getattr(np, f"float{precision}"))
        assert dm.set_dataset_precision(dm.data[0], new_precision).dtype == np.dtype(
            getattr(np, f"float{new_precision}")
        )


def test_setup_load_dir(tmp_path, get_darts_example_dm):
    """Test that state of datamodule can be saved and loaded."""
    dm = copy.deepcopy(get_darts_example_dm)
    dm.hparams.processing_pipeline = hydra.utils.instantiate(_SCALER_PIPELINE_CONFIG)
    dm.setup("fit")
    dm.save_state(tmp_path)

    dm_load = copy.deepcopy(get_darts_example_dm)
    dm_load.hparams.train_val_test_split = dm.hparams.train_val_test_split
    del dm_load.hparams.train_val_test_split[0]["train"]

    dm_noload = copy.deepcopy(dm_load)

    dm_load.setup("fit", load_dir=tmp_path)
    assert dm_load.get_data(["target"], "val") == dm.get_data(["target"], "val")

    dm_noload.hparams.processing_pipeline = hydra.utils.instantiate(_SCALER_PIPELINE_CONFIG)
    with pytest.raises(RuntimeError):  # no train-data and not fitted pipeline should raise error
        dm_noload.setup("fit")


@pytest.mark.parametrize(
    "split",
    [
        {"train": [[0.05, 0.15], [0.2, 0.3]], "val": [[0.4, 0.5]]},
        {"train": [[0.0, 0.16], [0.16, 0.2]], "test": [0.5, 1.0]},
        {"train": 0.5, "val": 0.25, "test": 0.25},
        {"test": 0.25, "val": 0.25, "train": 0.5},
        {"test": 0.25, "train": 0.5},
        {"train": [0.05, 0.55], "test": [0.65, 0.85]},
        {"train": 100, "test": 50},
        {"val": [5, 50], "train": [50, 150], "test": [155, 158]},
    ],
)
def test_split_float_int(get_darts_example_dm, split):
    """Test various edge cases and combinations of dataset splits for float and int."""
    dm = get_darts_example_dm
    dm.hparams.train_val_test_split = copy.deepcopy(split)
    dm.hparams.processing_pipeline = None
    dm.setup("fit")
    split_data = {
        split_name: dm.get_data(["target"], main_split=split_name) for split_name in split
    }
    split_order = list(split.keys())

    split_sizes = {}
    for split_name, split_values in split.items():
        if not isinstance(split_values, list):
            split_sizes[split_name] = split_values
        elif isinstance(split_values[0], list):
            split_sizes[split_name] = sum(s[1] - s[0] for s in split_values)
        else:
            split_sizes[split_name] = split_values[1] - split_values[0]

    split_sum = sum(split_sizes.values())

    if isinstance(split_sum, float) and split_sum == 1.0:
        assert sum(len(s) for sd in split_data.values() for s in sd["target"]) == len(dm.data[0])

    for split_name in ["train", "val", "test"]:
        if split_name not in split:
            assert getattr(dm, f"data_{split_name}") is None

    # assert correct size
    for split_name, split_size in split_sizes.items():
        # rounding errors in float conversion and integer to timestamp index and non-overlap logic can cause up to 2 difference
        if isinstance(split_size, float):
            target_size = split_size * len(dm.data[0])
        else:
            target_size = split_size
        split_actual_size = sum(len(d) for d in split_data[split_name]["target"])
        assert abs(split_actual_size - target_size) <= 2

    # assert not overlapping
    if len(split) > 1:
        split_combinations = itertools.combinations(split, 2)
        for splits in split_combinations:
            s1, s2 = split_data[splits[0]]["target"], split_data[splits[1]]["target"]
            s1_inds = s2_inds = [None]
            if not isinstance(s1, darts.TimeSeries):
                s1_inds = range(len(s1))
            if not isinstance(s2, darts.TimeSeries):
                s2_inds = range(len(s2))
            ds_combs = [[x, y] for x in s1_inds for y in s2_inds]
            for i1, i2 in ds_combs:
                if i1 is None:
                    assert isinstance(s1, darts.TimeSeries), "Faulty testing logic"
                    ds1 = s1
                else:
                    ds1 = s1[i1]
                if i2 is None:
                    assert isinstance(s2, darts.TimeSeries), "Faulty testing logic"
                    ds2 = s2
                else:
                    ds2 = s2[i2]
                assert (
                    ds1.start_time() not in ds2.time_index and ds1.end_time() not in ds2.time_index
                )

    # assert correct order
    if len(split) > 1:
        for split_i in range(len(split_order) - 1):
            assert (
                split_data[split_order[split_i]]["target"][-1].start_time()
                < split_data[split_order[split_i + 1]]["target"][0].start_time()
            )


# TODO: test multiple splits per split
@pytest.mark.parametrize(
    "split",
    [
        {"train": "1990-01-01", "val": "2001-01-01", "test": "2007-01-01"},
        {"train": ["start", "1990-01-01"], "val": ["1990-01-01", "end"]},
    ],
)
def test_split_timestamp(get_darts_example_dm, split):
    """Test splits defined as timestamps, including special values 'start' and 'end'."""
    dm = get_darts_example_dm
    dm.hparams.train_val_test_split = split
    dm.hparams.processing_pipeline = None
    dm.setup("fit")

    split_data = {
        split_name: dm.get_data(["target"], main_split=split_name) for split_name in split
    }

    # assert correct start / stop
    for split_name, data in split_data.items():
        if isinstance(split[split_name], list):
            if split[split_name][0] == "start":
                split_start = dm.data[0].start_time()
            else:
                split_start = pd.Timestamp(split[split_name][0])
            if split[split_name][1] == "end":
                split_end = dm.data[0].end_time()
            else:
                split_end = pd.Timestamp(split[split_name][1])
            assert dm.data[0].get_index_at_point(data["target"].start_time()) == dm.data[
                0
            ].get_index_at_point(split_start)
            assert dm.data[0].get_index_at_point(data["target"].end_time()) == dm.data[
                0
            ].get_index_at_point(split_end)
        else:
            # overlap logic can shift one index
            assert (
                abs(
                    dm.data[0].get_index_at_point(data["target"].end_time())
                    - dm.data[0].get_index_at_point(pd.Timestamp(split[split_name]))
                )
                <= 1
            )

    # assert not overlapping
    if len(split_data) > 1 and not isinstance(
        split["train"], list
    ):  # TODO: make nonoverlapping for list?
        split_combinations = itertools.combinations(split, 2)
        for comb in split_combinations:
            s1, s2 = split_data[comb[0]]["target"], split_data[comb[1]]["target"]
            assert s1.start_time() not in s2.time_index and s1.end_time() not in s2.time_index


@pytest.mark.parametrize("dataset_name", ["example_ettm1"])
@pytest.mark.parametrize("train_val_test_split", ["original", "no-val"])
def test_get_data(get_darts_example_dm, train_val_test_split):
    """Test the helper function to get data from datamodule."""

    def get_split_data(dm, split_name, covariate=None):
        res = dm.data[0].slice(*dm.hparams.train_val_test_split[0][split_name])
        if covariate is not None:
            res = res.drop_columns(
                [
                    col
                    for col in dm.data[0].columns
                    if col not in dm.hparams.data_variables[covariate]
                ]
            )
        return res

    dm = get_darts_example_dm
    if train_val_test_split == "no-val":
        dm.hparams.train_val_test_split["val"] = None
    dm.setup("fit")

    data_translator = {
        "target": "target",
        "series": "target",
        "past_covariates": "past_covariates",
        "future_covariates": "future_covariates",  # TODO: static covariates
        "actual_anomalies": "actual_anomalies",
    }

    # TODO: test anomalies?

    requested_data = ["series", "target", "past_covariates", "future_covariates"]

    for split in ["train", "val", "test"]:
        if split == "val" and train_val_test_split == "no-val":
            continue
        res_data = dm.get_data(requested_data, main_split=split)
        assert len(res_data) == len(requested_data)
        assert all(req_d in res_data for req_d in requested_data)
        assert all(
            get_split_data(dm, split, data_translator[req_d]) == res_data[req_d]
            for req_d in requested_data
        )

    val_requested_data = ["val_series", "val_past_covariates"]
    requested_data.extend(val_requested_data)

    res_data = dm.get_data(requested_data)
    if train_val_test_split == "original":
        assert len(res_data) == len(requested_data)
        assert all(req_d in res_data for req_d in requested_data)
    else:
        assert len(res_data) == len(requested_data) - len(val_requested_data)
        assert all(
            req_d in res_data for req_d in requested_data if req_d not in val_requested_data
        )

    for req_d in requested_data:
        if req_d.startswith("val_"):
            if train_val_test_split == "no-val":
                res_data[req_d] = None
            else:
                base_name = "_".join(req_d.split("_")[1:])
                assert get_split_data(dm, "val", data_translator[base_name]) == res_data[req_d]
        else:
            assert get_split_data(dm, "train", data_translator[req_d]) == res_data[req_d]


@pytest.mark.parametrize("resample_method", ["interpolate", "mean", "sum", "non_existent"])
@pytest.mark.parametrize("freq", ["1Y", "invalid", "2Y"])
def test_resample(get_darts_example_dm, resample_method, freq):
    """Test the resample feature of the datamodule."""
    dm = copy.deepcopy(get_darts_example_dm)
    dm.hparams.resample = dict(freq=freq, method=resample_method)
    if resample_method == "non_existent" or freq == "invalid":
        with pytest.raises(ValueError):
            dm.setup("fit")
        return
    else:
        dm.setup("fit")

        dm_original = get_darts_example_dm
        dm_original.setup("fit")

    if freq == "1Y":
        assert len(dm.data[0]) == 53
    elif freq == "2Y":
        assert len(dm.data[0]) == 27
    else:
        raise ValueError

    assert dm.data[0].freq == pd.tseries.frequencies.to_offset(freq)
    index_diffs = dm.data[0].time_index.to_series().diff()[1:]
    accepted_diffs = [index_diffs[0]]
    if "Y" in freq:
        accepted_diffs.append(index_diffs[0] + pd.Timedelta("1D"))  # leap years...
    assert all([index_diffs[i] in accepted_diffs for i in range(1, len(index_diffs))])

    if resample_method == "interpolate":
        for i, ts in enumerate(dm.data[0].time_index):
            original_index_pre = (
                dm_original.data[0].time_index.get_indexer([ts], method="ffill").item()
            )
            original_index_after = (
                dm_original.data[0].time_index.get_indexer([ts], method="bfill").item()
            )
            if original_index_pre == -1:
                raise ValueError("How is this possible?")
            elif original_index_after == -1:
                continue  # Index is after end of original data. Should we remove this datapoint perhaps?
            if original_index_pre != original_index_after:
                original_values = [
                    dm_original.data[0][original_index_pre].values(),
                    dm_original.data[0][original_index_after].values(),
                ]
                index_time_delta = (
                    dm_original.data[0].time_index[original_index_after]
                    - dm_original.data[0].time_index[original_index_pre]
                )
                interpolation_point = (
                    1
                    - (dm_original.data[0].time_index[original_index_after] - ts)
                    / index_time_delta
                )
                assert np.isclose(
                    dm.data[0][ts].values(),
                    original_values[0] * (1 - interpolation_point)
                    + original_values[1] * interpolation_point,
                )
            elif dm_original.data[0].time_index[original_index_after] == ts:
                assert dm_original.data[0][ts] == dm.data[0][ts]
            else:
                print("how")
    else:
        # TODO: this one depends on the type of aggregation...
        # assert np.isclose(dm.data[0].values().mean(), dm_original.data.values().mean(), rtol=1e-3)
        # TODO: maybe test more robustly, i.e. as for interpolate. Loop through etc.
        assert np.all(
            np.isclose(
                dm.data[0].values(),
                getattr(
                    dm_original.data[0].pd_dataframe().resample(freq), resample_method
                )().values,
            )
        )


def test_non_unique_data_variables(get_darts_example_dm):
    """Test that datamodule raises error when configuring duplicate data variables in different
    data types."""
    dm = get_darts_example_dm
    dm.hparams.data_variables = {"target": "test", "past_covariates": "test"}

    with pytest.raises(ValueError):
        dm.setup("fit")


@pytest.mark.skip("data-source feature is not in use")
@pytest.mark.parametrize("dataset_name", ["example_basic_dataset"])
def test_data_source(get_darts_example_dm):
    """Test that data_source logic for darts example datasets datamodule is working as expected."""
    dm = get_darts_example_dm
    assert "data_source" in dm.hparams
    assert dm.__class__ is src.datamodules.TimeSeriesDataModule
    if not os.path.exists(
        os.path.join(dm.hparams.data_dir, dm.hparams.data_source["relative_file_path"])
    ):
        if not dm.hparams.data_source["relative_file_path"] == "air_passengers.csv":
            return
        darts_dataset = darts.datasets.AirPassengersDataset()
        darts_dataset._root_path = dm.hparams.data_dir
        darts_dataset._download_dataset()
    dm.setup("fit")
    assert isinstance(dm.get_data(["target", "past_covariates"], main_split="train"), dict)
    assert (
        isinstance(dm.get_data(["target"], main_split="train"), darts.timeseries.TimeSeries)
        and len(dm.get_data(["target"], main_split="train")) > 0
    )


@pytest.mark.parametrize("dataset_name", ["example_ettm1_multiple-series"])
# @pytest.mark.parametrize("slice_range", [None, [3, 10]])
@pytest.mark.parametrize("separate_splits", [True, False])
def test_plot_data(get_darts_example_dm, separate_splits):
    """Test that the plot_data function returns the expected number of figures."""
    dm = get_darts_example_dm
    dm.setup("fit")

    if False:  # TODO: figure out correct number of plots
        if dm.hparams.train_val_test_split is not None:
            n_splits = len(dm.hparams.train_val_test_split[0])
            n_series = len(dm.hparams.train_val_test_split[0]["train"]) * len(dm.data)
        else:
            n_splits = 1
            n_series = len(dm.data)
        expected_n_figures = sum(
            (
                len(dvs)
                if len(dvs) <= src.datamodules.components.timeseries_datamodule._PLOT_SEPARATE_MAX
                else 1
            )
            * n_splits
            * n_series
            for dvs in dm.hparams.data_variables.values()
            if dvs is not None
        )
    figs = dm.plot_data(separate_splits=separate_splits)
    # assert len(figs) == expected_n_figures
    for split in ["train", "val", "test"]:
        split_figs = dm.plot_data(split=split)
        if dm.hparams.train_val_test_split is None and split != "train":
            assert split_figs is None
        else:
            pass
            # assert len(split_figs) == expected_n_figures


@pytest.mark.parametrize("dataset_name", ["electricity"])
def test_illegal_chunk_idx(get_darts_example_dm):
    """Test that errors are raised when an invalid chunk_idx is given."""
    dm = get_darts_example_dm

    dm.hparams.chunk_idx = -1
    with pytest.raises(ValueError):
        dm.setup("fit")
    dm.hparams.chunk_idx = dm.num_chunks
    with pytest.raises(ValueError):
        dm.setup("fit")

    dm.hparams.chunk_idx = dm.num_chunks - 1
    dm.setup("fit")


# TODO: use multiple sets as parameter in the other tests as well?
@pytest.mark.parametrize(
    "train_val_test_split",
    [{"train": 0.5, "val": 0.5}, {"first": {"train": 0.5, "val": 0.5}, "second": None}],
)
def test_multiple_datasets(get_darts_example_dm, train_val_test_split):
    """Test using multiple datasets with datamodule."""
    dm = get_darts_example_dm
    dm.data = dm.dataset.load()
    dm.data = dict(first=dm.data, second=copy.deepcopy(dm.data))
    dm.hparams.train_val_test_split = train_val_test_split
    dm._finalize_setup()
    train_data = dm.get_split_data("train")
    assert len(train_data["target"]) == 2
    if "second" in train_val_test_split:
        assert len(train_data["target"][1]) == len(dm.data["second"])
        assert len(train_data["target"][1]) > len(train_data["target"][0])


@pytest.mark.parametrize("data_variable", ["target", "past_covariates", "future_covariates"])
def test_empty_data_variable_list(get_darts_example_dm, data_variable):
    dm = get_darts_example_dm
    dm.hparams.data_variables[data_variable] = []

    if data_variable == "target":
        with pytest.raises(AssertionError):
            dm.setup("fit")
    else:
        # should not error
        dm.setup("fit")
