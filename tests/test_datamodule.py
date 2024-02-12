import copy
from pathlib import Path

import hydra.utils
import pytest
import sklearn.preprocessing
import torch
import itertools
import pandas as pd
import numpy as np
import darts.dataprocessing
import os
import darts.datasets

import src.datamodules.components


# TODO: test illegal splits
# TODO: hvorfor ikke bare en sekvens av datamoduler?
    # * hvordan velger man da processing pipeline? Kanskje fitte en pipeline ved å ta alle datasettene?
    # * hvordan blir det med plotting av dataset?
    # * hvordan skal det rent praktisk implementeres? Wrapper rundt liste med dm's?


_SCALER_PIPELINE_CONFIG = dict(_target_="darts.dataprocessing.Pipeline",
          transformers=[
              dict(_target_="darts.dataprocessing.transformers.Scaler",
                   scaler=dict(_target_="sklearn.preprocessing.StandardScaler"))
            ]
    )


@pytest.mark.parametrize("lags", [[4], [4, 1], 4])
def test_pipeline_inverse_diff(get_darts_example_dm, lags):
    dm = get_darts_example_dm
    dm.hparams.train_val_test_split = {"train": 0.5, "val": 0.25, "test": 0.25}
    if isinstance(lags, list):
        sum_lags = sum(lags)
    else:
        sum_lags = lags

    dm.hparams.processing_pipeline = darts.dataprocessing.Pipeline([darts.dataprocessing.transformers.Diff(lags=lags),
                                                                    darts.dataprocessing.transformers.Scaler(scaler=sklearn.preprocessing.StandardScaler())])
    dm.setup("fit")

    assert np.allclose(dm.data.values(), dm.inverse_transform_data(dm.transform_data(dm.data)).values())
    for split in ["train", "val", "test"]:
        split_data_transformed = dm.get_data(["target"], main_split=split)["target"]
        original_split_data = dm.data.slice_n_points_before(split_data_transformed.end_time(), len(split_data_transformed) + sum_lags)
        split_data_inversed = dm.inverse_transform_data(split_data_transformed, partial=True)
        assert np.allclose(original_split_data.values(), split_data_inversed.values())


@pytest.mark.parametrize("dataset_name", ["example_ettm1"])
@pytest.mark.parametrize("pipeline_config", [_SCALER_PIPELINE_CONFIG])
def test_pipeline_subset(get_darts_example_dm, pipeline_config):
    dm = get_darts_example_dm
    dm.hparams.processing_pipeline = hydra.utils.instantiate(pipeline_config)
    dm.setup("fit")

    data_train = dm.get_data(["target"], main_split="train")["target"]
    assert getattr(dm.hparams.processing_pipeline, "_fit_called", False)
    original_train_data = dm.data.slice_intersect(data_train)
    pipeline = hydra.utils.instantiate(pipeline_config)
    transformed_train_data = dm.transform_data(original_train_data, fit_pipeline=True, pipeline=pipeline)
    transformed_train_data_subset = transformed_train_data.univariate_component(2)

    assert np.allclose(original_train_data.univariate_component(2).values(), dm.inverse_transform_data(transformed_train_data_subset).values())


@pytest.mark.parametrize("dataset_name", ["example_ettm1", "example_aus_beer"])
@pytest.mark.parametrize("pipeline_config", [_SCALER_PIPELINE_CONFIG, None])
def test_pipeline(get_darts_example_dm, pipeline_config):
    dm = get_darts_example_dm
    dm.hparams.processing_pipeline = hydra.utils.instantiate(pipeline_config)
    dm.setup("fit")

    data_train = dm.get_data(["target"], main_split="train")["target"]
    data_val = dm.get_data(["target"], main_split="val")["target"]
    if pipeline_config is None:
        assert dm.data.slice_intersect(data_train).univariate_component(data_train.components[0]) == data_train
    else:
        assert getattr(dm.hparams.processing_pipeline, "_fit_called", False)
        original_train_data = dm.data.slice_intersect(data_train)
        original_val_data = dm.data.slice_intersect(data_val)
        assert not original_train_data == data_train

        pipeline = hydra.utils.instantiate(pipeline_config)
        transformed_train_data = dm.transform_data(original_train_data, fit_pipeline=True, pipeline=pipeline)
        assert getattr(pipeline, "_fit_called", False)
        assert transformed_train_data.univariate_component(data_train.components[0]) == data_train
        transformed_val_data = dm.transform_data(original_val_data, pipeline=pipeline)
        assert transformed_val_data.univariate_component(data_val.components[0]) == data_val, "Pipeline was not fitted on just training data"

        assert np.allclose(dm.data.values(), dm.inverse_transform_data(dm.transform_data(dm.data)).values(), atol=1e-6)


@pytest.mark.parametrize("crop_data_range", [["1980-01-01", "2000-01-01"], ["1980-01-01", "2050-01-01"]])
def test_crop_data_range(get_darts_example_dm, crop_data_range):
    dm = get_darts_example_dm
    dm.hparams.crop_data_range = crop_data_range
    dm.setup("fit")

    assert dm.data.start_time() >= pd.Timestamp(crop_data_range[0]) and dm.data.end_time() <= pd.Timestamp(crop_data_range[1])


@pytest.mark.parametrize("precision", [64, 32, 16, "invalid"])
def test_precision(get_darts_example_dm, precision):
    dm = get_darts_example_dm
    dm.hparams.precision = precision

    if precision == "invalid":
        with pytest.raises(AttributeError):
            dm.setup("fit")
    else:
        dm.setup("fit")

        new_precision = 64 if precision != 64 else 32

        assert dm.data.dtype == np.dtype(getattr(np, f"float{precision}"))
        assert dm.set_dataset_precision(dm.data, new_precision).dtype == np.dtype(getattr(np, f"float{new_precision}"))


def test_setup_load_dir(tmp_path, get_darts_example_dm):
    dm = copy.deepcopy(get_darts_example_dm)
    dm.hparams.processing_pipeline = hydra.utils.instantiate(_SCALER_PIPELINE_CONFIG)
    dm.setup("fit")
    dm.save_state(tmp_path)

    dm_load = copy.deepcopy(get_darts_example_dm)
    dm_load.hparams.train_val_test_split = dm.hparams.train_val_test_split
    del dm_load.hparams.train_val_test_split["train"]

    dm_noload = copy.deepcopy(dm_load)

    dm_load.setup("fit", load_dir=tmp_path)
    assert dm_load.data_val["target"] == dm.data_val["target"]

    dm_noload.hparams.processing_pipeline = hydra.utils.instantiate(_SCALER_PIPELINE_CONFIG)
    with pytest.raises(AssertionError):
        dm_noload.setup("fit")


@pytest.mark.parametrize("split", [ {"train": [[0.05, 0.15], [0.2, 0.3]], "val": [[0.4, 0.5]]},
                                    {"train": [[0.0, 0.16], [0.16, 0.2]], "test": [0.5, 1.0]},
                                    {"train": 0.5, "val": 0.25, "test": 0.25},
                                   {"test": 0.25, "val": 0.25, "train": 0.5},
                                   {"test": 0.25, "train": 0.5},
                                   {"train": [0.05, 0.55], "test": [0.65, 0.85]},
                                   {"train": 100, "test": 50},
                                   {"val": [5, 50], "train": [50, 150], "test": [155, 158]}
                                    ])
def test_split_float_int(get_darts_example_dm, split):
    dm = get_darts_example_dm
    dm.hparams.train_val_test_split = split
    dm.hparams.processing_pipeline = None
    dm.setup("fit")
    split_data = {split_name: dm.get_data(["target"], main_split=split_name) for split_name in split}
    split_order = list(split.keys())

    split_sizes = {}
    for split_name, split_values in split.items():
        if not isinstance(split_values, list):
            split_sizes[split_name] = split_values
        elif isinstance(split_values[0], list):
            split_sizes[split_name] = sum([s[1] - s[0] for s in split_values])
        else:
            split_sizes[split_name] = split_values[1] - split_values[0]

    split_sum = sum(split_sizes.values())

    if isinstance(split_sum, float) and split_sum == 1.0:
        assert sum(len(s) for sd in split_data.values() for s in sd["target"]) == len(dm.data)

    for split_name in ["train", "val", "test"]:
        if split_name not in split:
            assert getattr(dm, f"data_{split_name}") is None

    # assert correct size
    for split_name, split_size in split_sizes.items():
        # rounding errors in float conversion and integer to timestamp index and non-overlap logic can cause up to 2 difference
        if isinstance(split_size, float):
            target_size = split_size * len(dm.data)
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
                assert ds1.start_time() not in ds2.time_index and ds1.end_time() not in ds2.time_index

    # assert correct order
    if len(split) > 1:
        for split_i in range(len(split_order) - 1):
            assert split_data[split_order[split_i]]["target"][-1].start_time() < split_data[split_order[split_i + 1]]["target"][0].start_time()


# TODO: test multiple splits per split
@pytest.mark.parametrize("split", [{"train": "1990-01-01", "val": "2001-01-01", "test": "2007-01-01"},
                                   {"train": ["start", "1990-01-01"], "val": ["1990-01-01", "end"]}
                                   ])
def test_split_timestamp(get_darts_example_dm, split):
    dm = get_darts_example_dm
    dm.hparams.train_val_test_split = split
    dm.hparams.processing_pipeline = None
    dm.setup("fit")

    split_data = {split_name: dm.get_data(["target"], main_split=split_name) for split_name in split}

    # assert correct start / stop
    for split_name, data in split_data.items():
        if isinstance(split[split_name], list):
            if split[split_name][0] == "start":
                split_start = dm.data.start_time()
            else:
                split_start = pd.Timestamp(split[split_name][0])
            if split[split_name][1] == "end":
                split_end = dm.data.end_time()
            else:
                split_end = pd.Timestamp(split[split_name][1])
            assert dm.data.get_index_at_point(data["target"].start_time()) == dm.data.get_index_at_point(split_start)
            assert dm.data.get_index_at_point(data["target"].end_time()) == dm.data.get_index_at_point(split_end)
        else:
            # overlap logic can shift one index
            assert abs(dm.data.get_index_at_point(data["target"].end_time()) - dm.data.get_index_at_point(pd.Timestamp(split[split_name]))) <= 1

    # assert not overlapping
    if len(split_data) > 1 and not isinstance(split["train"], list):  # TODO: make nonoverlapping for list?
        split_combinations = itertools.combinations(split, 2)
        for comb in split_combinations:
            s1, s2 = split_data[comb[0]]["target"], split_data[comb[1]]["target"]
            assert s1.start_time() not in s2.time_index and s1.end_time() not in s2.time_index


@pytest.mark.parametrize("dataset_name", ["example_ettm1"])
def test_get_data(get_darts_example_dm):
    dm = get_darts_example_dm
    dm.setup("fit")

    data_translator = {
        "target": "target",
        "series": "target",
        "past_covariates": "past_covariates",
        "future_covariates": "future_covariates",  # TODO: static covariates
        "actual_anomalies": "actual_anomalies"
    }

    # TODO: test anomalies?

    requested_data = ["series", "target", "past_covariates", "future_covariates"]

    for split in ["train", "val", "test"]:
        res_data = dm.get_data(requested_data, main_split=split)
        assert len(res_data) == len(requested_data)
        assert all(req_d in res_data for req_d in requested_data)
        assert all(getattr(dm, f"data_{split}")[data_translator[req_d]][0] == res_data[req_d] for req_d in requested_data)

    requested_data.extend(["val_series", "val_past_covariates"])

    res_data = dm.get_data(requested_data)
    assert len(res_data) == len(requested_data)
    assert all(req_d in res_data for req_d in requested_data)

    for req_d in requested_data:
        if req_d.startswith("val_"):
            base_name = "_".join(req_d.split("_")[1:])
            assert dm.data_val[data_translator[base_name]][0] == res_data[req_d]
        else:
            assert dm.data_train[data_translator[req_d]][0] == res_data[req_d]


@pytest.mark.parametrize("resample_method", ["interpolate", "mean", "sum", "non_existent"])
@pytest.mark.parametrize("freq", ["1Y", "invalid", "2Y"])
def test_resample(get_darts_example_dm, resample_method, freq):
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
        assert len(dm.data) == 53
    elif freq == "2Y":
        assert len(dm.data) == 27
    else:
        raise ValueError

    assert dm.data.freq == pd.tseries.frequencies.to_offset(freq)
    index_diffs = dm.data.time_index.to_series().diff()[1:]
    accepted_diffs = [index_diffs[0]]
    if "Y" in freq:
        accepted_diffs.append(index_diffs[0] + pd.Timedelta("1D"))  # leap years...
    assert all([index_diffs[i] in accepted_diffs for i in range(1, len(index_diffs))])

    if resample_method == "interpolate":
        for i, ts in enumerate(dm.data.time_index):
            original_index_pre = dm_original.data.time_index.get_indexer([ts], method="ffill").item()
            original_index_after = dm_original.data.time_index.get_indexer([ts], method="bfill").item()
            if original_index_pre == -1:
                raise ValueError("How is this possible?")
            elif original_index_after == -1:
                continue  # Index is after end of original data. Should we remove this datapoint perhaps?
            if original_index_pre != original_index_after:
                original_values = [dm_original.data[original_index_pre].values(), dm_original.data[original_index_after].values()]
                index_time_delta = dm_original.data.time_index[original_index_after] - dm_original.data.time_index[original_index_pre]
                interpolation_point = 1 - (dm_original.data.time_index[original_index_after] - ts) / index_time_delta
                assert np.isclose(dm.data[ts].values(), original_values[0] * (1 - interpolation_point) + original_values[1] * interpolation_point)
            elif dm_original.data.time_index[original_index_after] == ts:
                assert dm_original.data[ts] == dm.data[ts]
            else:
                print("how")
    else:
        # TODO: this one depends on the type of aggregation...
        #assert np.isclose(dm.data.values().mean(), dm_original.data.values().mean(), rtol=1e-3)
        # TODO: maybe test more robustly, i.e. as for interpolate. Loop through etc.
        assert np.all(np.isclose(dm.data.values(), getattr(dm_original.data.pd_dataframe().resample(freq), resample_method)().values))


def test_non_unique_data_variables(get_darts_example_dm):
    dm = get_darts_example_dm
    dm.hparams.data_variables = {"target": "test", "past_covariates": "test"}

    with pytest.raises(ValueError):
        dm.setup("fit")


@pytest.mark.parametrize("dataset_name", ["example_basic_dataset"])
def test_data_source(get_darts_example_dm):
    dm = get_darts_example_dm
    assert "data_source" in dm.hparams
    assert dm.__class__ is src.datamodules.TimeSeriesDataModule
    if not os.path.exists(os.path.join(dm.hparams.data_dir, dm.hparams.data_source["relative_file_path"])):
        if not dm.hparams.data_source["relative_file_path"] == "air_passengers.csv":
            return
        darts_dataset = darts.datasets.AirPassengersDataset()
        darts_dataset._root_path = dm.hparams.data_dir
        darts_dataset._download_dataset()
    dm.setup("fit")
    assert isinstance(dm.data_train, dict)
    assert isinstance(dm.data_train["target"][0], darts.timeseries.TimeSeries) and len(dm.data_train) > 0


@pytest.mark.parametrize("dataset_name", ["example_basic_dataset", "example_ettm1_multiple-series"])
def test_plot_data(get_darts_example_dm):
    dm = get_darts_example_dm
    dm.setup("fit")

    dm.plot_data()
    for split in ["train", "val", "test"]:
        dm.plot_data(split=split)




