import os
from pathlib import Path

import darts
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

import src.models.utils
import src.utils
from src.predict import predict, process_predict_index
from tests.helpers.retrain_utils import expected_retrain_status
from tests.helpers.set_config import cfg_set_paths_and_hydra_args

# TODO: test that we can run _DEFAULT_MODELS using different covariates that they support


# make fixture that trains model for 1 epoch
# def test_covariates():
#    pass
#    # test that it works with covariates
#    # test that fig has as many axes as expected?


# test different types of prediction indices (int, timestamp, float)

# test inverse transformation
# metric scores should be higher?

# test RangeIndex and DatetimeIndex

# test different n
# invalid (not provided)

# perhaps set up a conftest that trains a model we can use for all tests


# TODO: test for fit_ensemble_weights=True
def test_predictable_indices():
    """Test processing of predict indices."""
    with initialize(
        version_base="1.3", config_path="../configs"
    ):  # TODO: refactor this into conftest fixtures somehow
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=["datamodule=example_ettm1", "model=xgboost"],
        )

    with open_dict(cfg):
        del cfg.logger
        del cfg.callbacks
        del cfg.trainer
        cfg.model.lags = 4
        cfg.model.lags_past_covariates = 2
        cfg.model.lags_future_covariates = (3, 3)
        cfg.predict = {"kwargs": {"n": 5}, "split": "test"}

    HydraConfig().set_config(cfg)

    with open_dict(cfg):  # can not resolve hydra config, therefore remove after setting config
        cfg_hydra = cfg.hydra
        del cfg.hydra
    object_dict = src.utils.instantiate_objects(cfg)

    model, datamodule = object_dict["model"], object_dict["datamodule"]
    datamodule.setup("fit")

    data = datamodule.get_data(
        ["series", "future_covariates", "past_covariates"], main_split=cfg.predict.split
    )
    float_first = process_predict_index(
        0.0, cfg.predict.kwargs.n, model, data, fit_ensemble_weights=False
    )
    assert float_first == cfg.model.lags

    retrain_first = process_predict_index(0.0, cfg.predict.kwargs.n, model, data, retrain=True)
    assert retrain_first >= float_first and retrain_first == model.min_train_series_length

    float_last = process_predict_index(
        1.0, cfg.predict.kwargs.n, model, data, fit_ensemble_weights=False
    )
    assert float_last == len(data["future_covariates"]) - (
        cfg.model.lags_future_covariates[1] + cfg.predict.kwargs.n
    )

    assert (
        process_predict_index(0, cfg.predict.kwargs.n, model, data, fit_ensemble_weights=False)
        == 0
    )
    assert (
        process_predict_index(99, cfg.predict.kwargs.n, model, data, fit_ensemble_weights=False)
        == 99
    )

    timestamp = "2023-10-02 15-00-00"
    timestamp_index = process_predict_index(
        timestamp, cfg.predict.kwargs.n, model, data, fit_ensemble_weights=False
    )
    assert timestamp_index.tz is None
    assert timestamp_index == pd.Timestamp(timestamp).tz_localize(None)


def run_predict_default_tests(cfg, train_func):
    src.utils.enable_eval_resolver()

    train_metric_dict, train_objects = train_func
    model_dir = Path(train_objects["cfg"].paths.output_dir)

    with open_dict(cfg):
        cfg.model_dir = model_dir
        cfg.predict.split = "val"
        cfg.predict.kwargs.n = 2

    cfg = src.utils.verify_and_load_config(cfg)

    test_predict_indices = [[0.0, 1.0], ["2016-07-15T00"], [50]]

    for predict_indices in test_predict_indices:
        with open_dict(cfg):
            cfg.predict.indices = predict_indices

        HydraConfig().set_config(cfg)
        predict_metrics, predict_objects = predict(cfg)

        assert "figs" in predict_objects
        assert "predictions" in predict_objects
        assert len(predict_objects["figs"]) == len(cfg.predict.indices)
        assert len(predict_objects["predictions"]) == len(cfg.predict.indices)
        assert len(predict_objects["predictions"][0]) == cfg.predict.kwargs.n
        assert isinstance(predict_objects["predictions"][0], darts.TimeSeries)
        assert predict_objects["figs"][0][0] is None or isinstance(
            predict_objects["figs"][0][0], plt.Figure
        )

        assert expected_retrain_status(
            cfg, "predict.retrain", predict_objects["model"], train_objects["model"]
        )


def test_predict_torch(cfg_predict, get_trained_model_torch):
    """Test that prediction with 'predict.py' works as expected."""
    run_predict_default_tests(cfg_predict, get_trained_model_torch)


def test_predict_nontorch(cfg_predict, get_trained_model_nontorch):
    """Test that prediction with 'predict.py' works as expected."""
    run_predict_default_tests(cfg_predict, get_trained_model_nontorch)


def test_predict_local_model(tmp_path):
    """Test predict.py with missing model_dir and LocalForecastingModel instead."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="predict.yaml",
            return_hydra_config=True,
            overrides=["model=baseline_naive_seasonal", "datamodule=example_ettm1"],
        )
    cfg = cfg_set_paths_and_hydra_args(cfg.copy(), tmp_path)
    metric_dict, object_dict = predict(cfg)

    assert metric_dict is not None and len(metric_dict) > 0
    assert len(object_dict["predictions"]) > 0
    assert OmegaConf.is_missing(object_dict["cfg"], "model_dir")
    assert object_dict["model"].training_series is not None
    GlobalHydra.instance().clear()


def test_predict_retrain(cfg_predict, get_trained_model_nontorch, get_trained_model_torch):
    """Test that predict.retrain argument for 'predict.py' fits the model before predicting."""
    with open_dict(cfg_predict):
        cfg_predict.predict.retrain = True
    run_predict_default_tests(cfg_predict, get_trained_model_nontorch)
    run_predict_default_tests(cfg_predict, get_trained_model_torch)


@pytest.mark.parametrize("fit_ensemble_weights", [True, False])
def test_predict_ensemble(cfg_predict, get_trained_model_nontorch, fit_ensemble_weights):
    """Test evaluate.py with ReWTSEnsembleModel."""
    src.utils.enable_eval_resolver()

    _, train_objects = get_trained_model_nontorch
    model_dir = Path(train_objects["cfg"].paths.output_dir)

    with open_dict(cfg_predict):
        cfg_predict.model_dir = [model_dir, model_dir]
        cfg_predict.predict.ensemble_weights.fit = fit_ensemble_weights

    cfg_predict = src.utils.verify_and_load_config(cfg_predict)
    metric_dict, object_dict = predict(cfg_predict)

    assert len(object_dict.get("predictions")) > 0
    assert src.models.utils.is_rewts_model(object_dict["model"])
    if fit_ensemble_weights:
        assert len(object_dict["model"]._weights_history) > 0
        path_to_weights = os.path.join(
            cfg_predict.paths.output_dir, "ensemble_weights", "predict_0_weights.pkl"
        )
        assert os.path.exists(path_to_weights)
    else:
        assert len(object_dict["model"]._weights_history) == 0
    GlobalHydra.instance().clear()
