import os

import darts
import hydra.core.utils
import matplotlib.pyplot as plt
import numpy as np
import pyrootutils
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

import src.utils
from src.eval import evaluate
from src.predict import predict
from src.train import train
from tests.helpers.retrain_utils import expected_retrain_status

_DEFAULT_MODELS = [
    "arima",
    "elastic_net_cv",
    "exponential_smoothing",
    "exponential_smoothing_sf",
    "exponential_smoothing_complex_sf",
    "tcn",
    "block_rnn",
    "random_forest",
    "linear_regression",
    "baseline_naive_drift",
    "baseline_naive_seasonal",
    "baseline_naive_mean",
    "baseline_naive_moving_average",
    "auto_arima_pm",  # very slow
    # "catboost",    # As of darts v 0.25.0 catboost is not installed by default
    "croston",
    "dlinear",
    "fft",
    # "kalman_forecaster", # TODO: figure out why it fails, maybe something to do with configuration?
    # "lightgbm",    # As of darts v 0.25.0 catboost is not installed by default
    "nbeats",
    "nhits",
    "nlinear",
    # "prophet",     # v.1.1.2 is bugged https://github.com/facebook/prophet/issues/2354 # As of darts v 0.25.0 catboost is not installed by default
    "rnn",
    "auto_arima_sf",
    # "bats",             # very slow (6 min)
    # "tbats",            # Dont test by default since they are so slow.
    "tft",
    "theta",
    "four_theta",
    "theta_auto_sf",
    "transformer",
    # "varima",  # only for multivariate target
    "xgboost",
    "tide",
]


def run_train_eval_predict(
    tmp_path, model, train_overrides=("datamodule=example_ettm1", "ensemble=default")
):
    """Run training, evaluation, and prediction for a given model."""
    src.utils.enable_eval_resolver()

    root = pyrootutils.setup_root(
        search_from=os.getcwd(),
        indicator=".project-root",
        pythonpath=True,
        dotenv=True,
    )

    assert f"{model}.yaml" in os.listdir(
        root / "configs" / "model"
    ), "Missing model configuration file"
    assert isinstance(train_overrides, (list, tuple)), "train_overrides must be list or tuple"
    overrides = [f"model={model}"]
    overrides.extend(train_overrides)
    with initialize(
        version_base="1.3", config_path="../configs"
    ):  # TODO: refactor this into conftest fixtures somehow
        cfg_train = compose(
            config_name="train.yaml", return_hydra_config=True, overrides=overrides
        )

    retrain = isinstance(
        hydra.utils.instantiate(cfg_train.model),
        darts.models.forecasting.forecasting_model.LocalForecastingModel,
    )

    is_torch_model = cfg_train.get("trainer", None) is not None

    with open_dict(cfg_train):
        cfg_train.paths.output_dir = str(tmp_path)
        cfg_train.paths.log_dir = str(tmp_path)
        cfg_train.hydra.job.num = 0
        cfg_train.hydra.job.id = 0
        cfg_train.seed = 0
        cfg_train.plot_datasets = False
        if is_torch_model:
            cfg_train.trainer.max_epochs = 1
        else:
            assert (
                "fit" in cfg_train
            )  # TODO: how to set number of epochs for nontorch? Is it always 1?
            cfg_train.fit.verbose = False
            cfg_train.fit.max_samples_per_ts = 10
        cfg_train.eval.update(dict(split="test", mc_dropout=False))
        if "kwargs" not in cfg_train.eval:
            cfg_train.eval.kwargs = {}
        cfg_train.eval.kwargs.update(dict(retrain=retrain))
        cfg_train.validate = False
        cfg_train.test = True

    HydraConfig().set_config(cfg_train)

    with open_dict(
        cfg_train
    ):  # can not resolve hydra config, therefore remove after setting config
        cfg_hydra = cfg_train.hydra
        del cfg_train.hydra

    train_metric_dict, train_objects = train(cfg_train)

    log_dir_files = os.listdir(tmp_path)
    if is_torch_model:
        assert "_model.pth.tar" in log_dir_files
    else:
        assert "model.pkl" in log_dir_files

    assert os.path.exists(tmp_path / "datamodule" / "pipeline.pkl")

    hydra.core.utils._save_config(cfg_train, "config.yaml", tmp_path / ".hydra")

    GlobalHydra.instance().clear()

    if is_torch_model:
        assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with initialize(version_base="1.3", config_path="../configs"):
        cfg_eval = compose(
            config_name="eval.yaml", return_hydra_config=True, overrides=[f"model_dir={tmp_path}"]
        )

    cfg_eval = src.utils.verify_and_load_config(cfg_eval)

    with open_dict(cfg_eval):
        if is_torch_model:
            ckpt_path = train_objects["trainer"].checkpoint_callback.best_model_path
            if ckpt_path == "":
                cfg_eval.ckpt = "last.ckpt"
            else:
                cfg_eval.ckpt = ckpt_path
        cfg_eval.seed = 0
        cfg_eval.extras.print_config = None
        cfg_eval.paths.output_dir = tmp_path
        cfg_eval.eval.update(dict(split="test", mc_dropout=False))
        if "kwargs" not in cfg_train.eval:
            cfg_eval.eval.kwargs = {}
        cfg_eval.eval.kwargs.update(dict(retrain=retrain))

    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.model_dir

    HydraConfig().set_config(cfg_eval)
    test_metric_dict, eval_objects = evaluate(cfg_eval)

    # TODO: perhaps try manipulating the train split to ensure it still works
    assert train_objects["datamodule"].get_data(["target"], main_split="test") == eval_objects[
        "datamodule"
    ].get_data(["target"], main_split="test")

    # TODO: does not work for models that have retrain=True, because they will already retrain on the same thing in train
    # script, thus matching the model retraining during eval.
    # assert expected_retrain_status(
    #    cfg_eval, "eval.kwargs.retrain", eval_objects["model"], train_objects["model"]
    # )

    try:
        metric_name = cfg_eval["eval"]["kwargs"]["metric"][0]["_target_"].split(".")[-1]
    except Exception:
        metric_name = "mse"
    assert test_metric_dict[f"test_{metric_name}"] > 0.0
    assert np.isclose(
        train_metric_dict[f"test_{metric_name}"], test_metric_dict[f"test_{metric_name}"]
    )

    with initialize(version_base="1.3", config_path="../configs"):
        cfg_predict = compose(
            config_name="predict.yaml",
            return_hydra_config=True,
            overrides=[f"model_dir={tmp_path}"],
        )

    cfg_predict = src.utils.verify_and_load_config(cfg_predict)

    with open_dict(cfg_predict):
        cfg_predict.seed = 0
        cfg_predict.predict.indices = [0.0, 1.0]
        cfg_predict.paths.output_dir = tmp_path
        cfg_predict.extras.print_config = None

    HydraConfig().set_config(cfg_predict)

    predict_metrics, predict_objects = predict(cfg_predict)

    assert "figs" in predict_objects
    assert "predictions" in predict_objects
    assert len(predict_objects["figs"]) == len(cfg_predict.predict.indices)
    assert len(predict_objects["predictions"]) == len(cfg_predict.predict.indices)
    assert len(predict_objects["predictions"][0]) == cfg_predict.predict.kwargs.n
    assert isinstance(predict_objects["predictions"][0], darts.TimeSeries)
    assert predict_objects["figs"][0][0] is None or isinstance(
        predict_objects["figs"][0][0], plt.Figure
    )

    assert expected_retrain_status(
        cfg_eval, "predict.retrain", predict_objects["model"], train_objects["model"]
    )


@pytest.mark.slow
@pytest.mark.parametrize("model", _DEFAULT_MODELS)
def test_train_eval_predict(tmp_path, model):
    """Train for 1 epoch with `train.py`, evaluate with `eval.py`, and predict with
    'predict.py'."""
    run_train_eval_predict(tmp_path, model)


def test_multiple_series_train_eval_predict(tmp_path):
    """Run training, evaluation, and prediction for a model with multiple series for each split."""
    run_train_eval_predict(
        tmp_path,
        "rnn",
        train_overrides=(
            "datamodule=example_ettm1_multiple-series",
            "ensemble=default",
        ),
    )
