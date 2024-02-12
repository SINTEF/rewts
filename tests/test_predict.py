import os
from hydra.core.hydra_config import HydraConfig
import numpy as np

from src.predict import predict, process_predict_index
from src.train import train, initialize_objects
import src.utils
import pyrootutils
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict
import hydra.core.utils
import darts
import matplotlib.pyplot as plt

# TODO: test that we can run _DEFAULT_MODELS using different covariates that they support

# make fixture that trains model for 1 epoch
def test_covariates():
    pass
    # test that it works with covariates
    # test that fig has as many axes as expected?

# test different types of prediction indices (int, timestamp, float)

# test inverse transformation
    # metric scores should be higher?

# test RangeIndex and DatetimeIndex

# test different n
    #invalid (not provided)

# perhaps set up a conftest that trains a model we can use for all tests


def test_predictable_indices():
    with initialize(version_base="1.3", config_path="../configs"):  # TODO: refactor this into conftest fixtures somehow
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=["datamodule=example_ettm1", f"model=xgboost"])

    with open_dict(cfg):  # can not resolve hydra config, therefore remove after setting config
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
    object_dict = initialize_objects(cfg)

    model, datamodule = object_dict["model"], object_dict["datamodule"]
    datamodule.setup("fit")

    data = datamodule.get_data(["series", "future_covariates", "past_covariates"], main_split=cfg.predict.split)
    float_first = process_predict_index(0.0, cfg.predict.kwargs.n, model, data, fit_ensemble_weights=False)
    assert float_first == cfg.model.lags

    float_last = process_predict_index(1.0, cfg.predict.kwargs.n, model, data, fit_ensemble_weights=False)
    assert float_last == len(data["future_covariates"]) - (cfg.model.lags_future_covariates[1] + cfg.predict.kwargs.n)


@pytest.mark.parametrize("model", ["linear_regression", "rnn"])
def test_predict(tmp_path, model):
    """Train for 1 epoch with `train.py`, evaluate with `eval.py`, and predict with 'predict.py'"""

    src.utils.enable_eval_resolver()

    root = pyrootutils.setup_root(
        search_from=os.getcwd(),
        indicator=".project-root",
        pythonpath=True,
        dotenv=True,
    )

    assert f"{model}.yaml" in os.listdir(root / "configs" / "model"), "Missing model configuration file"
    with initialize(version_base="1.3", config_path="../configs"):  # TODO: refactor this into conftest fixtures somehow
        cfg_train = compose(config_name="train.yaml", return_hydra_config=True, overrides=["datamodule=example_ettm1", f"model={model}"])

    is_torch_model = cfg_train.get("trainer", None) is not None

    with open_dict(cfg_train):
        cfg_train.paths.output_dir = str(tmp_path)
        cfg_train.paths.log_dir = str(tmp_path)
        cfg_train.hydra.job.num = 0
        cfg_train.hydra.job.id = 0
        cfg_train.plot_datasets = False
        if is_torch_model:
            cfg_train.trainer.max_epochs = 1
        else:
            assert "fit" in cfg_train  # TODO: how to set number of epochs for nontorch? Is it always 1?
            cfg_train.fit.verbose = False
        cfg_train.eval.update(dict(eval_split="test", mc_dropout=False))
        cfg_train.validate = False
        cfg_train.test = True

    HydraConfig().set_config(cfg_train)

    with open_dict(cfg_train):  # can not resolve hydra config, therefore remove after setting config
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
        cfg_predict = compose(config_name="predict.yaml", return_hydra_config=True, overrides=[f"model_dir={tmp_path}"])

    cfg_predict = src.utils.load_saved_config(str(tmp_path), cfg_predict)  # TODO: this should be part of the eval script

    with open_dict(cfg_predict):
        cfg_predict.predict.indices = [0.0, 1.0]
        cfg_predict.paths.output_dir = tmp_path
        cfg_predict.extras.print_config = False

    HydraConfig().set_config(cfg_predict)
    predict_metrics, predict_objects = predict(cfg_predict)

    assert "figs" in predict_objects
    assert "predictions" in predict_objects
    assert len(predict_objects["figs"]) == len(cfg_predict.predict.indices)
    assert len(predict_objects["predictions"]) == len(cfg_predict.predict.indices)
    assert len(predict_objects["predictions"][0]) == cfg_predict.predict.kwargs.n
    assert isinstance(predict_objects["predictions"][0], darts.TimeSeries)
    assert predict_objects["figs"][0][0] is None or isinstance(predict_objects["figs"][0][0], plt.Figure)