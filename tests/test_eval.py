import os
from hydra.core.hydra_config import HydraConfig
import numpy as np

from src.eval import evaluate
from src.train import train
import src.utils
import pyrootutils
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict
import hydra.core.utils


# Test evaluation of one torch model and one non-torch model
@pytest.mark.parametrize("model", ["rnn", "xgboost"])
def test_eval(tmp_path, model):#, cfg_train, cfg_eval):
    """Train for 1 epoch with `train.py` and evaluate with `eval.py`"""

    src.utils.enable_eval_resolver()

    root = pyrootutils.setup_root(
        search_from=os.getcwd(),
        indicator=".project-root",
        pythonpath=True,
        dotenv=True,
    )

    assert f"{model}.yaml" in os.listdir(root / "configs" / "model"), "Missing model configuration file"
    with initialize(version_base="1.3", config_path="../configs"):  # TODO: refactor this into conftest fixtures somehow
        cfg_train = compose(config_name="train.yaml", return_hydra_config=True, overrides=["datamodule=example_ettm1", f"model={model}", "eval=backtest"])

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
        cfg_train.eval.update(dict(split="test", mc_dropout=False))
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
        cfg_eval = compose(config_name="eval.yaml", return_hydra_config=True, overrides=[f"model_dir={tmp_path}", "eval=backtest"])

    cfg_eval = src.utils.load_saved_config(str(tmp_path), cfg_eval)  # TODO: this should be part of the eval script

    with open_dict(cfg_eval):
        if is_torch_model:
            ckpt_path = train_objects["trainer"].checkpoint_callback.best_model_path
            if ckpt_path == "":
                cfg_eval.ckpt = "last.ckpt"
            else:
                cfg_eval.ckpt = ckpt_path
        cfg_eval.extras.print_config = False
        cfg_eval.paths.output_dir = tmp_path
        cfg_eval.eval.update(dict(split="test", mc_dropout=False))

    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.model_dir

    HydraConfig().set_config(cfg_eval)
    test_metric_dict, eval_objects = evaluate(cfg_eval)

    # TODO: perhaps try manipulating the train split to ensure it still works
    assert train_objects["datamodule"].data_test["target"] == eval_objects["datamodule"].data_test["target"]

    if cfg_eval.eval.get("kwargs", {}).get("metric") is None:
        metric_name = "mse"
    else:
        metric_name = cfg_eval.eval.get("kwargs", {}).get("metric")
        if src.utils.is_sequence(metric_name):
            metric_name = metric_name[0]
        metric_name = metric_name._target_.split(".")[-1]
    assert test_metric_dict[f"test_{metric_name}"] > 0.0
    assert np.isclose(train_metric_dict[f"test_{metric_name}"], test_metric_dict[f"test_{metric_name}"])