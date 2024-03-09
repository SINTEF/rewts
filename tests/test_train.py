import os

import pyrootutils

import src.utils

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

import src.models.utils
from src.train import train
from tests.helpers.run_if import RunIf

# TODO: test that we can run _DEFAULT_MODELS using different covariates that they support

# TODO: test that datasets are plotted


def test_train_fast_dev_run(cfg_train_torch):
    """Run for 1 train, val and test step."""
    cfg_train = cfg_train_torch
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
    train(cfg_train)


@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_train_torch):
    """Run for 1 train, val and test step on GPU."""
    cfg_train = cfg_train_torch
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
    train(cfg_train)


@pytest.mark.skip(reason="Not relevant?")
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu_amp(cfg_train):
    """Train 1 epoch on GPU with mixed-precision."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.precision = 16
    train(cfg_train)


@pytest.mark.skip(reason="Not interesting")
@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_train):
    """Train 1 epoch with validation loop twice per epoch."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.val_check_interval = 0.5
    train(cfg_train)


@pytest.mark.skip(reason="Not relevant for our use?")
@pytest.mark.slow
def test_train_ddp_sim(cfg_train):
    """Simulate DDP (Distributed Data Parallel) on 2 CPU processes."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 2
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 2
        cfg_train.trainer.strategy = "ddp_spawn"
    train(cfg_train)


@pytest.mark.skip(reason="Deterministic is unsupported with CUDA > 10.2")
@pytest.mark.slow
def test_train_resume(tmp_path, cfg_train_torch):
    """Run 1 epoch, finish, and resume for another epoch."""
    cfg_train = cfg_train_torch

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.deterministic = True
        cfg_train.test = False
        cfg_train.seed = 3

    HydraConfig().set_config(cfg_train)

    with open_dict(
        cfg_train
    ):  # can not resolve hydra config, therefore remove after setting config
        cfg_hydra = cfg_train.hydra
        del cfg_train.hydra

    metric_dict_1, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "last.ckpt" in files
    assert "epoch_000.ckpt" in files

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_train.trainer.max_epochs = 2
        cfg_train.trainer.min_epochs = 2
        cfg_train.trainer.deterministic = True
        cfg_train.test = False
        cfg_train.seed = 3

    metric_dict_2, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "epoch_001.ckpt" in files
    assert "epoch_002.ckpt" not in files

    # TODO: this is a bit random it seems? Not guaranteed that another epoch will actually reduce the loss
    # assert metric_dict_1["train_loss"] > metric_dict_2["train_loss"]
    # assert metric_dict_1["val_loss"] > metric_dict_2["val_loss"]


def test_lr_tuner(tmp_path, cfg_train_torch):
    """Test that lr_tuner runs, produces plot, produces a valid suggestion for lr, and that the
    suggested lr is applied to the model."""
    cfg_train = cfg_train_torch

    with open_dict(cfg_train):
        cfg_train.train = False
        cfg_train.predict = False
        cfg_train.test = False
        cfg_train.validate = False
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.deterministic = True
        cfg_train.seed = 3
        cfg_train.plot_datasets = False
        cfg_train.lr_tuner = OmegaConf.load(root / "configs" / "lr_tuner" / "default.yaml")
        cfg_train.lr_tuner.plot = True

    HydraConfig().set_config(cfg_train)

    with open_dict(
        cfg_train
    ):  # can not resolve hydra config, therefore remove after setting config
        cfg_hydra = cfg_train.hydra
        del cfg_train.hydra

    _, object_dict = train(cfg_train)

    assert "lr_tuner" in object_dict

    lr_suggestion = object_dict["lr_tuner"].suggestion(**cfg_train.lr_tuner.get("suggestion", {}))
    assert lr_suggestion is not None
    assert object_dict["model"].model_params["optimizer_kwargs"]["lr"] == lr_suggestion

    plot_files = os.listdir(tmp_path / "plots")
    assert any(["lr_find_results" in str(plot_file) for plot_file in plot_files])


def test_train(get_trained_model_torch, get_trained_model_nontorch):
    """Test that the expected objects are returned by train function and that the expected files
    are created."""
    train_funcs = [get_trained_model_torch, get_trained_model_nontorch]
    for train_func in train_funcs:
        train_metric_dict, train_objects = train_func
        expected_objects = ["cfg", "datamodule", "model", "logger", "trainer", "callbacks"]
        assert all([obj in train_objects for obj in expected_objects])
        model_dir = Path(train_objects["cfg"].paths.output_dir)

        is_torch_model = src.models.utils.is_torch_model(train_objects["cfg"])

        log_dir_files = os.listdir(model_dir)
        if is_torch_model:
            assert "_model.pth.tar" in log_dir_files
        else:
            assert "model.pkl" in log_dir_files

        assert os.path.exists(model_dir / "datamodule" / "pipeline.pkl")

        if is_torch_model:
            assert "last.ckpt" in os.listdir(model_dir / "checkpoints")
        # TODO: more tests for non-torch models
