import numpy as np
import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

import src.metrics.soft_dtw
from src.train import train
from tests.helpers.run_if import RunIf


def run_soft_dtw_tests(cfg_train_torch, accelerator="cpu"):
    """Test that using SoftDTW as loss function for torch models works."""
    cfg_train = cfg_train_torch

    with open_dict(cfg_train):
        cfg_train.train = True
        cfg_train.predict = False
        cfg_train.test = False
        cfg_train.validate = False
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.deterministic = True
        cfg_train.seed = 3
        cfg_train.plot_datasets = False
        cfg_train.lr_tuner = None
        cfg_train.trainer.accelerator = accelerator
        cfg_train.model.loss_fn = dict(
            _target_="src.metrics.soft_dtw.SoftDTWMetric", normalize=True, accelerator=accelerator
        )

    HydraConfig().set_config(cfg_train)

    with open_dict(
        cfg_train
    ):  # can not resolve hydra config, therefore remove after setting config
        cfg_hydra = cfg_train.hydra
        del cfg_train.hydra

    _, object_dict = train(cfg_train)

    assert isinstance(object_dict["model"].model.criterion, src.metrics.soft_dtw.SoftDTWMetric)

    GlobalHydra.instance().clear()


@RunIf(max_cuda="10.2")
def test_soft_dtw_cpu(cfg_train_torch):
    """Test that CPU implementation of SoftDTW works."""
    run_soft_dtw_tests(cfg_train_torch.copy(), accelerator="cpu")
    run_soft_dtw_tests(cfg_train_torch, accelerator="auto")


@RunIf(min_gpus=1, max_cuda="10.2")
def test_soft_dtw_gpu(cfg_train_torch):
    """Test that CPU implementation of SoftDTW works."""
    run_soft_dtw_tests(cfg_train_torch, accelerator="gpu")


@pytest.mark.skip("For development purposes")
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("bandwidth", [None, 0.5, 1.0])
@pytest.mark.parametrize("gamma", [0.01, 1.0, 10.0])
@pytest.mark.parametrize("use_cuda", [True, False])
def test_soft_dtw_metric_correctness(normalize, bandwidth, gamma, use_cuda):
    input_data = torch.randn((2, 30, 1), device="cuda" if use_cuda else "cpu")
    target_data = torch.randn((2, 30, 1), device="cuda" if use_cuda else "cpu")

    dtw_original = src.metrics.soft_dtw.SoftDTW(
        use_cuda=use_cuda, normalize=normalize, bandwidth=bandwidth, gamma=gamma
    )
    dtw_metric = src.metrics.soft_dtw.SoftDTWMetric(
        accelerator="gpu" if use_cuda else "cpu",
        normalize=normalize,
        bandwidth=bandwidth,
        gamma=gamma,
    )

    assert torch.equal(
        dtw_original(input_data, target_data).mean(), dtw_metric(input_data, target_data)
    )
