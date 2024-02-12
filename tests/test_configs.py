import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import pytest


def test_train_torch_config(cfg_train_torch: DictConfig):
    cfg_train = cfg_train_torch
    assert cfg_train
    assert cfg_train.datamodule
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.datamodule)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_train_nontorch_config(cfg_train_nontorch: DictConfig):
    cfg_train = cfg_train_nontorch
    assert cfg_train
    assert cfg_train.datamodule
    assert cfg_train.model
    assert cfg_train.fit

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.datamodule)
    hydra.utils.instantiate(cfg_train.model)


def test_eval_config(cfg_eval: DictConfig):
    assert cfg_eval
    assert cfg_eval.model_dir
    assert cfg_eval.eval

    HydraConfig().set_config(cfg_eval)

    #hydra.utils.instantiate(cfg_eval.datamodule)
    #hydra.utils.instantiate(cfg_eval.model)


def test_predict_config(cfg_predict: DictConfig):
    assert cfg_predict
    assert cfg_predict.model_dir
    assert cfg_predict.predict
