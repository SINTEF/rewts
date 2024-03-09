import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def test_train_torch_config(cfg_train_torch: DictConfig):
    """Test that train config for torch model has required arguments and is instantiable."""
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
    """Test that train config for non-torch model has required arguments and is instantiable."""
    cfg_train = cfg_train_nontorch
    assert cfg_train
    assert cfg_train.datamodule
    assert cfg_train.model
    assert cfg_train.fit

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.datamodule)
    hydra.utils.instantiate(cfg_train.model)


def test_eval_config(cfg_eval: DictConfig):
    """Test that eval config has required arguments."""
    assert cfg_eval
    assert cfg_eval.eval

    HydraConfig().set_config(cfg_eval)


def test_predict_config(cfg_predict: DictConfig):
    """Test that predict config required arguments."""
    assert cfg_predict
    assert cfg_predict.predict
