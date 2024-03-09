import os
from pathlib import Path

import hydra.utils
import pyrootutils
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.train import train
from tests.helpers.set_config import cfg_set_paths_and_hydra_args


@pytest.fixture(scope="package")
def cfg_train_torch_global() -> DictConfig:
    """Default config for torch models."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=["model=rnn", "ensemble=default"],
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root())
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 100
            cfg.trainer.limit_val_batches = 10
            cfg.trainer.limit_test_batches = 10
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.model.batch_size = 8
            cfg.lr_tuner.lr_find.num_training = 2
            # cfg.datamodule.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.eval.kwargs.start = -2 * cfg.model.input_chunk_length
            cfg.eval.kwargs.start_format = "position"
            cfg.logger = None
            cfg.plot_datasets = False

    return cfg


@pytest.fixture(scope="package")
def cfg_train_nontorch_global() -> DictConfig:
    """Default config for non-torch models."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=["model=linear_regression", "ensemble=default"],
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root())
            cfg.fit.max_samples_per_ts = 10
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None
            cfg.plot_datasets = False

    return cfg


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    """Default eval config."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="eval.yaml", return_hydra_config=True)

    return cfg


@pytest.fixture(scope="package")
def cfg_predict_global() -> DictConfig:
    """Default predict config."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="predict.yaml", return_hydra_config=True)

    return cfg


# this is called by each test which uses `cfg_train` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_train_torch(cfg_train_torch_global, tmp_path) -> DictConfig:
    """Function-scoped fixture for torch train config."""
    yield cfg_set_paths_and_hydra_args(cfg_train_torch_global.copy(), tmp_path)

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_train_nontorch(cfg_train_nontorch_global, tmp_path) -> DictConfig:
    """Function-scoped fixture for non-torch train config."""
    yield cfg_set_paths_and_hydra_args(cfg_train_nontorch_global.copy(), tmp_path)

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global, tmp_path) -> DictConfig:
    """Function-scoped fixture for eval config."""
    yield cfg_set_paths_and_hydra_args(cfg_eval_global.copy(), tmp_path)

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_predict(cfg_predict_global, tmp_path) -> DictConfig:
    """Function-scoped fixture for predict config."""
    yield cfg_set_paths_and_hydra_args(cfg_predict_global.copy(), tmp_path)

    GlobalHydra.instance().clear()


@pytest.fixture
def dataset_name():
    """Name of default darts example datamodule."""
    return "example_aus_beer"


@pytest.fixture(scope="function")
def get_darts_example_dm(dataset_name):
    """Fixture to get darts example datamodule that supports argument to control dataset."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[f"datamodule={dataset_name}"],
        )

    # set defaults for all tests
    with open_dict(cfg):
        cfg.paths.root_dir = str(pyrootutils.find_root())
        if "dataset_name" in cfg.datamodule:  # is darts example
            cfg.datamodule.load_as_dataframe = True

    return hydra.utils.instantiate(cfg.datamodule, _convert_="partial")


def train_model_by_name(log_path, model_name, datamodule_name="example_ettm1"):
    root = pyrootutils.setup_root(
        search_from=os.getcwd(),
        indicator=".project-root",
        pythonpath=True,
        dotenv=True,
    )

    assert f"{model_name}.yaml" in os.listdir(
        root / "configs" / "model"
    ), "Missing model configuration file"
    with initialize(version_base="1.3", config_path="../configs"):
        cfg_train = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[f"datamodule={datamodule_name}", f"model={model_name}", "ensemble=default"],
        )

    is_torch_model = cfg_train.get("trainer", None) is not None

    with open_dict(cfg_train):
        cfg_train.paths.output_dir = str(log_path)
        cfg_train.paths.log_dir = str(log_path)
        cfg_train.hydra.job.num = 0
        cfg_train.hydra.job.id = 0
        cfg_train.plot_datasets = False
        if is_torch_model:
            cfg_train.trainer.max_epochs = 1
        else:
            assert (
                "fit" in cfg_train
            )  # TODO: how to set number of epochs for nontorch? Is it always 1?
            cfg_train.fit.verbose = False
        cfg_train.eval.update(dict(split="test", mc_dropout=False))
        cfg_train.validate = False
        cfg_train.test = True
        cfg_train.plot_datasets = False

    HydraConfig().set_config(cfg_train)

    with open_dict(
        cfg_train
    ):  # can not resolve hydra config, therefore remove after setting config
        cfg_hydra = cfg_train.hydra
        del cfg_train.hydra

    return train(cfg_train)


# TODO: check if cleanup between each use of the fixture is necessary
@pytest.fixture(scope="module")
def get_trained_model_torch(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("logs", numbered=True)
    metric_dict, objects = train_model_by_name(tmp_path, "rnn")
    hydra.core.utils._save_config(
        objects["cfg"], "config.yaml", Path(objects["cfg"].paths.output_dir) / ".hydra"
    )
    yield metric_dict, objects


@pytest.fixture(scope="module")
def get_trained_model_nontorch(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("logs", numbered=True)
    metric_dict, objects = train_model_by_name(tmp_path, "linear_regression")
    hydra.core.utils._save_config(
        objects["cfg"], "config.yaml", Path(objects["cfg"].paths.output_dir) / ".hydra"
    )
    yield metric_dict, objects
