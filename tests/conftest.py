import hydra.utils
import pyrootutils
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict


@pytest.fixture(scope="package")
def cfg_train_torch_global() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=["model=rnn"])

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
            cfg.datamodule.num_loader_workers = 0
            #cfg.datamodule.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.eval.kwargs.start = - 2 * cfg.model.input_chunk_length
            cfg.eval.kwargs.start_format = "position"
            cfg.logger = None

    return cfg


@pytest.fixture(scope="package")
def cfg_train_nontorch_global() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root())
            cfg.fit.max_samples_per_ts = 10
            #cfg.datamodule.num_workers = 0
            #cfg.datamodule.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="eval.yaml", return_hydra_config=True, overrides=["model_dir=."])

    return cfg


@pytest.fixture(scope="package")
def cfg_predict_global() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="predict.yaml", return_hydra_config=True, overrides=["model_dir=."])

    return cfg


# this is called by each test which uses `cfg_train` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_train_torch(cfg_train_torch_global, tmp_path) -> DictConfig:
    cfg = cfg_train_torch_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_train_nontorch(cfg_train_nontorch_global, tmp_path) -> DictConfig:
    cfg = cfg_train_nontorch_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


# this is called by each test which uses `cfg_eval` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global, tmp_path) -> DictConfig:
    cfg = cfg_eval_global.copy()

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_predict(cfg_predict_global, tmp_path) -> DictConfig:
    cfg = cfg_predict_global.copy()

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_train_eval(tmp_path, model) -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg_train = compose(config_name="train.yaml", return_hydra_config=True, overrides=[f"datamodule=example_aus_beer model={model}"])

    with open_dict(cfg_train):
        cfg_train.paths.output_dir = str(tmp_path)
        cfg_train.paths.log_dir = str(tmp_path)


@pytest.fixture
def dataset_name():
    return "example_aus_beer"


@pytest.fixture(scope="function")
def get_darts_example_dm(dataset_name):
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[f"datamodule={dataset_name}"])

    # set defaults for all tests
    with open_dict(cfg):
        cfg.paths.root_dir = str(pyrootutils.find_root())
        if "dataset_name" in cfg.datamodule:  # is darts example
            cfg.datamodule.load_as_dataframe = True

    return hydra.utils.instantiate(cfg.datamodule, _convert_="partial")