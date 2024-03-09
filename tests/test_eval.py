import os
import pickle
from pathlib import Path

import darts
import numpy as np
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict

import src.models.utils
from src.eval import evaluate
from src.models.ensemble_model import ReWTSEnsembleModel
from tests.helpers.retrain_utils import expected_retrain_status
from tests.helpers.set_config import cfg_set_paths_and_hydra_args


def run_eval_default_tests(cfg, train_func):
    """Test evaluation of model with `eval.py`"""
    src.utils.enable_eval_resolver()

    train_metric_dict, train_objects = train_func
    model_dir = Path(train_objects["cfg"].paths.output_dir)
    is_torch_model = src.models.utils.is_torch_model(train_objects["cfg"])

    with open_dict(cfg):
        cfg.model_dir = model_dir
        if is_torch_model:
            ckpt_path = train_objects["trainer"].checkpoint_callback.best_model_path
            if ckpt_path == "":
                cfg.ckpt = "last.ckpt"
            else:
                cfg.ckpt = ckpt_path
        cfg.eval.update(dict(split="test", mc_dropout=False))

    cfg = src.utils.verify_and_load_config(cfg)

    test_metric_dict, eval_objects = evaluate(cfg)

    # TODO: perhaps try manipulating the train split to ensure it still works
    assert (
        train_objects["datamodule"].data_test["target"]
        == eval_objects["datamodule"].data_test["target"]
    )

    metric_name = "mse"
    assert test_metric_dict[f"test_{metric_name}"] > 0.0
    assert np.isclose(
        train_metric_dict[f"test_{metric_name}"], test_metric_dict[f"test_{metric_name}"]
    )

    assert expected_retrain_status(
        cfg, "eval.kwargs.retrain", eval_objects["model"], train_objects["model"]
    )


def test_eval_torch(cfg_eval, get_trained_model_torch):
    """Train torch model for 1 epoch with `train.py` and evaluate with `eval.py`"""
    run_eval_default_tests(cfg_eval, get_trained_model_torch)


def test_eval_nontorch(cfg_eval, get_trained_model_nontorch):
    """Train non-torch model for 1 epoch with `train.py` and evaluate with `eval.py`"""
    run_eval_default_tests(cfg_eval, get_trained_model_nontorch)


def test_eval_predictions(cfg_eval, get_trained_model_nontorch):
    """Test that predictions are saved and returned when configured to do so."""
    train_metric_dict, train_objects = get_trained_model_nontorch
    model_dir = Path(train_objects["cfg"].paths.output_dir)
    with open_dict(cfg_eval):
        cfg_eval.model_dir = model_dir
        cfg_eval.eval.predictions = {"return": {"data": True}, "save": {"data": True}}

    cfg_eval = src.utils.verify_and_load_config(cfg_eval)

    test_metric_dict, eval_objects = evaluate(cfg_eval)

    assert isinstance(eval_objects.get("predictions"), darts.TimeSeries)
    predictions_data = eval_objects.get("predictions_data")
    assert isinstance(predictions_data, dict) and isinstance(
        predictions_data["series"], darts.TimeSeries
    )

    assert os.path.exists(
        os.path.join(cfg_eval.paths.output_dir, "predictions", "predictions.pkl")
    )
    assert os.path.exists(os.path.join(cfg_eval.paths.output_dir, "predictions", "data.pkl"))

    with open(
        os.path.join(cfg_eval.paths.output_dir, "predictions", "data.pkl"), "rb"
    ) as pkl_file:
        loaded_data = pickle.load(pkl_file)

    with open(
        os.path.join(cfg_eval.paths.output_dir, "predictions", "predictions.pkl"), "rb"
    ) as pkl_file:
        loaded_predictions = pickle.load(pkl_file)

    assert predictions_data == loaded_data
    assert eval_objects.get("predictions") == loaded_predictions


def test_eval_plot(cfg_eval, get_trained_model_nontorch):
    """Test plotting of predictions with eval.py."""
    train_metric_dict, train_objects = get_trained_model_nontorch
    model_dir = Path(train_objects["cfg"].paths.output_dir)
    with open_dict(cfg_eval):
        cfg_eval.model_dir = model_dir
        cfg_eval.eval.plot = True

    cfg_eval = src.utils.verify_and_load_config(cfg_eval)

    test_metric_dict, eval_objects = evaluate(cfg_eval)
    assert len(eval_objects.get("figs", [])) > 0


@pytest.mark.parametrize("fh_stride", [(1, 1), (3, 1), (1, 5), (2, 5)])
def test_eval_stride(cfg_eval, get_trained_model_nontorch, fh_stride):
    train_metric_dict, train_objects = get_trained_model_nontorch
    model_dir = Path(train_objects["cfg"].paths.output_dir)

    forecast_horizon, stride = fh_stride
    with open_dict(cfg_eval):
        cfg_eval.model_dir = model_dir
        cfg_eval.eval.plot = True
        cfg_eval.eval.kwargs.forecast_horizon = forecast_horizon
        cfg_eval.eval.kwargs.stride = stride

    cfg_eval = src.utils.verify_and_load_config(cfg_eval)

    test_metric_dict, eval_objects = evaluate(cfg_eval)


def test_eval_local_model(tmp_path):
    """Test evaluate.py with missing model_dir and LocalForecastingModel instead."""
    src.utils.enable_eval_resolver()

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=["model=baseline_naive_seasonal", "datamodule=example_ettm1"],
        )
    cfg = cfg_set_paths_and_hydra_args(cfg.copy(), tmp_path)
    metric_dict, object_dict = evaluate(cfg)

    assert metric_dict is not None and len(metric_dict) > 0
    assert OmegaConf.is_missing(object_dict["cfg"], "model_dir")
    GlobalHydra.instance().clear()


def test_eval_ensemble(cfg_eval, get_trained_model_nontorch):
    """Test evaluate.py with ReWTSEnsembleModel."""
    src.utils.enable_eval_resolver()

    _, train_objects = get_trained_model_nontorch
    model_dir = Path(train_objects["cfg"].paths.output_dir)

    with open_dict(cfg_eval):
        cfg_eval.model_dir = [model_dir, model_dir]
        cfg_eval.eval.ensemble_weights.save = True

    cfg_eval = src.utils.verify_and_load_config(cfg_eval)
    metric_dict, object_dict = evaluate(cfg_eval)

    assert metric_dict is not None and len(metric_dict) > 0
    assert src.models.utils.is_rewts_model(object_dict["model"])
    path_to_weights = os.path.join(
        cfg_eval.paths.output_dir, "ensemble_weights", f"eval_{cfg_eval.eval.split}_weights.pkl"
    )
    assert os.path.exists(path_to_weights)

    def compare_weights(structure1, structure2):
        if len(structure1) != len(structure2):
            return False

        for (array1, index1), (array2, index2) in zip(structure1, structure2):
            if not np.array_equal(array1, array2) or index1 != index2:
                return False

        return True

    loaded_weights = np.load(path_to_weights, allow_pickle=True)
    assert compare_weights(loaded_weights, object_dict["model"]._weights_history)
    GlobalHydra.instance().clear()
