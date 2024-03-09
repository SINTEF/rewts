import pyrootutils

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from omegaconf import OmegaConf

import src.models.utils
import src.train
import src.utils


def test_is_torch_model(cfg_train_torch, cfg_train_nontorch):
    """Test utility function to check if model (object or config) is a torch model."""
    torch_models = [
        "rnn",
        "tcn",
        "block_rnn",
        "tft",
        "transformer",
        "dlinear",
        "nbeats",
        "nhits",
        "nlinear",
    ]
    nontorch_models = ["linear_regression", "arima", "xgboost"]

    for model in torch_models:
        assert src.models.utils.is_torch_model(
            OmegaConf.load(root / "configs" / "model" / f"{model}.yaml")
        )

    for model in nontorch_models:
        assert not src.models.utils.is_torch_model(
            OmegaConf.load(root / "configs" / "model" / f"{model}.yaml")
        )

    torch_objects = src.utils.instantiate_objects(cfg_train_torch)
    nontorch_objects = src.utils.instantiate_objects(cfg_train_nontorch)

    assert src.models.utils.is_torch_model(torch_objects["model"])
    assert not src.models.utils.is_torch_model(nontorch_objects["model"])


def test_is_local_model(cfg_train_torch, cfg_train_nontorch):
    """Test utility function to check if model (object or config) is a local model."""
    local_models = [
        "arima",
        "auto_arima_pm",
        "auto_arima_sf",
        "baseline_naive_drift",
        "baseline_naive_mean",
        "baseline_naive_moving_average",
        "baseline_naive_seasonal",
        "croston",
        "exponential_smoothing",
        "exponential_smoothing_complex_sf",
        "exponential_smoothing_sf",
        "fft",
        "four_theta",
        "kalman_forecaster",
        "theta",
        "theta_auto_sf",
        "varima",
    ]
    nonlocal_models = ["rnn", "xgboost"]

    for model in local_models:
        assert src.models.utils.is_local_model(
            OmegaConf.load(root / "configs" / "model" / f"{model}.yaml")
        )

    for model in nonlocal_models:
        assert not src.models.utils.is_local_model(
            OmegaConf.load(root / "configs" / "model" / f"{model}.yaml")
        )


def test_is_transferable_model(cfg_train_torch, cfg_train_nontorch):
    """Test utility function to check if model (object or config) is a transferable model."""
    transferable_models = ["rnn", "xgboost", "arima", "kalman_forecaster", "varima"]

    nontransferable_models = [
        "auto_arima_pm",
        "auto_arima_sf",
        "baseline_naive_drift",
        "baseline_naive_mean",
        "baseline_naive_moving_average",
        "baseline_naive_seasonal",
        "croston",
        "exponential_smoothing",
        "exponential_smoothing_complex_sf",
        "exponential_smoothing_sf",
        "fft",
        "four_theta",
        "theta",
        "theta_auto_sf",
    ]

    for model in transferable_models:
        assert src.models.utils.is_transferable_model(
            OmegaConf.load(root / "configs" / "model" / f"{model}.yaml")
        )

    for model in nontransferable_models:
        assert not src.models.utils.is_transferable_model(
            OmegaConf.load(root / "configs" / "model" / f"{model}.yaml")
        )


def test_is_rewts_model(cfg_eval, get_trained_model_nontorch):
    """Test utility function to check if model (object or config) is a ReWTS Ensemble model."""
    assert src.models.utils.is_rewts_model(
        OmegaConf.load(root / "configs" / "ensemble" / "default.yaml").ensemble
    )

    assert not src.models.utils.is_rewts_model(
        OmegaConf.load(root / "configs" / "model" / "arima.yaml")
    )

    assert not src.models.utils.is_rewts_model(
        OmegaConf.load(root / "configs" / "model" / "rnn.yaml")
    )

    # objects are tested in test_ensemble_model.py::test_instantiation
