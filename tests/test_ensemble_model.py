from pathlib import Path

import numpy as np
import pytest
from omegaconf import open_dict

import src.models.utils
import src.utils


def test_instantiation(cfg_eval, get_trained_model_nontorch):
    """Test that the ensemble model is instantiated when setting multiple values for model_dir."""
    _, train_objects = get_trained_model_nontorch
    model_dir = Path(train_objects["cfg"].paths.output_dir)

    with open_dict(cfg_eval):
        cfg_eval.model_dir = model_dir

    cfg_eval_global = src.utils.verify_and_load_config(cfg_eval)
    global_objects = src.utils.instantiate_saved_objects(cfg_eval_global)

    assert not src.models.utils.is_rewts_model(global_objects["model"])

    with open_dict(cfg_eval):
        cfg_eval.model_dir = [model_dir, model_dir]

    cfg_eval_ensemble = src.utils.verify_and_load_config(cfg_eval)
    ensemble_objects = src.utils.instantiate_saved_objects(cfg_eval_ensemble)

    assert src.models.utils.is_rewts_model(ensemble_objects["model"])


# TODO: test logic for reusing fit_data
