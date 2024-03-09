from omegaconf import OmegaConf

import src.models.utils


def should_retrain(cfg, retrain_key, model):
    return OmegaConf.select(
        cfg, retrain_key, default=False
    ) or not src.models.utils.is_transferable_model(model)


# TODO: find better condition here
def has_retrained(model1, model2):
    return model1.training_series != model2.training_series


def expected_retrain_status(cfg, retrain_key, model1, model2):
    return should_retrain(cfg, retrain_key, model1) == has_retrained(model1, model2)
