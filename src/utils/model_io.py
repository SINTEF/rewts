from typing import Union
import hydra
import darts.models.forecasting.torch_forecasting_model
import os
import src.utils as utils
import glob
import logging

log = utils.get_pylogger(__name__)


class SuppressMissingCheckpointWarning(logging.Filter):
    def filter(self, record):
        if "Model was loaded without weights since no PyTorch LightningModule checkpoint ('.ckpt') could be found at" in record.getMessage():
            return False
        return True


logging.getLogger('darts.models.forecasting.torch_forecasting_model').addFilter(SuppressMissingCheckpointWarning())


def get_best_checkpoint(checkpoint_dir) -> Union[str, None]:
    """
    Returns the path to the best checkpoint file in the given directory. The best checkpoint is identified by matching
    the names of the files in the directory against the pattern default pattern for best checkpoints,
    i.e. "epoch_*.ckpt". If no files match this pattern, the last checkpoint is returned. If no checkpoint files are
    found, None is returned.

    :param checkpoint_dir: Path to the directory containing the checkpoint files.

    :return: Path to the best checkpoint file if it exists, otherwise None.
    """
    if not os.path.exists(checkpoint_dir):
        log.warning(f"No checkpoint directory exists at {os.path.split(checkpoint_dir)[0]}")
        return None

    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "epoch_*.ckpt"))
    if len(ckpt_files) == 0:
        ckpt = "last.ckpt"
        assert os.path.exists(os.path.join(checkpoint_dir, ckpt)), "No checkpoints found."
        log.info("Best checkpoint was requested but no checkpoints matching default pattern were found, using last.ckpt.")
    elif len(ckpt_files) == 1:
        ckpt = os.path.basename(ckpt_files[0])
        log.info(f"Found best checkpoint file: {ckpt}")
    else:
        ckpt = os.path.basename(ckpt_files[-1])
        log.info(f"Multiple checkpoints matching best pattern were found, selected the following checkpoint: {ckpt}")

    return ckpt


def load_model(model_cfg, model_dir, ckpt=None):
    """
    Loads and returns a model from the given directory. If the model is a TorchForecastingModel, the checkpoint must be
    provided, otherwise an AssertionError is raised. If the model is not a TorchForecastingModel, the checkpoint is
    ignored.

    :param model_cfg: The model configuration object.
    :param model_dir: The directory containing the model file
    :param ckpt: The name of the checkpoint file to load weights and state from.

    :return: The loaded model.
    """
    model_class = hydra.utils.get_class(model_cfg._target_)
    if utils.is_torch_model(model_cfg):
        assert ckpt is not None, "For TorchForecastingModels the model parameters are saved in the checkpoint object. The name of the checkpoint to load must therefore be provided"
        if ckpt == "best":
            ckpt = get_best_checkpoint(checkpoint_dir=os.path.join(model_dir, "checkpoints"))
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(model_dir, "checkpoints", ckpt)

        try:
            model = model_class.load(os.path.join(model_dir, getattr(darts.models.forecasting.torch_forecasting_model, "INIT_MODEL_NAME")))
            model.model = model._load_from_checkpoint(ckpt, **model.pl_module_params)  # TODO: probably should not be necessary to provide pl_module params.
        except RuntimeError:
            log.info("Model could not be loaded, attempting to map model to and load on CPU.")
            model = model_class.load(os.path.join(model_dir, getattr(darts.models.forecasting.torch_forecasting_model, "INIT_MODEL_NAME")), map_location="cpu")
            model.model = model._load_from_checkpoint(ckpt, map_location="cpu", **model.pl_module_params)
        model.load_cpkt_path = ckpt
        model._fit_called = True  # TODO: gets set to False in the load method, can we check somehow if the model was actually fit?
    else:
        model = model_class.load(os.path.join(model_dir, "model.pkl"))
    
    return model


def ensure_torch_model_saving(model, model_work_dir) -> None:
    """
    Ensures that the given model is configured to save checkpoints in the given directory. If the model is not a
    TorchForecastingModel, a warning is logged and no operation is performed.

    :param model: The model to ensure saving for.
    :param model_work_dir: The directory to save checkpoints in.

    :return: None
    """
    if utils.is_torch_model(model):
        model.save_checkpoints = True
        model.work_dir = model_work_dir
        model.model_name = ""
    else:
        log.info("function was called with non torch model as argument, no operation was performed.")


def save_model(model, save_dir, save_name="model") -> None:
    """
    Saves the given model to the given directory. If the model is a TorchForecastingModel, the method performs no
    operation apart from logging an info message, as the model is saved through pytorch lightning callbacks.

    :param model: The model to save.
    :param save_dir: The directory to save the model in.

    :return: None
    """
    if utils.is_torch_model(model):
        log.info(f"Torch model saving is configured through pytorch lightning callbacks, no operation was done.")
        # TODO: maybe should save anyway? e.g. if something has changed in the object.
    else:
        save_path = os.path.join(save_dir, save_name + ".pkl")
        model.save(save_path)
        log.info(f"Model was saved to {save_path}")
