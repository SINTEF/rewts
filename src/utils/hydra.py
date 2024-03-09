# initialize hydra
# load config
import pyrootutils

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import copy
import glob
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

import src.utils
from src.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)
import src.models.utils


def get_absolute_project_path(project_path: Union[str, Path]):
    """Returns the absolute path to the provided path in the project. If the provided path is
    already an absolute path, it is returned as is. If it is a relative path, it is joined with the
    root directory.

    :param project_path: The provided project_path, possibly relative to project root.
    :return: The absolute path to the provided project_path.
    """
    if os.path.isabs(project_path):
        return project_path
    else:
        return root / project_path


def generate_dotpath_value_pairs(cfg: DictConfig, parent_key=None):
    """Recursively generate all dotpath-value pairs from config.

    :param cfg: Hydra config.
    :param parent_key: Used to recursively generate dotpaths.
    :return: Collection of two-tuples of dotpaths and values.
    """
    for key, value in cfg.items():
        if isinstance(value, Mapping):
            yield from generate_dotpath_value_pairs(
                value, parent_key=f"{parent_key}.{key}" if parent_key is not None else key
            )
        else:
            if parent_key is None:
                yield key, value
            else:
                yield f"{parent_key}.{key}", value


def is_ensemble_model(log_dir):
    if src.utils.is_sequence(log_dir):
        return True
    else:
        if os.path.exists(os.path.join(get_absolute_project_path(log_dir), "multirun.yaml")):
            return True
        elif len(glob.glob(str(get_absolute_project_path(log_dir)))) > 1:
            return True
        else:
            return False


def _load_saved_config(
    log_dir: str, cfg_overrides: Optional[DictConfig] = None, print_config: bool = False
):
    """Load saved config from log_dir with optional overrides from cfg_overrides. Will first try to
    load resolved config, if that fails, will load non-resolved config. Note that loading non-
    resolved config can yield non-reproducible configuration, e.g. for paths that resolve the
    current time.

    :param log_dir: Path to log directory.
    :param cfg_overrides: Config overrides.
    :param print_config: Whether to print config.
    :return: Config.
    """
    # TODO: perhaps check if cfg.model_dir is missing and if so set to log_dir?
    resolved_config_path = os.path.join(log_dir, ".hydra", "resolved_config.yaml")

    config_path = os.path.join(log_dir, ".hydra", "config.yaml")
    cfg = OmegaConf.load(config_path)
    if not os.path.exists(resolved_config_path):  # backwards compatibility
        log.warning(
            "Could not find resolved config. Loading non-resolved config instead. Note that this can yield non-reproducible configuration, e.g. for paths that resolve the current time."
        )
    else:
        resolved_cfg = OmegaConf.load(resolved_config_path)

        # replace paths referencing project-root in resolved config (which is an absolute path) so that models can be transferred between projects
        dotpath_values = list(generate_dotpath_value_pairs(resolved_cfg))
        original_root_dir = resolved_cfg.paths.root_dir
        for dotpath, value in dotpath_values:
            if not src.utils.is_sequence(value):
                value = [value]
            for v in value:
                if isinstance(v, str) and original_root_dir in v:
                    OmegaConf.update(
                        resolved_cfg, dotpath, v.replace(original_root_dir, cfg.paths.root_dir)
                    )
        cfg = resolved_cfg

    if cfg_overrides is not None:
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, cfg_overrides)

    with open_dict(cfg):
        if src.utils.is_sequence(cfg.model_dir):
            cfg.model_dir = [get_absolute_project_path(project_path=md) for md in cfg.model_dir]
        else:
            cfg.model_dir = get_absolute_project_path(project_path=cfg.model_dir)
            if cfg.get("ckpt") == "best" and src.models.utils.is_torch_model(cfg):
                cfg.ckpt = src.models.utils.get_best_checkpoint(
                    checkpoint_dir=os.path.join(cfg.model_dir, "checkpoints")
                )

    if print_config:
        rich_utils.print_config_tree(cfg)

    return cfg


def load_saved_config(
    log_dir: Union[str, Sequence[str]],
    cfg_overrides: Optional[DictConfig] = None,
    print_config: bool = False,
):  # TODO: shouldn't callbacks also be per model? To ensure checkpoints are correct and stuff
    """Load saved config from log_dir with optional overrides from cfg_overrides. Will first try to
    load resolved config, if that fails, will load non-resolved config. Note that loading non-
    resolved config can yield non-reproducible configuration, e.g. for paths that resolve the
    current time.

    :param log_dir: Path to log directory.
    :param cfg_overrides: Config overrides.
    :param print_config: Whether to print config.
    :return: Config.
    """
    if src.utils.is_sequence(log_dir) and len(log_dir) == 1:
        log_dir = log_dir[0]
    if is_ensemble_model(
        log_dir
    ):  # TODO: if is sequence and not any wildcards, pass through as is?
        model_dir = log_dir
        if not src.utils.is_sequence(log_dir):
            model_dir = [log_dir]
        model_dirs = []
        for md in model_dir:
            if os.path.exists(os.path.join(get_absolute_project_path(md), "multirun.yaml")):
                md_candidates = [
                    os.path.join(md, fname) for fname in os.listdir(get_absolute_project_path(md))
                ]
            else:
                md_candidates = glob.glob(str(get_absolute_project_path(project_path=md)))
                if len(md_candidates) == 0:
                    log.warning(f"Model directory not found: {md}")
            for md_candidate in md_candidates:
                model_name = os.path.basename(md_candidate)
                if "multiruns" in md_candidate:
                    if re.match("[0-9]+", model_name):
                        model_dirs.append(md_candidate)
                else:
                    model_dirs.append(md_candidate)
        if not len(model_dirs) > 1:
            log.error(
                "When creating an ensemble model, at least two model_dirs must be specified. See configs/eval.yaml for documentation."
            )
            # TODO: consider more granular error message here. E.g. because specified model directories cannot be found
            # on disk
            raise ValueError("Invalid ensemble configuration: model_dir")

        with open_dict(cfg_overrides):
            cfg_overrides.model_dir = model_dirs
        model_cfgs = []
        for model_dir in cfg_overrides.model_dir:
            model_cfgs.append(
                _load_saved_config(
                    get_absolute_project_path(project_path=model_dir),
                    cfg_overrides,
                    print_config=print_config,
                )
            )
        cfg = model_cfgs[0]
        with open_dict(cfg):
            cfg.ensemble_models = [m_cfg.model for m_cfg in model_cfgs]
    else:
        cfg = _load_saved_config(
            get_absolute_project_path(project_path=log_dir),
            cfg_overrides,
            print_config=print_config,
        )

    return cfg


def verify_and_load_config(cfg: DictConfig) -> DictConfig:
    """Helper function for pipelines taking an existing model, or a LocalForecastingModel which
    does not need to be trained beforehand.

    If cfg.model_dir is set, the existing config of the model is loaded and merged with the
    arguments in cfg.
    """
    model_dir = OmegaConf.select(cfg, "model_dir")
    if model_dir is None:
        if not src.models.utils.is_local_model(OmegaConf.select(cfg, "model")):
            raise ValueError(
                "Either cfg.model_dir must be set and pointing to a log folder resulting from train.py, or a LocalForecastingModel must be set at cfg.model"
            )
    else:
        cfg = load_saved_config(cfg.model_dir, cfg)

    return cfg


def initialize_hydra(
    config_path: str,
    overrides_dot: Optional[List[str]] = None,
    overrides_dict: Optional[Dict[str, Any]] = None,
    return_hydra_config=False,
    print_config: bool = False,
    job_name=None,
) -> DictConfig:
    """Initialize hydra and compose config. Optionally, override config values with overrides_dict
    and overrides_dot. Overrides_dot is a list of strings in dot notation, e.g.
    ["model.transformer.encoder.layers=6"], and is useful for overriding whole sections of the
    config, e.g. ["model=xgboost"]. Overrides_dict is a dictionary of overrides in the form
    {"model": {"transformer": {"encoder": {"layers": 6}}}}, and is useful when doing many overrides
    of nested config values.

    :param config_path: Path to config file.
    :param overrides_dot: List of overrides in dot notation.
    :param overrides_dict: Dictionary of overrides.
    :param return_hydra_config: Whether to return hydra config. If true a run directory will be
        created at the path specified by cfg.hydra.run.dir.
    :param print_config: Whether to print config.
    :param job_name: Optional job name for hydra run.
    :return: Config.
    """
    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    hydra.initialize(
        version_base="1.3", config_path=os.path.dirname(config_path), job_name=job_name
    )
    cfg = hydra.compose(
        config_name=os.path.basename(config_path),
        return_hydra_config=return_hydra_config,
        overrides=overrides_dot,
    )

    if overrides_dict is not None:
        cfg_overrides = OmegaConf.create(overrides_dict)
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, cfg_overrides)

    if return_hydra_config:
        # Generate and set run output directory
        with open_dict(cfg):
            cfg.paths.output_dir = cfg.hydra.run.dir
        os.makedirs(os.path.join(cfg.paths.output_dir, ".hydra"), exist_ok=True)

        # Save the config without Hydra config entry
        cfg_save = copy.deepcopy(cfg)
        cfg_save_hydra = cfg_save.pop("hydra", None)
        OmegaConf.save(cfg_save_hydra, os.path.join(cfg.paths.output_dir, ".hydra", "hydra.yaml"))
    else:
        cfg_save = cfg
    try:
        output_dir = cfg.get("paths", {}).get("output_dir")
        if output_dir is not None:
            OmegaConf.save(cfg_save, os.path.join(cfg.paths.output_dir, ".hydra", "config.yaml"))
    except ValueError:
        log.info("Could not resolve cfg.paths.output_dir. Config was not saved.")

    if "extras" in cfg:
        original_extras = cfg.extras
        # there is a separate print_config argument of this function, use that instead of config value
        with open_dict(cfg):
            if cfg.extras.get("print_config") is not None:
                cfg.extras.print_config = False
            if cfg.extras.get("enforce_tags", False) and not return_hydra_config:
                cfg.extras.enforce_tags = False
        src.utils.extras(cfg)
        with open_dict(cfg):
            cfg.extras = original_extras

    if cfg.get("model_dir"):
        cfg = load_saved_config(cfg.model_dir, cfg, print_config=False)

    if "hydra" in cfg:
        HydraConfig().set_config(cfg)
    if print_config:
        rich_utils.print_config_tree(cfg)

    return cfg
