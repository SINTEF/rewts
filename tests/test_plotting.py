import os
import platform

import matplotlib.pyplot as plt
import numpy as np
import pyrootutils
import pytest
import pytorch_lightning.loggers
from omegaconf import OmegaConf, open_dict

import src.utils
import src.utils.plotting
from tests.helpers.package_available import _MLFLOW_AVAILABLE


def test_create_figure():
    """Test creation of figures helper function."""
    fig, axs = src.utils.plotting.create_figure(1, 1)
    assert isinstance(fig, plt.Figure)
    assert isinstance(axs, np.ndarray) and len(axs) == 1

    fig, axs = src.utils.plotting.create_figure(1, 2, sharex=True)
    assert isinstance(fig, plt.Figure)
    assert isinstance(axs, np.ndarray) and len(axs) == 2

    fig, axs = src.utils.plotting.create_figure(3, 2)
    assert isinstance(fig, plt.Figure)
    assert isinstance(axs, np.ndarray) and len(axs) == 6

    figsize = (5, 4)
    fig, axs = src.utils.plotting.create_figure(1, 1, figsize=figsize)
    assert isinstance(fig, plt.Figure)
    assert isinstance(axs, np.ndarray) and len(axs) == 1
    assert all(fig.get_size_inches() == figsize)


def test_has_valid_extension():
    """Test verification of valid figure filenames and extension."""
    fig, _ = src.utils.plotting.create_figure(1, 1)
    valid_fnames = [
        "some.figure/fig.png",
        "fig.png",
        "fig.pdf",
        "fig.tiff",
        "fig.eps",
        "fig.has.other.dots.png",
    ]

    invalid_fnames = ["some.figure/fig.aklwdmawk", "fig.has.other.dots.jpgpng", "test.test"]

    assert all(
        src.utils.plotting.has_valid_extension(fig, valid_fname) for valid_fname in valid_fnames
    )

    assert not any(
        src.utils.plotting.has_valid_extension(fig, invalid_fname)
        for invalid_fname in invalid_fnames
    )


@pytest.mark.parametrize("presenter", src.utils.plotting.PRESENTERS + ["invalid"])
def test_present(tmp_path, presenter):
    """Test presenting figure with a single presenter."""
    fig = plt.figure()
    if presenter == "invalid":
        with pytest.raises(ValueError):
            src.utils.plotting.present_figure(fig, presenter=presenter)
    elif presenter is None:
        ret_fig = src.utils.plotting.present_figure(fig, presenter=presenter)
        assert fig == ret_fig, "The same figure was not returned by None presenter"
    else:
        if presenter == "show":
            src.utils.plotting.present_figure(fig, presenter=presenter)
        elif presenter == "savefig":
            fig_path = tmp_path / "test_fig.png"
            src.utils.plotting.present_figure(fig, presenter=presenter, fname=fig_path)
            assert os.path.exists(fig_path)
        elif issubclass(presenter, pytorch_lightning.loggers.Logger):
            if presenter is pytorch_lightning.loggers.MLFlowLogger and not _MLFLOW_AVAILABLE:
                pytest.skip("mlflow package not available")
            if presenter is pytorch_lightning.loggers.logger.DummyLogger:
                logger = pytorch_lightning.loggers.logger.DummyLogger()
            else:
                presenter_name = presenter.__name__[:-6].lower()
                lg_conf = OmegaConf.load(
                    os.path.join(
                        pyrootutils.find_root(), "configs", "logger", presenter_name + ".yaml"
                    )
                )
                if "extras" in lg_conf:
                    src.utils.extras(lg_conf.extras)
                    lg_conf = lg_conf.logger
                elif "logger" in lg_conf:
                    lg_conf = lg_conf.logger
                with open_dict(lg_conf):
                    if presenter_name == "mlflow":
                        lg_conf.mlflow.tracking_uri = f"file:{tmp_path}"
                        del lg_conf.mlflow.tags
                        del lg_conf.mlflow.run_name
                    elif presenter_name == "tensorboard":
                        lg_conf.tensorboard.save_dir = tmp_path
                logger = src.utils.instantiate_loggers(lg_conf)[0]
            if presenter is pytorch_lightning.loggers.MLFlowLogger:
                with pytest.raises(AttributeError):
                    src.utils.plotting.present_figure(fig, presenter=logger)
                test_title = "test title"
                fig.suptitle(test_title)
                src.utils.plotting.present_figure(fig, presenter=logger)
                assert len(logger.experiment.list_artifacts(logger.run_id)) == 1
                assert (
                    logger.experiment.list_artifacts(logger.run_id)[-1].path == f"{test_title}.png"
                )

                new_fname = os.path.join("test_dir", f"{test_title}2.png")
                src.utils.plotting.present_figure(fig, presenter=logger, fname=new_fname)
                assert len(logger.experiment.list_artifacts(logger.run_id)) == 2
                if platform.system() == "Windows":
                    assert logger.experiment.list_artifacts(logger.run_id)[
                        -1
                    ].path == os.path.basename(new_fname)
                else:
                    assert logger.experiment.list_artifacts(logger.run_id)[-1].is_dir
                    assert logger.experiment.list_artifacts(logger.run_id)[
                        -1
                    ].path == os.path.dirname(new_fname)
            elif presenter is pytorch_lightning.loggers.TensorBoardLogger:
                with pytest.raises(AssertionError):
                    src.utils.plotting.present_figure(fig, presenter=logger)

                src.utils.plotting.present_figure(fig, presenter=logger, global_step=0, tag="test")
                # TODO: how to test if image was written?
            elif presenter is pytorch_lightning.loggers.logger.DummyLogger:
                src.utils.plotting.present_figure(fig, presenter=logger)
            else:
                raise NotImplementedError(f"Test not written for presenter {presenter}")
        else:
            raise NotImplementedError(f"Test not written for presenter {presenter}")
        assert not plt.fignum_exists(fig.number), "Figure was not closed"


def test_multiple_presenters(tmp_path):
    """Test presenting figure with multiple presenters."""
    presenters = ["savefig", None, "savefig"]
    presenter_kwargs = [
        {"fname": tmp_path / "presenter1.png"},
        None,
        {"fname": tmp_path / "presenter2.png"},
    ]
    fig = plt.figure()

    with pytest.raises(AssertionError):
        src.utils.plotting.multiple_present_figure(
            fig, presenters, presenter_kwargs=presenter_kwargs + presenter_kwargs
        )

    with pytest.raises(AssertionError):
        src.utils.plotting.multiple_present_figure(fig, presenters, [presenter_kwargs[0]])

    ret_fig = src.utils.plotting.multiple_present_figure(
        fig, presenters, presenter_kwargs=presenter_kwargs
    )
    assert fig == ret_fig, "The same figure was not returned by None presenter"

    for p_i in range(len(presenter_kwargs)):
        if presenter_kwargs[p_i] is not None:
            assert os.path.exists(tmp_path / presenter_kwargs[p_i]["fname"])
