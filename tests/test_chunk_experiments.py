import glob
import os

import pyrootutils
import pytest

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_bash_command, run_sh_python_command

train_file = str(root / "src" / "train.py")
eval_file = str(root / "src" / "eval.py")
hopt_file = str(root / "src" / "train_hopt.py")
iterative_experiment_file = str(root / "scripts" / "run_iterative_experiment.sh")


@RunIf(sh=True)
@pytest.mark.slow
@pytest.mark.parametrize(
    "model",
    [None, "electricity_rnn", "electricity_xgboost", "electricity_elastic_net", "electricity_tcn"],
)
def test_train_eval_experiments(tmp_path, model):
    """Test running chunk experiments with the four model architectures in the paper."""
    for experiment in ["global", "global_iterative", "ensemble"]:
        train_command = [
            train_file,
            "-m",
            f"experiment={experiment}",
            "trainer.max_epochs=2",
            "++fit.max_samples_per_ts=10",
            "hydra.sweep.dir=" + str(tmp_path) + "/" + experiment,
            "++eval.kwargs.stride=1000",
            "logger=[]",
        ]
        if model is not None:
            train_command.append("model=" + model)
        run_sh_python_command(train_command)

    eval_single_chunk_command = [
        eval_file,
        "-m",
        "experiment=chunk_eval",
        "++datamodule.chunk_idx=3",
        "++ensemble.fit_weights_every=100000",
        "ensemble_model_dir=" + str(tmp_path) + "/ensemble",
        "global_model_dir=" + str(tmp_path) + "/global_iterative",
        "hydra.sweep.dir=" + str(tmp_path) + "/eval_single",
        "++eval.kwargs.stride=1000",
        "model_type=global,ensemble",
    ]
    run_sh_python_command(eval_single_chunk_command)

    eval_iterative_chunk_command = [
        eval_file,
        "-m",
        "experiment=chunk_eval_iterative",
        "++ensemble.fit_weights_every=100000",
        "ensemble_model_dir=" + str(tmp_path) + "/ensemble",
        "global_model_dir=" + str(tmp_path) + "/global_iterative",
        "hydra.sweep.dir=" + str(tmp_path) + "/eval_iterative",
        "++eval.kwargs.stride=1000",
        "model_type=global,ensemble",
        "chunk_idx_end=3",
    ]
    run_sh_python_command(eval_iterative_chunk_command)


@RunIf(sh=True)
@pytest.mark.slow
def test_iterative_experiment_script(tmp_path):
    """Test the run_iterative_experiment.sh script running an integrated train and evaluation
    pipeline."""
    os.environ["LOGS_ROOT"] = str(tmp_path)
    script_command = [
        iterative_experiment_file,
        "++eval.kwargs.stride=5000",
        "-train",
        "datamodule=electricity",
        "model=electricity_elastic_net",
        "datamodule.chunk_length=15104",  # gives 3 chunks
        "trainer.max_epochs=2",
        "++fit.max_samples_per_ts=10",
        "logger=[]",
        "-eval",
        "++ensemble.fit_weights_every=10000",
    ]
    run_sh_bash_command(script_command)

    def filter_only_run_dirs(folders):
        return [folder for folder in folders if folder.split("/")[-1].isdigit()]

    NUM_TRAINING_CHUNKS = 3
    NUM_EVAL_CHUNKS = NUM_TRAINING_CHUNKS - 2
    model_types = ["ensemble", "global"]

    train_logs_dir = tmp_path / "train" / "multiruns"
    assert len(glob.glob(str(train_logs_dir / "*"))) == len(model_types)
    for model_type in model_types:
        assert len(glob.glob(str(train_logs_dir / f"*_{model_type}"))) == 1
        assert (
            len(filter_only_run_dirs(glob.glob(str(train_logs_dir / f"*_{model_type}" / "*"))))
            == NUM_TRAINING_CHUNKS
        )

    eval_logs_dir = tmp_path / "eval" / "multiruns"
    assert len(glob.glob(str(eval_logs_dir / "*_itexp"))) == 1
    assert (
        len(filter_only_run_dirs(glob.glob(str(eval_logs_dir / "*_itexp" / "*"))))
        == len(model_types) * NUM_EVAL_CHUNKS
    )


@RunIf(sh=True)
@pytest.mark.slow
@pytest.mark.parametrize("model_type", ["global", "ensemble"])
def test_hopt(tmp_path, model_type):
    """Test optuna hyperparameter optimization."""
    hparam_config = "electricity_elastic_net"
    if model_type == "ensemble":
        hparam_config += "_ensemble"
    command = [
        hopt_file,
        "-m",
        "hparams_search=" + hparam_config,
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_jobs=1",
        "hydra.sweeper.n_trials=2",
        "hydra.sweeper.sampler.n_startup_trials=1",
        "hydra.sweeper.storage=sqlite:////" + str(tmp_path) + "optuna.db",
        "++fit.max_samples_per_ts=10",
        "++ensemble.fit_weights_every=10000",
        "++eval.kwargs.stride=1000",
        "logger=[]",
    ]
    run_sh_python_command(command)
