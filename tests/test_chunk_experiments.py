import pytest
import pyrootutils
root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command


train_file = str(root / "src/train.py")
eval_file = str(root / "src/eval.py")
hopt_file = str(root / "src/train_hopt.py")


@RunIf(sh=True)
@pytest.mark.slow
@pytest.mark.parametrize("model", [None, "electricity_rnn", "electricity_xgboost", "electricity_elastic_net", "electricity_tcn"])
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
        run_sh_command(train_command)

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
        f"model_type=global,ensemble"
    ]
    run_sh_command(eval_single_chunk_command)

    eval_iterative_chunk_command = [
        eval_file,
        "-m",
        "experiment=chunk_eval_iterative",
        "++ensemble.fit_weights_every=100000",
        "ensemble_model_dir=" + str(tmp_path) + "/ensemble",
        "global_model_dir=" + str(tmp_path) + "/global_iterative",
        "hydra.sweep.dir=" + str(tmp_path) + "/eval_iterative",
        "++eval.kwargs.stride=1000",
        f"model_type=global,ensemble",
        "chunk_idx_end=4",
    ]
    run_sh_command(eval_iterative_chunk_command)


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
        "logger=[]"
    ]
    run_sh_command(command)
