import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "src/train.py"
overrides = ["logger=[]"]


@pytest.mark.skip(reason="Not set up for this to be meaningful currently")
@RunIf(sh=True)
@pytest.mark.slow
def test_experiments(tmp_path):
    """Test running all available experiment configs with fast_dev_run=True."""
    command = [
        startfile,
        "-m",
        "experiment=glob(*)",
        "hydra.sweep.dir=" + str(tmp_path),
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


@pytest.mark.skip(reason="Use chunk specific tests instead")
@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(tmp_path):
    """Test default hydra sweep."""
    command = [
        startfile,
        "-m",
        "hydra.sweep.dir=" + str(tmp_path),
        "model=arima",
        "model.p=2,4"
    ] + overrides

    run_sh_command(command)


@pytest.mark.skip(reason="Not relevant")
@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep_ddp_sim(tmp_path):
    """Test default hydra sweep with ddp sim."""
    command = [
        startfile,
        "-m",
        "hydra.sweep.dir=" + str(tmp_path),
        "trainer=ddp_sim",
        "trainer.max_epochs=3",
        "+trainer.limit_train_batches=0.01",
        "+trainer.limit_val_batches=0.1",
        "+trainer.limit_test_batches=0.1",
        "model.optimizer.lr=0.005,0.01,0.02",
    ] + overrides
    run_sh_command(command)


@pytest.mark.skip(reason="Use chunk specific tests instead")
@RunIf(sh=True)
@pytest.mark.slow
def test_optuna_sweep_nontorch(tmp_path):
    """Test optuna sweep."""
    command = [
        startfile,
        "-m",
        "hparams_search=example_optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=5",
        "hydra.sweeper.sampler.n_startup_trials=2",
        "++fit.max_samples_per_ts=50",
    ] + overrides
    run_sh_command(command)


@pytest.mark.skip(reason="Use chunk specific tests instead")
@RunIf(sh=True)
@pytest.mark.slow
def test_optuna_sweep_torch(tmp_path):
    """Test optuna sweep."""
    command = [
        startfile,
        "-m",
        "hparams_search=example_optuna",
        "model=tcn",
        "optimized_metric=val_loss",
        "hydra.sweeper.params={model.lr: interval(0.001, 0.01)}"
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=5",
        "hydra.sweeper.sampler.n_startup_trials=2",
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


@pytest.mark.skip(reason="Not relevant")
@RunIf(wandb=True, sh=True)
@pytest.mark.slow
def test_optuna_sweep_ddp_sim_wandb(tmp_path):
    """Test optuna sweep with wandb and ddp sim."""
    command = [
        startfile,
        "-m",
        "hparams_search=mnist_optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=5",
        "trainer=ddp_sim",
        "trainer.max_epochs=3",
        "+trainer.limit_train_batches=0.01",
        "+trainer.limit_val_batches=0.1",
        "+trainer.limit_test_batches=0.1",
        "logger=wandb",
    ]
    run_sh_command(command)
