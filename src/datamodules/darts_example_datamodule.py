from typing import Any, Dict, Optional, Tuple, List

import darts.datasets

from src.datamodules.components.chunked_timeseries_datamodule import ChunkedTimeSeriesDataModule


class DartsExampleDataModule(ChunkedTimeSeriesDataModule):
    """Data module for anomaly detection in Hydro's fuel cells.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """
    def __init__(self,
                 dataset_name: str,
                 load_as_dataframe: bool = False,  # For debugging purposes to test pd.DataFrame pipeline
                 **kwargs):
        super().__init__(**kwargs)
        self.dataset = None
        self.save_hyperparameters(logger=False)
        dataset_class = getattr(darts.datasets, f"{self.hparams.dataset_name}Dataset")
        assert dataset_class is not None, f"Can not find dataset with name {self.hparams.dataset_name}Dataset in darts.datasets"
        self.dataset = dataset_class()
        self.dataset._root_path = self.hparams.data_dir

    def prepare_data(self):
        if not self.dataset._is_already_downloaded():
            self.dataset.load()

    def setup(self, stage: Optional[str] = None, load_dir: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data = self.dataset.load()
            if self.hparams.data_variables is None or self.hparams.data_variables.get("target", None) is None:
                self.hparams.data_variables = {"target": self.data.components.values.tolist()}
            if self.hparams.load_as_dataframe:
                self.data = self.data.pd_dataframe()
            self._finalize_setup(load_dir=load_dir)  # callback?
