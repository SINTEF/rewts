from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
import numpy as np
import os

from src.datamodules.components.timeseries_datamodule import TimeSeriesDataModule

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


class ChunkedTimeSeriesDataModule(TimeSeriesDataModule):
    """Example data module for custom dataset.

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
                 *args,
                 chunk_length: int,
                 chunk_idx: Optional[int] = None,
                 dataset_length: Optional[int] = None,
                 # any argument listed here is configurable through the yaml config files and the command line.
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._chunk_called = False
        self.save_hyperparameters(logger=False)

    def chunk_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to select a chunk from the dataset, controlled by the arguments chunk_idx and chunk_length. This
        function will automatically be called during .setup if it has not been called already.

        :param df: Data to be chunked
        :return: Chunked data
        """
        # This could be bcause of resampling etc.
        if self.hparams.dataset_length is not None and len(df) < self.hparams.dataset_length:
            log.warning(f"The dataset_length argument used to calculate chunk indices is bigger than the actual dataset ({self.hparams.dataset_length} > {len(df)})")

        if self.hparams.chunk_idx is not None:
            if not (self.hparams.chunk_length is not None and self.hparams.chunk_idx is not None):
                log.warning("chunk_idx is specified, but chunk_length is not defined. Chunking is disabled.")
            else:
                chunk_range = [self.hparams.chunk_idx * self.hparams.chunk_length, (self.hparams.chunk_idx + 1) * self.hparams.chunk_length]
                df = self.crop_dataset_range(df, chunk_range)

        self._chunk_called = True
        return df

    @property
    def num_chunks(self):
        if self.hparams.dataset_length is None:
            return None
        return self.hparams.dataset_length // self.hparams.chunk_length

    def _finalize_setup(self, load_dir: Optional[str] = None):
        """
        This function must be called by the setup function of any subclass of TimeSeriesDataModule. It performs the
        final steps of the setup function, including fitting the processing pipeline, splitting the data into train, val
        and test sets and transforming the sets, and resampling the data if needed. It also checks that the data is
        valid and that the hparams are valid.

        :param load_dir: The folder to which the state of the datamodule is saved for later reproduction.

        :return: None
        """
        if not self._chunk_called:
            self.data = self.chunk_dataset(self.data)
        super()._finalize_setup(load_dir=load_dir)


if __name__ == "__main__":
    pass