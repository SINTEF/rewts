import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import scipy.io.arff

from src.datamodules.components.chunked_timeseries_datamodule import (
    ChunkedTimeSeriesDataModule,
)


# https://www.openml.org/search?type=data&sort=runs&id=151&status=active
class ElectricityDatamodule(ChunkedTimeSeriesDataModule):
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

    def __init__(
        self,
        *args,  # you can add arguments unique to your dataset here that you use in .setup
        filename: str = "electricity-normalized.arff",  # e.g. argument for filename of data file.
        # any argument listed here is configurable through the yaml config files and the command line.
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None, load_dir: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!

        :param stage: The pytorch lightning stage to prepare the dataset for
        :param load_dir: The folder from which to load state of datamodule (e.g. fitted scalers
            etc.).
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # **************** INSERT CODE HERE TO LOAD DATASET INTO A COLUMN-FORMAT PANDAS DATAFRAME ****************

            # YOU MUST SET YOUR DATASET TO THE self.data ATTRIBUTE
            # If your data has a DatetimeIndex, you must localize timezone information (i.e. pd.tz_localize).
            self.data, meta = scipy.io.arff.loadarff(
                os.path.join(self.hparams.data_dir, self.hparams.filename)
            )
            self.data = pd.DataFrame(self.data)
            self.data["day"] = pd.to_numeric(self.data["day"]) / 7

            # If you want to process the chunk before passing it to finalize_setup you can explicitly call chunk_dataset
            # self.data = self.chunk_dataset(self.data)
            # and then process the data. Otherwise, it is called automatically in _finalize_setup

            # **************** INSERT CODE HERE TO LOAD DATASET INTO A COLUMN-FORMAT PANDAS DATAFRAME ****************

            # Finally, call the _finalize_data_processing function from the base class which performs operations such as
            # data splitting and scaling etc.
            self._finalize_setup(load_dir=load_dir)


if __name__ == "__main__":
    import hydra

    import src.utils

    # You can run this script to test if your datamodule sets up without errors.
    #   Note that data_variables.target needs to be defined
    # Then check notebooks/data_explorer.ipynb to inspect if data looks as expected.

    cfg = src.utils.initialize_hydra(
        os.path.join(os.pardir, os.pardir, "configs", "datamodule", "electricity.yaml"),
        overrides_dict=dict(data_dir=os.path.join("..", "..", "data"), chunk_idx=3),
        print_config=False,
    )

    dm = hydra.utils.instantiate(cfg, _convert_="partial")
    dm.setup("fit")
