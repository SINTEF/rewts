from typing import Any, Dict, Optional, Tuple, List, Union

import darts.timeseries
import pandas as pd
import numpy as np
import os

from src.datamodules.components.timeseries_datamodule import TimeSeriesDataModule


class SineDataModule(TimeSeriesDataModule):
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
                 *args,  # you can add arguments unique to your dataset here that you use in .setup
                 data_args: Union[List[Dict[str, Any]], Dict[str, Any]],
                 chunk_idx: Optional[int] = None,
                 chunk_length: Optional[int] = None,
                 split_per_segment: bool = False,
                 dataset_name: str = "sine",
                 # any argument listed here is configurable through the yaml config files and the command line.
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None, load_dir: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!

        :param stage: The pytorch lightning stage to prepare the dataset for
        :param load_dir: The folder from which to load state of datamodule (e.g. fitted scalers etc.).
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # **************** INSERT CODE HERE TO LOAD DATASET INTO A COLUMN-FORMAT PANDAS DATAFRAME ****************

            if not isinstance(self.hparams.data_args, list):
                assert self.hparams.chunk_idx is None or self.hparams.chunk_idx == 0
                self.hparams.data_args = [self.hparams.data_args]
            elif self.hparams.chunk_idx is not None:
                if isinstance(self.hparams.chunk_idx, int):
                    self.hparams.data_args = [self.hparams.data_args[self.hparams.chunk_idx]]
                elif isinstance(self.hparams.chunk_idx, list):
                    self.hparams.data_args = [self.hparams.data_args[idx] for idx in self.hparams.chunk_idx]
                else:
                    raise ValueError

            data_segments = []
            for data_args in self.hparams.data_args:
                segment_t = getattr(np, data_args["t"]["np_func"])(**data_args["t"]["kwargs"])
                segment = data_args["amplitude"] * np.sin(data_args["frequency"] * segment_t + data_args["phase"])
                data_segments.append(segment)

            self.data = np.concatenate(data_segments)
            self.data = pd.DataFrame({"sine": self.data})

            if len(data_segments) > 1 and self.hparams.split_per_segment:
                assert all((isinstance(v, float) or v is None) or isinstance(v, (list, {}.values().__class__)) and all((isinstance(vs, float) or vs is None) for vs in v) for v in self.hparams.train_val_test_split.values()), "Only float splits are supported with split_per_segment"
                splits = self.process_train_val_test_split(darts.timeseries.TimeSeries.from_dataframe(self.data),
                                                           self.hparams.train_val_test_split)
                new_splits = {k: [] for k in splits}
                base_index = 0
                for segment in data_segments:
                    for split_name, split_values in splits.items():
                        new_splits[split_name].append([round(segment.shape[0] * split_values[0]) + base_index,
                                                       round(segment.shape[0] * split_values[1]) + base_index])
                    base_index += segment.shape[0]
                self.hparams.train_val_test_split = new_splits

            # **************** INSERT CODE HERE TO LOAD DATASET INTO A COLUMN-FORMAT PANDAS DATAFRAME ****************

            # Finally, call the _finalize_data_processing function from the base class which performs operations such as
            # data splitting and scaling etc.
            self._finalize_setup(load_dir=load_dir)


if __name__ == "__main__":
    import src.utils
    import hydra
    import os

    # You can run this script to test if your datamodule sets up without errors.
    #   Note that data_variables.target needs to be defined
    # Then check notebooks/data_explorer.ipynb to inspect if data looks as expected.

    cfg = src.utils.initialize_hydra(os.path.join(os.pardir, os.pardir, "configs", "datamodule", "sine.yaml"),
                                     overrides_dict=dict(data_dir=os.path.join("..", "..", "data")),  # path to data
                                     print_config=False)

    dm = hydra.utils.instantiate(cfg, _convert_="partial")
    dm.setup("fit")