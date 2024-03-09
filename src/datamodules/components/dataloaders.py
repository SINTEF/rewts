import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
from darts.logging import get_logger

logger = get_logger(__name__)

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# TODO: pass data_variables to dataloader so it only loads necessary data
_SUPPORTED_FORMATS = ("csv", "h5")


class DataLoader:
    _DEFAULT_DIRECTORY = Path(root / "data")

    def __init__(
        self,
        relative_file_path: str,
        file_format: str,
        header_time: str,
        format_time: Optional[str] = None,
        pivot: Optional[Dict[str, str]] = None,
        root_path: Path = _DEFAULT_DIRECTORY,
    ):
        """Utility class to load simple datasets from file into a pandas Dataframe. Requires the
        path to the file (relative to the configured paths.data_dir in hydra), the format of the
        data (currently supports: csv, h5), and the name of the column in the data containing the
        time index. Additionally supports a pivot operation to transform data from a group-format
        to column-format.

        :param relative_file_path: Path to file relative to hydra paths.data_dir (default is
            <project_root>/data/)
        :param file_format: Format of data, e.g. csv or h5.
        :param header_time: Name of column in data file that contains the time index. Will be set
            as index of dataframe.
        :param format_time: Format of time index. See pandas.read_csv format_time argument.
        :param pivot: If not None, will call the Dataframe pivot_table function with these
            arguments.
        :param root_path: Override the default root path for which the relative_file_path is
            relative to. (default is <project_root>/data/)
        """
        self.relative_file_path = relative_file_path
        assert (
            file_format in _SUPPORTED_FORMATS
        ), f"The following file formats are supported: {_SUPPORTED_FORMATS}, you have {file_format}."
        self.file_format = file_format
        self.header_time = header_time
        self.format_time = format_time
        self.pivot = pivot
        self.root_path = root_path

    def load(self) -> pd.DataFrame:
        return self._load_from_disk(self._get_path_dataset())

    def _load_from_disk(self, path_to_file: Path) -> pd.DataFrame:
        if self.file_format == "csv":
            df = pd.read_csv(path_to_file)
        elif self.file_format == "h5":
            df = pd.read_hdf(path_to_file)
        else:
            raise ValueError(
                f"The following file formats are supported: {_SUPPORTED_FORMATS}, you have {self.file_format}."
            )

        if self.pivot is not None:
            df = df.drop_duplicates().pivot_table(**self.pivot).reset_index()
        if self.header_time is not None:
            df = self._format_time_column(df)
            df = df.set_index(self.header_time)
        else:
            df.sort_index(inplace=True)
        return df

    def _format_time_column(self, df):
        df[self.header_time] = pd.to_datetime(
            df[self.header_time],
            format=self.format_time,
            errors="raise",
        )
        df[self.header_time] = df[self.header_time].dt.tz_localize(None)
        return df

    def _get_path_dataset(self) -> Path:
        return Path(os.path.join(self.root_path, self.relative_file_path))
