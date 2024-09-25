import copy
from typing import Dict, List, Set, Union

import darts.dataprocessing
import pandas as pd


def get_all_data_components(data_variables: Dict[str, str]) -> Set[str]:
    """Utility function to get a set of all components given the data_variables dictionary indexed
    by covariate types (target, past_covariates, future_covariates, etc.) and values being the
    components of the that covariate."""
    components = set()
    for cov_type, cov_components in data_variables.items():
        if cov_components is None:
            continue
        if isinstance(cov_components, str):
            cov_components = [cov_components]
        components.update(cov_components)

    return components


def ensure_pipeline_per_component(
    pipeline: Union[darts.dataprocessing.Pipeline, Dict[str, darts.dataprocessing.Pipeline]],
    data_variables: Dict[str, str],
) -> Dict[str, darts.dataprocessing.Pipeline]:
    """Ensure pipeline is a dictionary with keys data components and values darts dataprocessing
    Pipelines. If pipeline is a single Pipeline, then it will be copied for each component. If
    pipeline is a dictionary but with a subset of the components, None values will be set for the
    pipelines of the missing components.

    :param pipeline: Pipeline object or dictionary with component keys and pipeline per component
    :param data_variables: Dictionary with keys covariate type (target, past_covariates,
        future_covariates, etc.) and values the data components for that covariate.
    :return: pipeline with expected structure of a dictionary indexed by components with Pipeline
        or None values.
    """
    components = get_all_data_components(data_variables)

    if not isinstance(pipeline, dict):
        assert isinstance(pipeline, darts.dataprocessing.Pipeline), "Unexpected type for pipeline"
        pipeline = {component: copy.deepcopy(pipeline) for component in components}

    for component in components:
        if component not in pipeline:
            pipeline[component] = None
            continue

    return pipeline


def pipeline_is_fitted(
    pipeline: Union[Dict[str, darts.dataprocessing.Pipeline], darts.dataprocessing.Pipeline]
) -> bool:
    """Utility function to check if a datamodule pipeline object has been fitted."""
    if isinstance(pipeline, dict):
        return all(p is None or getattr(p, "_fit_called", False) for p in pipeline.values())
    else:
        return getattr(pipeline, "_fit_called", False)


def generate_cross_validation_folds(
    start_time: Union[pd.Timestamp, str, float, int],
    end_time: Union[pd.Timestamp, str, float, int],
    min_length: Union[pd.Timedelta, str, float, int],
    train_fraction: float = 0.75,
) -> List[Dict[str, List[List[Union[pd.Timestamp, float, int]]]]]:
    """Generates the possible set of cross validation folds following three conditions: i) Splits will interleave train
    and validation datasets that each have a minimum length of min_length (also equal to the validation length), ii)
    all datasets should be contiguous, iii) The training set should have train_fraction fraction of the total datapoints
    , leaving the rest for validation. This function will therefore generate train_fraction / (1 - train_fraction) + 1
    number of folds: 1 that starts with a validation set of min_length, and 1 for each start with a training dataset
    for multiples of min_length up to train_fraction / (1 - train_fraction).

    :param start_time: Start time of the dataset to create cross-validation folds for.
    :param end_time: End time of the dataset to create cross-validation folds for.
    :param min_length: Minimum length of each generated dataset. Equal to the length of each validation dataset, while
    the traininig datasets will be multiples of this up to train_fraction / (1 - train_fraction).
    :param train_fraction: Fraction of the total datapoints to be included in the training set.

    :returns: A list of dictionary splits where keys are train/val. This can then be set to the train_val_test_split
    variable of the datamodule."""

    def generate_fold(split_to_add: str, first_split_ratio: int):
        if split_to_add == "val":
            fold_val = [[start_time, start_time + min_length]]
            fold_train = [
                [start_time + min_length, start_time + min_length + first_split_ratio * min_length]
            ]

            current_time = fold_train[-1][-1]

            split_to_add = "val"
        else:
            fold_train = [[start_time, start_time + first_split_ratio * min_length]]
            fold_val = [
                [
                    start_time + first_split_ratio * min_length,
                    start_time + (first_split_ratio + 1) * min_length,
                ]
            ]

            current_time = fold_val[-1][-1]

            split_to_add = "train"

        while (
            current_time + min_length * (train_ratio if split_to_add == "train" else 1) <= end_time
        ):
            if split_to_add == "train":
                fold_train.append([current_time, current_time + train_ratio * min_length])
                current_time += train_ratio * min_length
                split_to_add = "val"
            else:
                fold_val.append([current_time, current_time + min_length])
                current_time += min_length
                split_to_add = "train"

        if end_time - current_time > min_length:
            if split_to_add == "train":
                fold_train.append([current_time, end_time])
            else:
                fold_val.append([current_time, end_time])
        elif end_time > current_time:
            # now reversed, we want to extend the last one that was made
            if split_to_add == "train":
                fold_val[-1][-1] = end_time
            else:
                fold_train[-1][-1] = end_time

        return {"train": fold_train, "val": fold_val}

    if isinstance(start_time, str):
        start_time = pd.to_datetime(start_time)

    if isinstance(end_time, str):
        end_time = pd.to_datetime(end_time)

    if isinstance(min_length, str):
        min_length = pd.Timedelta(min_length)

    train_ratio = int(train_fraction / (1 - train_fraction))

    folds = []

    # folds starting with training
    for train_start_ratio in range(train_ratio):
        folds.append(generate_fold("train", train_start_ratio + 1))

    # fold starting with validation
    folds.append(generate_fold("val", train_ratio))

    return folds


# TODO: move more utilities here from DataModule class / src.utils.utils
