import copy
from typing import Dict, Set, Union

import darts.dataprocessing


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


# TODO: move more utilities here from DataModule class / src.utils.utils
