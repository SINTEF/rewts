import functools
from typing import Callable, List, Optional, Tuple, Union


def get_metric_name(split: str, metric_func: Callable) -> str:
    """Returns the name of the metric function to use for logging on the form.

    <split>_<metric_func.__name__>. Gets the name of the metric from the function name.

    :param split: The split to use for the metric name.
    :param metric_func: The metric function to use.
    :return: The name of the metric.
    """
    if isinstance(metric_func, functools.partial):
        metric_name = metric_func.func.__name__
    else:
        metric_name = metric_func.__name__

    return f"{split}_{metric_name}"


def process_metric_funcs(
    metric_funcs: Union[Callable, List[Callable]],
    split: str,
    inverse_transform_data_func: Optional[Callable] = None,
) -> Tuple[List[Callable], List[str]]:
    """
    Helper function to ensure metric funcs is an iterable and to optionally inverse transform data before calculating
    metrics.
    Args:
        metric_funcs: Callable or collection of callables that return metric scores.
        split: String that will prepend the metric names.
        inverse_transform_data_func: Callable that inverse transforms data

    Returns: List of Callable metric functions.
    """
    if callable(metric_funcs):
        metric_funcs = [metric_funcs]

    metric_names = [get_metric_name(split, m_func) for m_func in metric_funcs]

    if inverse_transform_data_func is not None:
        for i, metric_func in enumerate(metric_funcs):
            # TODO: room here for optimization
            # TODO: perhaps smart here to only inverse transform the intersection?
            metric_funcs[i] = lambda actuals, preds, *args, m_func=metric_func, **kwargs: m_func(
                inverse_transform_data_func(actuals),
                inverse_transform_data_func(preds),
                *args,
                **kwargs,
            )

    return metric_funcs, metric_names
