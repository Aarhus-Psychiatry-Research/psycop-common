from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import bootstrap  # type: ignore

np.random.seed(42)


def bootstrap_estimates(
    metric: Callable,
    input_1: pd.Series,
    input_2: pd.Series,
    n_bootstraps: int = 100,
    ci_width: float = 0.95,
    **kwargs: Any,
) -> pd.Series:
    _kwargs = {
        "method": "basic",
        "n_resamples": n_bootstraps,
    }
    _kwargs.update(kwargs)

    # Calculate the confidence interval
    def metric_wrapper(
        true: np.ndarray,
        pred: np.ndarray,
        **kwargs: Any,  # noqa: ARG001
    ) -> float:
        # bootstrap function requires the metric function to
        # be able to take additional arguments (notably the length of the array)
        try:
            return metric(true, pred)
        except ValueError as e:
            print(repr(e))
            return np.nan

    boot = bootstrap(
        (input_1, input_2),
        statistic=metric_wrapper,
        confidence_level=ci_width,
        paired=True,
        **_kwargs,  # type: ignore
    )

    low, high = boot.confidence_interval.low, boot.confidence_interval.high

    return pd.Series(
        {"ci": (low, high)},
    )
