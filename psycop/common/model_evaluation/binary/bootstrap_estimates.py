from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import bootstrap  # type: ignore
from sklearn.utils import resample


def bootstrap_estimates(
    metric: Callable,  # type: ignore
    input_1: pd.Series,  # type: ignore
    input_2: pd.Series,  # type: ignore
    n_bootstraps: int = 100,
    random_state: int = 42,
    ci_width: float = 0.95,
    stratified: bool = False,
    **kwargs: Any,
) -> pd.Series:  # type: ignore
    _kwargs = {"method": "basic", "n_resamples": n_bootstraps}
    _kwargs.update(kwargs)

    # Calculate the confidence interval
    def metric_wrapper(
        true: np.ndarray,  # type: ignore
        pred: np.ndarray,  # type: ignore
        **kwargs: Any,  # noqa: ARG001
    ) -> float:
        # bootstrap function requires the metric function to
        # be able to take additional arguments (notably the length of the array)
        try:
            return metric(true, pred)
        except ValueError as e:
            print(repr(e))
            return np.nan

    rng = np.random.default_rng(random_state)

    if stratified:
        boot = stratified_bootstrap(
            y_true=input_1,
            y_pred=input_2,
            metric=metric_wrapper,  # type: ignore
            ci_width=ci_width,
            random_state=rng,
            **_kwargs,
        )

        low, high = boot[0], boot[1]

    else:
        boot = bootstrap(
            data=(input_1, input_2),
            statistic=metric_wrapper,
            confidence_level=ci_width,
            paired=True,
            random_state=rng,
            **_kwargs,  # type: ignore
        )

        low, high = boot.confidence_interval.low, boot.confidence_interval.high

    return pd.Series({"ci": (low, high)})


def stratified_bootstrap(
    y_true: pd.Series,  # type: ignore
    y_pred: pd.Series,  # type: ignore
    metric: Callable[[pd.Series, pd.Series], float],  # type: ignore
    random_state: Any,
    n_resamples: Any = 200,
    ci_width: Any = 0.95,
    **metric_kwargs: Any,
) -> tuple[float, float]:
    """
    Compute a stratified bootstrap confidence interval for a given metric.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Predicted values (scores or labels, depending on metric).
        metric (Callable): Scoring function taking (y_true, y_pred, **kwargs).
        n_resamples (int): Number of bootstrap resamples.
        ci_width (float): Confidence interval width (e.g. 0.95 for 95% CI).
        random_state (int): Seed for reproducibility.
        **metric_kwargs: Additional arguments for the metric function.

    Returns:
        Tuple[float, float]: Lower and upper bounds of the confidence interval.
    """
    y_true = np.asarray(y_true)  # type: ignore
    y_pred = np.asarray(y_pred)  # type: ignore

    classes = np.unique(y_true)
    indices_by_class = {cls: np.where(y_true == cls)[0] for cls in classes}

    scores = []

    for _ in range(n_resamples):
        resampled_indices = []
        for cls in classes:
            cls_indices = indices_by_class[cls]
            resampled_cls_indices = resample(
                cls_indices,
                replace=True,
                n_samples=len(cls_indices),
                random_state=random_state.integers(0, 1_000_000),
            )
            resampled_indices.append(resampled_cls_indices)

        indices = np.concatenate(resampled_indices)
        score = metric(y_true[indices], y_pred[indices], **metric_kwargs)
        scores.append(score)

    lower = (1 - ci_width) / 2 * 100
    upper = (1 + ci_width) / 2 * 100

    return tuple(np.percentile(scores, [lower, upper]))
