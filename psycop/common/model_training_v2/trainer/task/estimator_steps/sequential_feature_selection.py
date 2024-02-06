from collections.abc import Sequence
from typing import Literal

from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import BaseCrossValidator

from ....config.baseline_registry import BaselineRegistry
from ..model_step import ModelStep


@BaselineRegistry.estimator_steps.register("sequential_feature_selection")
def sequential_feature_selection_step(
    estimator: ModelStep,
    k_features: Sequence[int],  # A sequence of 2 ints, [min_features, max_features]
    scoring: Literal["roc_auc"],
    cv: BaseCrossValidator | int,
    forward: bool,
    n_jobs: int = 1,
    verbose: Literal[0, 1, 2] = 2,
) -> ModelStep:
    return (
        "sequential_feature_selection",
        SequentialFeatureSelector(
            estimator=estimator[1],
            k_features=tuple(k_features),  # type: ignore
            scoring=scoring,
            cv=cv,  # type: ignore
            forward=forward,
            n_jobs=n_jobs,
            clone_estimator=False,
            verbose=verbose,
        ),
    )
