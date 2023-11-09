from dataclasses import dataclass
from typing import Any, Sequence

import optuna
import pytest

from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.logistic_regression_suggester import (
    FloatSpace,
    LogisticRegressionSuggester,
    Suggester,
)


@dataclass(frozen=True)
class DependentSearchSpace:
    suggestor: Sequence[Suggester]


def fill_suggestions(config: dict[str, Any]) -> dict[str, Any]:
    """Traverse the tree and find all instances of SearchSpace"""
    ...

def test_dependent_search_space():
    input_dict = {
        "estimator_steps": [
            DependentSearchSpace(
                suggestor=[
                    LogisticRegressionSuggester(
                        C=FloatSpace(low=0, high=1, logarithmic=False),
                        l1_ratio=FloatSpace(low=0, high=1, logarithmic=False),
                    )
                ]
            )
        ]
    }

    with pytest.raises(optuna.TrialPruned):
        pass
