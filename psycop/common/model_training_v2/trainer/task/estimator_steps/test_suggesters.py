from dataclasses import dataclass

import pytest

from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.test_suggesters import (
    suggester_tester,
)
from psycop.common.model_training_v2.trainer.task.estimator_steps.logistic_regression import (
    LogisticRegressionSuggester,
)

from ....hyperparameter_suggester.suggesters.base_suggester import Suggester


@dataclass(frozen=True)
class SuggesterExample:
    should: str
    suggester: Suggester


@pytest.mark.parametrize(
    ("example"),
    [
        SuggesterExample(
            should="Logistic regression with mapping resolves correctly",
            suggester=LogisticRegressionSuggester(
                C={"low": 0.1, "high": 1, "logarithmic": False},
                l1_ratio={"low": 0.1, "high": 1, "logarithmic": False},
                solvers=("saga", "lbfgs"),
            ),
        ),
        SuggesterExample(
            should="Logistic regression with list resolves correctly",
            suggester=LogisticRegressionSuggester(
                C=[0.1, 1, False],
            ),
        ),
    ],
)
def test_logistic_regression_suggester(example: SuggesterExample):
    suggester_tester(
        suggester=example.suggester,
    )
