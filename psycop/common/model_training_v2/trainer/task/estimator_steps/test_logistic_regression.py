from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.test_suggesters import (
    suggester_tester,
)
from psycop.common.model_training_v2.trainer.task.estimator_steps.logistic_regression import (
    LogisticRegressionSuggester,
)


def test_logistic_regression_suggester():
    suggester_tester(
        suggester=LogisticRegressionSuggester(
            C={"low": 0.1, "high": 1, "logarithmic": False},
            l1_ratio={"low": 0.1, "high": 1, "logarithmic": False},
            solvers=("saga","lbfgs")
        ),
    )