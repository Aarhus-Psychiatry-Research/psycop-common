from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.test_suggesters import (
    suggester_tester,
)
from psycop.common.model_training_v2.trainer.task.estimator_steps.logistic_regression import (
    LogisticRegressionSuggester,
)


def test_logistic_regression_suggester():
    suggester_tester(
        suggester=LogisticRegressionSuggester(
            C_low=0.1,
            C_high=1,
            C_log=False,
            l1_ratio_low=0.1,
            l1_ratio_high=1,
            l1_ratio_log=False,
            solvers=("saga","lbfgs")
        ),
    )