from psycop.common.model_training_v2.hyperparameter_suggester.hyperparameter_suggester import (
    SearchSpace,
)
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.logistic_regression_suggester import (
    FloatSpace,
    LogisticRegressionSuggester,
)
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.test_suggesters import (
    float_space_for_test,
)


def test_hyperparameter_suggester():
    base_cfg = {
        "model": {
            SearchSpace(
                suggesters={"logistic_regression":
                    LogisticRegressionSuggester(
                        C=float_space_for_test(), l1_ratio=float_space_for_test()
                    )
                }
            )
        }
    }

    suggestion = hyperparameter_suggester(base_cfg=base_cfg)
