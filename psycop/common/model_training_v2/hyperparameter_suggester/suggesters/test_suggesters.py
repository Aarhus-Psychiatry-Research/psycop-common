from typing import Any, Callable

import optuna
from optuna.testing.storage import StorageSupplier

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.populate_registry import (
    populate_baseline_registry,
)
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.logistic_regression_suggester import (
    FloatSpace,
    LogisticRegressionSuggester,
    Suggester,
)


def float_space_for_test() -> FloatSpace:
    return FloatSpace(low=0, high=1, logarithmic=False)


def suggester_tester(suggester: Suggester) -> dict[str, Any]:
    sampler = optuna.samplers.RandomSampler()

    with StorageSupplier("inmemory") as storage:
        study = optuna.create_study(storage=storage, sampler=sampler)
        trial = optuna.Trial(study, study._storage.create_new_trial(study._study_id))  # type: ignore
        result = suggester.suggest_hyperparameters(trial=trial)

        populate_baseline_registry()
        cfg = BaselineRegistry.resolve(result)

    return cfg


def test_logistic_regression_suggester():
    suggester_tester(
        suggester=LogisticRegressionSuggester(
            C=float_space_for_test(), l1_ratio=float_space_for_test()
        )
    )

    # XXX: Ensure this tests more than resolution. E.g. that the return value is meaningful.
