from dataclasses import dataclass
from typing import Any

import optuna
from optuna.testing.storage import StorageSupplier

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.populate_registry import (
    populate_baseline_registry,
)
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.base_suggester import (
    Suggester,
)
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.logistic_regression_suggester import (
    FloatSpace,
)


def float_space_for_test() -> FloatSpace:
    return FloatSpace(low=0, high=1, logarithmic=False)


@dataclass(frozen=True)
class TestSuggestion:
    pre_resolution: dict[str, Any]
    resolved: dict[str, Any]


def suggester_tester(suggester: Suggester) -> TestSuggestion:
    """Test utility function which ensures that the suggester:
    1. Interfaces correctly with Optuna
    2. Can be resolved from the BaselineRegistry
    """
    sampler = optuna.samplers.RandomSampler(seed=42)

    with StorageSupplier("inmemory") as storage:
        study = optuna.create_study(storage=storage, sampler=sampler)
        trial = optuna.Trial(study, study._storage.create_new_trial(study._study_id))  # type: ignore
        result = suggester.suggest_hyperparameters(trial=trial)

        populate_baseline_registry()
        cfg = BaselineRegistry.resolve(result)

    return TestSuggestion(pre_resolution=result, resolved=cfg)


