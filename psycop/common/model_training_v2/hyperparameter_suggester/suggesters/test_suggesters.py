from dataclasses import dataclass
from typing import Any

import optuna
from optuna.testing.storage import StorageSupplier

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.base_suggester import (
    Suggester,
)
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.filter_suggester import (
    LookbehindCombinationFilterSuggester,
)
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.suggester_spaces import (
    FloatSpace,
)
from psycop.common.test_utils.str_to_df import str_to_pl_df


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
        cfg = BaselineRegistry.resolve({"test_key": result})

    return TestSuggestion(pre_resolution=result, resolved=cfg)


def test_lookbehind_combination_suggester():
    df = str_to_pl_df(
        """pred_age,pred_age_within_2_days,pred_age_within_3_days,pred_diagnosis_within_4_days
        3,4,3,2
        3,4,3,3
        4,3,4,1
        """
    ).lazy()

    lookbehind_combinations_sets = [
        {"within_2_days", "within_3_days"},
        {"within_3_days", "within_4_days"},
        {"within_2_days", "within_4_days"},
    ]
    lookbehind_combinations = ["{2, 3}", "{3, 4}", "{2, 4}"]

    suggestions = suggester_tester(
        LookbehindCombinationFilterSuggester(
            lookbehinds=lookbehind_combinations,  # type: ignore
            pred_col_prefix="pred_",
        )
    )

    assert suggestions.resolved["test_key"].lookbehinds in lookbehind_combinations_sets
