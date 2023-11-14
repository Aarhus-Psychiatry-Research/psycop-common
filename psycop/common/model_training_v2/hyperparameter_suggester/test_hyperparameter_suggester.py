from typing import Any

import optuna
from optuna.testing.storage import StorageSupplier

from psycop.common.model_training_v2.hyperparameter_suggester.hyperparameter_suggester import (
    SearchSpace,
    suggest_hyperparams_from_cfg,
)
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.logistic_regression_suggester import (
    LogisticRegressionSuggester,
)
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.test_suggesters import (
    float_space_for_test,
)


class TestHyperparameterSuggester:
    def _get_suggestions(self, base_cfg: dict[str, Any]) -> dict[str, Any]:
        sampler = optuna.samplers.RandomSampler()

        with StorageSupplier("inmemory") as storage:
            study = optuna.create_study(storage=storage, sampler=sampler)
            trial = optuna.Trial(
                study,
                study._storage.create_new_trial(study._study_id),  # type: ignore
            )

            return suggest_hyperparams_from_cfg(base_cfg=base_cfg, trial=trial)

    def test_hyperparameter_suggester(self):
        base_cfg = {
            "model": SearchSpace(
                suggesters=(
                    LogisticRegressionSuggester(
                        C=(0,1,False),
                        l1_ratio=(0,1,False),
                    ),
                ),
            ),
        }

        suggestion = self._get_suggestions(base_cfg=base_cfg)

        model = suggestion["model"]
        hyperparam_keys = {"C", "l1_ratio"}

        assert set(model["logistic_regression"].keys()) == {"@estimator_steps"}.union(
            hyperparam_keys,
        )
        assert model["logistic_regression"]["@estimator_steps"] == "logistic_regression"

    def test_nested_hyperparameter_suggestion(self):
        base_cfg = {
            "level_1": {
                "level_2": SearchSpace(
                    suggesters=(
                        LogisticRegressionSuggester(
                            C=(0,1,False),
                            l1_ratio=(0,1,False),
                        ),
                    ),
                ),
            },
        }

        suggestion = self._get_suggestions(base_cfg=base_cfg)
        model = suggestion["level_1"]["level_2"]

        hyperparam_keys = {"C", "l1_ratio"}
        assert set(model["logistic_regression"].keys()) == {"@estimator_steps"}.union(
            hyperparam_keys,
        )
        assert model["logistic_regression"]["@estimator_steps"] == "logistic_regression"

    # XXX: Add a test using confection

    def test_no_search(self):
        base_cfg = {"int": 1, "str": "a", "float": 1.0, "list": [1, 2, 3]}
        suggestion = self._get_suggestions(base_cfg=base_cfg)
        assert suggestion == base_cfg
