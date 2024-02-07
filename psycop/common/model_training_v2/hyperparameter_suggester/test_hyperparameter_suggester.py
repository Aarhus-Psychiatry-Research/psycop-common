from pathlib import Path
from typing import Any

import optuna
from optuna.testing.storage import StorageSupplier

from psycop.common.model_training_v2.config.config_utils import resolve_and_fill_config
from psycop.common.model_training_v2.hyperparameter_suggester.hyperparameter_suggester import (
    SuggesterSpace,
    suggest_hyperparams_from_cfg,
)
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.base_suggester import (
    Suggester,
)
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.suggester_spaces import (
    MockSuggester,
)


def parametrised_suggester() -> Suggester:
    return MockSuggester(value_low=0.01, value_high=0.99, log=True)


class TestHyperparameterSuggester:
    def _get_suggestions(self, base_cfg: dict[str, Any], seed: int = 42) -> dict[str, Any]:
        sampler = optuna.samplers.RandomSampler(seed=seed)

        with StorageSupplier("inmemory") as storage:
            study = optuna.create_study(storage=storage, sampler=sampler)
            trial = optuna.Trial(
                study,
                study._storage.create_new_trial(study._study_id),  # type: ignore
            )

            return suggest_hyperparams_from_cfg(base_cfg=base_cfg, trial=trial)

    def test_hyperparameter_suggester(self):
        base_cfg = {"model": SuggesterSpace(parametrised_suggester(), parametrised_suggester())}

        suggestion = self._get_suggestions(base_cfg=base_cfg)

        model = suggestion["model"]
        assert isinstance(model["mock_value"], float)

    def test_nested_hyperparameter_suggestion(self):
        base_cfg = {"level_1": {"level_2": SuggesterSpace(parametrised_suggester())}}

        suggestion = self._get_suggestions(base_cfg=base_cfg)
        assert set(suggestion["level_1"]["level_2"].keys()) == {"mock_value"}

    def test_no_search(self):
        base_cfg = {"int": 1, "str": "a", "float": 1.0, "list": [1, 2, 3]}
        suggestion = self._get_suggestions(base_cfg=base_cfg)
        assert suggestion == base_cfg

    def test_confection_integration(self):
        cfg = resolve_and_fill_config(
            Path(__file__).parent / "test_hyperparam_search.cfg", fill_cfg_with_defaults=True
        )

        suggestions = []

        n_suggestions = 10
        for i in range(n_suggestions):
            suggestions.append(self._get_suggestions(base_cfg=cfg, seed=i))

        suggestion_keys = {tuple(suggestions[i]["model"].keys()) for i in range(n_suggestions)}
        assert suggestion_keys == {("value_1",), ("value_2",)}
