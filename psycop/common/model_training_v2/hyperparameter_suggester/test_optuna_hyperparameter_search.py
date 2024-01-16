# pyright: ignore[reportPrivateUsage]
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from confection import Config
from optuna import Trial

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.hyperparameter_suggester.optuna_hyperparameter_search import (
    OptunaHyperParameterOptimization,
)
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.base_suggester import (
    Suggester,
)
from psycop.common.model_training_v2.trainer.task.estimator_steps.logistic_regression import (
    LogisticRegressionSuggester,
)


@BaselineRegistry.suggesters.register("mock_log_reg_suggester")
@dataclass
class MockLogisticRegression(Suggester):
    c_high: float = 1

    def suggest_hyperparameters(self, trial: Trial) -> dict[str, Any]:
        return {
            "logistic_regression": {
                "@estimator_steps": "logistic_regression",
                "C": trial.suggest_float("C", low=0.1, high=self.c_high, log=False),
                "l1_ratio": 0.5,
                "solver": "saga",
                "penalty": "elasticnet",
            },
        }


def test_validate_configspace_no_suggesters():
    cfg = {"param1": 1, "param2": {"nested_param": 2}}

    assert (
        OptunaHyperParameterOptimization()._check_if_suggester_in_configspace(
            cfg,
        )
        is None
    )


def test_validate_configspace_with_suggester():
    cfg = {
        "param1": 1,
        "param2": {
            "nested_suggester": LogisticRegressionSuggester(
                C={"low": 0.1, "high": 1, "logarithmic": False},
                l1_ratio={"low": 0.1, "high": 1, "logarithmic": False},
                solvers=("saga", "lbfgs"),
            ),
        },
    }
    assert (
        OptunaHyperParameterOptimization()._check_if_suggester_in_configspace(cfg) == 1
    )


def test_resolve_only_suggesters():
    cfg = Config(
        {
            "param1": 1,
            "mock_suggester": {
                "@suggesters": "mock_log_reg_suggester",
                "c_high": 1,
            },
            "age_filter": {
                "@preprocessing": "age_filter",
                "age_col_name": "age",
                "max_age": 100,
                "min_age": 0,
            },
        },
    )
    resolved_suggesters = OptunaHyperParameterOptimization()._resolve_only_suggestors(
        cfg,
    )
    assert isinstance(resolved_suggesters["mock_suggester"], MockLogisticRegression)
    assert isinstance(resolved_suggesters["age_filter"], dict)


def test_hyperparameter_optimization_from_file():
    study = OptunaHyperParameterOptimization().conduct_hyperparameter_optimization_from_file(
        (Path(__file__).parent / "test_optuna_hyperparameter_search.cfg"),
        n_trials=2,
        n_jobs=1,
    )
    assert len(study.trials) == 2
