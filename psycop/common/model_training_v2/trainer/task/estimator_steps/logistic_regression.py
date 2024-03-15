from collections.abc import Sequence
from typing import Any, Literal

import optuna
from sklearn.linear_model import LogisticRegression

from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.base_suggester import (
    Suggester,
)
from psycop.common.model_training_v2.hyperparameter_suggester.suggesters.suggester_spaces import (
    CategoricalSpace,
    FloatSpace,
    FloatSpaceT,
)
from psycop.common.model_training_v2.trainer.task.model_step import ModelStep

LogRegSolvers = Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
LogRegPenalties = Literal["l1", "l2", "elasticnet"]


@BaselineRegistry.estimator_steps.register("logistic_regression")
def logistic_regression_step(
    penalty: LogRegPenalties = "elasticnet",
    solver: LogRegSolvers = "saga",
    C: float = 1.0,
    l1_ratio: float = 0.5,
) -> ModelStep:
    """Initialize logistic regression model with hparams specified as kwargs.
    The 'missing' hyperparameter specifies the value to be treated as missing
    and is set to np.nan by default."""

    return (
        "classifier",
        LogisticRegression(
            penalty=penalty, solver=solver, C=C, l1_ratio=l1_ratio, random_state=41
        ),  # Random_state is required for reproducibility, e.g. getting the same result on every test
    )


@BaselineRegistry.estimator_steps_suggesters.register("logistic_regression_suggester")
class LogisticRegressionSuggester(Suggester):
    def __init__(
        self,
        C: FloatSpaceT = {"low": 1e-8, "high": 10, "logarithmic": False},  # noqa: B006
        l1_ratio: FloatSpaceT = {  # noqa: B006
            "low": 0.0,
            "high": 1,
            "logarithmic": False,
        },
        solvers: Sequence[LogRegSolvers] = ("saga",),
        penalties: Sequence[LogRegPenalties] = ("l1", "l2", "elasticnet"),
    ):
        self.C = FloatSpace.from_list_or_mapping(C)
        self.l1_ratio = FloatSpace.from_list_or_mapping(l1_ratio)
        self.solver = CategoricalSpace(solvers)
        self.penalties = CategoricalSpace(penalties)

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "@estimator_steps": "logistic_regression",
            "C": self.C.suggest(trial, "C"),
            "l1_ratio": self.l1_ratio.suggest(trial, "l1_ratio"),
            "solver": self.solver.suggest(trial, "solver"),
            "penalty": self.penalties.suggest(trial, "penalty"),
        }
