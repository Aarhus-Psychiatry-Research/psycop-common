from typing import Literal

from sklearn.linear_model import LogisticRegression

from psycop.common.model_training_v2.classifier_pipelines.model_step import ModelStep


def logistic_regression_step(
    penalty: Literal["l1", "l2", "elasticnet"] = "elasticnet",
    solver: Literal[
        "lbfgs",
        "liblinear",
        "newton-cg",
        "newton-cholesky",
        "sag",
        "saga",
    ] = "saga",
    C: float = 1.0,
    l1_ratio: float = 0.5,
) -> ModelStep:
    """Initialize logistic regression model with hparams specified as kwargs.
    The 'missing' hyperparameter specifies the value to be treated as missing
    and is set to np.nan by default."""

    return (
        "logistic_regression",
        LogisticRegression(penalty=penalty, solver=solver, C=C, l1_ratio=l1_ratio, random_state=41),
    )
