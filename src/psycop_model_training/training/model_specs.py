"""Specifications of models to be evaluated."""
import numpy as np

# from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


def init_logistic_regression(**kwargs):
    """Initialize LogisticRegression model."""
    if "penalty_solver" in kwargs:
        kwargs["penalty"], kwargs["solver"] = kwargs.pop("penalty_solver").split("_")
    return LogisticRegression(**kwargs)


logistic = {"model": init_logistic_regression, "static_hyperparameters": {}}
xgboost = {"model": XGBClassifier, "static_hyperparameters": {"missing": np.nan}}
nb = {"model": GaussianNB, "static_hyperparameters": {}}

# ebm = {"model": ExplainableBoostingClassifier, "static_hyperparameters": {}} - removed because of issue with github actions.
# See https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/pull/194 for thoughts on root cause

MODELS = {
    "logistic-regression": logistic,
    "xgboost": xgboost,
    "naive-bayes": nb,
}
