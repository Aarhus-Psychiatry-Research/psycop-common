import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def InitLogisticRegression(**kwargs):
    if "penalty_solver" in kwargs:
        kwargs["penalty"], kwargs["solver"] = kwargs.pop("penalty_solver").split("_")
    return LogisticRegression(**kwargs)


logistic = {"model": InitLogisticRegression, "static_hyperparameters": {}}
xgboost = {"model": XGBClassifier, "static_hyperparameters": {"missing": np.nan}}
ebm = {"model": ExplainableBoostingClassifier, "static_hyperparameters": {}}

MODELS = {"logistic-regression": logistic, "xgboost": xgboost, "ebm": ebm}
