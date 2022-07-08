import catalogue
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

model_catalogue = catalogue.create("psycopt2d", "models")

xgboost = {"model": XGBClassifier, "static_hyperparameters": {"missing": np.nan}}
model_catalogue.register("xgboost", func=xgboost)


def InitLogisticRegression(**kwargs):
    if "penalty_solver" in kwargs:
        kwargs["penalty"], kwargs["solver"] = kwargs.pop("penalty_solver").split("_")
    return LogisticRegression(**kwargs)


logistic = {"model": InitLogisticRegression, "static_hyperparameters": {}}
model_catalogue.register("logistic-regression", func=logistic)
