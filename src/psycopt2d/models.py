import catalogue
import numpy as np
from xgboost import XGBClassifier

model_hyperparameters = catalogue.create("psycopt2d", "models")

xgboost = {"model": XGBClassifier, "static_hyperparameters": {"missing": np.nan}}
model_hyperparameters.register("xgboost", func=xgboost)
