import catalogue
import numpy as np
from xgboost import XGBClassifier

model_catalogue = catalogue.create("psycopt2d", "models")

xgboost = {"model": XGBClassifier, "static_hyperparameters": {"missing": np.nan}}
model_catalogue.register("xgboost", func=xgboost)
