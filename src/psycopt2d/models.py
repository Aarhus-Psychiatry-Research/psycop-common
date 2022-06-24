import numpy as np
from xgboost import XGBClassifier

xgboost = {"model": XGBClassifier, "static_hyperparameters": {"missing": np.nan}}
MODELS = {"xgboost": xgboost}
