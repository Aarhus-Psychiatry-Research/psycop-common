import numpy as np
from xgboost import XGBClassifier

models_dict = {
    "xgboost": {"model": XGBClassifier, "static_hyperparameters": {"missing": np.nan}},
}
