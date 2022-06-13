import datetime as dt

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from src.features.load_features import *
from wasabi import Printer
from xgboost import XGBClassifier

if __name__ == "__main__":
    msg = Printer(timestamp=True)
    outcome_col_name = "t2d_within_1826.25_days_max_fallback_0"

    train_X, train_y = load_train(outcome_col_name=outcome_col_name)
    train_X["timestamp"] = pd.to_datetime(train_X["timestamp"]).map(
        dt.datetime.toordinal
    )

    val_X, val_y = load_val(outcome_col_name=outcome_col_name)
    val_X["timestamp"] = pd.to_datetime(val_X["timestamp"]).map(dt.datetime.toordinal)

    # This is a hack! My first_t2d csv file is probably from an older dataset,
    # resulting in prediction_times without a t2d value. Huh, but should be replaced
    # by a 0. Have to look into that.
    msg.warn("Replacing NAs in outcome col with 0")
    train_y = train_y.fillna(0)
    val_y = val_y.fillna(0)

    # my_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    # train_X = my_imputer.fit_transform(train_X)
    # val_X = my_imputer.transform(val_X)

    msg.info("Fitting model")
    model = XGBClassifier(n_jobs=20, missing=np.nan)
    model.fit(train_X, train_y, verbose=True)
    msg.good("Model fit!")

    msg.info("Generating predictinos")
    predictions = model.predict_proba(val_X)

    auc_predictions = predictions[:, 1]

    msg.info(f"auc: {roc_auc_score(val_y.astype(int), auc_predictions)}")
