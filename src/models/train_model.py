import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from src.features.load_features import *
from wasabi import Printer
from xgboost import XGBClassifier


def impute(
    train_X,
    val_X,
):
    msg.info("Imputing!")
    my_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    train_X_imputed = my_imputer.fit_transform(train_X)
    val_X_imputed = my_imputer.transform(val_X)
    return train_X_imputed, val_X_imputed


def generate_predictions(train_y, train_X, val_X):
    msg.info("Fitting model")
    model = XGBClassifier(n_jobs=58, missing=np.nan)
    model.fit(train_X, train_y, verbose=True)
    msg.good("Model fit!")

    msg.info("Generating predictions")

    pred_probs = model.predict_proba(val_X)
    preds = model.predict(val_X)
    return preds, pred_probs, model


def print_auc(val_y, pred_probs):
    auc_predictions = pred_probs[:, 1]

    roc_auc = round(roc_auc_score(val_y, auc_predictions), 4)
    msg.info(f"auc: {roc_auc}")


def difference_in_days(start_date_series: pd.Series, end_date_series: pd.Series):
    """Calculate difference in days between two pandas datetime series
    Args:
        series1 (pd.Series): First datetime64[ns] series
        series2 (pd.Series): Second datetime64[ns] series
    """
    return (start_date_series - end_date_series) / np.timedelta64(1, "D")


def round_floats_to_edge(series: pd.Series, bins: List[float]):
    """Rounds a float to the lowest value it is larger than.
    Args:
        edges (List[floats]): Values to round to.
    """
    _, edges = pd.cut(series, bins=bins, retbins=True)
    labels = [f"({abs(edges[i]):.0f}, {edges[i+1]:.0f}]" for i in range(len(bins) - 1)]

    return pd.cut(series, bins=bins, labels=labels)


def plot_tpr_by_time_to_event(
    df: pd.DataFrame,
    outcome_col_name: str,
    predicted_outcome_col_name: str,
    outcome_timestamp_col_name: str,
    prediction_timestamp_col_name: str = "timestamp",
    bins=[0, 1, 7, 14, 28, 182, 365, 730, 1825],
):
    eval_pos_df = df[df[outcome_timestamp_col_name].notnull()]

    # Calculate difference in days between columns
    eval_pos_df["days_to_outcome"] = difference_in_days(
        eval_pos_df[outcome_timestamp_col_name],
        eval_pos_df[prediction_timestamp_col_name],
    )

    true_vals = eval_pos_df[outcome_col_name]
    pred_vals = eval_pos_df[predicted_outcome_col_name]

    eval_pos_df["true_positive"] = (true_vals == 1) & (pred_vals == 1)
    eval_pos_df["false_negative"] = (true_vals == 1) & (pred_vals == 0)

    eval_pos_df["days_to_outcome_binned"] = round_floats_to_edge(
        eval_pos_df["days_to_outcome"], bins=bins
    )

    tpr_by_time_to_outcome_df = (
        eval_pos_df[["days_to_outcome_binned", "true_positive", "false_negative"]]
        .groupby("days_to_outcome_binned")
        .sum()
    )

    tpr_by_time_to_outcome_df["tpr"] = tpr_by_time_to_outcome_df["true_positive"] / (
        tpr_by_time_to_outcome_df["true_positive"]
        + tpr_by_time_to_outcome_df["false_negative"]
    )

    plt.bar(
        tpr_by_time_to_outcome_df.reset_index()["days_to_outcome_binned"],
        tpr_by_time_to_outcome_df["tpr"],
    )
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.ylabel("TPR")
    plt.xlabel("Days until incident T2D")
    plt.xticks(rotation=45)

    plt.subplots_adjust(left=0.1, bottom=0.171, right=0.92, top=0.92)
    plt.show()


def plot_xgb_feature_importances(val_X_column_names, model):
    sorted_idx = model.feature_importances_.argsort()
    plt.barh(val_X_column_names[sorted_idx], model.feature_importances_[sorted_idx])
    plt.autoscale()
    plt.subplots_adjust(left=0.5, bottom=0.171, right=0.92, top=0.92)
    plt.show()


if __name__ == "__main__":
    msg = Printer(timestamp=True)

    IMPUTE = False
    LOAD_ALL = True
    FORCE_ALL_BINARY = False

    OUTCOME_COL_NAME = "t2d_within_1826.25_days_max_fallback_0"
    PREDICTED_OUTCOME_COL_NAME = f"pred_{OUTCOME_COL_NAME}"

    if LOAD_ALL:
        n_to_load = None
    else:
        n_to_load = 5_000

    outcome_timestamp_col_name = f"timestamp_{OUTCOME_COL_NAME}"

    cols_to_drop_before_training = [
        "dw_ek_borger",
        "prediction_time_uuid",
        outcome_timestamp_col_name,
    ]

    # Val set
    val_X, val_y = load_val(outcome_col_name=OUTCOME_COL_NAME, n_to_load=n_to_load)
    eval_X = val_X.copy()

    # Train set
    train_X, train_y = load_train(
        outcome_col_name=OUTCOME_COL_NAME, n_to_load=n_to_load
    )

    # Prep for training
    for ds in train_X, val_X:
        msg.info("Dropping columns that won't generalise")
        ds.drop(cols_to_drop_before_training, axis=1, errors="ignore", inplace=True)

        if FORCE_ALL_BINARY:
            msg.info("Rounding all to binary")
            cols_to_round = [
                colname
                for colname in ds.columns
                if ds[colname].dtype != "datetime64[ns]"
            ]

            [cols_to_round.remove(c) for c in ["age_in_years", "male"]]

            for col in cols_to_round:
                ds[col] = ds[col].map(lambda x: 1 if x > 0 else np.NaN)

        timestamp_cols = [col for col in ds.columns if "timestamp" in col]
        msg.info("Converting timestamps to ordinal")
        for colname in timestamp_cols:
            ds[colname] = ds[colname].map(dt.datetime.toordinal)

    # Make a copy of the datasets so you can inspect during debugging
    if IMPUTE:
        train_X_processed, val_X_processed = impute(train_X=train_X, val_X=val_X)
    else:
        train_X_processed, val_X_processed = train_X, val_X

    preds, pred_probs, model = generate_predictions(
        train_y, train_X_processed, val_X_processed
    )

    print_auc(val_y, pred_probs)

    # Evaluation
    ## Feature importance
    plot_xgb_feature_importances(val_X.columns, model)

    eval_df = eval_X
    eval_df[OUTCOME_COL_NAME] = val_y
    eval_df[PREDICTED_OUTCOME_COL_NAME] = preds

    plot_tpr_by_time_to_event(
        df=eval_df,
        outcome_col_name=OUTCOME_COL_NAME,
        predicted_outcome_col_name=PREDICTED_OUTCOME_COL_NAME,
        outcome_timestamp_col_name=outcome_timestamp_col_name,
        prediction_timestamp_col_name="timestamp",
        bins=[0, 1, 7, 14, 28, 182, 365, 730, 1825],
    )
