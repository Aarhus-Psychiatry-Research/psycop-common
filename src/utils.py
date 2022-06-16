from typing import List, Union

import numpy as np
import pandas as pd
import wandb
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from wasabi import msg
from xgboost import XGBClassifier


def difference_in_days(end_date_series: pd.Series, start_date_series: pd.Series):
    """Calculate difference in days between two pandas datetime series (start_date - end_date).

    Args:
        end_date_series (pd.Series): First datetime64[ns] series
        start_date_series (pd.Series): Second datetime64[ns] series
    """
    return (start_date_series - end_date_series) / np.timedelta64(1, "D")


def drop_records_if_datediff_days_smaller_than(
    df: pd.DataFrame,
    t2_col_name: str,
    t1_col_name: str,
    threshold_days: Union[float, int],
    inplace: bool = True,
):
    """Drop rows where datediff is smaller than threshold_days.

    Args:
        df (pd.DataFrame): Dataframe.
        t2_col_name (str): _description_
        t1_col_name (str): _description_
        threshold_days (Union[float, int]): _description_
        inplace (bool, optional): Defaults to True.
    """
    if inplace:
        df.drop(
            df[
                difference_in_days(df[t2_col_name], df[t1_col_name]) < threshold_days
            ].index,
            inplace=True,
        )
    else:
        return df[difference_in_days(df[t2_col_name], df[t1_col_name]) < threshold_days]


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


def round_floats_to_edge(series: pd.Series, bins: List[float]):
    """Rounds a float to the lowest value it is larger than.
    Args:
        edges (List[floats]): Values to round to.
    """
    _, edges = pd.cut(series, bins=bins, retbins=True)
    labels = [f"({abs(edges[i]):.0f}, {edges[i+1]:.0f}]" for i in range(len(bins) - 1)]

    return pd.cut(series, bins=bins, labels=labels)


def log_tpr_by_time_to_event(
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
    wandb.log({"TPR by time to event": plt})


def plot_xgb_feature_importances(val_X_column_names, model):
    sorted_idx = model.feature_importances_.argsort()
    plt.barh(val_X_column_names[sorted_idx], model.feature_importances_[sorted_idx])
    plt.autoscale()
    plt.subplots_adjust(left=0.5, bottom=0.171, right=0.92, top=0.92)


def convert_all_to_binary(ds, skip):
    msg.info("Rounding all to binary")
    cols_to_round = [
        colname for colname in ds.columns if ds[colname].dtype != "datetime64[ns]"
    ]

    [cols_to_round.remove(c) for c in skip]

    for col in cols_to_round:
        ds[col] = ds[col].map(lambda x: 1 if x > 0 else np.NaN)
