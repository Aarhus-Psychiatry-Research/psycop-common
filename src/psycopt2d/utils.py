from collections.abc import MutableMapping
from typing import Dict, Iterable, List, Union

import numpy as np
import pandas as pd
from psycopmlutils.model_performance import ModelPerformance
from sklearn.impute import SimpleImputer
from wasabi import msg
from xgboost import XGBClassifier


def flatten_nested_dict(
    d: Dict,
    parent_key: str = "",
    sep: str = ".",
) -> Dict:
    """Recursively flatten an infinitely nested dict.

    E.g. {"level1": {"level2": "level3": {"level4": 5}}}} becomes
    {"level1.level2.level3.level4": 5}.

    Args:
        dict (Dict): Dict to flatten.
        separator (str, optional): How to separate each level in the dict. Defaults to
            ".".

    Returns:
        Dict: The flattened dict.
    """

    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_nested_dict(d=v, parent_key=new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def drop_records_if_datediff_days_smaller_than(
    df: pd.DataFrame,
    t2_col_name: str,
    t1_col_name: str,
    threshold_days: Union[float, int],
    inplace: bool = True,
):
    """Drop rows where datediff is smaller than threshold_days. datediff = t2 - t1.

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
                (df[t2_col_name] - df[t1_col_name]) / np.timedelta64(1, "D")
                < threshold_days
            ].index,
            inplace=True,
        )
    else:
        return df[
            (df[t2_col_name] - df[t1_col_name]) / np.timedelta64(1, "D")
            < threshold_days
        ]


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
        bins (List[floats]): Values to round to.
    """
    _, edges = pd.cut(series, bins=bins, retbins=True)
    labels = [f"({abs(edges[i]):.0f}, {edges[i+1]:.0f}]" for i in range(len(bins) - 1)]

    return pd.cut(series, bins=bins, labels=labels)


def convert_all_to_binary(ds, skip):
    msg.info("Rounding all to binary")
    cols_to_round = [
        colname for colname in ds.columns if ds[colname].dtype != "datetime64[ns]"
    ]

    [cols_to_round.remove(c) for c in skip]

    for col in cols_to_round:
        ds[col] = ds[col].map(lambda x: 1 if x > 0 else np.NaN)


def calculate_performance_metrics(
    eval_df: pd.DataFrame,
    outcome_col_name: str,
    prediction_probabilities_col_name: str,
    id_col_name: str = "dw_ek_borger",
):
    """Log performance metrics to WandB.

    Args:
        eval_df (pd.DataFrame): DataFrame with predictions, labels, and id
        outcome_col_name (str): Name of the column containing the outcome (label)
        prediction_probabilities_col_name (str): Name of the column containing predicted
            probabilities
        id_col_name (str): Name of the id column
    """
    performance_metrics = ModelPerformance.performance_metrics_from_df(
        eval_df,
        prediction_col_name=prediction_probabilities_col_name,
        label_col_name=outcome_col_name,
        id_col_name=id_col_name,
        metadata_col_names=None,
        to_wide=True,
    )

    performance_metrics = performance_metrics.to_dict("records")[0]
    return performance_metrics


def bin_continuous_data(series: pd.Series, bins: List[int]) -> pd.Series:
    """For prettier formatting of continuous binned data such as age.

    Args:
        series (pd.Series): Series with continuous data such as age
        bins (List[int]): Desired bins

    Returns:
        pd.Series: Binned data

    Example:
    >>> ages = pd.Series([15, 18, 20, 30, 32, 40, 50, 60, 61])
    >>> age_bins = [0, 18, 30, 50, 110]
    >>> bin_Age(ages, age_bins)
    0     0-18
    1     0-18
    2    19-30
    3    19-30
    4    31-50
    5    31-50
    6    31-50
    7      51+
    8      51+
    """
    labels = []
    for i, bin in enumerate(bins):
        if i == 0:
            labels.append(f"{bin}-{bins[i+1]}")
        elif i < len(bins) - 2:
            labels.append(f"{bin+1}-{bins[i+1]}")
        elif i == len(bins) - 2:
            labels.append(f"{bin+1}+")
        else:
            continue

    return pd.cut(series, bins=bins, labels=labels)


def pred_proba_to_threshold_percentiles(
    pred_probs: pd.Series,
    threshold_percentiles: Iterable,
) -> List:
    """Get thresholds for a set of percentiles. E.g. if percentile == 1, return
    the threshold that 1% of the probabilities are above.

    Args:
        pred_probs (pd.Sereis): Predicted probabilities.
        threshold_percentiles (Iterable): Threshold-percentiles

    Returns:
        pd.Series: Threshodls for each percentile
    """
    # Check if percentiles provided as whole numbers, e.g. 99, 98 etc.
    # If so, convert to float.
    if max(threshold_percentiles) > 1:
        threshold_percentiles = [x / 100 for x in threshold_percentiles]

    return pd.Series(pred_probs).quantile(threshold_percentiles).tolist()
