"""Misc.

utilities.
"""
import math
import sys
import tempfile
from collections.abc import Iterable, MutableMapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional, Union

import dill as pkl
import numpy as np
import pandas as pd
import wandb
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Extra
from sklearn.pipeline import Pipeline

SHARED_RESOURCES_PATH = Path(r"E:\shared_resources")
FEATURE_SETS_PATH = SHARED_RESOURCES_PATH / "feature_sets"
OUTCOME_DATA_PATH = SHARED_RESOURCES_PATH / "outcome_data"
RAW_DATA_VALIDATION_PATH = SHARED_RESOURCES_PATH / "raw_data_validation"
FEATURIZERS_PATH = SHARED_RESOURCES_PATH / "featurizers"
MODEL_PREDICTIONS_PATH = SHARED_RESOURCES_PATH / "model_predictions"

PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_PLOT_PATH = PROJECT_ROOT / "tests" / "plots_from_tests"


class BaseModel(PydanticBaseModel):
    """."""

    class Config:
        """An pydantic basemodel, which doesn't allow attributes that are not
        defined in the class."""

        allow_mutation = False
        arbitrary_types_allowed = True
        extra = Extra.forbid


def format_dict_for_printing(d: dict) -> str:
    """Format a dictionary for printing. Removes extra apostrophes, formats
    colon to dashes, separates items with underscores and removes curly
    brackets.

    Args:
        d (dict): dictionary to format.

    Returns:
        str: Formatted dictionary.

    Example:
        >>> d = {"a": 1, "b": 2}
        >>> print(format_dict_for_printing(d))
        >>> "a-1_b-2"
    """
    return (
        str(d)
        .replace("'", "")
        .replace(": ", "-")
        .replace("{", "")
        .replace("}", "")
        .replace(", ", "_")
    )


def flatten_nested_dict(
    d: dict,
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """Recursively flatten an infinitely nested dict.

    E.g. {"level1": {"level2": "level3": {"level4": 5}}}} becomes
    {"level1.level2.level3.level4": 5}.

    Args:
        d (dict): dict to flatten.
        parent_key (str): The parent key for the current dict, e.g. "level1" for the first iteration.
        sep (str): How to separate each level in the dict. Defaults to ".".

    Returns:
        dict: The flattened dict.
    """

    items: list[dict[str, Any]] = []

    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(
                flatten_nested_dict(d=v, parent_key=new_key, sep=sep).items(),  # type: ignore
            )  # typing: ignore
        else:
            items.append((new_key, v))  # type: ignore

    return dict(items)  # type: ignore


def drop_records_if_datediff_days_smaller_than(
    df: pd.DataFrame,
    t2_col_name: str,
    t1_col_name: str,
    threshold_days: float,
) -> pd.Series:
    """Drop rows where datediff is smaller than threshold_days. datediff = t2 - t1.

    Args:
        df (pd.DataFrame): Dataframe.
        t2_col_name (str): Column name of a time column
        t1_col_name (str): Column name of a time column
        threshold_days (float): Drop if datediff is smaller than this.

    Returns:
        A pandas dataframe without the records where datadiff was smaller than threshold_days.
    """
    return df[
        (df[t2_col_name] - df[t1_col_name]) / np.timedelta64(1, "D") > threshold_days  # type: ignore
    ]


def round_floats_to_edge(series: pd.Series, bins: Sequence[float]) -> pd.Series:
    """Rounds a float to the lowest value it is larger than. E.g. if bins = [0, 1, 2, 3],
    0.9 will be rounded to 0, 1.8 will be rounded to 1, etc.

    Args:
        series (pd.Series): The series of floats to round to bin edges.
        bins (list[floats]): Values to round to.

    Returns:
        A numpy ndarray with the borders.
    """
    _, edges = pd.cut(series, bins=bins, retbins=True, duplicates="drop")
    labels = [f"({abs(edges[i]):.0f}, {edges[i+1]:.0f}]" for i in range(len(bins) - 1)]

    return pd.cut(series, bins=bins, labels=labels)


def bin_continuous_data(
    series: pd.Series,
    bins: Sequence[float],
    min_n_in_bin: int = 5,
    use_min_as_label: bool = False,
    bin_decimals: Optional[int] = None,
) -> tuple[pd.Series, pd.Series]:
    """For prettier formatting of continuous binned data such as age.

    Args:
        series (pd.Series): Series with continuous data such as age
        bins (list[int]): Desired bins. Last value creates a bin from the last value to infinity.
        min_n_in_bin (int, optional): Minimum number of observations in a bin. If fewer than this, the bin is dropped. Defaults to 5.
        use_min_as_label (bool, optional): If True, the minimum value in the bin is used as the label. If False, the maximum value is used. Defaults to False.
        bin_decimals: Number of decimals to round bins to. Defaults to None, in which case all decimals are removed.

    Returns:
        Two ungrouped series, e.g. a row for each observation in the original dataset, each containing:

        pd.Series: Binned categories for values in data
        pd.Series: Number of samples in binned category
    """
    labels = []

    if not isinstance(bins, list):
        bins = list(bins)

    # Append maximum value from series to bins set upper cut-off if larger than maximum bins value
    if not series.isna().all() and series.max() > max(bins):
        # Round max value up
        max_value_rounded = math.ceil(series.max())
        bins.append(max_value_rounded)

    # Round bins to specified number of decimals
    bins = [round(b, bin_decimals) for b in bins]

    # Create bin labels
    for i, bin_v in enumerate(bins):
        # If not the final bin
        if i < len(bins) - 2:
            # If the difference between the current bin and the next bin is 1, the bin label is a single value and not an interval
            if (bins[i + 1] - bin_v) == 1 or use_min_as_label:
                labels.append(f"{bin_v}")
            # Else generate bin labels as intervals
            elif i == 0:
                labels.append(f"{bin_v}-{bins[i+1]}")
            else:
                labels.append(f"{bin_v+1}-{bins[i+1]}")
        elif i == len(bins) - 2:
            labels.append(f"{bin_v+1}+")
        else:
            continue

    df = pd.DataFrame(
        {
            "series": series,
            "bin": pd.cut(
                series,
                bins=bins,
                labels=labels,
                duplicates="drop",
                include_lowest=True,
            ),
        },
    )

    # Drop all rows where bin is NaN
    df = df.dropna()

    # Add a column with counts for the bin each row belongs to
    df["n_in_bin"] = df.groupby("bin")["bin"].transform("count")

    # Mask n_in_bin if less than min_n_in_bin
    df["n_in_bin"] = df["n_in_bin"].mask(df["n_in_bin"] < min_n_in_bin, np.nan)

    return df["bin"], df["n_in_bin"]


def positive_rate_to_pred_probs(
    pred_probs: pd.Series,
    positive_rates: Iterable,
) -> list[Any]:
    """Get thresholds for a set of percentiles. E.g. if one
    positive_rate == 1, return the value where 1% of predicted
    probabilities lie above.

    Args:
        pred_probs (pd.Sereis): Predicted probabilities.
        positive_rates (Iterable): positive_rates

    Returns:
        pd.Series: Thresholds for each percentile
    """

    # Check if percentiles provided as whole numbers, e.g. 99, 98 etc.
    # If so, convert to float.
    if max(positive_rates) > 1:
        positive_rates = [x / 100 for x in positive_rates]

    # Invert thresholds for quantile calculation
    thresholds = [1 - threshold for threshold in positive_rates]

    return pd.Series(pred_probs).quantile(thresholds).tolist()


def read_pickle(path: Union[str, Path]) -> Any:
    """Reads a pickled object from a file.

    Args:
        path (str): Path to pickle file.

    Returns:
        Any: Pickled object.
    """
    with Path(path).open(mode="rb") as f:
        return pkl.load(f)


def write_df_to_file(
    df: pd.DataFrame,
    file_path: Path,
):
    """Write dataset to file. Handles csv and parquet files based on suffix.

    Args:
        df: Dataset
        file_path (str): File path. Infers file type from suffix.
    """

    file_suffix = file_path.suffix

    if file_suffix == ".csv":
        df.to_csv(file_path, index=False)
    elif file_suffix == ".parquet":
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"Invalid file suffix {file_suffix}")


def get_feature_importance_dict(pipe: Pipeline) -> Union[None, dict[str, float]]:
    """Returns feature importances as a dict.

    Args:
        pipe (Pipeline): Sklearn pipeline.

    Returns:
        Union[None, dict[str, float]]: Dictionary of feature importances.
    """
    return dict(
        zip(pipe["model"].feature_names, pipe["model"].feature_importances_),  # type: ignore
    )


def get_selected_features_dict(
    pipe: Pipeline,
    train_col_names: list[str],
) -> Union[None, dict[str, int]]:
    """Returns results from feature selection as a dict.

    Args:
        pipe (Pipeline): Sklearn pipeline.
        train_col_names (list[str]): List of column names in the training set.

    Returns:
        Union[None, dict[str, int]]: Dictionary of selected features. 0 if not selected, 1 if selected.
    """
    is_selected = [
        int(i) for i in pipe["preprocessing"]["feature_selection"].get_support()  # type: ignore
    ]
    return dict(
        zip(train_col_names, is_selected),
    )


def create_wandb_folders():
    """Creates folders to store logs on Overtaci."""
    if sys.platform == "win32":
        (Path(tempfile.gettempdir()) / "debug-cli.onerm").mkdir(
            exist_ok=True,
            parents=True,
        )
        (PROJECT_ROOT / "wandb" / "debug-cli.onerm").mkdir(exist_ok=True, parents=True)


def coerce_to_datetime(date_repr: Union[str, date]) -> datetime:
    """Coerce date or str to datetime."""
    if isinstance(date_repr, str):
        date_repr = date.fromisoformat(
            date_repr,
        )

    if isinstance(date_repr, date):
        date_repr = datetime.combine(
            date_repr,
            datetime.min.time(),
        )

    return date_repr


def get_percent_lost(n_before: float, n_after: float) -> float:
    """Get the percent lost."""
    return round((100 * (1 - n_after / n_before)), 2)


def output_table(
    output_format: str,
    df: pd.DataFrame,
) -> Union[pd.DataFrame, wandb.Table, str]:
    """Output table in specified format."""
    if output_format == "html":
        return df.reset_index(drop=True).to_html()
    if output_format == "df":
        return df.reset_index(drop=True)
    if output_format == "wandb_table":
        return wandb.Table(dataframe=df)

    raise ValueError("Output format does not match anything that is allowed")
