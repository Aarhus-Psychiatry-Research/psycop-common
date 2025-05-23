"""Loaders for synth data."""

from __future__ import annotations

import pandas as pd

from psycop.common.global_utils.paths import PSYCOP_PKG_ROOT


def load_raw_test_csv(filename: str, n_rows: int | None = None) -> pd.DataFrame:
    """Load raw csv.

    Args:
        filename (str): Name of the file to load.
        n_rows (int, optional): Number of rows to load. Defaults to None.
    """
    df = pd.read_csv(PSYCOP_PKG_ROOT / "tests" / "test_data" / "raw" / filename, nrows=n_rows)

    # Convert timestamp col to datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def synth_predictor_float(n_rows: int | None = None) -> pd.DataFrame:
    """Load synth predictor data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """
    return load_raw_test_csv("synth_raw_float_1.csv", n_rows=n_rows)


def synth_predictor_binary(n_rows: int | None = None) -> pd.DataFrame:
    """Load synth predictor data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """
    return load_raw_test_csv("synth_raw_binary_1.csv", n_rows=n_rows)


def load_synth_outcome(n_rows: int | None = None) -> pd.DataFrame:
    """Load synth predictor data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """
    # Get first row for each id
    df = load_raw_test_csv("synth_raw_binary_2.csv", n_rows=n_rows)
    df = df.groupby("dw_ek_borger").last().reset_index()
    return df


def load_synth_prediction_times(n_rows: int | None = None) -> pd.DataFrame:
    """Load synth predictor data.".

    Args:
        n_rows: Number of rows to return. Defaults to None which returns entire coercion data view.

    Returns:
        pd.DataFrame
    """
    return load_raw_test_csv("synth_prediction_times.csv", n_rows=n_rows)
