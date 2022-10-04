"""Refactor this to be an application of the module in psycop-ml-utils.

Generator of synthetic data.
"""

# pylint: disable=too-many-locals,invalid-name,missing-function-docstring

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


def generate_synth_data(
    predictors: dict,
    outcome_column_name: str,
    n_samples: int,
    logistic_outcome_model: str,
    intercept: Union[int, float, complex, str, bytes] = 0,
    na_prob: float = 0.01,
    na_ignore_cols: Optional[tuple[str]] = None,
    prob_outcome: float = 0.08,
    noise_mean_sd: tuple[float, float] = (0, 1),
) -> pd.DataFrame:
    """Takes a dict and generates synth data from it.

    Args:
        predictors (dict): A dict representing each column of shape:
            {"col_name":
                {
                "column_type": str,
                "output_type": ["float"|"int"],
                "min": int,
                "max":int
                }
            }
        outcome_column_name: str, name of the outcome column
        n_samples (int): Number of samples (rows) to generate.
        logistic_outcome_model (str): The statistical model used to generate outcome
            values, e.g. specified as'1*col_name+1*col_name2'
        intercept (float, optional): The intercept of the logistic outcome model. Defaults to 0.
        na_prob (float, optional): Probability of changing a value in a predictor column
            to NA.
        na_ignore_cols (list[str], optional): Columns to ignore when creating NAs
        prob_outcome (float): Probability of a given row receiving "1" for the outcome.
        noise_mean_sd (tuple[float, float], optional): mean and sd of the noise.
            Increase SD to obtain more uncertain models.

    Returns:
        pd.DataFrame: The synthetic dataset
    """

    # Initialise dataframe
    df = pd.DataFrame(columns=list(predictors.keys()))

    # Generate data
    df = generate_data_columns(predictors, n_samples, df)

    # Linear model with columns
    y_ = intercept
    for var in logistic_outcome_model.split("+"):
        effect, col = var.split("*")
        y_ = float(effect) * df[col] + y_

    noise = np.random.normal(
        loc=noise_mean_sd[0],
        scale=noise_mean_sd[1],
        size=n_samples,
    )
    # Z-score normalise and add noise
    y_ = stats.zscore(y_) + noise

    # randomly replace predictors with NAs
    if na_prob:
        mask = np.random.choice([True, False], size=df.shape, p=[na_prob, 1 - na_prob])
        df_ = df.mask(mask)

        # ignore nan values for non-nan columns.
        if na_ignore_cols:
            for col in na_ignore_cols:
                df_[col] = df[col]

    # Sigmoid it to get probabilities with mean = 0.5
    df[outcome_column_name] = 1 / (1 + np.exp(y_))

    df[outcome_column_name] = np.where(df[outcome_column_name] < prob_outcome, 1, 0)

    return df


def generate_data_columns(predictors, n_samples, df):
    for col_name, col_props in predictors.items():
        column_type = col_props["column_type"]

        if column_type == "uniform_float":
            df[col_name] = np.random.uniform(
                low=col_props["min"],
                high=col_props["max"],
                size=n_samples,
            )
        elif column_type == "uniform_int":
            df[col_name] = np.random.randint(
                low=col_props["min"],
                high=col_props["max"],
                size=n_samples,
            )
        elif column_type == "normal":
            df[col_name] = np.random.normal(
                loc=col_props["mean"],
                scale=col_props["sd"],
                size=n_samples,
            )
        elif column_type == "datetime_uniform":
            df[col_name] = pd.to_datetime(
                np.random.uniform(
                    low=col_props["min"],
                    high=col_props["max"],
                    size=n_samples,
                ),
                unit="D",
            ).round("min")
        else:
            raise ValueError(f"Unknown distribution: {column_type}")

        # If column has min and/or max, floor and ceil appropriately
        if df[col_name].dtype not in ["datetime64[ns]"]:
            if "min" in col_props:
                df[col_name] = df[col_name].clip(lower=col_props["min"])
            if "max" in col_props:
                df[col_name] = df[col_name].clip(upper=col_props["max"])

    return df


if __name__ == "__main__":
    column_specifications = {
        "citizen_ids": {"column_type": "uniform_int", "min": 0, "max": 1_200_000},
        "timestamp": {"column_type": "datetime_uniform", "min": 0, "max": 5 * 365},
        "timestamp_outcome": {
            "column_type": "datetime_uniform",
            "min": 1 * 365,
            "max": 6 * 365,
        },
        "pred_hba1c": {"column_type": "normal", "mean": 48, "sd": 5},
        "pred_hdl": {"column_type": "normal", "mean": 1, "sd": 0.5, "min": 0},
    }

    df = generate_synth_data(
        predictors=column_specifications,
        outcome_column_name="outc_dichotomous_t2d_within_30_days_max_fallback_0",
        n_samples=10_000,
        logistic_outcome_model="1*pred_hba1c+1*pred_hdl",
        prob_outcome=0.08,
    )

    df.describe()

    save_path = Path(__file__).parent.parent.parent.parent
    df.to_csv(save_path / "tests" / "test_data" / "synth_prediction_data.csv")
