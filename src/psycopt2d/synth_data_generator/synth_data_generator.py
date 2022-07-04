from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


def generate_synth_data(dict: Dict, len: int) -> pd.DataFrame:
    """Takes a dict and generates synth data from it.

    Args:
        dict (Dict): A dict representing each column of shape: {"col_name": {"distribution": str, "output_type": ["float"|"int"], "min": int, "max": int}}
        len (int): Number of rows to generate.

    Returns:
        pd.DataFrame
    """

    # Initialise dataframe
    df = pd.DataFrame(columns=list(dict.keys()))

    # Generate data
    for col_name, col_props in dict.items():
        distribution = col_props["distribution"]

        if distribution == "uniform_float":
            df[col_name] = np.random.uniform(
                low=col_props["min"],
                high=col_props["max"],
                size=len,
            )
        elif distribution == "uniform_int":
            df[col_name] = np.random.randint(
                low=col_props["min"],
                high=col_props["max"],
                size=len,
            )
        elif distribution == "normal":
            df[col_name] = np.random.normal(
                loc=col_props["mean"],
                scale=col_props["sd"],
                size=len,
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # If column has min and/or max, floor and ceil appropriately
        if "min" in col_props:
            df[col_name] = df[col_name].clip(lower=col_props["min"])
        if "max" in col_props:
            df[col_name] = df[col_name].clip(upper=col_props["max"])

    return df


if __name__ == "__main__":
    column_specifications = {
        "citizen_ids": {"distribution": "uniform_int", "min": 0, "max": 1_200_000},
        "pred_hba1c": {"distribution": "normal", "mean": 48, "sd": 5},
        "pred_hdl": {"distribution": "normal", "mean": 1, "sd": 0.5, "min": 0},
    }

    OUTCOME_COL_NAME = "outc_dichotomous_t2d_within_30_days_max_fallback_0"

    n_samples = 10_000

    df = generate_synth_data(dict=column_specifications, len=n_samples)
    noise = np.random.normal(loc=0, scale=1, size=n_samples)

    # Linear model with columns
    x = df["pred_hba1c"] + df["pred_hdl"]

    # Z-score normalise and add noise
    x = stats.zscore(x) + noise

    # Sigmoid it to get probabilities with mean = 0.5
    df[OUTCOME_COL_NAME] = 1 / (1 + np.exp(x))

    threshold = 0.08
    df[OUTCOME_COL_NAME] = np.where(df[OUTCOME_COL_NAME] < threshold, 1, 0)
    df.describe()

    df.to_csv("synth_prediction_data.csv")
