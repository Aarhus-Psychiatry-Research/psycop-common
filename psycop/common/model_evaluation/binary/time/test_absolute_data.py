import random

import numpy as np
import pandas as pd

from psycop.common.model_evaluation.binary.time.absolute_data import (
    create_roc_auc_by_absolute_time_df,
)


def test_create_roc_auc_by_absolute_time_df():
    DF_LENGTH = 100
    np.random.seed(42)
    random.seed(42)

    input_df = pd.DataFrame(
        {
            "y": [random.randint(0, 1) for _ in range(DF_LENGTH)],
            "y_hat_probs": [random.random() for _ in range(DF_LENGTH)],
            "timestamp": [pd.Timestamp("2020-01-01") for _ in range(DF_LENGTH)],
        }
    )

    output_df = create_roc_auc_by_absolute_time_df(
        labels=input_df["y"],
        y_hat_probs=input_df["y_hat_probs"],
        timestamps=input_df["timestamp"],
        bin_period="M",
        confidence_interval=True,
        n_bootstraps=5,
    )

    assert output_df["n_in_bin"][0] == 100
    assert 0.55 > output_df["auroc"][0] > 0.45
