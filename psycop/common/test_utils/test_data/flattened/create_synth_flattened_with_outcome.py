"""Generate synth data with outcome."""

from pathlib import Path

import numpy as np

from psycop.common.global_utils.synth_data_generator.synth_prediction_times_generator import (
    generate_synth_data,
)

if __name__ == "__main__":
    column_specifications = {
        "citizen_ids": {"column_type": "uniform_int", "min": 0, "max": 1_200_001},
        "timestamp": {"column_type": "datetime_uniform", "min": 0, "max": 5 * 365},
        "timestamp_outcome": {"column_type": "datetime_uniform", "min": 1 * 365, "max": 6 * 365},
        "pred_hba1c_within_100_days_max_fallback_np.nan": {
            "column_type": "normal",
            "mean": 48,
            "sd": 5,
            "fallback": np.nan,
        },
        "pred_hdl_within_100_days_max_fallback_np.nan": {
            "column_type": "normal",
            "mean": 1,
            "sd": 0.5,
            "min": 0,
            "fallback": np.nan,
        },
    }

    synth_df = generate_synth_data(
        predictors=column_specifications,  # type: ignore
        outcome_column_name="outc_t2d_within_30_days_max_fallback_0_dichotomous",
        n_samples=10_000,
        logistic_outcome_model="1*pred_hba1c_within_100_days_max_fallback_nan+1*pred_hdl_within_100_days_max_fallback_nan",
        prob_outcome=0.08,
    )

    synth_df.describe()

    synth_df.to_csv(Path(__file__).parent / "synth_prediction_data.csv")
