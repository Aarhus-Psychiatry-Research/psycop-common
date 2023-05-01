"""Generate synth data with outcome."""
import numpy as np
from psycop_ml_utils.synth_data_generator.synth_prediction_times_generator import (
    generate_synth_data,
)
from psycop_model_training.utils.utils import PROJECT_ROOT

from tests.test_data.model_eval.generate_synthetic_dataset_for_eval import (
    add_age_is_female,
)


def test_synth_data_generator():
    """Test synth data generator."""
    override_dataset_on_test_run = False

    column_specifications = [
        {"citizen_ids": {"column_type": "uniform_int", "min": 0, "max": 1_200_001}},
        {"timestamp": {"column_type": "datetime_uniform", "min": 0, "max": 10 * 365}},
        {
            "timestamp_exclusion": {
                "column_type": "datetime_uniform",
                "min": 0,
                "max": 5 * 365,
            },
        },
        {
            "pred_age": {
                "column_type": "uniform_int",
                "min": 18,
                "max": 90,
            },
        },
        {
            "hba1c_within_9999_days_count_nan": {
                "column_type": "uniform_int",
                "min": 0,
                "max": 8,
            },
        },
        {
            "timestamp_outcome": {
                "column_type": "datetime_uniform",
                "min": 1 * 365,
                "max": 6 * 365,
            },
        },
        {
            "pred_hba1c_within_30_days_max_fallback_np.nan": {
                "column_type": "normal",
                "mean": 48,
                "sd": 5,
                "fallback": np.nan,
            },
        },
        {
            "pred_hba1c_within_60_days_max_fallback_np.nan": {
                "column_type": "normal",
                "mean": 48,
                "sd": 5,
                "fallback": np.nan,
            },
        },
        {
            "pred_hba1c_within_100_days_max_fallback_np.nan": {
                "column_type": "normal",
                "mean": 48,
                "sd": 5,
                "fallback": np.nan,
            },
        },
        {
            "pred_hdl_within_100_days_max_fallback_np.nan": {
                "column_type": "normal",
                "mean": 1,
                "sd": 0.5,
                "min": 0,
                "fallback": np.nan,
            },
        },
    ]

    for split in ("train", "val", "test"):
        n_samples = 70000 if split == "train" else 30000

        outcome_col_name = "outc_dichotomous_t2d_within_30_days_max_fallback_0"

        synth_df = generate_synth_data(
            predictors=column_specifications,
            outcome_column_name=outcome_col_name,
            n_samples=n_samples,
            logistic_outcome_model="1*pred_hba1c_within_100_days_max_fallback_nan+1*pred_hdl_within_100_days_max_fallback_nan",
            prob_outcome=0.08,
            na_ignore_cols=[outcome_col_name],
        )

        synth_df["pred_time_uuid"] = synth_df["citizen_ids"].astype(str) + synth_df[
            "timestamp"
        ].dt.strftime(
            "-%Y-%m-%d-%H-%M-%S",
        )

        synth_df = add_age_is_female(synth_df, id_column_name="citizen_ids")

        if override_dataset_on_test_run:
            # Save to csv
            synth_df.to_csv(
                PROJECT_ROOT
                / "tests"
                / "test_data"
                / "synth_splits"
                / f"synth_{split}.csv",
                index=False,
            )

        synth_df.describe()

        assert synth_df.shape == (n_samples, len(column_specifications) + 4)
