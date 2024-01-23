import datetime as dt

import pandas as pd
import polars as pl

from psycop.common.feature_generation.loaders.raw.load_lab_results import hba1c


def time_from_first_pos_pred_to_next_hba1c(
    pos_preds: pl.LazyFrame, hba1cs: pl.LazyFrame
) -> pl.LazyFrame:
    first_pos_pred_colname = "timestamp_first_pos_pred"
    delta_time_col_name = "time_from_first_pos_pred_to_next_hba1c"

    first_pos_pred = pos_preds.groupby("patient_id").agg(
        pl.col("pred_timestamps").min().alias(first_pos_pred_colname)
    )

    filtered = (
        first_pos_pred.join(hba1cs, on="patient_id", how="inner")
        .with_columns(
            (pl.col("timestamp") - pl.col(first_pos_pred_colname)).alias(delta_time_col_name)
        )
        .filter(pl.col(delta_time_col_name) >= 0)
    )

    return filtered.groupby("patient_id").agg(
        pl.col(delta_time_col_name).min().alias("min_time_to_next_hba1c")
    )


if __name__ == "__main__":
    from psycop.projects.t2d.paper_outputs.selected_runs import get_best_eval_pipeline

    pipeline = get_best_eval_pipeline()
    eval_ds = pipeline.pipeline_outputs.get_eval_dataset()

    positive_predictions = (
        pl.from_pandas(
            pd.DataFrame(
                {
                    "pred": eval_ds.get_predictions_for_positive_rate(
                        desired_positive_rate=pipeline.paper_outputs.pos_rate
                    )[0],
                    "y": eval_ds.y,
                    "patient_id": eval_ds.ids,
                    "pred_timestamps": eval_ds.pred_timestamps,
                    "outcome_timestamps": eval_ds.outcome_timestamps,
                }
            )
        )
        .lazy()
        .filter(pl.col("pred") == 1)
        .filter(pl.col("y") == 1)
    )

    hba1cs = hba1c().rename({"dw_ek_borger": "patient_id", "timestamp": "timestamp"}, axis=1)
    hba1cs_with_fuzz = pl.from_pandas(hba1cs).with_columns(
        (pl.col("timestamp") + dt.timedelta(days=1)).alias("timestamp")
    )

    delta_time_df = time_from_first_pos_pred_to_next_hba1c(
        pos_preds=positive_predictions, hba1cs=hba1cs_with_fuzz.lazy()
    ).collect()

    description_df = delta_time_df.describe()

    print(delta_time_df)
