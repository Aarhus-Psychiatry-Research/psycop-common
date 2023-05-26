import pandas as pd
import polars as pl
from psycop.projects.t2d.feature_generation.eligible_prediction_times.loader import (
    get_eligible_prediction_times_as_polars,
)
from psycop.projects.t2d.paper_outputs.selected_runs import BEST_EVAL_PIPELINE
from wasabi import Printer

if __name__ == "__main__":
    pipeline_inputs = BEST_EVAL_PIPELINE.inputs
    pos_rate = BEST_EVAL_PIPELINE.paper_outputs.pos_rate

    flattened_dataset = pl.concat(
        [
            pipeline_inputs.get_flattened_split_as_lazyframe(split=split)  # type: ignore
            for split in ["train", "test", "val"]
        ],
        how="vertical",
    )

    eval_dataset = BEST_EVAL_PIPELINE.pipeline_outputs.get_eval_dataset()
    eval_df = pl.from_pandas(
        pd.DataFrame(
            {
                "y": eval_dataset.y,
                "y_pred": eval_dataset.get_predictions_for_positive_rate(
                    desired_positive_rate=pos_rate
                )[0],
                "pred_time_uuids": eval_dataset.pred_time_uuids,
            }
        )
    )

    false_positives = eval_df.filter(((pl.col("y") == 0) & (pl.col("y_pred") == 1)))
    true_negatives = eval_df.filter(((pl.col("y") == 0) & (pl.col("y_pred") == 0)))
    hba1c_within_lookahead = flattened_dataset.select(pl.col("*hba1c*")).schema

    outcome_columns = flattened_dataset.select(pl.col("^outc_.*$")).columns

    for col_name in outcome_columns:
        lookahead_distance = int(col_name.split("within_")[1].split("_days")[0])

        outcomes = flattened_dataset.select(pl.col(col_name)).collect()
        total = outcomes.shape[0]

        n_positive = outcomes.filter(pl.col(col_name) == 1).shape[0]
        percent_positive = f"{round(n_positive / total*100, 1)}%"

        n_negative = (
            outcomes.select(pl.col(col_name)).filter(pl.col(col_name) == 0).shape[0]
        )
        percent_negative = f"{round(n_negative / total*100, 1)}%"

        msg.info(
            f"{lookahead_distance}: pos: {n_positive:,} ({percent_positive}) | neg: {n_negative:,} ({percent_negative})"
        )
