from collections.abc import Sequence
from typing import NewType

import pandas as pd
import polars as pl

from psycop.common.cohort_definition import OutcomeTimestampFrame, PredictionTimeFrame
from psycop.common.global_utils.cache import shared_cache
from psycop.common.model_evaluation.binary.time.timedelta_data import (
    get_sensitivity_by_timedelta_df,
)
from psycop.common.model_training.training_output.dataclasses import (
    get_predictions_for_positive_rate,
)
from psycop.projects.cvd.model_evaluation.single_run.single_run_artifact import (
    RunSelector,
    get_eval_df,
)

SensitivityByTTEDF = NewType("SensitivityByTTEDF", pl.DataFrame)


def add_dw_ek_borger(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("pred_time_uuid").str.split("-").list.first().cast(pl.Int64).alias("dw_ek_borger")
    )


@shared_cache.cache()
def sensitivity_by_time_to_event_model(
    run: RunSelector,
    pred_timestamps: PredictionTimeFrame,
    outcome_timestamps: OutcomeTimestampFrame,
    pprs: Sequence[float] = (0.01, 0.03, 0.05),
) -> SensitivityByTTEDF:
    eval_dataset = get_eval_df(run=run)

    # Add dw_ek_borger, extract from pred_time_uuid
    eval_dataset = (
        add_dw_ek_borger(eval_dataset)
        .join(pred_timestamps.stripped_df, on="dw_ek_borger", suffix="_pred")
        .join(outcome_timestamps.stripped_df, on="dw_ek_borger", suffix="_outcome")
    ).to_pandas()

    dfs = []
    for ppr in pprs:
        df = get_sensitivity_by_timedelta_df(
            y=eval_dataset.y,  # type: ignore
            y_hat=get_predictions_for_positive_rate(
                desired_positive_rate=ppr, y_hat_probs=eval_dataset.y_hat_prob
            )[0],
            time_one=eval_dataset["timestamp"],
            time_two=eval_dataset["timestamp_outcome"],
            direction="t2-t1",
            bins=range(0, 60, 6),
            bin_unit="M",
            bin_continuous_input=True,
            drop_na_events=True,
        )

        # Convert to string to allow distinct scales for color
        df["actual_positive_rate"] = str(ppr)
        dfs.append(df)

    plot_df = pd.concat(dfs)
    return SensitivityByTTEDF(pl.from_pandas(plot_df))
