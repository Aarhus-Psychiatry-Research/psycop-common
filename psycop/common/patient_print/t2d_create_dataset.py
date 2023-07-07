import datetime
from dataclasses import dataclass

import polars as pl

from psycop.common.cohort_definition import CohortDefiner
from psycop.common.feature_generation.loaders.raw.load_lab_results import hba1c
from psycop.common.feature_generation.sequences.timeseries_windower.timeseries_windower import (
    window_timeseries,
)
from psycop.common.feature_generation.sequences.timeseries_windower.types.event_dataframe import (
    EventColumnNames,
    EventDataframeBundle,
)
from psycop.common.feature_generation.sequences.timeseries_windower.types.prediction_time_dataframe import (
    PredictiontimeColumnNames,
    PredictiontimeDataframeBundle,
)
from psycop.common.patient_print.healthprints_config import HEALTHPRINTS_DATASETS_DIR
from psycop.projects.t2d.feature_generation.cohort_definition.t2d_cohort_definer import (
    T2DCohortDefiner,
)


@dataclass(frozen=True)
class PredictionTimes:
    positive: pl.DataFrame
    negative: pl.DataFrame


def get_n_prediction_times_per_citizen(input_df: pl.DataFrame, n: int) -> pl.DataFrame:
    return input_df.groupby("dw_ek_borger").first().sample(n=n)


def get_healthprint_prediction_times(
    cohort_definer: CohortDefiner, n_per_class: int
) -> PredictionTimes:
    all_prediction_times = (
        cohort_definer.get_filtered_prediction_times_bundle().prediction_times
    )
    outcome_timestamps = cohort_definer.get_outcome_timestamps()

    prediction_times_with_outcome = all_prediction_times.join(
        outcome_timestamps, on="dw_ek_borger", how="left", suffix="_outcome"
    )

    positive_prediction_times = prediction_times_with_outcome.filter(
        (pl.col("timestamp") < pl.col("timestamp_outcome"))
        & (pl.col("source") == "lab_results")
    )
    negative_prediction_times = prediction_times_with_outcome.filter(
        (pl.col("timestamp_outcome").is_null())
    )

    return PredictionTimes(
        positive=get_n_prediction_times_per_citizen(
            positive_prediction_times, n=n_per_class
        ),
        negative=get_n_prediction_times_per_citizen(
            negative_prediction_times, n=n_per_class
        ),
    )


def z_score_normalise(
    input_df: pl.LazyFrame, value_col_name: str, type_col_name: str
) -> pl.LazyFrame:
    return input_df.with_columns(
        (
            (pl.col(value_col_name) - pl.col(value_col_name).mean())
            / (pl.std(value_col_name))
        ).over(type_col_name)
    )


if __name__ == "__main__":
    healthprint_prediction_times = get_healthprint_prediction_times(
        cohort_definer=T2DCohortDefiner(), n_per_class=1_000
    )

    hba1c_df = (
        pl.from_dataframe(hba1c())
        .lazy()
        .with_columns([pl.lit("hba1c").alias("type"), pl.lit("lab").alias("source")])
    ).rename({"timestamp": "event_timestamp"})

    event_bundles = [
        EventDataframeBundle(
            df=z_score_normalise(
                hba1c_df, value_col_name="value", type_col_name="type"
            ),
            cols=EventColumnNames(
                entity_id="dw_ek_borger",
                timestamp="event_timestamp",
                event_source="source",
                event_type="type",
                event_value="value",
            ),
        )
    ]

    lookbehind = datetime.timedelta(days=365 * 5)
    pred_time_columns = PredictiontimeColumnNames(
        entity_id="dw_ek_borger", timestamp="timestamp"
    )

    negative_windowed = window_timeseries(
        prediction_times_bundle=PredictiontimeDataframeBundle(
            df=healthprint_prediction_times.negative.lazy(),
            cols=pred_time_columns,
        ),
        event_bundles=event_bundles,
        lookbehind=lookbehind,
    )
    positive_windowed = window_timeseries(
        prediction_times_bundle=PredictiontimeDataframeBundle(
            df=healthprint_prediction_times.positive.lazy(),
            cols=pred_time_columns,
        ),
        event_bundles=event_bundles,
        lookbehind=lookbehind,
    )

    for ds in ("positive", "negative"):
        if ds == "positive":
            df, cols = positive_windowed.unpack()
        else:
            df, cols = negative_windowed.unpack()

        df.collect().write_parquet(HEALTHPRINTS_DATASETS_DIR / f"{ds}.parquet")

    pass
