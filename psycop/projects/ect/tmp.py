# get prediction times after filtering
# add time of first ect (or all)
# check if any are within the last 3 years

import polars as pl

from psycop.projects.ect.feature_generation.cohort_definition.ect_cohort_definition import (
    ECTCohortDefiner,
)
from psycop.projects.ect.feature_generation.cohort_definition.outcome_specification.combined import (
    get_first_ect_indicator,
)

if __name__ == "__main__":
    pred_times = ECTCohortDefiner.get_filtered_prediction_times_bundle().prediction_times

    first_ect = pl.from_pandas(get_first_ect_indicator()).drop("cause").rename({"timestamp": "first_ect"})

    pred_times.frame.join(first_ect, on="dw_ek_borger").filter(pl.col("first_ect").is_not_null()).with_columns(
        (pl.col("first_ect") - pl.col("timestamp")).alias("timedelta")
    ).filter((pl.col("timedelta") < 0) & (pl.col("timedelta") > -pl.duration(days=365*3)))

