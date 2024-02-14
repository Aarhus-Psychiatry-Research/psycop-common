import datetime as dt
import logging
from dataclasses import KW_ONLY, dataclass

import numpy as np

from psycop.common.feature_generation.loaders.raw.load_lab_results import ldl
from psycop.common.feature_generation.loaders.raw.load_structured_sfi import (
    smoking_categorical,
    smoking_continuous,
    systolic_blood_pressure,
)
from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    CVDCohortDefiner,
)
from psycop.timeseriesflattener.src.timeseriesflattenerv2.aggregators import (
    MaxAggregator,
    MeanAggregator,
)
from psycop.timeseriesflattener.src.timeseriesflattenerv2.feature_specs import (
    PredictionTimeFrame,
    PredictorSpec,
    ValueFrame,
)
from psycop.timeseriesflattener.src.timeseriesflattenerv2.flattener import Flattener

log = logging.getLogger(__file__)
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/&m/%d %H:%M:%S",
    )
    lookbehind_distances = [dt.timedelta(days=days) for days in range(30, 60)]
    aggregators = [MeanAggregator(), MaxAggregator()]

    filtered_prediction_times = (
        CVDCohortDefiner.get_filtered_prediction_times_bundle()
        .prediction_times.frame.rename({"timestamp": "pred_timestamp"})
        .lazy()
    )

    flattener = Flattener(
        predictiontime_frame=PredictionTimeFrame(
            init_df=filtered_prediction_times, entity_id_col_name="dw_ek_borger"
        ),
        lazy=False,
    )

    log.info("Starting flattening")
    df = flattener.aggregate_timeseries(
        specs=[
            PredictorSpec(
                value_frame=ValueFrame(
                    init_df=ldl().rename({"value": "ldl"}, axis=1),
                    value_col_name="ldl",
                    value_timestamp_col_name="timestamp",
                ),
                lookbehind_distances=lookbehind_distances,
                aggregators=aggregators,
                fallback=np.nan,
            ),
            PredictorSpec(
                value_frame=ValueFrame(
                    init_df=systolic_blood_pressure().rename(
                        {"value": "systolic_blood_pressure"}, axis=1
                    ),
                    value_col_name="systolic_blood_pressure",
                ),
                lookbehind_distances=lookbehind_distances,
                aggregators=aggregators,
                fallback=np.nan,
            ),
            PredictorSpec(
                value_frame=ValueFrame(
                    init_df=smoking_continuous().rename({"value": "smoking_continuous"}, axis=1),
                    value_col_name="smoking_continuous",
                ),
                lookbehind_distances=lookbehind_distances,
                aggregators=aggregators,
                fallback=np.nan,
            ),
            PredictorSpec(
                value_frame=ValueFrame(
                    init_df=smoking_categorical().rename({"value": "smoking_categorical"}, axis=1),
                    value_col_name="smoking_categorical",
                ),
                lookbehind_distances=lookbehind_distances,
                aggregators=aggregators,
                fallback=np.nan,
            ),
        ]
    ).df

    log.info("Finished flattening")

    pass
