import datetime as dt

import numpy as np

from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    CVDCohortDefiner,
)
from psycop.timeseriesflattener.src.timeseriesflattenerv2.flattener import Flattener

from ....common.feature_generation.loaders.raw.load_lab_results import ldl
from ....common.feature_generation.loaders.raw.load_structured_sfi import (
    smoking_categorical,
    smoking_continuous,
    systolic_blood_pressure,
)
from ....timeseriesflattener.src.timeseriesflattenerv2.aggregators import MeanAggregator
from ....timeseriesflattener.src.timeseriesflattenerv2.feature_specs import (
    PredictionTimeFrame,
    PredictorSpec,
    ValueFrame,
)

if __name__ == "__main__":
    lookbehind_distances = [dt.timedelta(days=90), dt.timedelta(days=365), dt.timedelta(days=730)]
    aggregators = [MeanAggregator()]

    df = (
        Flattener(
            predictiontime_frame=PredictionTimeFrame(
                df=CVDCohortDefiner.get_filtered_prediction_times_bundle().prediction_times.frame.lazy(),
                entity_id_col_name="dw_ek_borger",
            )
        )
        .aggregate_timeseries(
            specs=[
                PredictorSpec(
                    value_frame=ValueFrame(
                        df=ldl().rename({"value": "ldl"}, axis=1), value_col_name="ldl"
                    ),
                    lookbehind_distances=lookbehind_distances,
                    aggregators=aggregators,
                    fallback=np.nan,
                ),
                PredictorSpec(
                    value_frame=ValueFrame(
                        df=systolic_blood_pressure().rename(
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
                        df=smoking_continuous().rename({"value": "smoking_continuous"}, axis=1),
                        value_col_name="smoking_continuous",
                    ),
                    lookbehind_distances=lookbehind_distances,
                    aggregators=aggregators,
                    fallback=np.nan,
                ),
                PredictorSpec(
                    value_frame=ValueFrame(
                        df=smoking_categorical().rename({"value": "smoking_categorical"}, axis=1),
                        value_col_name="smoking_categorical",
                    ),
                    lookbehind_distances=lookbehind_distances,
                    aggregators=aggregators,
                    fallback=np.nan,
                ),
            ]
        )
        .df.collect()
    )
